from collections import Counter
import inspect
import os
import re
from types import ModuleType
from typing import Literal, cast

import libcst as cst
from black import format_str
from black.mode import Mode

from .analyzing_transformer import analyze_module
from .type_normalizer import is_trivial, normalize_type
from .base_transformer import BaseTransformer
from .sklearn_native import write_sklearn_native_stubs
from .traces import combine_types, init_trace_loader
from .utils import Sections, State, load_map, load_docstrings, load_type_maps, process_module


_tlmodule: str = ''  # top-level module; kludge added late that would be a pain to plumb through all the way

_total_return_annotations = 0  
_total_return_annotations_missing = 0
_total_param_annotations = 0  
_total_param_annotations_missing = 0
_total_attr_annotations = 0  
_total_attr_annotations_missing = 0
_union_returns = {}


def _get_code(n) -> str:
    return cst.Module(body=()).code_for_node(n)

class StubbingTransformer(BaseTransformer):
    def __init__(self, tlmodule: str, modname: str, fname: str, state: State,
        strip_defaults=False, 
        infer_types_from_defaults=True):
        super().__init__(modname, fname)
        self._state = state
        self._tlmodule = tlmodule
        assert(state.maps is not None)
        self._maps: Sections[dict[str, str]] = state.maps
        self._docstrings: Sections[dict[str,str]|dict[str,dict[str,str]]] = state.docstrings[modname]
        self._returnstrings = state.creturns
        self._strip_defaults: bool = strip_defaults
        self._infer_types: bool = infer_types_from_defaults
        self._method_names = set()
        self._local_class_names = set()
        self._need_imports: dict[str, str] = {} # map class -> module to import from
        self._ident_re = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
        self._class_re = re.compile(r'class [A-Za-z_][A-Za-z0-9_]*')
        self._trace_sigs = state.trace_sigs[modname] if modname in state.trace_sigs else {}
        self._noreturns = []
        self._dropped_decorators = set()  # Decorators that were dropped because they were not supported
        # We collect those just to log them in case we are missing something we should have kept.
        self._keep_imports = set()  # Things that were explicitly imported in the original source
        self._seen_attrs = set() # Attributes that were explicitly defined in the original source

    @staticmethod
    def get_value_type(node: cst.CSTNode) -> str|None:
        typ: str|None= None
        if isinstance(node, cst.Name):
            if node.value in [ 'True', 'False']:
                typ = 'bool'
            elif node.value == 'None':
                typ = 'None'
        elif isinstance(node, cst.SimpleString):
            typ = f'Literal[{node.value}]'
        else:
            for k, v in {
                cst.Integer: 'int',
                cst.Float: 'float',
                cst.Imaginary: 'complex',
                cst.BaseString: 'str',
                cst.BaseDict: 'dict',
                cst.BaseList: 'list',
                cst.BaseSlice: 'slice',
                cst.BaseSet: 'set',
                # TODO: check the next two
                cst.Lambda: 'Callable',
                cst.MatchPattern: 'pattern',
            }.items():
                if isinstance(node, k):
                    typ = v
                    break
        if typ is None:
            pass
        return typ

    def get_assign_value(self, node: cst.Assign|cst.AnnAssign) -> cst.BaseExpression:
        # See if this is an alias, in which case we want to
        # preserve the value; else we set the new value to ...
        new_value = None
        if isinstance(node.value, cst.Name) and not self.in_function():
            check = set()
            if self.at_top_level():
                check = self._local_class_names
            elif self.at_top_level_class_level(): # Class level
                check = self._method_names
            if node.value.value in check:
                new_value = node.value
        if new_value is None:
            new_value = cst.parse_expression("...")  
        return new_value

    def docstring2type(self, doctyp: str, map: dict[str, str], is_param: bool = False) -> \
            tuple[str | None, Literal['mapped', 'trivial', '']]:
        """
        Convert a docstring to a type, either by getting it from the corresponding map (if
        present), or by running it through the trivial type converter.
        Returns None if the type is non-trivial and not in the map.
        Also returns a label identifying how the conversion was done; useful for logging.
        """
        typ = None
        how = ''
        if doctyp in map:
            typ = map[doctyp]
            how = 'mapped'
        elif is_trivial(doctyp, self._modname):
            typ = normalize_type(doctyp, self._modname, is_param)
            how = 'trivial'
        return typ, how

    def update_imports(self, imports: dict[str, list[str]]):
        # imports has file -> list of classes
        # self._need_imports has class -> file
        for module, classlist in imports.items():
            for cls in classlist:
                self._need_imports[cls] = module

    def _get_new_annotation(self, which: Literal['attribute', 'parameter', 'return'], 
                            nodename: str, trace_context: str, doctyp: str|None, \
                            valtyp: str|None = None, is_classvar: bool = False) -> cst.Annotation | None:
        map = {
            'attribute': self._maps.attrs,
            'parameter': self._maps.params,
            'return': self._maps.returns,
        }[which]
        trace_annotation = None
        if trace_context in self._trace_sigs:
            _trace_sig = self._trace_sigs[trace_context]
            if _trace_sig is not None:
                if nodename:
                    p = _trace_sig.parameters.get(nodename, None)
                    if p is not None and p.annotation != inspect._empty:
                        trace_annotation = p.annotation
                else:
                    trace_annotation = _trace_sig.return_annotation

        typ = None
        if doctyp:
            if which != 'return':  # Return types are already handled in leaveFunctionDef
                typ, how = self.docstring2type(doctyp, map, which == 'parameter')
                if typ:
                    # If the default value is None, make sure we include it in the type
                    is_optional = 'None' in typ.split('|')
                    if not is_optional and valtyp == 'None' and typ != 'Any' and typ != 'None':
                        typ += '|None'
            else:
                typ = doctyp
        n = None

        try:
            typ, imports = combine_types(self._tlmodule, trace_annotation, typ, valtyp)
            if typ is None or (which != 'return' and typ == 'None'):
                if self._infer_types and valtyp and valtyp != 'None':
                    # Use the inferred type from default value as long as it is not None
                    if is_classvar:
                        valtyp = f'ClassVar[{valtyp}]'
                        self._need_imports['ClassVar'] = 'typing'
                    n = cst.Annotation(annotation=cst.parse_expression(valtyp))
                elif which != 'return':
                    print(f'Could not annotate {which} {trace_context}.{nodename} from {doctyp}; no mapping or default value')
            elif typ:
                self._need_imports.update(imports)
                if is_classvar:
                    typ = f'ClassVar[{typ}]'
                    self._need_imports['ClassVar'] = 'typing'
                n = cst.Annotation(annotation=cst.parse_expression(typ))
                print(f'Annotated {which} {trace_context}.{nodename} with {typ} from {doctyp} and trace {trace_annotation}')
            else:
                print(f'Could not annotate {which} {trace_context}.{nodename} from {doctyp}; no mapping')
        except:
            print(f'Could not annotate {which} {trace_context}.{nodename} with {typ} from {doctyp} and trace {trace_annotation}; fails to parse')
        return n

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode|cst.RemovalSentinel:
        # If not at top level or top-level class level, we don't want to do anything
        if not self.at_top_level() and not self.at_top_level_class_level():
            return cst.RemoveFromParent()
        
        value=self.get_assign_value(original_node)
        if  len(original_node.targets) > 1:
            # We don't handle multiple targets for stubbing for now.
            # TODO: fix this; see the augmenter
            return updated_node.with_changes(value=value)
                
        target = original_node.targets[0].target

        # If the assignment includes [] or similar we want to drop it.
        if not isinstance(target, (cst.Name, cst.Attribute)):
            return cst.RemoveFromParent()

        # Don't touch assignments to __all__
        if isinstance(target, cst.Name) and target.value == '__all__':
            return updated_node
        
        global _total_attr_annotations, _total_attr_annotations_missing
        try:
            attr_name: str = target.value # type: ignore
            self._seen_attrs.add(attr_name)  # keep track of those we have seen
            context = f'{self.context()}.{attr_name}'
            is_classvar = self.at_top_level_class_level()
            doctyp = cast(str|None, self._docstrings.attrs.get(context)) if is_classvar else None
            valtyp = StubbingTransformer.get_value_type(original_node.value)
            annotation = self._get_new_annotation('attribute', '', context, doctyp, valtyp, \
                                                  is_classvar = is_classvar)
            if annotation:
                node = cst.AnnAssign(target=target, annotation=annotation, value=value)
                _total_attr_annotations += 1
                return node
        except:
            pass
        _total_attr_annotations_missing += 1
        return updated_node.with_changes(value=value)

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.CSTNode:
        value=self.get_assign_value(original_node)
        if self.at_top_level_class_level():
            self._need_imports['ClassVar'] = 'typing'
            code = f'ClassVar[{_get_code(original_node.annotation.annotation)}]'
            return updated_node.with_changes(annotation=cst.Annotation(annotation=cst.parse_expression(code)), \
                                              value=value)
        else:
            return updated_node.with_changes(value=value)

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.CSTNode:
        param_context = self.context()
        super().leave_Param(original_node, updated_node)
        function_context = self.context()
        annotation = original_node.annotation
        default = original_node.default
        defaultvaltyp = None

        if default:
            defaultvaltyp = StubbingTransformer.get_value_type(default) # Inferred type from default
            if (not defaultvaltyp or self._strip_defaults):
                # Default is something too complex for a stub or should be stripped; replace with '...'
                default = cst.parse_expression("...")

        if not annotation:
            annotation = self._get_new_annotation('parameter', original_node.name.value, function_context,
                                                cast(str, self._docstrings.params.get(param_context)), 
                                                defaultvaltyp)
            
        global _total_param_annotations, _total_param_annotations_missing
        if annotation:
            _total_param_annotations += 1
            return updated_node.with_changes(annotation=annotation, default=default)
        
        _total_param_annotations_missing += 1
        return updated_node.with_changes(default=default)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Record the names of top-level classes
        if not self.in_class():
            self._local_class_names.add(node.name.value)
            self._seen_attrs = set()
        return super().visit_ClassDef(node)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        context = self.context()
        super().leave_ClassDef(original_node, updated_node)
        if not self.in_class(): # were we in a top-level class?
            # Clear the method name set
            self._method_names = set()

            # Add base classes to needed imports
            #for base in updated_node.bases:
            #    if isinstance(base.value, cst.Name):
            #        needed = base.value.value
            #        # We have to figure out where this comes from.
             #       # It could be in the existing imports, or the file
             #       # itself.

            # See if there are documented attributes that we haven't seen yet,
            # and add them to the class body.

            # TODO: is node cloning needed here?
            body = [n for n in updated_node.body.body if not isinstance(n, cst.Pass)]
            prefix = context + '.'  # Add '.' in case there are classes with same name prefix
            for attr, doctyp in self._docstrings.attrs.items():
                if attr.startswith(prefix):
                    attr_name = attr[len(context)+1:]
                    if attr_name not in self._seen_attrs:
                        global _total_attr_annotations, _total_attr_annotations_missing
                        dtyp, _ = self.docstring2type(cast(str, doctyp), self._maps.attrs, False)
                        # Need this for further normalizing and getting the imports
                        typ, imps = combine_types(self._tlmodule, None, dtyp, None)
                        if typ:
                            self._need_imports.update(imps)
                            node = cst.parse_statement(f'{attr_name}: {typ} = ...')
                            _total_attr_annotations += 1
                        else:
                            node = cst.parse_statement(f'{attr_name} = ...')
                            _total_attr_annotations_missing += 1
                        
                        body.insert(0, node)
            # If the class body is empty insert '...' statement
            if len(body) == 0:
                body.append(cst.parse_statement('...'))
            return updated_node.with_changes(body=cst.IndentedBlock(body=tuple(body))) # type: ignore
        else:
            # Nested class; return ...
            return cst.parse_statement('...')

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name = node.name.value
        is_private = name.startswith('_') and not name.startswith('__')
        if not is_private and self.at_top_level_class_level():
            # Record the method name
            self._method_names.add(name)
        rtn = super().visit_FunctionDef(node)
        return False if is_private else rtn
    
            # Handle known return types for dunder methods.
    dunders = {
            '__init__': 'None',
            '__len__':  'int',
            '__hash__': 'int',
            '__eq__':   'bool',
            '__ne__':   'bool',
            '__lt__':   'bool',
            '__le__':   'bool',
            '__gt__':   'bool',
            '__ge__':   'bool',
            '__bool__': 'bool',
            '__str__':  'str',
            '__repr__': 'str',
            '__format__':'str',
            '__new__':  'Self',
            '__bytes__':'bytes',
            '__setattr__':'None',
            '__delattr__':'None',
            '__dir__':  'list[str]',
        }


    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode|cst.RemovalSentinel:
        """Remove function bodies and add return type annotations """
        context = self.context()
        name = original_node.name.value
        ckey = f'{self._modname}.{context}'
        returnstring = self._returnstrings.get(ckey)
        if returnstring:
            doctyp = {"" : returnstring}
        else:
            doctyp = cast(dict[str,str], self._docstrings.returns.get(context))
        annotation = original_node.returns
        super().leave_FunctionDef(original_node, updated_node)
        if (name.startswith('_') and not name.startswith('__')) or self.in_function(): 
            # Nested or private function; return nothing.
            return cst.RemoveFromParent()
        
        # Are we in a class? If so, add the function name to the seen attributes.
        if self.in_class():
            self._seen_attrs.add(name)

        # Remove decorators that are not needed in stubs
        decorators = []
        keep = ['abstractmethod', 'classmethod', 'dataclass_transform', 'deprecated', 'final', 'override', 'property', 'setter', 'staticmethod']
        try:
            for d in original_node.decorators:
                if isinstance(d.decorator, cst.Attribute):
                    if d.decorator.attr.value in keep:
                        decorators.append(d.deep_clone())
                    else:
                        self._dropped_decorators.add(d.decorator.attr.value)
                elif isinstance(d.decorator, cst.Name):
                    if d.decorator.value in keep:
                        if d.decorator.value == 'property' and not doctyp:
                            # Property with no docstring; we may get it from class docstring
                            doctyp = cast(dict[str,str], self._docstrings.attrs.get(context))
                            if doctyp:
                                doctyp = {"" : doctyp}
                        decorators.append(d.deep_clone())
                    else:
                        self._dropped_decorators.add(d.decorator.value)
                elif isinstance(d.decorator, cst.Call):
                    if d.decorator.func.value in keep: # type: ignore
                        decorators.append(d.deep_clone())
                    elif isinstance(d.decorator.func, cst.Name):
                        self._dropped_decorators.add(d.decorator.func.value)
                    else:
                        self._dropped_decorators.add(d.decorator.func.value.value)  # type: ignore
        except:
            pass

        if not annotation or name in StubbingTransformer.dunders:
            rtntyp = None
            if name in StubbingTransformer.dunders:
                rtntyp = StubbingTransformer.dunders[name]
            elif doctyp:
                map = self._maps.returns
                v = []
                for t in doctyp.values():
                    typ, how = self.docstring2type(cast(str, t), map)
                    if typ:
                        v.append(typ)
                    else:
                        print(f'Could not annotate {context}-> from {doctyp}')
                        v = None
                        break
                if v:
                    if len(v) > 1:
                        rtntyp = 'tuple[' + ', '.join(v) + ']'
                    else:
                        rtntyp = v[0]
            else:
                pass
                # TODO: maybe one day....
                # Try to infer it from the body, or if this is an overload in a derived class,
                # infer from the base class.
                #rtntyp = self._infer_return_type(original_node.body)
                #if rtntyp:
                #    print(f'Inferred return type {rtntyp} for {context}')
                
            annotation = self._get_new_annotation('return', '', context, rtntyp)

        global _total_return_annotations, _total_return_annotations_missing
        if annotation:
            # Is this a union return? If so, keep track of these as they are
            # problematic and may need rework; e.g. with overloads
            is_union =  (isinstance(annotation.annotation, cst.BinaryOperation) and \
                isinstance(annotation.annotation.operator, cst.BitOr)) or \
            (isinstance(annotation.annotation, cst.Subscript) and \
             isinstance(annotation.annotation.value, cst.Name) and \
                annotation.annotation.value.value == 'Union')
            if is_union:
                _union_returns[f'{self._modname}.{context}'] = _get_code(annotation.annotation)
                pass

            _total_return_annotations += 1
            try:
                print(f'Annotating {self.context}-> as {annotation}')  
            except:
                # I have had cases where __repr__ recurses infinitely... :-(
                print(f'Annotating {self.context} (cannot print annotation)') 
            return updated_node.with_changes(body=cst.parse_statement("..."), returns=annotation, \
                                                decorators=decorators)
        _total_return_annotations_missing += 1    
        self._noreturns.append(ckey)
        # Remove the body only
        return updated_node.with_changes(body=cst.parse_statement("..."), decorators=decorators)

    def _get_module_name_from_node(self, n: cst.Attribute|cst.Name) -> str:
        if isinstance(n, cst.Attribute):
            return self._get_module_name_from_node(n.value) + '.' + n.attr.value  # type: ignore
        else:
            return n.value

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.CSTNode:
        # Save the "from foo impprt bar" imports for later use. We will remove them from
        # the tree here and then re-inject them later when we add all the necessary imports. 
        # The aim is to remove unused or duplicate imports.
        keep = [] # for "from foo import *""
        for node in updated_node.body:
            if isinstance(node, cst.ImportFrom):
                node = cast(cst.ImportFrom, node)
                #if not node.module:
                #    continue
                m = self._get_module_name_from_node(node.module) if node.module else ''
                if node.relative:
                    m = '.' * len(node.relative) + m
                if isinstance(node.names, cst.ImportStar):
                    keep.append(node)
                else:
                    for t in node.names:
                        if t.asname:
                            n = f'{t.name.value} as {t.asname.name.value}' # type: ignore
                        else:
                            n = t.name.value
                        n = cast(str, n)
                        self._need_imports[n] = m
                        if not n.startswith('_'):
                            # keep only the part after " as " if that is present
                            self._keep_imports.add(n[n.rfind(' ')+1:])

        newbody = [
            node
            for node in updated_node.body 
            if isinstance(node, cst.Assign) or isinstance(node, cst.AnnAssign) or isinstance(node, cst.Import) or node in keep
        ]
        return updated_node.with_changes(body=newbody)

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        """Remove everything from the body that is not an import,
        class def, function def, or assignment.
        """
        newbody = [
            node
            for node in updated_node.body
            if isinstance(node, cst.ClassDef) or \
               isinstance(node,cst.SimpleStatementLine) or \
              (isinstance(node, cst.FunctionDef) and not node.name.value.startswith('_'))
        ]
        return updated_node.with_changes(body=newbody)

    def leave_ImportAlias(
        self, original_node: cst.ImportAlias, updated_node: cst.ImportAlias
    ) -> cst.ImportAlias:
        # For __init__ files, change "import foo" to "import foo as foo",
        # which is needed in stubs.
        if self._fname.endswith('__init__.py') and original_node.asname is None and isinstance(original_node.name, cst.Name):
            return updated_node.with_changes(asname=cst.AsName(name=cst.Name(original_node.name.value)))
        else:
            return updated_node


_stub_folder = 'typings'
_dropped_decorators = set()


def make_imports_relative(m: str, module: str, fname: str) -> str:
    """Make a module name _module_ relative to the current file _fname_ in module _m_. """
    # We can import from same module but don't try import from the same file
    if m == module and (fname == '__init__.py' or fname.endswith(m + '.pyi')):
        return ''
    # Find the longest common prefix of m and module
    parts_m = m.split('.')
    parts_module = module.split('.')
    i = 0
    while i < min(len(parts_m), len(parts_module)):
        if parts_m[i] != parts_module[i]:
            break
        i += 1
    if i == 0:
        # Independent modules
        return module
    # For the parts not in common in m, add a '..'
    rel = '.' * (len(parts_m) - i)
    if fname.endswith('__init__.py'):
        rel += '.'
    rel += '.'.join(parts_module[i:])
    if rel == '':
        rel = '.'  # See if this fixes MinCovDet in covariance/_elliptic_envelope.py
    return rel
    

def patch_source(tlmodule: str, m: str, fname: str, source: str, state: State, strip_defaults: bool = False) -> str|None:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None

    patcher = StubbingTransformer(tlmodule, m, fname, state, strip_defaults=strip_defaults)
    modified = cstree.visit(patcher)
    modified_code = modified.code
    _dropped_decorators.update(patcher._dropped_decorators)

    with open(f'analysis/{tlmodule}.creturns.map.missing', 'a') as f:
        for context in patcher._noreturns:
            f.write(f'1#{context}#\n')

    # Get all the identifiers in the modified code using a regular expression.
    # We use this to tell if an import is actually used or we can drop it.
    # We can get false-positives from comments and docstrings.
    classes = set([x[6:] for x in patcher._class_re.findall(modified_code)]) 
    idents = set([x[x.rfind('.')+1:] for x in patcher._ident_re.findall(modified_code) if x not in classes]) 

    import_statements = ''
    for module in set(patcher._need_imports.values()):
        if module == 'typing':
            # We deal with typing separately below.
            continue

        ityps = []
        for k, v in patcher._need_imports.items():
            if not k:
                continue
            x = k.rfind(' ')
            t = k if x < 0 else k[x+1:]  # Use the asname when searching code for use.
            # We only retain imports if they are used in the code, or this is an __init__.py file,
            # or they were explict imports in the original code.
            if v == module:
                if t in idents:
                    ityps.append(k)
                elif k.startswith('_'):
                    pass
                elif t in patcher._keep_imports or fname.endswith('__init__.py'):
                    if k != t:
                        ityps.append(k)
                    else:  # re-export
                        ityps.append(f'{t} as {t}')
                    
        if len(ityps) > 0:
            relpath = make_imports_relative(m, module, fname)
            if relpath:
                import_statements += f'from {relpath} import {", ".join(ityps)}\n'
            else:
                pass

    # Add imports from typing. We don't require these to be qualified with "typing." in .map files so have
    # to handle them specially.
    # TODO: long term we should update any that are using old typing stuff like Union, Tuple, etc.
    typing_imports = ['Any', 'Callable', 'ClassVar', 'FileLike', 'IO', 'Iterable', 'Iterator', 'Literal', 'Mapping', 'NamedTuple', 
                      'Self', 'Sequence', 'SupportsIndex', 'Type', 'TypeVar',
                      'Dict', 'List', 'Optional', 'Set', 'Tuple', 'Union']
                      
    need = [t for t in typing_imports if t in idents]
    if need:
        import_statements = f'from typing import {", ".join(need)}\n' + import_statements
    
    code = import_statements + modified.code
    try:
        return format_str(code, mode=Mode()) 
    except Exception as e:
        print(f'Error formatting stub of {fname}: {e}')
        return code


def _stub(mod: ModuleType, m: str, fname: str, source: str, state: State, **kwargs) -> str|None:
    return patch_source(_tlmodule, m, fname, source, state, **kwargs)


def _targeter(fname: str) -> str:
    return f'{_stub_folder}/{fname[fname.find("/site-packages/") + 15 :]}i'


def stub_module(m: str, include_submodules: bool = True, strip_defaults: bool = False, skip_analysis: bool = False,
stub_folder: str = _stub_folder, trace_folder: str = "tracing") -> None:
    global _stub_folder, _tlmodule
    # TODO: I think the below is true; just being lazy for now and using an assert
    # to prove myself right. If so, come back to this and remove the assert,
    # and stop using the global _tlmodule. Eventually I want to get rid of
    # that gloabl everywhere.
    assert(_tlmodule == m.split('.')[0])
    _stub_folder = stub_folder
    if m.find('.') < 0:
        _tlmodule = m
    else:
        _tlmodule = m[:m.find('.')]
    init_trace_loader(trace_folder, _tlmodule)
    if skip_analysis:
        state = State(None, load_docstrings(_tlmodule), load_type_maps(_tlmodule), {}, {}, {}, {})
    else:   
        state = analyze_module(m, include_submodules=include_submodules)
    if state is not None:
        state.creturns.update(load_map(_tlmodule, 'creturns'))
        creturns = f'analysis/{_tlmodule}.creturns.map.missing'
        if os.path.exists(creturns):
            os.remove(creturns)

        process_module(m, state, _stub, _targeter, include_submodules=include_submodules,
            strip_defaults=strip_defaults)

    print(f'Annotated {_total_param_annotations} parameters, {_total_attr_annotations} attributes and {_total_return_annotations} returns')
    print(f'Failed to annotate {_total_param_annotations_missing} parameters, {_total_attr_annotations_missing} attributes and {_total_return_annotations_missing} returns')

    if _dropped_decorators:
        print(f'Dropped decorators:')
        for d in _dropped_decorators:
            print(f'  {d}')

    print(f'{len(_union_returns)} functions had union returns')
    for f, r in _union_returns.items():
        print(f'    {f}: {r}')
    with open(f'typings/{_tlmodule}/_typing.pyi', 'w') as f:
        f.write('''
# This file is generated by docs2stub. These types are intended
# to simplify the stubs generated by docs2stub. They are not
# intended to be used directly by other users.

import decimal
import io
import numpy.typing
import numpy as np
import pandas as pd
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.sparse import spmatrix
from typing import TypeAlias


Decimal: TypeAlias = decimal.Decimal
PythonScalar: TypeAlias = str | int | float | bool

ArrayLike: TypeAlias = numpy.typing.ArrayLike
MatrixLike: TypeAlias = np.ndarray | pd.DataFrame | spmatrix
FileLike: TypeAlias = io.IOBase
PathLike: TypeAlias = str
Int: TypeAlias = int | np.int8 | np.int16 | np.int32 | np.int64
Float: TypeAlias = float | np.float16 | np.float32 | np.float64

PandasScalar: TypeAlias = pd.Period | pd.Timestamp | pd.Timedelta | pd.Interval
Scalar: TypeAlias = PythonScalar | PandasScalar

Estimator: TypeAlias = BaseEstimator
Classifier: TypeAlias = ClassifierMixin
Regressor: TypeAlias = RegressorMixin

Color = tuple[float, float, float] | str

        ''')
    if m == 'sklearn':
        write_sklearn_native_stubs()
