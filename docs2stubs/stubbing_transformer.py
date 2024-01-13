from collections import Counter
import inspect
import logging
import os
import re
from types import ModuleType
from typing import Literal, cast, _type_repr

import libcst as cst
from black import format_str
from black.mode import Mode

from .analyzing_transformer import analyze_module
from .type_normalizer import is_trivial, normalize_type
from .base_transformer import BaseTransformer
from .sklearn_native import write_sklearn_native_stubs
from .traces import combine_types, init_trace_loader
from .utils import Sections, State, load_map, load_docstrings, load_type_maps, process_module


_total_return_annotations = 0  
_total_return_annotations_missing = 0
_total_param_annotations = 0  
_total_param_annotations_missing = 0
_total_attr_annotations = 0  
_total_attr_annotations_missing = 0
_union_returns = {}


def _get_code(n) -> str:
    return cst.Module(body=()).code_for_node(n)


def docstring2type(modname: str, doctyp: str, map: dict[str, str], is_param: bool = False) -> \
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
    elif is_trivial(doctyp, modname):
        typ = normalize_type(doctyp, modname, is_param)
        how = 'trivial'
    return typ, how
    
class StubbingTransformer(BaseTransformer):
    def __init__(self, tlmodule: str, modname: str, fname: str, state: State,
        strip_defaults=False):
        super().__init__(modname, fname)
        self._state = state
        self._tlmodule = tlmodule
        assert(state.maps is not None)
        self._maps: Sections[dict[str, str]] = state.maps
        self._docstrings: Sections[dict[str,str]|dict[str,dict[str,str]]] = state.docstrings[modname]
        self._returnstrings = state.creturns
        self._strip_defaults: bool = strip_defaults
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
        self._typevars = set()  # Typevars needed for Self returns

    def typevars(self) -> set[str]:
        return self._typevars
    
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

    def update_imports(self, imports: dict[str, list[str]]):
        # imports has file -> list of classes
        # self._need_imports has class -> file
        for module, classlist in imports.items():
            for cls in classlist:
                self._need_imports[cls] = module

    def _is_union(self, annotation: cst.Annotation) -> bool:
        return (isinstance(annotation.annotation, cst.BinaryOperation) and \
            isinstance(annotation.annotation.operator, cst.BitOr)) or \
        (isinstance(annotation.annotation, cst.Subscript) and \
            isinstance(annotation.annotation.value, cst.Name) and \
            annotation.annotation.value.value == 'Union')

    def _get_new_annotation(self, which: Literal['attribute', 'parameter', 'return'], 
                            context: str, param_name: str = '', doctyp: str|None = None, \
                            valtyp: str|None = None, is_classvar: bool = False) -> cst.Annotation | None:
        map = {
            'attribute': self._maps.attrs,
            'parameter': self._maps.params,
            'return': self._maps.returns,
        }[which]

        result: cst.Annotation|None = None # The annotation we return at the end of this
        reason = ''                   # Reason we succeeded or failed

        # A fair amount of the work here is giving good reasons for failure so we can
        # act on them. We also keep the various types used at different points in distinct
        # variables in case we want to leverage them in more detail later.
        # We have a bunch of possible sources we use here:
        # doctyp is the docstring type
        # valtyp is the inferred type from a default parameter value or attribute assignment
        maptyp: str|None = None       # mapped or trivially converted type from doctyp
        trace_annotation: type|None = None # Python type from Monkeytype trace
        tracetyp: str|None = None     # String representation of trace_annotation
        usedtyp: str|None = None      # What did we end up using? 
        is_warning = True

        if context in self._trace_sigs:
            _trace_sig = self._trace_sigs[context]
            if _trace_sig is not None:
                if param_name:
                    p = _trace_sig.parameters.get(param_name, None)
                    if p is not None and p.annotation != inspect._empty:
                        trace_annotation = p.annotation
                else:
                    trace_annotation = _trace_sig.return_annotation

                #if trace_annotation:
                #    tracetyp = _type_repr(trace_annotation)

        if doctyp:
            if which == 'return':  # Return types are already mapped/converted in leaveFunctionDef
                # TODO: see later if we can avoid special casing them here.
                maptyp = doctyp
            else:
                # Get the mapped (or trivially converted) version of the docstring
                maptyp, how = docstring2type(self._modname, doctyp, map, which == 'parameter')
                if maptyp:
                    # If the default value is None, make sure we include it in the type
                    is_optional = 'None' in maptyp.split('|')
                    if not is_optional and valtyp == 'None' and maptyp != 'Any' and maptyp != 'None':
                        maptyp += '|None'

        combtyp, imports = combine_types(self._tlmodule, context, 
                                         tracetyp=trace_annotation, doctyp=maptyp, valtyp=valtyp)

        if maptyp:
            reason += "mapped docstring"
        elif doctyp:
            reason += f'unmapped docstring <{doctyp}>'
        if trace_annotation:
            if reason:
                reason += " and "
            reason += "execution trace"

        if combtyp and (which == 'return' or combtyp != 'None'):
            try:
                if is_classvar:
                    combtyp = f'ClassVar[{combtyp}]'
                    self._need_imports['ClassVar'] = 'typing'
                result = cst.Annotation(annotation=cst.parse_expression(combtyp))
                usedtyp = combtyp
                self._need_imports.update(imports)
            except Exception as e:
                reason = f"failed to parse type <{combtyp}> from {reason} ({e})"
        elif which == 'return':
            if reason:
                if combtyp == 'None':
                    if maptyp:
                        if doctyp:
                            reason = f"type from mapped type {doctyp} -> {maptyp} would be None"
                        else:
                            reason = f"type from trace type {tracetyp} -> {maptyp} would be None"
                    elif doctyp:
                        # Probably not possible
                        reason = f"type from docstring {doctyp} would be None"
                    else:
                        reason = f"type from {reason} would be None"
                else:
                    reason = f"failed to combine {reason}" # I don't think this can happen
            else:
                reason = "no docstring or trace"
        elif valtyp == 'None':
            reason = f"no mapping for <{doctyp}>, no trace, and assigned value is None"
        elif valtyp:
            try:
                if is_classvar:
                    valtyp = f'ClassVar[{valtyp}]'
                    self._need_imports['ClassVar'] = 'typing'
                maptyp = valtyp
                result = cst.Annotation(annotation=cst.parse_expression(valtyp))
                reason = "inferred from assignment"
                usedtyp = valtyp
            except Exception as e:
                reason = f"failed to parse inferred type <{valtyp}> from assignment ({e})"  # I don't think this can happen
        elif doctyp is None:
            reason = "no docstring, trace or assigned value"
            is_warning = False  # just log at info level
        else:
            reason = f"no mapping for <{doctyp}>, trace or assigned value"

        # We put the reason before the context in the failed messages so we can sort these
        #  messages and easily cluster reason types
        if usedtyp:
            logging.debug(f'Annotated {which} {context}.{param_name} with {usedtyp}: {reason}')
        elif is_warning:
            logging.warning(f'Failed to annotate {which}: {reason} [{context}.{param_name}]')
        else:
            logging.info(f'Failed to annotate {which}: {reason} [{context}.{param_name}]')

        if result and which == "return" and self._is_union(result):
            # Keep track of union returns as they are
            # problematic and may need rework; e.g. with overloads
            _union_returns[f'{self._modname}.{context}'] = usedtyp 
        return result

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
        if isinstance(target, cst.Name):
            attr_name: str = target.value
            if not isinstance(target, cst.Name):
                pass
            self._seen_attrs.add(attr_name)  # keep track of those we have seen
            context = f'{self.context()}.{attr_name}'
            is_classvar = self.at_top_level_class_level()
            doctyp = cast(str|None, self._docstrings.attrs.get(context)) if is_classvar else None
            valtyp = StubbingTransformer.get_value_type(original_node.value)
            annotation = self._get_new_annotation('attribute', context, doctyp=doctyp, valtyp=valtyp, \
                                                  is_classvar = is_classvar)
            if annotation:
                node = cst.AnnAssign(target=target, annotation=annotation, value=value)
                _total_attr_annotations += 1
                return node
            
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
            annotation = self._get_new_annotation('parameter', function_context, param_name=original_node.name.value,
                                                doctyp=cast(str, self._docstrings.params.get(param_context)), 
                                                valtyp=defaultvaltyp)
            
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

            # See if there are documented attributes that we haven't seen yet,
            # and add them to the class body.
            body = [n for n in updated_node.body.body if not isinstance(n, cst.Pass)]
            global _total_attr_annotations, _total_attr_annotations_missing
            prefix = context+'.'  # Add '.' in case there are classes with same name prefix
            for attr, doctyp in self._docstrings.attrs.items():
                if not attr.startswith(prefix):
                    continue
                attr_name = attr[len(context)+1:]
                if attr_name in self._seen_attrs:
                    continue
                dtyp, _ = docstring2type(self._modname, cast(str, doctyp), self._maps.attrs, False)
                # Need this for further normalizing and getting the imports
                typ, imps = combine_types(self._tlmodule, context, doctyp=dtyp)
                if typ:
                    self._need_imports.update(imps)
                    stmt = f'{attr}: {typ} = ...'
                    _total_attr_annotations += 1
                else:
                    stmt = f'{attr} = ...'
                    _total_attr_annotations_missing += 1
                body.insert(0, cst.parse_statement(stmt))

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
            doctyp: dict[str, str]|None = cast(dict[str,str]|None, self._docstrings.returns.get(context))
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
        keep = ['abstractmethod', 'classmethod', 'dataclass_transform', 'deprecated', \
                'final', 'override', 'property', 'setter', 'staticmethod']
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
                            attrtyp: str = cast(str, self._docstrings.attrs.get(context))
                            assert(isinstance(attrtyp, str))
                            if attrtyp:
                                doctyp = {"" : attrtyp}
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

        self_annotation = None

        if not annotation or name in StubbingTransformer.dunders:
            rtntyp: str|None = None
            if name in StubbingTransformer.dunders:
                rtntyp = StubbingTransformer.dunders[name]
            elif doctyp:
                # We have the docstrings; still need to map them
                map = self._maps.returns
                v = []
                for n, t in doctyp.items():
                    typ, _ = docstring2type(self._modname, cast(str, t), map)

                    if typ:
                        if n == 'self':
                            # if typ is Any (probably came from object) or the containing class, this
                            # is almost certainly a Self return.
                            if typ == 'Any' or context.split('.')[0] == typ.split('.')[-1]:
                                typ = 'Self'
                            # else TODO: see what other cases of Self we can catch here
                        v.append(typ)
                    else:
                        if n == 'self':
                            pass
                        logging.warning(f'Could not get {context} return type from docstring {t}; no mapping and not trivial')
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

            # Special workaround for Self for now. Self is only in 3.11, and many people aren't using 3.11 yet. So we have to introduce 
            # typevars instead for now. There's a slight risk of a name collision here.
            if rtntyp == 'Self':
                containing_class = context.split('.')[0]
                if name == '__new__':
                    # Don't use Self here anyway
                    rtntyp = containing_class
                else:
                    # Create a typevar we can both return and use to annotate the self parameter.
                    rtntyp = f'{containing_class}_Self'
                    self_annotation = cst.Annotation(annotation=cst.parse_expression(rtntyp))
                    typevar = f'{rtntyp} = TypeVar("{rtntyp}", bound="{containing_class}")'
                    self._typevars.add(typevar)

            annotation = self._get_new_annotation('return', context, doctyp=rtntyp)

        global _total_return_annotations, _total_return_annotations_missing
        if annotation:
            _total_return_annotations += 1
            if self_annotation:
                # We have to change the annotation on self to be the typevar
                newparams = []
                for p in updated_node.params.params:
                    if newparams:
                        newparams.append(p)
                    else:
                        newparams.append(p.with_changes(annotation=self_annotation))
                return updated_node.with_changes(body=cst.parse_statement("..."), returns=annotation, \
                                                decorators=decorators, 
                                                params=updated_node.params.with_changes(params=newparams))
            
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
        # Save the "from foo import bar" imports for later use. We will remove them from
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
    

def _stub_python_module(m: str, fname: str, source: str, state: State, strip_defaults: bool = False) -> str|None:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None

    tlmodule = m.split('.')[0]
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
    typevars = list(patcher.typevars())
    if typevars:
        need.append('TypeVar')
        typevars.append('\n')  # make sure we have a \n after last typevar

    if need:
        import_statements = f'from typing import {", ".join(need)}\n' + import_statements
    
    code = import_statements + '\n'.join(typevars) + modified.code
    try:
        return format_str(code, mode=Mode()) 
    except Exception as e:
        logging.error(f'Error formatting stub of {fname}: {e}')
        return code


def _stub_native_module(mod: ModuleType, module_name: str, file_name: str, state: State, **kwargs) -> str|None:
    # Right now this doesn't work so just return ''
    # During analysis we get the docstrings by importing and then inspecting the module.
    # That can't distinguish between imported functions/classes and those defined in the module.
    # Plus we completely skip anything that doesn't have a docstring.
    # There's probably no good alternative here to writing a parser for .pyx files.
    return ''

    tlmodule = module_name.split('.')[0]
    result = []
    docstrings: Sections[dict[str,str]|dict[str,dict[str,str]]] 
    returnstrings = state.creturns

    def _handle_native_func(func, context: str) -> None:
        sig = inspect.signature(func)
        return_type = returnstrings.get(context, '')
        args = [f"{param.name}: {param.annotation.__name__ if hasattr(param.annotation, '__name__') else param.annotation}" 
                for param in sig.parameters.values()]
        if return_type:
            result.append(f"def {name}({', '.join(args)}) -> {return_type}: ")
        else:
            result.append(f"def {name}({', '.join(args)}): ")
        if func.__doc__ is not None:
            result.append(f" # {func.__doc__}")
        result.append("\n    ...\n\n")
    

    assert(state.maps is not None)
    for name, obj in inspect.getmembers(mod):
        if name.startswith('_'):
            continue
        if not inspect.isbuiltin(obj):
            continue

        if inspect.isclass(obj):
            result.append(f"class {name}:\n") # TODO: base classes
            body = '    ...\n'

            # Collect this class's attributes from its docstring
            attribs = {attr[len(name)+1:]: doctyp 
                       for attr, doctyp in docstrings.attrs.items() 
                       if attr.startswith(name+'.')}

            for member_name, member in inspect.getmembers(obj):
                if inspect.isfunction(member):
                    _handle_native_func(member, f'{name}.{member_name}')
                    body = ''
                else:
                    # Attribute
                    doctyp = attribs.get(member_name, '')
                    dtyp, _ = docstring2type(module_name, cast(str, doctyp), state.maps.attrs, False)
                    # Need this for further normalizing and getting the imports
                    typ, imps = combine_types(tlmodule, name, doctyp=dtyp)
                    global _total_attr_annotations, _total_attr_annotations_missing
                    if typ:
                        result.append(f'    {member_name}: {typ} = ...')
                        _total_attr_annotations += 1
                    else:
                        result.append(f"    {member_name}: {type(member).__name__} = ...")
                        _total_attr_annotations_missing += 1
                    body = ''
            result.append(body)
            
        elif inspect.isfunction(obj):
            # top-level function
            _handle_native_func(obj, name)
        elif obj:
            # Global variable
            result.append(f"{name}: {type(obj).__name__}")

    # Insert imports

    result.insert(0, f"# Type stub file for module {module_name}\n\n")
    return ''.join(result)


def _targeter(fname: str) -> str:
    return f'{_stub_folder}/{fname[fname.find("/site-packages/") + 15 :]}i'


def stub_module(m: str, strip_defaults: bool = False, skip_analysis: bool = False,
stub_folder: str = _stub_folder, trace_folder: str = "tracing") -> None:
    global _stub_folder

    logging.basicConfig(level=logging.WARNING)
    _stub_folder = stub_folder
    if m.find('.') < 0:
        tlmodule = m
    else:
        tlmodule = m[:m.find('.')]
    init_trace_loader(trace_folder, tlmodule)
    if skip_analysis:
        state = State(None, load_docstrings(tlmodule), load_type_maps(tlmodule), {}, {}, {}, {})
    else:   
        state = analyze_module(m)
    if state is not None:
        state.creturns.update(load_map(tlmodule, 'creturns'))
        creturns = f'analysis/{tlmodule}.creturns.map.missing'
        if os.path.exists(creturns):
            os.remove(creturns)

        process_module("Stubbing", m, state, _stub_python_module, _stub_native_module, _targeter,
                       strip_defaults=strip_defaults)
 
    print(f'Annotated {_total_param_annotations} parameters, {_total_attr_annotations} attributes and {_total_return_annotations} returns')
    print(f'Failed to annotate {_total_param_annotations_missing} parameters, {_total_attr_annotations_missing} attributes and {_total_return_annotations_missing} returns')

    if _dropped_decorators:
        print('Dropped decorators:')
        for d in _dropped_decorators:
            print(f'  {d}')

    print(f'{len(_union_returns)} functions had union returns')
    for f, r in _union_returns.items():
        print(f'    {f}: {r}')
    with open(f'typings/{tlmodule}/_typing.pyi', 'w') as f:
        f.write('''
# This file is generated by docs2stub. These types are intended
# to simplify the stubs generated by docs2stub. They are not
# intended to be used directly outside of this package.

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
