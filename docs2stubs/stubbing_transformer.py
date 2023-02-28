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
from .traces import combine_types, get_toplevel_function_signature, init_trace_loader
from .utils import Sections, State, load_map, load_docstrings, load_type_maps, process_module


_tlmodule: str = ''  # top-level module; kludge added late that would be a pain to plumb through all the way

class StubbingTransformer(BaseTransformer):
    def __init__(self, tlmodule: str, modname: str, fname: str, state: State,
        strip_defaults=False, 
        infer_types_from_defaults=False):
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
        self._ident_re = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)')
        self._trace_sigs = state.trace_sigs[modname] if modname in state.trace_sigs else {}
        self._noreturns = []

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

    def get_assign_props(self, node: cst.Assign) -> tuple[str|None, cst.BaseExpression]:
         typ = StubbingTransformer.get_value_type(node.value)
         value=self.get_assign_value(node)
         return typ, value

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        # If this is an __all__ assignment, we want to preserve it
        if len(original_node.targets) == 1:
            target0 = original_node.targets[0].target
            if isinstance(target0, cst.Name) and target0.value == '__all__':
                return updated_node
        
        typ, value = self.get_assign_props(original_node)
        # Make sure the assignment was not to a tuple before
        # changing to AnnAssign
        # TODO: if this is an attribute, see if it had an annotation in 
        # the class docstring and use that
        if typ is not None and len(original_node.targets) == 1:
            try:
                return cst.AnnAssign(target=original_node.targets[0].target,
                    annotation=cst.Annotation(annotation=cst.Name(typ)),
                    value=value)
            except:
                pass
        return updated_node.with_changes(value=value)

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.CSTNode:
        value=self.get_assign_value(original_node)
        return updated_node.with_changes(value=value)

    def fixtype(self, doctyp: str, map: dict[str, str], is_param: bool = False) -> \
            tuple[str | None, dict[str, list[str]], Literal['mapped', 'trivial', '']]:
        typ = None
        imports = {}
        how = ''
        if doctyp in map:
            # We still call the normalizer, just to get the imports, althought there is a chance
            # these could now be wrong
            typ = map[doctyp]
            _, imports = normalize_type(typ, self._modname, self._state.imports, is_param)
            how = 'mapped'
        elif is_trivial(doctyp, self._modname, self._state.imports):
            typ, imports = normalize_type(doctyp, self._modname, self._state.imports, is_param)
            how = 'trivial'   
        return typ, imports, how

    def update_imports(self, imports: dict[str, list[str]]):
        # imports has file -> list of classes
        # self._need_imports has class -> file
        for module, classlist in imports.items():
            for cls in classlist:
                self._need_imports[cls] = module


    def _get_new_annotation(self, nodename: str, trace_context: str, doctyp: str|None, \
                            valtyp: str|None = None):
        which, isparam = ("parameter", True) if nodename else ("function", False)
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
            typ, imports, how = self.fixtype(doctyp, self._maps.params, isparam)
            if typ:
                if imports:
                    self.update_imports(imports)

                # If the default value is None, make sure we include it in the type
                is_optional = 'None' in typ.split('|')
                if not is_optional and valtyp == 'None' and typ != 'Any' and typ != 'None':
                    typ += '|None'

        n = None

        try:
            typ, imports = combine_types(self._tlmodule, trace_annotation, typ, valtyp)
            if typ is None or (isparam and typ == 'None'):
                if self._infer_types and valtyp and valtyp != 'None':
                    # Use the inferred type from default value as long as it is not None
                    annotation = cst.Annotation(annotation=cst.Name(valtyp))
                elif isparam:
                    print(f'Could not annotate {which} {trace_context}.{nodename} from {doctyp}; no mapping or default value')
            elif typ is None:
                print(f'Could not annotate {which} {trace_context}.{nodename} from {doctyp}; no mapping')
            else:
                for k, v in imports:
                    if v == 'np':
                        v = 'numpy'
                    self._need_imports[k] = v
                n = cst.Annotation(annotation=cst.parse_expression(typ))
                print(f'Annotated {which} {trace_context}.{nodename} with {typ} from {doctyp} and trace {trace_annotation}')
        except:
            print(f'Could not annotate {which} {trace_context}.{nodename} with {typ} from {doctyp} and trace {trace_annotation}; fails to parse')
        return n

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
            annotation = self._get_new_annotation(original_node.name.value, function_context,
                                                cast(str, self._docstrings.params.get(param_context)), 
                                                defaultvaltyp)
            if annotation:
                return updated_node.with_changes(annotation=annotation, default=default)
        return updated_node.with_changes(default=default)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Record the names of top-level classes
        if not self.in_class():
            self._local_class_names.add(node.name.value)
        return super().visit_ClassDef(node)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        super().leave_ClassDef(original_node, updated_node)
        if not self.in_class():
            # Clear the method name set
            self._method_names = set()
            # Check if the class has an empty body and insert '...' statement if so
            if len(original_node.body.body) == 0:
                newbody = updated_node.body.with_changes(body=[cst.parse_statement('...')])
                return updated_node.with_changes(body=newbody)
            return updated_node
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
        
        # Remove decorators that are not needed in stubs
        decorators = []
        keep = ['abstractmethod', 'classmethod', 'dataclass_transform', 'deprecated', 'final', 'override', 'property', 'staticmethod']
        try:
            for d in original_node.decorators:
                if isinstance(d.decorator, cst.Attribute) and d.decorator.attr.value in keep:
                    decorators.append(d.deep_clone())
                elif isinstance(d.decorator, cst.Name) and d.decorator.value in keep:
                    decorators.append(d.deep_clone())
                elif isinstance(d.decorator, cst.Call) and d.decorator.func.value in keep: # type: ignore
                     decorators.append(d.deep_clone())
        except:
            pass

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

        if not annotation:
            rtntyp = None
            if doctyp:
                map = self._maps.returns
                v = []
                for t in doctyp.values():
                    typ, imp, how = self.fixtype(t, map)
                    if typ:
                        v.append(typ)
                        self.update_imports(imp)
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
                # If its a dunder method we can make a good guess in most cases
                if name in dunders:
                    rtntyp = dunders[name]
                else:
                    pass
                    # TODO: maybe one day....
                    # Try to infer it from the body, or if this is an overload in a derived class,
                    # infer from the base class.
                    #rtntyp = self._infer_return_type(original_node.body)
                    #if rtntyp:
                    #    print(f'Inferred return type {rtntyp} for {context}')
                
            annotation = self._get_new_annotation('', context, rtntyp)

        if annotation:
            print(f'Annotating {self.context}-> as {annotation}')   
            return updated_node.with_changes(body=cst.parse_statement("..."), returns=annotation, \
                                                decorators=decorators)
            
        self._noreturns.append(ckey)
        # Remove the body only
        return updated_node.with_changes(body=cst.parse_statement("..."), decorators=decorators)

    def _get_module_name_from_node(self, n: cst.Attribute|cst.Name) -> str:
        if isinstance(n, cst.Attribute):
            return self._get_module_name_from_node(n.value) + '.' + n.attr.value
        else:
            return n.value

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.CSTNode:
        # Save the imports for later use
        keep = [] # for "from foo import *""
        for node in updated_node.body:
            if isinstance(node, cst.ImportFrom):
                node = cast(cst.ImportFrom, node)
                if not node.module:
                    continue
                m = self._get_module_name_from_node(node.module)
                if node.relative:
                    m = '.' * len(node.relative) + m
                if isinstance(node.names, cst.ImportStar):
                    keep.append(node)
                else:
                    for t in node.names:
                        if t.asname:
                            self._need_imports[f'{t.name.value} as {t.asname.name.value}'] = m
                        else:
                            self._need_imports[t.name.value] = m

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
        # If the containing file is an __init__.py, then we need to
        # then we need to add the alias to the import statement if not
        # present.
        if self._fname.endswith('__init__.py') and original_node.asname is None and isinstance(original_node.name, cst.Name):
            return updated_node.with_changes(asname=cst.AsName(name=cst.Name(original_node.name.value)))
        else:
            return updated_node


def make_imports_relative(m: str, module: str, fname: str) -> str:
    """Make a module name _module_ relative to the current file _fname_ in module _m_. """
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
    return rel
    
         
def patch_source(tlmodule: str, m: str, fname: str, source: str, state: State, strip_defaults: bool = False) -> str|None:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None

    patcher = StubbingTransformer(tlmodule, m, fname, state, strip_defaults=strip_defaults)
    modified = cstree.visit(patcher)
    modified_code = modified.code

    with open(f'analysis/{tlmodule}.creturns.map.missing', 'a') as f:
        for context in patcher._noreturns:
            f.write(f'1#{context}#\n')
                    
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
            if v == module and modified_code.find(t) >= 0:
                ityps.append(k)

        if module == 'np':
            module = 'numpy'
        elif module == 'pd':
            module = 'pandas'

        if len(ityps) > 0:
            # TODO: check if the file exists and if not, comment out the import. Note 
            # we must be careful here as the file may not have been created yet. May
            # need to check against site-packages instead of typings. This is mostly
            # to remove imports from Cython modules. 
            relpath = make_imports_relative(m, module, fname)
            if relpath:
                import_statements += f'from {relpath} import {", ".join(ityps)}\n'
            else:
                pass

    # TODO: long term we should update any that are using old typing stuff like Union, Tuple, etc.
    typing_imports = ['Any', 'Callable', 'FileLike', 'IO', 'Iterable', 'Iterator', 'Literal', 'Mapping', 'NamedTuple', 'Sequence', 'Type', 'TypeVar',
                      'Dict', 'List', 'Optional', 'Set', 'Tuple', 'Union']
                      
    need = [t for t in typing_imports if modified_code.find(t) >= 0]
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


_stub_folder = 'typings'


def _targeter(fname: str) -> str:
    return f'{_stub_folder}/{fname[fname.find("/site-packages/") + 15 :]}i'


def stub_module(m: str, include_submodules: bool = True, strip_defaults: bool = False, skip_analysis: bool = False,
stub_folder: str = _stub_folder, trace_folder: str = "tracing") -> None:
    global _stub_folder, _tlmodule
    _stub_folder = stub_folder
    _tlmodule = m
    init_trace_loader(trace_folder, m)
    imports = load_map(m, 'imports')
    if skip_analysis:
        state = State(None, imports, load_docstrings(m), load_type_maps(m), {}, {}, {}, {})
    else:   
        state = analyze_module(m, include_submodules=include_submodules)
    if state is not None:
        imports = state.imports
        if imports:
            # Add any extra imports paths found in the imports file
            for k, v in imports.items():
                if k not in state.imports:
                    state.imports[k] = v

        state.creturns.update(load_map(m, 'creturns'))
        creturns = f'analysis/{m}.creturns.map.missing'
        if os.path.exists(creturns):
            os.remove(creturns)

        process_module(m, state, _stub, _targeter, include_submodules=include_submodules,
            strip_defaults=strip_defaults)

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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from typing import TypeAlias


Decimal: TypeAlias = decimal.Decimal
PythonScalar: TypeAlias = str | int | float | bool

ArrayLike: TypeAlias = numpy.typing.ArrayLike
MatrixLike: TypeAlias = np.ndarray | pd.DataFrame 
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