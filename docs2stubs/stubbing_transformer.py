from collections import Counter
import inspect
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


class StubbingTransformer(BaseTransformer):
    def __init__(self, modname: str, fname: str, state: State,
        strip_defaults=False, 
        infer_types_from_defaults=False):
        super().__init__(modname, fname)
        self._state = state
        assert(state.maps is not None)
        self._maps: Sections[dict[str, str]] = state.maps
        self._docstrings: Sections[dict[str,str]|dict[str,dict[str,str]]] = state.docstrings[modname]
        self._strip_defaults: bool = strip_defaults
        self._infer_types: bool = infer_types_from_defaults
        self._method_names = set()
        self._local_class_names = set()
        self._need_imports: dict[str, str] = {} # map class -> module to import from
        self._ident_re = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)')
        self._trace_sigs = state.trace_sigs[modname] if modname in state.trace_sigs else {}

    @staticmethod
    def get_value_type(node: cst.CSTNode) -> str|None:
        typ: str|None= None
        if isinstance(node, cst.Name):
            if node.value in [ 'True', 'False']:
                typ = 'bool'
            elif node.value == 'None':
                typ = 'None'
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
                cst.MatchPattern: 'pattern'
            }.items():
                if isinstance(node, k):
                    typ = v
                    break
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
        typ = StubbingTransformer.get_value_type(original_node.value)
        # Make sure the assignment was not to a tuple before
        # changing to AnnAssign
        # TODO: if this is an attribute, see if it had an annotation in 
        # the class docstring and use that
        if typ is not None and len(original_node.targets) == 1:
            return cst.AnnAssign(target=original_node.targets[0].target,
                annotation=cst.Annotation(annotation=cst.Name(typ)),
                value=value)
        else:
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


    def _get_new_annotation(self, which: str, nodename: str, trace_context: str, doctyp: str|None, \
                            valtyp: str|None = None):
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
            typ, imports, how = self.fixtype(doctyp, self._maps.params, True)
            if typ:
                if imports:
                    self.update_imports(imports)

                # If the default value is None, make sure we include it in the type
                is_optional = 'None' in typ.split('|')
                if not is_optional and valtyp == 'None' and typ != 'Any' and typ != 'None':
                    typ += '|None'

        n = None
        if trace_annotation is None:
            if typ == 'None':
                print(f'Could not annotate {which} {trace_context}.{nodename} from {doctyp}; would be None')
            elif typ is None:
                if self._infer_types and valtyp and valtyp != 'None':
                    # Use the inferred type from default value as long as it is not None
                    annotation = cst.Annotation(annotation=cst.Name(valtyp))
                else:
                    print(f'Could not annotate {which} {trace_context}.{nodename} from {doctyp}; no mapping')
            else:
                try:
                    n = cst.Annotation(annotation=cst.parse_expression(typ))
                    print(f'Annotated {which} {trace_context}.{nodename} with {typ} from doctyp {doctyp}')
                except:
                    print(f'Could not annotate {which} {trace_context}.{nodename} with {typ} from doctyp {doctyp}; fails to parse')
        else:
            try:
                typ = combine_types(trace_annotation, typ)
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
        valtyp = None

        if default:
            valtyp = StubbingTransformer.get_value_type(default) # Inferred type from default
            if (not valtyp or self._strip_defaults):
                # Default is something too complex for a stub or should be stripped; replace with '...'
                default = cst.parse_expression("...")

        if not annotation:
            annotation = self._get_new_annotation("parameter", original_node.name.value, function_context,
                                                   cast(str, self._docstrings.params.get(param_context)), valtyp)
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
        doctyp = cast(dict[str,str], self._docstrings.returns.get(context))
        annotation = original_node.returns
        super().leave_FunctionDef(original_node, updated_node)
        if (name.startswith('_') and not name.startswith('__')) or self.in_function(): 
            # Nested or private function; return nothing (TODO: could this be None?)
            return cst.RemoveFromParent()

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

            annotation = self._get_new_annotation("function", '', context, rtntyp)
            if annotation:
                print(f'Annotating {self.context}-> as {annotation}')   
                return updated_node.with_changes(body=cst.parse_statement("..."), returns=annotation) 
            
        # Remove the body only
        return updated_node.with_changes(body=cst.parse_statement("..."))

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.CSTNode:
        newbody = [
            node
            for node in updated_node.body
            if any(
                isinstance(node, cls)
                for cls in [cst.Assign, cst.AnnAssign, cst.Import, cst.ImportFrom]
            )
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
    
         
def patch_source(m: str, fname: str, source: str, state: State, strip_defaults: bool = False) -> str|None:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None

    patcher = StubbingTransformer(m, fname, state, strip_defaults=strip_defaults)
    modified = cstree.visit(patcher)

    import_statements = ''
    #print(f"Need imports: {patcher._need_imports}")
    for module in set(patcher._need_imports.values()):
        ityps = []
        for k, v in patcher._need_imports.items():
            if v == module:
                ityps.append(k)
        # TODO: make these relative imports if appropriate
        import_statements += f'from {module} import {",".join(ityps)}\n'
        
    import_statements += f'from {m[:m.find(".")]}._typing import Int, Float, MatrixLike\n\n'
    #print(f"Add imports: {import_statements}")

    code = import_statements + modified.code
    return format_str(code, mode=Mode()) 


def _stub(mod: ModuleType, m: str, fname: str, source: str, state: State, **kwargs) -> str|None:
    return patch_source(m, fname, source, state, **kwargs)


_stub_folder = 'typings'


def _targeter(fname: str) -> str:
    return f'{_stub_folder}/{fname[fname.find("/site-packages/") + 15 :]}i'


def stub_module(m: str, include_submodules: bool = True, strip_defaults: bool = False, skip_analysis: bool = False,
stub_folder: str = _stub_folder, trace_folder: str = "tracing") -> None:
    global _stub_folder
    _stub_folder = stub_folder
    init_trace_loader(trace_folder, m)
    imports = load_map(m, 'imports')
    if skip_analysis:
        state = State(None, imports, load_docstrings(m), load_type_maps(m), {}, {}, {})
    else:   
        state = analyze_module(m, include_submodules=include_submodules)
    if state is not None:
        imports = state.imports
        if imports:
            # Add any extra imports paths found in the imports file
            for k, v in imports.items():
                if k not in state.imports:
                    state.imports[k] = v
        process_module(m, state, _stub, _targeter, include_submodules=include_submodules,
            strip_defaults=strip_defaults)

