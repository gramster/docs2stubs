import re
from types import ModuleType
import libcst as cst
from black import format_str
from black.mode import Mode

from .analyzing_transformer import analyze_module
from .type_normalizer import is_trivial, normalize_type
from .base_transformer import BaseTransformer
from .utils import Sections, load_map, load_type_maps, process_module


class StubbingTransformer(BaseTransformer):
    def __init__(self, modname: str, fname: str, 
        maps: Sections, 
        imports: dict[str, str], 
        typs: dict[str, Sections], 
        strip_defaults=False, 
        infer_types_from_defaults=False):
        super().__init__(modname, fname)
        self._maps = maps
        self._classes = imports
        self._typs: Sections = typs[modname]
        self._strip_defaults: bool = strip_defaults
        self._infer_types: bool = infer_types_from_defaults
        self._method_names = set()
        self._local_class_names = set()
        self._need_imports: dict[str, str] = {} # map class -> module to import from
        self._ident_re = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)')

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

    def get_assign_value(self, node: cst.Assign) -> cst.BaseExpression:
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
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        value=self.get_assign_value(original_node)
        return updated_node.with_changes(value=value)

    def fixtype(self, doctyp: str, map: dict[str, str], is_param: bool = False):
        typ = None
        imports = {}
        how = ''
        if doctyp in map:
            # We still call the normalizer, just to get the imports
            typ = map[doctyp]
            _, imports = normalize_type(typ, self._modname, self._classes, is_param)
            how = 'mapped'
        elif is_trivial(doctyp, self._modname, self._classes):
            typ, imports = normalize_type(doctyp, self._modname, self._classes, is_param)
            how = 'trivial'   
        return typ, imports, how

    def update_imports(self, imports: dict[str, list[str]]):
        # imports has file -> list of classes
        # self._need_imports has class -> file
        for module, classlist in imports.items():
            for cls in classlist:
                self._need_imports[cls] = module

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.CSTNode:
        doctyp = self._typs.params.get(self.context())
        super().leave_Param(original_node, updated_node)
        annotation = original_node.annotation
        default = original_node.default
        valtyp = None
        is_optional = False

        if default:
            valtyp = StubbingTransformer.get_value_type(default) # Inferred type from default
            if (not valtyp or self._strip_defaults):
                # Default is something too complex for a stub or should be stripped; replace with '...'
                default = cst.parse_expression("...")

        if doctyp and not annotation:
            typ, imports, how = self.fixtype(doctyp, self._maps.params, True)
            if typ == 'None':
                print(f'Could not annotate parameter {self.context()} from {doctyp}; would be None')
            elif typ:
                if imports:
                    self.update_imports(imports)

                # If the default value is None, make sure we include it in the type
                is_optional = 'None' in typ.split('|')
                if not is_optional and valtyp == 'None' and typ != 'Any':
                    typ += '|None'

                print(f'Annotated {self.context()} with {typ} from {doctyp} ({how})')
                try:
                    annotation = cst.Annotation(annotation=cst.parse_expression(typ))
                except:
                    print(f'Could not annotate parameter {self.context()} with {typ} ({how}); fails to parse')
            else:
                print(f'Could not annotate parameter {self.context()} from {doctyp}; no mapping')

        if self._infer_types and valtyp and not annotation and valtyp != 'None':
            # Use the inferred type from default value as long as it is not None
            annotation = cst.Annotation(annotation=cst.Name(valtyp))
            
        return updated_node.with_changes(annotation=annotation, default=default)

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
            return updated_node
        else:
            # Nested class; return ...
            return cst.parse_statement('...')

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self.at_top_level_class_level():
            # Record the method name
            self._method_names.add(node.name.value)
        return super().visit_FunctionDef(node)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        """Remove function bodies and add return type annotations """
        doctyp = self._typs.returns.get(self.context())
        annotation = original_node.returns
        super().leave_FunctionDef(original_node, updated_node)
        if self.in_function(): 
            # Nested function; return ...
            return cst.parse_statement('...')

        if not annotation and doctyp:
            map = self._maps.returns
            v = []
            for t in doctyp.values():
                typ, imp, how = self.fixtype(t, map)
                if typ:
                    v.append(typ)
                    self.update_imports(imp)
                else:
                    print(f'Could not annotate {self.context()}-> from {doctyp}')
                    v = None
                    break
            
            if v:
                if len(v) > 1:
                    rtntyp = 'tuple[' + ', '.join(v) + ']'
                else:
                    rtntyp = v[0]
                try: 
                    n = updated_node.with_changes(body=cst.parse_statement("..."), 
                        returns=cst.Annotation(annotation=cst.parse_expression(rtntyp))) 
                    print(f'Annotating {self.context()}-> as {rtntyp}')   
                    return n
                except:
                    print(f'Could not annotate {self.context()}-> as {rtntyp}: parse failed')
            else:
                print(f'Could not annotate {self.context()}-> from {doctyp}') 

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
            if any(
                isinstance(node, cls)
                for cls in [cst.ClassDef, cst.FunctionDef, cst.SimpleStatementLine]
            )
        ]
        return updated_node.with_changes(body=newbody)


def patch_source(m: str, fname: str, source: str, maps: Sections, imports: dict, typs: dict, strip_defaults: bool = False) -> str|None:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None

    patcher = StubbingTransformer(m, fname, maps, imports, typs, strip_defaults=strip_defaults)
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
    #print(f"Add imports: {import_statements}")

    code = import_statements + modified.code
    return format_str(code, mode=Mode()) 


def _stub(mod: ModuleType, m: str, fname: str, source: str, state: tuple, **kwargs) -> str|None:
    return patch_source(m, fname, source, state[0], state[1], state[2], **kwargs)


def _targeter(fname: str) -> str:
    return "typings/" + fname[fname.find("/site-packages/") + 15 :] + "i"


def stub_module(m: str, include_submodules: bool = True, strip_defaults: bool = False, skip_analysis: bool = False):
    imports = load_map(m, 'imports')
    rtn = analyze_module(m, include_submodules=include_submodules)
    if rtn is not None:
        if imports:
            # Add any extra imports paths found in the imports file
            for k, v in imports.items():
                if k not in rtn[1]:
                    rtn[1][k] = v
        process_module(m, rtn, _stub, _targeter, include_submodules=include_submodules,
            strip_defaults=strip_defaults)

