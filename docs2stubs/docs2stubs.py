import glob
import inspect
import os
from types import ModuleType
import libcst as cst
from .basetransformer import BaseTransformer
from .utils import process_module


class StubbingTransformer(BaseTransformer):
    def __init__(self, strip_defaults=False):
        super().__init__()
        self.strip_defaults = strip_defaults
        self.method_names = set()
        self.class_names = set()

    @staticmethod
    def get_value_type(node: cst.CSTNode) -> str|None:
        typ: str|None= None
        if isinstance(node, cst.Name) and node.value in [
            "True",
            "False",
        ]:
            typ = "bool"
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

    def get_assign_value(self, node: cst.Assign) -> cst.CSTNode:
        # See if this is an alias, in which case we want to
        # preserve the value; else we set the new value to ...
        new_value = None
        if isinstance(node.value, cst.Name) and not self.in_function():
            check = set()
            if self.at_top_level():
                check = self.class_names
            elif self.at_top_level_class_level(): # Class level
                check = self.method_names
            if node.value.value in check:
                new_value = node.value
        if new_value is None:
            new_value = cst.parse_expression("...")  
        return new_value

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        typ = StubbingTransformer.get_value_type(original_node.value)
        # Make sure the assignment was not to a tuple before
        # changing to AnnAssign
        if typ is not None and len(original_node.targets) == 1:
            return cst.AnnAssign(target=original_node.targets[0].target,
                annotation=cst.Annotation(annotation=cst.Name(typ)),
                value=cst.parse_expression("..."))
        else:
            return updated_node.with_changes(\
                value=self.get_assign_value(updated_node))

    def leave_AnnAssign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        return updated_node.with_changes(value=cst.parse_expression("..."))

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.CSTNode:
        annotation = original_node.annotation   
        if original_node.annotation is None and original_node.default is not None:
            typ = StubbingTransformer.get_value_type(original_node.default)
            if typ is not None:
                annotation = cst.Annotation(annotation=cst.Name(typ))

        default = original_node.default
        """Remove default values, replace with ..."""
        if self.strip_defaults and default is not None:
            default=cst.parse_expression("...")

        return updated_node.with_changes(default=default, annotation=annotation)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Record the names of top-level classes
        if not self.in_class():
            self.class_names.add(node.name.value)
        return super().visit_ClassDef(node)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        super().leave_ClassDef(original_node, updated_node)
        if not self.in_class():
            # Clear the method name set
            self.method_names = set()
            return updated_node
        else:
            # Nested class; return ...
            return cst.parse_statement('...')

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self.at_top_level_class_level():
            # Record the method name
            self.method_names.add(node.name.value)
        return super().visit_FunctionDef(node)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        """Remove function bodies"""
        super().leave_FunctionDef(original_node, updated_node)
        if self.in_function(): 
            # Nested function; return ...
            return cst.parse_statement('...')
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


def patch_source(source: str, strip_defaults: bool = False) -> str|None:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None
    try:
        patcher = StubbingTransformer(strip_defaults=strip_defaults)
        modified = cstree.visit(patcher)
    except:  # Exception as e:
        # Note: I know that e is undefined below; this actually lets me
        # successfully see the stack trace from the original excception
        # as traceback.print_exc() was not working for me.
        print(f"Failed to patch file: {e}")
        return None
    return modified.code


def _stub(mod: ModuleType, fname: str, source: str, **kwargs):
    return patch_source(source, **kwargs)

def _targeter(fname: str) -> str:
    return "typings/" + fname[fname.find("/site-packages/") + 15 :] + "i"

def stub_module(m: str, strip_defaults: bool = False):
    process_module(m, _stub, _targeter, strip_defaults=strip_defaults)

