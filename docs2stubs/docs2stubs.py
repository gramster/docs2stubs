import glob
import importlib
import inspect
import os
import libcst as cst


class StubbingTransformer(cst.CSTTransformer):
    def __init__(self, strip_defaults=False):
        self.strip_defaults = strip_defaults
        self.in_class_count = 0
        self.in_function_count = 0
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
            }.items():
                if isinstance(node, k):
                    typ = v
                    break
        return typ

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        # See if this is an alias, in which case we want to
        # preserve the value; else we set the new value to ...
        new_value = None
        if isinstance(original_node.value, cst.Name) and \
           self.in_function_count == 0:
            check = set()
            if self.in_class_count == 0: # Top-level
                check = self.class_names
            elif self.in_class_count == 1: # Class level
                check = self.method_names
            if original_node.value.value in check:
                new_value = updated_node.value
        if new_value is None:
            new_value = cst.parse_expression("...")
            
        typ = StubbingTransformer.get_value_type(original_node.value)
        # Make sure the assignment was not to a tuple before
        # changing to AnnAssign
        if typ is not None and len(original_node.targets) == 1:
            return cst.AnnAssign(target=original_node.targets[0].target,
                annotation=cst.Annotation(annotation=cst.Name(typ)),
                value=new_value)
        else:
            return updated_node.with_changes(value=new_value)

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
        if self.in_class_count == 0:
            self.class_names.add(node.name.value)

        self.in_class_count += 1
        # No point recursing if we are at nested function level
        return self.in_class_count == 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        self.in_class_count -= 1
        if self.in_class_count == 0:
            # Clear the method name set
            self.method_names = set()
            return updated_node
        else:
            # Nested class; return ...
            return cst.parse_statement('...')

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self.in_class_count == 1 and self.in_function_count == 0:
            # Record the method name
            self.method_names.add(node.name.value)
        self.in_function_count += 1
        # No point recursing if we are at nested function level
        return self.in_function_count == 1

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        """Remove function bodies"""
        self.in_function_count -= 1  
        if self.in_function_count == 0 or \
            (self.in_function_count == 1 and self.in_class_count == 1):
            return updated_node.with_changes(body=cst.parse_statement("..."))
        else:
            # Nested function; return ...
            return cst.parse_statement('...')

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


def stub_module(m: str, strip_defaults: bool = False):
    try:
        mod = importlib.import_module(m)
        print(f"Imported module {m} for patching")
    except Exception:
        print(f"Could not import module {m} for patching")
        return
    file = inspect.getfile(mod)
    if file.endswith("/__init__.py"):
        # Get the parent directory and all the files in that directory
        folder = file[:-12]
        files = glob.glob(folder + "/*.py")
    else:
        files = [file]

    for file in files:
        try:
            with open(file) as f:
                source = f.read()
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        modified = patch_source(source)
        if modified is None:
            print(f"Failed to parse {file}: {e}")
            continue

        target = "typings/" + file[file.find("/site-packages/") + 15 :] + "i"
        folder = target[: target.rfind("/")]
        os.makedirs(folder, exist_ok=True)
        with open(target, "w") as f:
            f.write(modified)
        print(f"Stubbed file {file}")
