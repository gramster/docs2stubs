import glob
import importlib
import inspect
import os
import libcst as cst


class StubbingTransformer(cst.CSTTransformer):

    def __init__(self, strip_defaults=False):
        self.strip_defaults = strip_defaults
        self.in_simple_statement_count = 0
        self.in_class_count = 0
        self.in_function_count = 0

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.CSTNode:
        return updated_node.with_changes(value=cst.parse_expression('...'))

    def leave_AnnAssign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.CSTNode:
        return updated_node.with_changes(value=cst.parse_expression('...'))
    
    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.CSTNode:
        """ Remove default values, replace with ..."""
        if self.strip_defaults and original_node.default is not None:
            return updated_node.with_changes(default=cst.parse_expression('...'))
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        """ Remove function bodies """
        return updated_node.with_changes(body=cst.parse_statement('...'))
     
    @staticmethod
    def keep_simple_statement(node: cst.CSTNode)-> bool:
        return isinstance(node, cst.Assign) or \
               isinstance(node, cst.AnnAssign) or \
               isinstance(node, cst.Import) or \
               isinstance(node, cst.ImportFrom)

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine) -> cst.CSTNode:
        newbody = [node for node in updated_node.body \
            if StubbingTransformer.keep_simple_statement(node)]
        return updated_node.with_changes(body=newbody)

        return super().leave_SimpleStatementLine(original_node, updated_node)
 
    @staticmethod
    def keep_module_statement(node: cst.CSTNode)-> bool:
        return isinstance(node, cst.ClassDef) or \
               isinstance(node, cst.FunctionDef) or \
               isinstance(node, cst.SimpleStatementLine)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """ Remove everything from the body that is not an import,
            class def, function def, or assignment.
        """
        newbody = [node for node in updated_node.body \
            if StubbingTransformer.keep_module_statement(node)]
        return updated_node.with_changes(body=newbody)

def stub_module(m: str, strip_defaults: bool = False):
    try:
        mod = importlib.import_module(m)
        print(f'Imported module {m} for patching')
    except Exception:
        print(f'Could not import module {m} for patching')
        return
    file = inspect.getfile(mod)
    if file.endswith('/__init__.py'):
        # Get the parent directory and all the files in that directory
        folder = file[:-12]
        files = glob.glob(folder + '/*.py')
    else:
        files = [file]

    for file in files:
        try:
            with open(file) as f:
                source = f.read()
        except Exception as e:
            print(f'Failed to read {file}: {e}')
            continue
        try:
            cstree = cst.parse_module(source)
        except Exception as e:
            print(f'Failed to parse {file}: {e}')
            continue
        try:
            patcher = StubbingTransformer(strip_defaults=strip_defaults)
            modified = cstree.visit(patcher)
        except:# Exception as e:
            # Note: I know that e is undefined below; this actually lets me
            # successfully see the stack trace from the original excception
            # as traceback.print_exc() was not working for me.
            print(f'Failed to patch file {file}: {e}')
            continue

        target = 'typings/' + file[file.find('/site-packages/')+15:] + 'i'
        folder = target[:target.rfind('/')]
        os.makedirs(folder, exist_ok=True)
        with open(target, 'w') as f:
            f.write(modified.code)
        print(f'Stubbed file {file}')

