from abc import ABC, abstractmethod
import glob
import importlib
import inspect
import sys
import libcst as cst


class StubbingTransformer(cst.CSTTransformer):

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.CSTNode:
        """ Remove default values, replace with ..."""
        if original_node.default is not None:
            return updated_node.with_changes(default=cst.parse_expression('...'))
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        """ Remove function bodies """
        return updated_node.with_changes(body=cst.parse_statement('...'))
     

def stub_module(m):
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
            patcher = StubbingTransformer()
            modified = cstree.visit(patcher)
            if modified.code == cstree.code:
                print(f'File {file} is unchanged')
            else:
                print(f'Patched file {file}')
                target = 'typings/' + file[file.find('/site-packages/')+15] + 'i'
                print(f'Writing {target}')
                with open(target, 'w') as f:
                    f.write(modified.code)
        except:# Exception as e:
            # Note: I know that e is undefined below; this actually lets me
            # successfully see the stack trace from the original excception
            # as traceback.print_exc() was not working for me.
            print(f'Failed to patch file {file}: {e}')


def docs2stubs(module: str):
    print(f'Patching module {module}')
    stub_module(module)

