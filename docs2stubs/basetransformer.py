import libcst as cst

class BaseTransformer(cst.CSTTransformer):
    def __init__(self, modname: str, fname: str, strip_defaults=False):
        self._modname = modname
        self._fname = fname
        self._in_class_count = 0
        self._in_function_count = 0
        self._context_stack = []

    def context(self) -> str:
        return '.'.join(self._context_stack)

    def in_class(self)-> bool:
        return self._in_class_count > 0

    def in_function(self)-> bool:
        return self._in_function_count > 0

    def at_top_level(self):
        return not(self.in_class() or self.in_function())

    def at_top_level_class_level(self) -> bool:
        return self._in_class_count == 1 and not self.in_function()

    def in_method(self) -> bool:
        # Strictly speaking this can happen if we define a class
        # in a top-level function too.
        # TODO: figure out how to detect that. It probably
        # doesn't matter though so punting for now.
        return self.in_class() and self.in_function()

    def at_top_level_function_level(self) -> bool:
        return not self.in_class() and self._in_function_count == 1

    def at_top_level_class_method_level(self) -> bool:
        return self._in_class_count == 1 and self._in_function_count == 1

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        assert(self._in_class_count <= 1)
        self._context_stack.append(node.name.value)
        self._in_class_count += 1
        # No point recursing if we are at nested function level
        # or this is a nested class.
        return self._in_class_count == 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        self._context_stack.pop()
        self._in_class_count -= 1
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        assert(self._in_function_count <= 1)
        self._context_stack.append(node.name.value)
        self._in_function_count += 1
        # No point recursing if we are at nested function level
        return self._in_function_count == 1

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        self._context_stack.pop() 
        self._in_function_count -= 1 
        return updated_node

    def visit_Param(self, node: cst.Param) -> bool:
        self._context_stack.append(node.name.value)
        return True

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.CSTNode:
        self._context_stack.pop() 
        return updated_node

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        # Avoid recursing into lambdas so we don't try annotate their params
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        # TODO: figure out how to handle attributes here
        #if len(node.targets) == 1:
        #    self._context_stack.append(node.targets[0].name.value)
        return False

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        #self._context_stack.pop() 
        return updated_node

