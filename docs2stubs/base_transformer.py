import libcst as cst


_stack_debug = False

class BaseTransformer(cst.CSTTransformer):
    """
    A base class for our CST transformers that keeps track of
    whether we are in a class or a function, and the nesting level
    thereof, so we can have an idea of the context of where we are
    at in each of the visitor functions. We also can create a 
    context key for storing state relevant to the context in a 
    dictionary.
    """

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
        if _stack_debug:
            print(f"{' '*4*len(self._context_stack)}Entering class {node.name.value}")
        self._context_stack.append(node.name.value)
        self._in_class_count += 1
        # No point recursing if we are at nested function level
        # or this is a nested class.
        return self._in_class_count == 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        n = self._context_stack.pop()
        if _stack_debug:
            print(f"{' '*4*len(self._context_stack)}Leaving class {n}")
        self._in_class_count -= 1
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        assert(self._in_function_count <= 1)
        if _stack_debug:
            print(f"{' '*4*len(self._context_stack)}Entering function {node.name.value}")
        self._context_stack.append(node.name.value)
        self._in_function_count += 1
        # No point recursing if we are at nested function level
        return self._in_function_count == 1

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        n = self._context_stack.pop() 
        if _stack_debug:
            print(f"{' '*4*len(self._context_stack)}Leaving function {n}")
        self._in_function_count -= 1 
        return updated_node

    def visit_Param(self, node: cst.Param) -> bool:
        if _stack_debug:
            print(f"{' '*4*len(self._context_stack)}Entering param {node.name.value}")
        self._context_stack.append(node.name.value)
        return True

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.CSTNode:
        n = self._context_stack.pop()
        if _stack_debug:
            print(f"{' '*4*len(self._context_stack)}Leaving param {n}")
        return updated_node

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        # Avoid recursing into lambdas so we don't try annotate their params
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        # TODO: figure out how to handle attributes here
        return False

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.CSTNode:
        return updated_node

