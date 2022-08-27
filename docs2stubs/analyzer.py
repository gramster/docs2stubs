from ast import Num
from collections import Counter
import inspect
from types import ModuleType
import libcst as cst
from .basetransformer import BaseTransformer
from .utils import process_module
from .parser import NumpyDocstringParser

# I suspect it would be easier to just import the module and
# recurse through the resuting __dict__ objects than use LibCST here,
# but I'm having fun with LibCST and this works, so....
class AnalyzingTransformer(BaseTransformer):

    def __init__(self, mod: ModuleType, fname: str, counter: Counter):
        super().__init__()
        self._mod = mod
        self._fname = fname
        self._classname = ''
        self._parser = NumpyDocstringParser()
        self._counter = counter

    def _analyze_obj(self, obj):
        doc = None
        if obj:
            doc = inspect.getdoc(obj)
        if not doc:
            return
        rtn = NumpyDocstringParser().parse(doc)
        for section in rtn:
            if section:
                for _, typs in section:
                    for typ in typs.split('|'):
                        self._counter[typ] += 1

    @staticmethod
    def get_top_level_obj(mod: ModuleType, fname: str, oname: str):
        try:
            return mod.__dict__[oname]
        except KeyError as e:
            try:
                submod = fname[fname.rfind('/')+1:-3]
                return mod.__dict__[submod].__dict__[oname]
            except Exception:
                print(f'{fname}: Could not get obj for {oname}')
                return None

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self.at_top_level():
            self._classname = node.name.value
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, node.name.value)
            self._analyze_obj(obj)
        return super().visit_ClassDef(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name = node.name.value
        obj = None
        if self.at_top_level():
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, name)
        elif self.at_top_level_class_level():
            parent = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, self._classname)
            if parent:
                if name in parent.__dict__:
                    obj = parent.__dict__[name]
                else:
                    print(f'{self._fname}: Could not get obj for {self._classname}.{name}')
        self._analyze_obj(obj)
        return super().visit_FunctionDef(node)


def _analyze(mod: ModuleType, fname: str, source: str, state: Counter, **kwargs):
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None
    try:
        patcher = AnalyzingTransformer(mod, fname, counter=state)
        cstree.visit(patcher)
    except:  # Exception as e:
        # Note: I know that e is undefined below; this actually lets me
        # successfully see the stack trace from the original excception
        # as traceback.print_exc() was not working for me.
        print(f"Failed to analyze file: {e}")
        return None
    return state

def _post_process(m: ModuleType, state: Counter):
    result = ''
    for typ, cnt in state.most_common():
        result += f'{cnt}#{typ}#{typ}\n'
    return result


def _targeter(m: str) -> str:
    """ Turn module name into map file name """
    return f"analysis/{m}.typ"


def analyze_module(m: str):
    process_module(m, _analyze, _targeter, post_processor=_post_process, state=Counter())

