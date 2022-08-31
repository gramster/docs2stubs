from ast import Num
from collections import Counter
import inspect
from types import ModuleType
import libcst as cst
from .basetransformer import BaseTransformer
from .utils import process_module
from .parser import NumpyDocstringParser

class AnalyzingTransformer(BaseTransformer):

    def __init__(self, mod: ModuleType, fname: str, counter: Counter, context: dict,
            imports: dict):
        super().__init__()
        self._mod = mod
        i = fname.find('site-packages')
        if i > 0:
            # Strip off the irrelevant part of the path
            self._fname = fname[i+14:]
        else:
            self._fname = fname
        self._classname = ''
        self._parser = NumpyDocstringParser()
        self._counter = counter
        self._context = context
        self._imports = imports

    def _analyze_obj(self, obj, context: str):
        doc = None
        if obj:
            doc = inspect.getdoc(obj)
        if not doc:
            return
        rtn = self._parser.parse(doc)
        for section in rtn:
            if section:
                for _, raw, typs in section:
                    for typ in typs.split('|'):
                        if typ not in self._context:
                            self._context[typ] = f'{self._fname}:{context} {raw}'
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
            self._imports[self._classname] = self._fname
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, node.name.value)
            self._analyze_obj(obj, self._classname)
        return super().visit_ClassDef(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name = node.name.value
        obj = None
        context = ''
        if self.at_top_level():
            context = name
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, name)
        elif self.at_top_level_class_level():
            context = f'{self._classname}.{name}'
            parent = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, self._classname)
            if parent:
                if name in parent.__dict__:
                    obj = parent.__dict__[name]
                else:
                    print(f'{self._fname}: Could not get obj for {self._classname}.{name}')
        self._analyze_obj(obj, context)
        return super().visit_FunctionDef(node)


def _analyze(mod: ModuleType, fname: str, source: str, state: tuple, **kwargs):
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None
    try:
        patcher = AnalyzingTransformer(mod, fname, 
            counter=state[0], 
            context=state[1],
            imports = state[2])
        cstree.visit(patcher)
    except:  # Exception as e:
        # Note: I know that e is undefined below; this actually lets me
        # successfully see the stack trace from the original excception
        # as traceback.print_exc() was not working for me.
        print(f"Failed to analyze file: {e}")
        return None
    return state

def _post_process(m: ModuleType, state: tuple):
    result = ''
    freq: Counter = state[0]
    context: dict = state[1]
    for typ, cnt in freq.most_common():
        result += f'{cnt}#{context[typ]}#{typ}#{typ}\n'
    return result


def _targeter(m: str) -> str:
    """ Turn module name into map file name """
    return f"analysis/{m}.typ"


def analyze_module(m: str):
    process_module(m, _analyze, _targeter, post_processor=_post_process, 
        state=(Counter(), {}, {}))

