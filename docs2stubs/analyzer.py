from ast import Num
from collections import Counter
import inspect
import json
import os
from types import ModuleType
from xml.etree.ElementInclude import include
import libcst as cst
from .basetransformer import BaseTransformer
from .utils import process_module, is_trivial, load_map
from .parser import NumpyDocstringParser
from .normalize import normalize_type

class AnalyzingTransformer(BaseTransformer):

    def __init__(self, 
            mod: ModuleType, 
            modname: str,
            fname: str, 
            counter: Counter,
            classes: dict,
            docs: dict):
        super().__init__(modname, fname)
        self._mod = mod
        self._parser = NumpyDocstringParser()
        self._counter = counter
        self._classes = classes
        self._docs = {}
        docs[modname] = self._docs
        self._classname = None
        

    def _analyze_obj(self, obj, context: str) -> tuple[dict[str, str]|None, ...]:
        doc = None
        if obj:
            doc = inspect.getdoc(obj)
        if not doc:
            return
        rtn = self._parser.parse(doc)
        for section in rtn:
            if section:
                for typ in section.values():
                    self._counter[typ] += 1
        return rtn

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
        rtn = super().visit_ClassDef(node)
        if self.at_top_level_class_level():
            self._classname = node.name.value
            self._classes[self._classname] = self._modname
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, node.name.value)
            self._docs[self.context()] = self._analyze_obj(obj, self._classname)
        return rtn

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        outer_context = self.context()
        rtn = super().visit_FunctionDef(node)
        name = node.name.value
        obj = None
        context = self.context()
        if self.at_top_level_function_level():
            #context = name
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, name)
        elif self.at_top_level_class_method_level():
            #context = f'{self._classname}.{name}'
            parent = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, self._classname)
            if parent:
                if name in parent.__dict__:
                    obj = parent.__dict__[name]
                else:
                    print(f'{self._fname}: Could not get obj for {context}')
        docs = self._analyze_obj(obj, context)
        self._docs[context] = docs

        if name == '__init__':
            # If we actually had a docstring with params section, we're done
            if docs and docs[0]:
                return rtn
            # Else use the class docstring for __init__
            self._docs[context] = self._docs.get(outer_context)

        return rtn

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        # Add a special entry for the return type
        context = self.context()
        doc = self._docs[context]
        if doc:
            self._docs[context + '->'] = doc[1]
        return super().leave_FunctionDef(original_node, updated_node)

    def visit_Param(self, node: cst.Param) -> bool:
        parent_context = self.context()
        parent_doc = self._docs.get(parent_context)
        rtn = super().visit_Param(node)
        if parent_doc and not isinstance(parent_doc, str):
             # The string check makes sure it's not a parameter of a lambda or function that was 
             # assigned as a default value of some other parameter
            param_docs = parent_doc[0]
            if param_docs:
                try:
                    self._docs[self.context()] = param_docs.get(node.name.value)
                except Exception as e:
                    print(e)
        return rtn


def _analyze(mod: ModuleType, m: str, fname: str, source: str, state: tuple, **kwargs):
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        return None
    try:
        patcher = AnalyzingTransformer(mod, m, fname, 
            counter=state[0], 
            classes = state[1],
            docs = state[2])
        cstree.visit(patcher)
    except:  # Exception as e:
        # Note: I know that e is undefined below; this actually lets me
        # successfully see the stack trace from the original excception
        # as traceback.print_exc() was not working for me.
        print(f"Failed to analyze file: {e}")
        return None
    return state


def _post_process(m: str, state: tuple):
    map = load_map(m)
    result = ''
    freq: Counter = state[0]
    classes: dict = state[1]
    docs: dict = state[2]
    for typ, cnt in freq.most_common():
        if typ not in map and not is_trivial(typ, m, classes):
            result += f'{typ}#{normalize_type(typ)}\n'
    return result, (map, classes, docs)


def _targeter(m: str) -> str:
    """ Turn module name into map file name """
    return f"analysis/{m}.map.missing"


def analyze_module(m: str, include_submodules: bool = True):
    return process_module(m, (Counter(), {}, {}), _analyze, _targeter, post_processor=_post_process,
        include_submodules=include_submodules)

