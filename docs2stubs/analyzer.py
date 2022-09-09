from collections import Counter
import inspect
from types import ModuleType
from typing import Any
import libcst as cst
from .basetransformer import BaseTransformer
from .utils import Sections, process_module, load_type_maps, save_result
from .parser import NumpyDocstringParser
from .normalize import is_trivial, normalize_type

class AnalyzingTransformer(BaseTransformer):

    def __init__(self, 
            mod: ModuleType, 
            modname: str,
            fname: str, 
            counters: Sections,
            locations: dict,
            typs: dict[str, Sections])-> None:
        """
        Params:
          mod - the module object. Used to get docstrings.
          modname - the module name.
          fname - the file name.
          counters - used to collect types and frequencies
          locations - used to collect names of classes defined in the module(s)
          typs,... - used to collect the types from docstrings

        Several of these are passed in so that they can be shared by all modules
        in a package.
        """
        super().__init__(modname, fname)
        self._mod = mod
        self._parser = NumpyDocstringParser()
        self._counters = counters
        self._locations = locations
        self._docs: dict[str, Sections|None] = {}
        self._attrtyps: dict[str, str] = {}
        self._paramtyps: dict[str, str] = {}
        self._returntyps: dict[str, str] = {}
        typs[modname] = Sections(
            params=self._paramtyps,
            returns=self._returntyps,
            attrs=self._attrtyps)
        self._classname = None
        

    def _analyze_obj(self, obj, context: str) -> Sections:
        doc = None
        rtn = Sections(params=None, returns=None, attrs=None)
        if obj:
            doc = inspect.getdoc(obj)
            if doc:
                rtn = self._parser.parse(doc)
        for section, counter in zip(rtn, self._counters):
            if section:
                for typ in section.values():
                    counter[typ] += 1
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
            self._locations[self._classname] = self._modname
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
        elif self._classname and self.at_top_level_class_method_level():
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
            if docs and docs.params:
                return rtn
            # Else use the class docstring for __init__
            self._docs[context] = self._docs.get(outer_context) 

        return rtn

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        # Add a special entry for the return type
        context = self.context()
        doc = self._docs.get(context)
        if doc:
            self._returntyps[context] = doc.returns
        return super().leave_FunctionDef(original_node, updated_node)

    def visit_Param(self, node: cst.Param) -> bool:
        parent_context = self.context()
        parent_doc = self._docs.get(parent_context)
        rtn = super().visit_Param(node)
        if parent_doc: # and isinstance(parent_doc, DocTypes):
             # The isinstance check makes sure it's not a parameter of a lambda or function that was 
             # assigned as a default value of some other parameter
            param_docs = parent_doc.params
            if param_docs:
                try:
                    self._paramtyps[self.context()] = param_docs.get(node.name.value)
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
            counters=state[0], 
            locations = state[1],
            typs = state[2],
            )
        cstree.visit(patcher)
    except Exception as e:
        print(f"Failed to analyze file: {fname}: {e}")
        return None
    return state


def _post_process(m: str, state: tuple, include_counts: bool = False):
    maps = load_type_maps(m)
    results = [[], [], []]
    freqs: Sections = state[0]
    locations: dict = state[1]
    typs: dict = state[2]
    total_trivial = 0
    total_mapped = 0
    total_missed = 0
    trivials = {}
    for result, freq, map in zip(results, freqs, maps):
        for typ, cnt in freq.most_common():
            if typ in map:
                total_mapped += cnt
            elif is_trivial(typ, m, locations):
                trivials[typ] = normalize_type(typ)
                total_trivial += cnt
            else:
                total_missed += cnt
                if include_counts:
                    result.append(f'{cnt}#{typ}#{normalize_type(typ)}\n')
                else:
                    result.append(f'{typ}#{normalize_type(typ)}\n')
    print(f'Trivial: {total_trivial}, Mapped: {total_mapped}, Missed: {total_missed}')
    print('\nTRIVIALS\n')
    for k, v in trivials.items():
        print(f'{k}#{v}')

    return Sections(params=''.join(results[0]), 
                    returns=''.join(results[1]),
                    attrs=''.join(results[2])), \
           (maps, locations, typs)


def _targeter(m: str, suffix: str) -> str:
    """ Turn module name into map file name """
    return f"analysis/{m}.{suffix}.map.missing"


def analyze_module(m: str, include_submodules: bool = True, include_counts = False) -> None|tuple:
    rtn = process_module(m, (
        Sections(params=Counter(), returns=Counter(), attrs=Counter()),
        {}, {}), 
        _analyze, _targeter, post_processor=_post_process,
        include_submodules=include_submodules,
        include_counts=include_counts)
    # Save imports and type contexts too
    imports= rtn
    if rtn:
        save_result(f"analysis/{m}.imports.map",
            ''.join([f"{k}#{v}\n" for k, v in rtn[1].items()]))
    return rtn


