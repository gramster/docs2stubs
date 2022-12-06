from collections import Counter
import inspect
from types import ModuleType
from typing import Any
import libcst as cst
from .base_transformer import BaseTransformer
from .utils import Sections, process_module, load_type_maps, save_fullmap, save_result
from .docstring_parser import NumpyDocstringParser
from .type_normalizer import is_trivial, normalize_type, print_norm1


_all_returns = {}
_all_params = {}
_all_attrs = {}

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
        

    def _update_fullmap(self, section, items, context, keysep='.'):
        if items:
            for name, typ in items.items():
                section[f'{context}{keysep}{name}'] = typ

    def _get_obj_name(self, obj):
        rtn = str(obj)
        if rtn.startswith('<class '):
            # Something like <class 'sklearn.preprocessing._discretization.KBinsDiscretizer'>
            return rtn[rtn.find(' ')+2:-2]
        elif rtn.startswith('<classmethod'):
            # Something like <classmethod(<function DistanceMetric.get_metric at 0x1277800d0>)>
            return rtn[rtn.find('.')+1:].split(' ')[0]
        elif rtn.find(' ') > 0:
            # Something like <function KBinsDiscretizer.__init__ at 0x169e70430>
            return rtn.split(' ')[1]
        else:
            return rtn

    def _update_full_context(self, sections: Sections, context: str):
        """
        As a side effect we collect all of these so they can be 
        written out at the end. This allows us to go from a type
        in the map file to the places it occurs in the source.
        We can also use this in the augmenter to show the 
        type annotation whenever we have a mismatch.
        """
        fullcontext = f'{self._modname}.{context}'   
        self._update_fullmap(_all_params, sections.params, fullcontext)
        self._update_fullmap(_all_returns, sections.returns, fullcontext, keysep='/')
        self._update_fullmap(_all_attrs, sections.attrs, fullcontext)

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
            context = self.context()
            docs = self._analyze_obj(obj, context)
            self._docs[context] = docs
            self._update_full_context(docs, context)
        return rtn

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        outer_context = self.context()
        rtn = super().visit_FunctionDef(node)
        name = node.name.value

        if name.startswith('_') and not name.startswith('__'):
            return False
        
        obj = None
        context = self.context()
        parent = None
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
        self._update_full_context(docs, context)
        if name == '__init__':
            # If we actually had a docstring with params section, we're done
            if docs and docs.params:
                return rtn
            # Else use the class docstring for __init__
            docs = self._docs.get(outer_context)
            self._docs[context] = docs
            if docs is not None:
                self._update_full_context(docs, context)

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


def _post_process(m: str, state: tuple, include_counts: bool = False, dump_all = False):
    print("Analyzing and normalizing types...")
    maps = load_type_maps(m)
    results = [[], [], []]
    freqs: Sections = state[0]
    locations: dict = state[1]
    typs: dict = state[2]
    total_trivial = 0
    total_mapped = 0
    total_missed = 0
    trivials = {}
    first = True # hacky way to tell we are in params
    for result, freq, map in zip(results, freqs, maps):
        for typ, cnt in freq.most_common():
            if typ in map:
                total_mapped += cnt
            else:
                normtype, _ = normalize_type(typ, m, locations, first)
                if normtype is None:
                    normtype = typ
                if not dump_all and is_trivial(typ, m, locations):
                    trivials[typ] = normtype
                    total_trivial += cnt
                else:
                    total_missed += cnt
                    if include_counts:
                        result.append(f'{cnt}#{typ}#{normtype}\n')
                    else:
                        result.append(f'{typ}#{normtype}\n')
        first = False        
    print(f'Trivial: {total_trivial}, Mapped: {total_mapped}, Missed: {total_missed}')
    print('\nTRIVIALS\n')
    for k, v in trivials.items():
        print(f'{k}#{v}')

    print_norm1()

    save_fullmap('analysis', m, _all_params, _all_returns, _all_attrs)

    for section, fullmap in zip(['params', 'returns', 'attrs'], [_all_params, _all_returns, _all_attrs]):
        with open(f'analysis/{m}.{section}.full', 'w') as f:
            for k, v in fullmap.items():
                f.write(f'{k}#{v}\n')

    return Sections(params=''.join(results[0]), 
                    returns=''.join(results[1]),
                    attrs=''.join(results[2])), \
           (maps, locations, typs)


def _targeter(m: str, suffix: str) -> str:
    """ Turn module name into map file name """
    return f"analysis/{m}.{suffix}.map.missing"


def analyze_module(m: str, include_submodules: bool = True, include_counts = False, dump_all = False) -> None|tuple:
    print("Gathering docstrings")
    rtn = process_module(m, (
        Sections(params=Counter(), returns=Counter(), attrs=Counter()),
        {}, {}), 
        _analyze, _targeter, post_processor=_post_process,
        include_submodules=include_submodules,
        include_counts=include_counts,
        dump_all=dump_all)
    # Save imports and type contexts too
    imports= rtn
    if rtn:
        save_result(f"analysis/{m}.imports.map",
            ''.join([f"{k}#{v}\n" for k, v in rtn[1].items()]))
    return rtn


