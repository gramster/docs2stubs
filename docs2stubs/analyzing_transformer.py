from collections import Counter
import inspect
from types import ModuleType
from typing import Any, cast
import libcst as cst

from .base_transformer import BaseTransformer
from .utils import Sections, State, process_module, load_type_maps, save_fullmap, save_result, save_import_map, save_docstrings
from .docstring_parser import NumpyDocstringParser
from .traces import get_method_signature, get_toplevel_function_signature, init_trace_loader
from .type_normalizer import is_trivial, normalize_type, print_norm1


# Collection of all the docstrings, for use by the augmenter
_fullmap = Sections[dict[str, str|dict[str,str]]]({}, {}, {})

class AnalyzingTransformer(BaseTransformer):

    def __init__(self, 
            mod: ModuleType, 
            modname: str,
            fname: str, 
            state: State)-> None:
        """
        Params:
          mod - the module object. Used to get docstrings.
          modname - the module name.
          fname - the file name.
          state - the state object used for collecting analysis results for all modules
        """
        super().__init__(modname, fname)
        self._mod = mod
        self._parser = NumpyDocstringParser()

        self._docs: dict[str, Sections[dict[str,str]|None]] = {}
        self._classname = None  # Current class we are in, if any

        # Initialize the state for this module
        self._attrtyps: dict[str, str] = {}
        self._paramtyps: dict[str, str] = {}
        self._returntyps: dict[str, dict[str, str]] = {}
        state.docstrings[modname] = Sections[dict[str,Any]](
            params=self._paramtyps,
            returns=self._returntyps,
            attrs=self._attrtyps)
        state.trace_sigs[modname] = self._trace_sigs = {}
        self._state = state
        assert(state.counters is not None)
        self._counters = state.counters

    def _get_obj_name(self, obj) -> str:
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

    def _update_fullmap(self, section, items, context) -> None:
        if items:
            for name, typ in items.items():
                section[f'{context}.{name}'] = typ

    def _update_full_context(self, sections: Sections[dict[str,str]|None], context: str) -> None:
        """
        As a side effect we collect all of these so they can be 
        written out at the end. This allows us to go from a type
        in the map file to the places it occurs in the source.
        We can also use this in the augmenter to show the tracing
        type annotation whenever we have a mismatch, although
        that is less useful now we are using tracing type annotations
        during this initial phase anyway as the mapped values.
        """
        fullcontext = f'{self._modname}.{context}'   
        self._update_fullmap(_fullmap.params, sections.params, fullcontext)
        self._update_fullmap(_fullmap.attrs, sections.attrs, fullcontext)
        if sections.returns is not None:
            types = list(sections.returns.values())
            if len(sections.returns) == 1:
                _fullmap.returns[context] = types[0]
            elif len(sections.returns) > 1:
                _fullmap.returns[context] = f'tuple[{",".join(types)}]'

    def _analyze_obj(self, obj, context: str) -> Sections[dict[str,str]|None]:
        doc = None
        rtn = Sections[dict[str,str]|None](params=None, returns=None, attrs=None)
        if obj:
            doc = inspect.getdoc(obj)
            if doc:
                rtn = self._parser.parse(doc)
        
        for section, counter in zip(rtn, self._counters):
            if section:
                section = cast(dict[str,str], section)
                counter = cast(Counter[str], counter)
                for typ in section.values():
                    counter[typ] += 1
        return rtn

    @staticmethod
    def get_top_level_obj(mod: ModuleType, fname: str, oname: str) -> Any:
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
        self._traced_methodsig = None
        rtn = super().visit_ClassDef(node)
        if self.at_top_level_class_level():
            self._classname = node.name.value
            self._state.imports[self._classname] = self._modname
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
            # TODO: make sure in this case we still call leave so that the stack
            # is correct; I am 99% sure we do and this is just to prevent the children
            # being visited.
            return False
        
        obj = None
        context = self.context()
        parent = None
        if self.at_top_level_function_level():
            #context = name
            obj = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, name)
            self._trace_sigs[context] = get_toplevel_function_signature(self._modname, name)
        elif self._classname and self.at_top_level_class_method_level():
            #context = f'{self._classname}.{name}'
            parent = AnalyzingTransformer.get_top_level_obj(self._mod, self._fname, self._classname)
            if parent:
                if name in parent.__dict__:
                    obj = parent.__dict__[name]
                    self._trace_sigs[context] = get_method_signature(self._modname, self._classname, name)
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
            if docs is not None:
                self._docs[context] = docs
                self._update_full_context(docs, context)
            else:
                del self._docs[context]

        return rtn

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        # Add a special entry for the return type
        context = self.context()
        doc = self._docs.get(context)
        if doc and doc.returns is not None:
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
                ptype = param_docs.get(node.name.value, None)
                if ptype is not None:
                    self._paramtyps[self.context()] = ptype
        return rtn


def _analyze(mod: ModuleType, m: str, fname: str, source: str, state: State, **kwargs) -> State:
    try:
        cstree = cst.parse_module(source)
    except Exception as e:
        raise Exception(f"Failed to parse file: {fname}: {e}")
    try:
        patcher = AnalyzingTransformer(mod, m, fname, state)
        cstree.visit(patcher)
    except Exception as e:
        raise Exception(f"Failed to analyze file: {fname}: {e}")
    return state


def _post_process(m: str, state: State, include_counts: bool = False, dump_all = False) -> Sections[str]:
    print("Analyzing and normalizing types...")
    maps = load_type_maps(m)
    results = [[], [], []]
    assert(state.counters is not None)
    freqs: Sections[Counter[str]] = state.counters
    imports: dict = state.imports
    total_trivial = 0
    total_mapped = 0
    total_missed = 0
    trivials = {}
    first = True # hacky way to tell we are in params
    for result, freq, map in zip(results, freqs, maps):
        freq = cast(Counter[str], freq)
        map = cast(dict[str, str], map)
        for typ, cnt in freq.most_common():
            if typ in map:
                total_mapped += cnt
            else:
                normtype, _ = normalize_type(typ, m, imports, first)
                if normtype is None:
                    normtype = typ
                if not dump_all and is_trivial(typ, m, imports):
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

    save_fullmap('analysis', m, _fullmap)

    return Sections[str](params=''.join(results[0]), 
                    returns=''.join(results[1]),
                    attrs=''.join(results[2]))


def _targeter(m: str, suffix: str) -> str:
    """ Turn module name into map file name """
    return f"analysis/{m}.{suffix}.map.missing"


def analyze_module(m: str, include_submodules: bool = True, include_counts = False, dump_all = False, trace_folder='tracing') -> None|State:
    print("Gathering docstrings")
    init_trace_loader(trace_folder, m)
    state = State(
        Sections[Counter[str]](params=Counter(), returns=Counter(), attrs=Counter()),
        {}, {}, {}, None)
    
    if process_module(m, state, 
            _analyze, _targeter, 
            post_processor=_post_process,
            include_submodules=include_submodules,
            include_counts=include_counts,
            dump_all=dump_all) is not None:
        save_import_map(m, state.imports)
        save_docstrings(m, state.docstrings)
        return state
    
    return None
