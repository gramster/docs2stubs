from collections import Counter
import logging
from types import ModuleType
from typing import  cast
import libcst as cst


from .base_transformer import BaseTransformer
from .utils import Sections, State, analyze_object, \
    collect_modules, get_top_level_obj, \
    load_type_maps, save_docstrings, save_result
from .docstring_parser import NumpyDocstringParser
from .traces import init_trace_loader, get_method_signature, get_toplevel_function_signature
from .type_normalizer import is_trivial, normalize_type, print_norm1


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
        state.docstrings[modname] = Sections(
            params=self._paramtyps,
            returns=self._returntyps,
            attrs=self._attrtyps)
        self._trace_sigs = state.trace_sigs[modname] = {}
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

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._traced_methodsig = None
        rtn = super().visit_ClassDef(node)
        if self.at_top_level_class_level():
            self._classname = node.name.value
            context = self.context()
            obj = get_top_level_obj(self._mod, self._fname, node.name.value)
            docs = analyze_object(obj, context, self._parser, self._counters, self._attrtyps)
            self._docs[context] = docs
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
            obj = get_top_level_obj(self._mod, self._fname, name)
            self._trace_sigs[context] = get_toplevel_function_signature(self._modname, name)
        elif self._classname and self.at_top_level_class_method_level():
            parent = get_top_level_obj(self._mod, self._fname, self._classname)
            if parent:
                if name in parent.__dict__:
                    obj = parent.__dict__[name]
                    self._trace_sigs[context] = get_method_signature(self._modname, self._classname, name)
                else:
                    logging.warning(f'{self._fname}: Could not get obj for {context}')

        docs = analyze_object(obj, context, self._parser, self._counters, self._attrtyps)
        self._docs[context] = docs
        if name == '__init__':
            # If we actually had a docstring with params section, we're done
            if docs and docs.params:
                return rtn
            # Else use the class docstring for __init__
            docs = self._docs.get(outer_context)
            if docs is not None:
                self._docs[context] = docs
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


def analyze_module(module_name: str, output_trivial_types: bool = True, trace_folder='tracing') -> State:
    logging.info("Gathering docstrings")
    state = State(
        Sections[Counter[str]](params=Counter(), returns=Counter(), attrs=Counter()),
        {}, 
        load_type_maps(module_name), 
        {}, 
        {},
        {}, 
        {}
    )    
    module_metadata = collect_modules("Analyzing", module_name, state)
    while module_metadata:
        mod, file_name, module_name = module_metadata.pop()
        if file_name.endswith('.py'):
            try:
                with open(file_name) as f:
                    source_code = f.read()
            except Exception as e:
                logging.error(f"Analyzing: Failed to read {file_name}: {e}")
                continue
            try:
                cstree = cst.parse_module(source_code)
            except Exception as e:
                logging.error(f"Failed to parse file: {file_name}: {e}")
                continue
            try:
                cstree.visit(AnalyzingTransformer(mod, module_name, file_name, state))
            except Exception as e:
                logging.error(f"Failed to analyze file: {file_name}: {e}")
                continue

            logging.info(f"Analyzing: Done {file_name}")

    logging.info("Analyzing and normalizing types...")
    maps = load_type_maps(module_name)
    results = [[], [], []]
    assert(state.counters is not None)
    freqs: Sections[Counter[str]] = state.counters
    total_trivial = 0
    total_mapped = 0
    total_missed = 0
    trivials = {}
    for section, result, freq, map in zip(['params', 'returns', 'attrs'], results, freqs, maps):
        freq = cast(Counter[str], freq)
        map = cast(dict[str, str], map)
        # Output in descending order of frequency
        for typ, cnt in freq.most_common():
            if typ in map:  # We already have a mapping for this type
                total_mapped += cnt
            else:
                normtype = normalize_type(typ, module_name, section=='params')
                trivial = is_trivial(typ, module_name)
                if not output_trivial_types and trivial:
                    # Just track this for logging below
                    total_trivial += cnt
                    trivials[typ] = normtype
                else:
                    # We don't have a mapping for this type. Add it to the missing types.
                    total_missed += cnt
                    result.append(f'{"@" if trivial else ""}{cnt}#{typ}#{normtype}\n')

    logging.info(f'Trivial: {total_trivial}, Mapped: {total_mapped}, Missed: {total_missed}')
    logging.info('\nTRIVIALS\n')
    for k, v in trivials.items():
        logging.info(f'{k}#{v}')
    print_norm1()

    for i, section in enumerate(['params', 'returns', 'attrs']):
        save_result(f"analysis/{module_name}.{section}.map.missing", ''.join(results[i]))
    save_docstrings(module_name, state.docstrings)
    return state
    
