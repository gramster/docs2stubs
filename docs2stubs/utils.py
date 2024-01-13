from collections import Counter
import glob
import importlib
import inspect
import logging
import os
import pickle
from types import ModuleType
from typing import Callable, Generator, Generic, NamedTuple, TypeVar, cast


# I tried using a generic class from NamedTuple here but
# pyright complains about not being able to do consistent
# method ordering.

T = TypeVar('T')

class Sections(Generic[T]):
    # Would use NamedTuple but it can't be generic
    def __init__(self, params: T, returns:T, attrs:T):
        self.params: T = params
        self.returns: T = returns
        self.attrs: T = attrs

    def __iter__(self) -> Generator[T, None, None]:
        yield self.params
        yield self.returns
        yield self.attrs

    def __getitem__(self, i: int) -> T:
        return [self.params, self.returns, self.attrs][i]


# State object
# counters: a Section of collections.Counter that count the frequency of each docstring
# imports: dict mapping class names to containing module names
# docstrings: dict mapping module name to Section, each Section has dicts mapping 'contexts' to type docstrings
# trace_sigs: dict mapping module name to Section, each Section has dicts mapping 'contexts' 
#     to inspect.Signatures derived from MonkeyType traces
# maps: dict mapping docstring types to annotations read from .map files
# 'contexts' are pseudo-pathnames identifying classes, functions, methods or parameters
# trace_types: dict mapping docstring types to sets of Python types derive from MonkeyType traces

State = NamedTuple("State", [
    ("counters", Sections[Counter[str]]|None), 
    ("docstrings", dict[str, Sections[dict[str, str]|dict[str,dict[str,str]]]]), 
    ("maps", None|Sections[dict[str,str]]),
    ("trace_param_types", dict[str, set[type]]),
    ("trace_return_types", dict[str, set[type]]),
    ("trace_sigs", dict[str, dict[str, inspect.Signature]]),
    ("creturns", dict[str,str]), 
])


def load_map(m: str, suffix: str|None = None) -> dict[str, str]:
    map = {}
    mapfile = f"analysis/{m}.{suffix}.map" if suffix else f"analysis/{m}.map" 
    if os.path.exists(mapfile):
        with open(mapfile) as f:
            lnum = 1
            try:

                for line in f:
                    parts = line.strip().split('#')    
                    # First field is optional count and triviality flag, so 
                    # index from the end.            
                    map[parts[-2]] = parts[-1]
                    lnum += 1
            except Exception as e:
                logging.error(f'Error in {mapfile} line {lnum}: {e}')
    return map


def load_type_maps(m: str) -> Sections[dict[str,str]]:
    return Sections[dict[str,str]](params=load_map(m, 'params'),
                    returns=load_map(m, 'returns'),
                    attrs=load_map(m, 'attrs'))


def save_docstrings(m:str, data: dict):
    with open(f'analysis/{m}.analysis.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_docstrings(m:str) -> dict:
    with open(f'analysis/{m}.analysis.pkl', 'rb') as f:
        return pickle.load(f)

     
def get_module_and_children(m: str) -> tuple[ModuleType|None, str|None, list[str], list[str]]:
    try:
        mod = importlib.import_module(m)
        file = inspect.getfile(mod)
    except Exception as e:
        logging.error(f'Could not import module {m}: {e}')
        return None, None, [], []

    submodules = []
    native_submodules = []
    if file.endswith("/__init__.py"):
        # Get the parent directory and all the files in that directory
        folder = file[:-12]
        for f in glob.glob(folder + "/*"):
            if f == file:
                continue
            if f.endswith('.py'):
                submodules.append(f'{m}.{f[f.rfind("/")+1:-3]}')
            # TODO: make this configureable. Right now it is geared 
            # to drop tests from sklearn.
            elif any([f.endswith(f'/{x}') for x in ['.so', '.dll', '.dylib']]):
                native_submodules.append(f'{m}.{f[f.rfind("/")+1:f[f.rfind(".")]]}')
            elif os.path.isdir(f) and not f.endswith('__pycache__') and not f.endswith('/tests'):
                submodules.append(f'{m}.{f[f.rfind("/")+1:]}')
    return mod, file, submodules, native_submodules


def save_result(target: str, result: str) -> None:
    """ Write `result` into file `target`. Creates intermediate
        directories if needed.
    """
    folder = target[: target.rfind("/")]
    os.makedirs(folder, exist_ok=True)
    with open(target, "w") as f:
        f.write(result)


def process_module(
        task_name: str,
        module_name: str, 
        state: State,
        python_module_processor: Callable, 
        native_module_processor: Callable,
        output_filename_generator: Callable|None = None,
        **kwargs) -> None|State:

    modules = [module_name]
    native_modules = []
    while modules or native_modules:
        if modules:
            module_name = modules.pop()
            mod, file, submodules, native = get_module_and_children(module_name)
            if file is None or mod is None:
                logging.error(f"{task_name}: Failed to import {module_name}")
                continue
            modules.extend(submodules)
            native_modules.extend(native)
            if file.endswith('.py'):
                try:
                    with open(file) as f:
                        source = f.read()
                except Exception as e:
                    logging.error(f"{task_name}: Failed to read {file}: {e}")
                    continue
                result = python_module_processor(mod, module_name, file, source, state, **kwargs)
            else:
                result = native_module_processor(mod, module_name, file, state, **kwargs)
        else:
            module_name = native_modules.pop()
            mod, file, _, _ = get_module_and_children(module_name)
            result = native_module_processor(mod, module_name, file, state, **kwargs)
            
        if result is None:
            logging.error(f"{task_name}: Failed to handle {file}")
        else:
            logging.info(f"{task_name}: Done {file}")
            if output_filename_generator is not None:
                save_result(output_filename_generator(file), result)

    return state


def get_top_level_obj(mod: ModuleType, fname: str, oname: str):
    try:
        return mod.__dict__[oname]
    except KeyError:
        try:
            submod = fname[fname.rfind('/')+1:-3]
            return mod.__dict__[submod].__dict__[oname]
        except Exception:
            logging.error(f'{fname}: Could not get obj for {oname}')
            return None


def analyze_object(obj, context: str, parser, 
                    counters: Sections[Counter[str]], 
                    attr_types: dict[str, str]|None=None) -> Sections[dict[str,str]|None]:
    # This gets the docstring for a class, function or method, parses it, and returns
    # the result. It also updates the counters for the types of docstrings found, and
    # adds entries for encountered class  attributes to the attr_types dictionary.

    doc = None
    rtn = Sections(params=None, returns=None, attrs=None)

    # Get and parse the docstring for the object. This uses inspect and so doesn't work for 
    # attributes that are documented in the class docstring. We handle those at the end.
    if obj:
        doc = inspect.getdoc(obj)
        if doc:
            rtn = parser.parse(doc)
    
    # Update the counters for the docstrings we found
    for section, counter in zip(rtn, counters):
        if section:
            section = cast(dict[str,str], section)
            counter = cast(Counter[str], counter)
            for typ in section.values():
                counter[typ] += 1

    # If we have attribute docstrings, we fake up the contexts and save them here.
    if rtn.attrs and attr_types:
        for k, v in rtn.attrs.items():
            attr_types[f'{context}.{k}'] = v

    return rtn # type: ignore


