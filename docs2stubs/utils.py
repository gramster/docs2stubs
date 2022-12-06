from collections import namedtuple
import glob
import importlib
import inspect
import os
import pickle
import re
from types import ModuleType
from typing import Callable


# I tried using a generic class from NamedTuple here but
# pyright complains about not being able to do consistent
# method ordering.
Sections = namedtuple("Sections", "params returns attrs")


def load_map(m: str, suffix: str|None = None) -> dict[str, str]:
    map = {}
    mapfile = f"analysis/{m}.{suffix}.map" if suffix else f"analysis/{m}.map" 
    if os.path.exists(mapfile):
        with open(mapfile) as f:
            for line in f:
                parts = line.strip().split('#')    
                # First field is optional count, so 
                # index from the end.            
                map[parts[-2]] = parts[-1]
    return map


def load_type_maps(m: str) -> Sections:
    return Sections(params=load_map(m, 'params'),
                    returns=load_map(m, 'returns'),
                    attrs=load_map(m, 'attrs'))


def save_import_map(m: str, imports: dict[str, str]):
    save_result(f"analysis/{m}.imports.map",
        ''.join([f"{k}#{v}\n" for k, v in imports.items()]))

    
def load_import_map(m: str) -> dict[str, str]:
    return load_map(m, 'imports')


def save_type_contexts(m:str, data: dict):
    with open(f'analysis/{m}.data.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_type_contexts(m:str) -> dict:
    with open(f'analysis/{m}.data.pkl', 'rb') as f:
        return pickle.load(f)

     
def get_module_and_children(m: str) -> tuple[ModuleType|None, str|None, list[str]]:
    try:
        mod = importlib.import_module(m)
        file = inspect.getfile(mod)
    except Exception as e:
        print(f'Could not import module {m}: {e}')
        return None, None, []

    submodules = []
    if file.endswith("/__init__.py"):
        # Get the parent directory and all the files in that directory
        folder = file[:-12]
        files = []
        for f in glob.glob(folder + "/*"):
            if f == file:
                continue
            if f.endswith('.py'):
                submodules.append(f'{m}.{f[f.rfind("/")+1:-3]}')
            # TODO: make this configureable. Right now it is geared 
            # to drop tests from sklearn.
            elif os.path.isdir(f) and not f.endswith('__pycache__') and not f.endswith('/tests'):
                submodules.append(f'{m}.{f[f.rfind("/")+1:]}')
    return mod, file, submodules


def save_result(target: str, result: str) -> None:
    """ Write `result` into file `target`. Creates intermediate
        directories if needed.
    """
    folder = target[: target.rfind("/")]
    os.makedirs(folder, exist_ok=True)
    with open(target, "w") as f:
        f.write(result)

# Returns None is no post-processor, else whatever tuple
# the post-processor returns. I think this wrapper turned 
# out to be a case of DRY-gone-wild and probably should
# be replaced.

def process_module(m: str, 
        state: tuple,
        processor: Callable, 
        targeter: Callable,
        post_processor: Callable|None = None, 
        include_submodules: bool = True,
        **kwargs) -> None|tuple:

    orig_m = m
    modules = [m]
    while modules:
        m = modules.pop()
        mod, file, submodules = get_module_and_children(m)
        if include_submodules:
            if not mod:
                continue
            modules.extend(submodules)
        else:
            if not mod:
                return state

        result = None

        if file is None:
            continue

        try:
            with open(file) as f:
                source = f.read()
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        result = processor(mod, m, file, source, state, **kwargs)
        if post_processor is None:
            if result is None:
                print(f"Failed to process {file}")
                continue
            else:
                target = targeter(file)
                save_result(target, result)

        print(f"Processed file {file}")

    if post_processor:
        result, rtn = post_processor(orig_m, state, **kwargs)
        if isinstance(result, str):
            target = targeter(orig_m)
            save_result(target, result)
        elif isinstance(result, Sections):
            save_result(targeter(orig_m, 'params'), result.params)
            save_result(targeter(orig_m, 'returns'), result.returns)
            save_result(targeter(orig_m, 'attrs'), result.attrs)

        return rtn
    return state


def save_fullmap(folder, module, params, returns, attrs):
    for section, fullmap in zip(['params', 'returns', 'attrs'], [params, returns, attrs]):
        with open(f'{folder}/{module}.{section}.full', 'w') as f:
            for k, v in fullmap.items():
                f.write(f'{k}#{v}\n')

                
def load_fullmap(folder, module):
    rtn = Sections({}, {}, {})
    for section, fullmap in zip(['params', 'returns', 'attrs'], rtn):
        fname = f'{folder}/{module}.{section}.full'
        if os.path.exists(fname):
            with open(fname) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        k, v = line.strip().split('#')
                        if section == 'returns': # Turn into a list/tuple
                            k = k.split('/')[0]
                            if k not in fullmap:
                                fullmap[k] = []
                            fullmap[k].append(v)
                        else:
                            fullmap[k] = v
    return rtn