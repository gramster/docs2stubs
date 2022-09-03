from genericpath import isdir
import glob
import importlib
import inspect
import os
import re
from types import ModuleType
from typing import Callable
from .normalize import normalize_type


def load_map(m: str):
    map = {}
    mapfile = f"analysis/{m}.map"
    if os.path.exists(mapfile):
        with open(mapfile) as f:
            for line in f:
                parts = line.strip().split('#')
                map[parts[0]] = parts[1]
    return map


def get_module_and_children(m: str) -> tuple[ModuleType|None, str, list[str]]:
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
            elif os.path.isdir(f) and not f.endswith('__pycache__'):
                submodules.append(f'{m}.{f[f.rfind("/")+1:]}')
    return mod, file, submodules


def process_module(m: str, 
        state: object,
        processor: Callable, 
        targeter: Callable,
        post_processor: Callable|None = None, 
        include_submodules: bool = True,
        **kwargs):

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
                return

        result = None

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
                folder = target[: target.rfind("/")]
                os.makedirs(folder, exist_ok=True)
                with open(target, "w") as f:
                    f.write(result)
        print(f"Processed file {file}")

    if post_processor:
        result, rtn = post_processor(m, state)
        if result:
            target = targeter(m)
            folder = target[: target.rfind("/")]
            os.makedirs(folder, exist_ok=True)
            with open(target, "w") as f:
                f.write(result)
        return rtn
    return None


# Start with {, end with }, comma-separated quoted words
_single_restricted = re.compile(r'^{([ ]*[\"\'][A-Za-z0-9\-_]+[\"\'][,]?)+}$') 


def is_trivial(s, m: str, classes: set = None):

    if s.find(' or ') > 0:
        if all([is_trivial(c.strip(), m, classes) for c in s.split(' or ')]):
            return True

    if _single_restricted.match(s):
        return True

    nt = normalize_type(s)

    if nt.lower() in ['float', 'int', 'bool', 'str', 'set', 'list', 'dict', 'tuple', 'array-like', 
                     'callable', 'none']:
        return True

    if classes:
        # Check unqualified classname

        if nt in classes: # 
            return True

        # Check full qualified classname
        if nt.startswith(m + '.'):
            if nt[nt.rfind('.')+1:] in classes:
                return True

    return False


_generic_type_map = {
    'float': 'float',
    'int': 'int',
    'bool': 'bool',
    'str': 'str',
    'dict': 'dict',
    'list': 'list',
}

_generic_import_map = {

}