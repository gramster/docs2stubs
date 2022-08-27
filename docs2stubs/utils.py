import glob
import importlib
import inspect
import os
from types import ModuleType
from typing import Callable


def get_module_and_files(m: str) -> tuple[ModuleType|None, list[str]]:
    try:
        mod = importlib.import_module(m)
    except Exception as e:
        print(f'Could not import module {m}: {e}')
        return None, []
    file = inspect.getfile(mod)
    if file.endswith("/__init__.py"):
        # Get the parent directory and all the files in that directory
        folder = file[:-12]
        files = glob.glob(folder + "/*.py")
    else:
        files = [file]
    return mod, files


def process_module(m: str, processor: Callable, 
        targeter: Callable,
        post_processor: Callable|None = None, 
        state: object = None,
        **kwargs):
    mod, files = get_module_and_files(m)
    if not mod:
        return

    result = None
    if state is None:
        state = {}

    for file in files:
        try:
            with open(file) as f:
                source = f.read()
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        result = processor(mod, file, source, state, **kwargs)
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
        result = post_processor(m, state)
        target = targeter(m)
        folder = target[: target.rfind("/")]
        os.makedirs(folder, exist_ok=True)
        with open(target, "w") as f:
            f.write(result)