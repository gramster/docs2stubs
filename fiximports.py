# This is for networkx but may be useful for
# other libraries. It recursively finds __init__.py
# files, then looks for "from <submodule> import *"
# lines, and then sees if that submodule has an 
# __all__ definition, and if so, rewrites the import
# to be:
#
# from <submodule> import (
#     item1 as item1,
#     item2 as item2,
#     ...
# )

import os


def recurse(dir, files):
  for entry in os.listdir(dir):
    path = os.path.join(dir, entry)
    if os.path.isdir(path):
      recurse(path, files)
    elif path.endswith('__init__.pyi'):
      files.append(path)
  return files


def get_module_from_import(file, line):
    # for a "from m import *" line in file <file>, get the path to m
    assert(line.endswith(' import *\n') > 0)
    mod = line[5:-10].strip()
    if mod.startswith('...'):
        mod = f'{file[:file.rfind("/")]}/../../{mod[3:].replace(".","/")}'
    elif mod.startswith('..'):
        mod = f'{file[:file.rfind("/")]}/../{mod[2:].replace(".","/")}'
    elif mod.startswith('.'):
        mod = f'{file[:file.rfind("/")]}/{mod[1:].replace(".","/")}'
    return mod



def get_all(modfile):
    with open(modfile) as f2:
        imps = ''
        for line2 in f2:
            if line2.find('__all__') >= 0:
                imps += line2
                if line2.find("]") > 0:
                    break
            elif imps:
                imps += line2
                if line2.find("]") >= 0:
                    break
    if imps:
        #print(f"====[raw]=============\n{imps}\n")
        imps =  imps.replace('\n', ' ')  # make a single line
        #print(f"====[single line]=============\n{imps}\n")
        imps = imps[imps.find('[')+1:imps.rfind(']')]  # remove __all__ = [....]
        #print(f"====[remove all]=============\n{imps}\n")
        imps = imps.replace(",", " ").replace('"', '').replace("'", '')  # remove punctuation
        #print(f"====[remove punc]=============\n{imps}\n")
        imps = [imp.strip() for imp in imps.split(' ') if imp.strip()]  # convert to list of names
        return imps
    return None



files = []
recurse('typings/networkx', files)
for file in files:
    rewritten = []
    changed = False
    with open(file) as f:
        for line in f:
            if line.endswith(' import *\n') > 0:
                mod = get_module_from_import(file, line)
                if os.path.isdir(mod):
                    rewritten.append(line)
                else:
                    imps = get_all(mod+ '.pyi')
                    if not imps:
                        print(f'{file}> {mod} has no __all__')
                        rewritten.append(line)
                    else:
                        print(f'{file}> {mod}{imps}')
                        if len(imps) > 1:
                            rewritten.append(line[:-2] + '(\n\t')
                            rewritten.append(",\n\t".join(imps))
                            rewritten.append('\n)\n')
                        else:
                            rewritten.append(line[:-2] + imps[0] + ' as ' + imps[0] + '\n')
                    changed = True
            else:
                rewritten.append(line)

    if changed:
        with open(file, 'w') as f:
            f.write(''.join(rewritten))




