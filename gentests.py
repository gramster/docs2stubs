import os
import sys
from docs2stubs.type_normalizer import check_normalizer
from docs2stubs.utils import load_map

# First argument is the file of test strings to read (either just types
# or count#type#mapping files). Second optional argument is the number 
# of initial lines to skip.

debug = False # Set to true to throw exception on failure for debugging; else failures are swallowed
onlyskips = False # If true only print the failure cases
modname = 'mymod'  # Not too important yet
use_counts = True # when counting, use the frequency counts in the input file

if __name__ == '__main__':
    input = ''
    skip = 0
    skipped = 0
    done = 0
    trivial = 0
    nontrivial = 0
    keyword = None # If set, only process lines with this keyword
    if len(sys.argv) >= 3:
        skip = int(sys.argv[2])
    if len(sys.argv) == 4:
        keyword = sys.argv[3]
    typefile = sys.argv[1]
    classes = None

    x = typefile.find('.')
    if x > 0:
        modname = typefile[:x]
        classes = load_map(modname)

    with open(typefile) as f:
        lnum = 1
        try:
            for line in f:
                if skip:
                    skip -= 1
                else:
                    input = line.strip()
                    if input:
                        count = 1
                        if input.find('#') > 0:
                            parts = input.split('#')
                            if use_counts:
                                count = int(parts[0])
                            count = int(parts[0])
                            input = parts[1]

                        if keyword and input.find(keyword) < 0:
                            continue
                        
                        if not onlyskips:
                            print(f'    # Line {lnum}')
                        if debug:
                            is_trivial, type, imports = check_normalizer(input, modname, classes)
                            if is_trivial:
                                trivial += count
                            else:
                                nontrivial += count
                            print(f'   {"" if is_trivial else "n"}tcheck("{input}", "{type}", {imports})')
                            continue
                        try:
                            is_trivial, type, imports = check_normalizer(input, modname, classes)
                            if is_trivial:
                                trivial += count
                            else:
                                nontrivial += count
                            if not onlyskips:
                                print(f'   {"" if is_trivial else "n"}tcheck("{input}", "{type}", {imports})')
                            done += count
                        except Exception:
                            nontrivial += count
                            print(f"    # SKIP: {input}")
                            skipped += count
                lnum += 1
        finally:
            print(input)
            
    print(f"SKIPPED: {skipped}\nDONE: {done}\nTOTAL: {skipped+done}\nTRIVIAL: {trivial}\nNON_TRIVIAL: {nontrivial}\nSCORE: {100.0*done/(done+skipped)}")
