import sys
from docs2stubs.type_normalizer import check_normalizer

# First argument is the file of test strings to read (either just types
# or count#type#mapping files). Second optional argument is the number 
# of initial lines to skip.

debug = False # Set to true to throw exception on failure for debugging; else failures are swallowed
onlyskips = True # If true only print the failure cases
modname = 'mymod'  # Not too important yet

if __name__ == '__main__':
    input = ''
    skip = 0
    skipped = 0
    done = 0
    if len(sys.argv) == 3:
        skip = int(sys.argv[2])
    with open(sys.argv[1]) as f:
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
                            count = int(parts[0])
                            input = parts[1]
                        if not onlyskips:
                            print(f'    # Line {lnum}')
                        if debug:
                            trivial, type, imports = check_normalizer(input, modname)
                            print(f'   {"" if trivial else "n"}tcheck("{input}", "{type}", {imports})')
                            continue
                        try:
                            trivial, type, imports = check_normalizer(input, modname)
                            if not onlyskips:
                                print(f'   {"" if trivial else "n"}tcheck("{input}", "{type}", {imports})')
                            done += count
                        except Exception:
                            print(f"    # SKIP: {input}")
                            skipped += count
                lnum += 1
        finally:
            print(input)
            
    print(f"SKIPPED: {skipped}\nDONE: {done}\nTOTAL: {skipped+done}\nSCORE: {100.0*done/(done+skipped)}")
