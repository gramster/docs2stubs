# Useful utility to migrate map files after changes to the code.
#
# - rename old .map files to .map.old
# - run the analysis to generate new .map.missing files
# - run this script to generate new .map files from the .map.missing files, 
#   filling in types from the map.old files.

import os
import sys

module = sys.argv[1]

for section in ["attrs", "params", "returns"]:

    old_map = {}
    old_map_file = f"analysis/{module}.{section}.map.old"
    if os.path.exists(old_map_file):
        with open(old_map_file) as f:
            for line in f:
                parts = line.strip().split('#')    
                # First field is optional count, so 
                # index from the end.            
                old_map[parts[-2]] = parts[-1]

    new_map = {}
    new_map_file = f"analysis/{module}.{section}.map.missing"
    if os.path.exists(new_map_file):
        with open(new_map_file) as f:
            for line in f:
                parts = line.strip().split('#')    
                # First field is optional count, so 
                # index from the end.            
                new_map[parts[-2]] = f'{parts[-3]}#' if len(parts) == 3 else ''

    with open(f"analysis/{module}.{section}.map", "w") as f:
        for name in new_map.keys():
            if name in old_map:
                f.write(f'{new_map[name]}{name}#{old_map[name]}\n')

