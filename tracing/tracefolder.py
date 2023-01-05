"""
tracefolder - run all files under a folder with monkeytrace.

Usage:
  tracefolder [-t timeout] <module> <folder> [<constraint>]
  tracefolder -h | --help
  tracefolder --version

Options:
  -t timeout   Timeout in seconds.
  <module>     The module to trace.
  <folder>     The folder containing the files to run.
  <constraint> An optional substring that must occur in path of file.
  -h --help    Show this screen.
  --version    Show version.

"""

import os
import subprocess
import sys
import sqlite3
from docopt import docopt, DocoptExit


__version__ = '1.0'


def dedup(db_file):
    conn = sqlite3.connect(db_file)
    with conn:
        conn.execute("""
create table dedupped
(
  count       INTEGER,
  module      TEXT,
  qualname    TEXT,
  arg_types   TEXT,
  return_type TEXT,
  yield_type  TEXT);
""")
        conn.execute("""
insert into dedupped(count, module, qualname, arg_types, return_type, yield_type)
select sum(count), module, qualname, arg_types, return_type, yield_type from monkeytype_call_traces
group by module, qualname, arg_types, return_type, yield_type;
""")
        conn.execute("drop table monkeytype_call_traces;")
        conn.execute("alter table dedupped rename to monkeytype_call_traces;")
    with conn:
        conn.execute("vacuum;")



def recurse(dir, files, constraint):
  for entry in os.listdir(dir):
    path = os.path.join(dir, entry)
    if os.path.isdir(path):
      recurse(path, files, constraint)
    elif path.endswith('.py'):
      if not constraint or path.find(constraint) >= 0:
          files.append(path)
  return files


if __name__ == '__main__':
    arguments = docopt(__doc__, version=__version__)
    root = os.path.realpath(arguments['<folder>'])
    module = arguments['<module>']
    timeout = arguments['-t'] 
    if timeout:
        timeout = int(timeout)
    else:
        timeout = 300
    dbfile = module + '.sqlite3'
    env = {}
    env.update(os.environ)
    env['MONKEYTYPE_TRACE_MODULES'] = module
    env['MT_DB_PATH'] = dbfile
    files = []
    recurse(root, files, arguments['<constraint>'])
    total = len(files)
    num = 1
    for item in files:
        print(f'====[ {num}/{total}  ]=============================================================')
        cmd = ["python3", "trace.py", item]
        try:
            subprocess.run(cmd, env=env, timeout=timeout)
        except Exception:
            pass
        num += 1
        if os.path.getsize(dbfile) > 100000000:
            dedup(dbfile)

    dedup(dbfile)

