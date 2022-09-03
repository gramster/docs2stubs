"""
docs2stubs.

Usage:
  docs2stubs analyze <module>
  docs2stubs bulk-analyze <module>...
  docs2stubs stub [--strip-defaults] <module>
  docs2stubs -h | --help
  docs2stubs --version

Options:
  --strip-defaults       Replace parameter default values with ...
  <module>               The target module (e.g. matplotlib.pyplot).

The package that has the module needs to be installed in the environment from which
you are running docs2stubs.

The analyze command parses numpydoc-format docstrings in the target
module and produces a file 'analysis/<module>.type' which lists the
discovered types in order of frequency, along with a normalized version
(separated by ':'). The normalized version is a 'best-effort' corrected
version of the type suitable for re-insertion in type stubs. If the
normalized version needs more cleanup it will be prefixed with '#'.

After analysis a 'human in the loop' pass of the .type file is 
recommended to do further cleanup of the normalized forms, removing
the '#' if they are now suitable for insertion.

The stub command will generate stubs for the module under a 'typings'
folder. It will make use of the .type file produced by analysis to 
determine types, and for assignments or default parameter values with
no normalized type available, will infer the type from the assigned
value if possible.

"""

__version__ = '0.1'

from docopt import docopt, DocoptExit
from .stubber import stub_module
from .analyzer import analyze_module


def main():
    arguments = docopt(__doc__, version=__version__)
    module = arguments['<module>']
    if arguments['analyze']:
      analyze_module(module[0])
    elif arguments['bulk-analyze']:
      analyze_module(module)
    elif arguments['stub']:
      strip_defaults = arguments['--strip-defaults']
      stub_module(module[0], strip_defaults)
    
