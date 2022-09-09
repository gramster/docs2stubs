"""
docs2stubs.

Usage:
  docs2stubs analyze (package|module) [--include-counts] <name>...
  docs2stubs stub (package|module) [--strip-defaults] <name>...
  docs2stubs -h | --help
  docs2stubs --version

Options:
  --strip-defaults  Replace parameter default values with ...
  <name>            The target package or module (e.g. matplotlib.pyplot).

The package/module needs to be installed in the environment from which
you are running docs2stubs.

The analyze command parses numpydoc-format docstrings in the target
and produces files 'analysis/<name>.<what>.map.missing' which lists the
discovered types in order of frequency, along with a normalized version
(separated by ':'). The normalized version is a 'best-effort' corrected
version of the type suitable for re-insertion in type stubs. The fields
are separated by '#'.

Separate files are generated for each of parameters, return types, or
attributes (the <what> in the filenames above).

If --include-counts is specified, then the output files include the
frequency counts in the first field.

After analysis a 'human in the loop' pass of the .map file is 
recommended to do further cleanup of the normalized forms. Once the 
map file is cleaned up, it can be renamed '.map' instead of 
'.map.missing', and will be used during stub generation to supplement
the logic of the type normalizer.

The stub command will generate stubs for the module/package under a 
'typings' folder. It will make use of the .map file produced by analysis to 
determine types, and for assignments or default parameter values with
no normalized type available, will infer the type from the assigned
value if possible.

"""

__version__ = '0.1'

from xml.etree.ElementInclude import include
from docopt import docopt, DocoptExit
from .stubber import stub_module
from .analyzer import analyze_module


def main():
    arguments = docopt(__doc__, version=__version__)
    name = arguments['<name>']
    include_submodules = False if arguments['module'] else True
    for n in name:
      if arguments['analyze']:
        include_counts = arguments['--include-counts']
        analyze_module(n, include_submodules=include_submodules, include_counts=include_counts)
      elif arguments['stub']:
        strip_defaults = arguments['--strip-defaults']
        stub_module(n, include_submodules=include_submodules, \
            strip_defaults=strip_defaults)
    