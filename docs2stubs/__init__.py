__doc__ = """
docs2stubs.

Usage:
  docs2stubs analyze [--skip-trivial] [--trace_folder FOLDER] <name>...
  docs2stubs stub [--strip-defaults] [--skip-analysis] [--stub_folder FOLDER] [--trace_folder FOLDER] <name>...
  docs2stubs test [<name>] <typestring>
  docs2stubs -h | --help
  docs2stubs --version

Options:
  --skip-trivial        Exclude trivial cases from the map files.
  --strip-defaults      Replace parameter default values with ...
  --skip-analysis       Skip analysis and use existing analysis file.
  --stub_folder FOLDER  Folder where stubs are stored [default: typings].
  --trace_folder FOLDER Folder where traces are stored [default: tracing].
  <name>                The target package or module (e.g. matplotlib.pyplot).

The module needs to be importable in the environment from which
you are running docs2stubs.

The `analyze` command parses numpydoc-format docstrings in the target
and produces files 'analysis/<name>.<what>.map.missing' which lists the
non-trivial discovered types in order of frequency, along with a normalized version
(separated by ':'). The normalized version is a 'best-effort' corrected
version of the type suitable for re-insertion in type stubs. The fields
are separated by '#'. The analysis phase also outputs a `.imports.map`
file enumerating the discovered classes and their modules; this file can
be augmented with additional class names and owner modules which will 
be used when stubbing to determine what imports might need to be added.

Separate files are generated for each of parameters, return types, or
attributes (the <what> in the filenames above).

After analysis a 'human in the loop' pass of the .map file is 
recommended to do further cleanup of the normalized forms. Once the 
map file is cleaned up, it can be renamed '.map' instead of 
'.map.missing', and will be used during stub generation to supplement
the logic of the type normalizer.

The `analyze` command just does the first phase of gathering types
and writing the `.map.missing` files. It will output a summary 
at the end of how many of the docstring types were found in the 
`.map` files (if present), how many were trivially convertible 
to type annotations, and how many were unhandled ('missed') and 
thus written to the `map.missing` files. Note that the map files
contain just one entry per type, while a type can occur multiple
times, so the output summary counts are typically much higher 
than the count of lines in the mapfiles.

The `stub` command will generate stubs for the module under a 
'typings' folder. It will first run an analysis pass and
then make use of the .map file produced by analysis to 
determine types, and for assignments or default parameter values with
no normalized type available, will infer the type from the assigned
value if possible. It can also make use of monkeytype traces to
both augment these types and add types elsewhere where there are 
no docstrings.

Parameter default values are usually preserved in the stubs if they are
scalars, but if you specify `--strip-defaults` they will all be removed
and replaced with '...'.

The `test` command can be used to test how a type string will be 
normalized by the program. A module or package name can optionally
be specified in which case the `.imports.map` file for that module
or package will be used (if it exists) to decide what are valid classnames.
"""

__version__ = '0.1'

from docopt import docopt
from .stubbing_transformer import stub_module
from .analyzing_transformer import analyze_module
from .traces import init_trace_loader
from .type_normalizer import check_normalizer


def main():
    arguments = docopt(__doc__, version=__version__)  # type: ignore
    name = arguments['<name>']
    for n in name:
      if arguments['analyze']:
        dump_all = not arguments['--skip-trivial']
        trace_folder = arguments['--trace_folder']  
        init_trace_loader(trace_folder, n)
        analyze_module(n, output_trivial_types=dump_all)
      elif arguments['stub']:
        strip_defaults = arguments['--strip-defaults']
        skip_analysis = arguments['--skip-analysis']
        stub_folder = arguments['--stub_folder']
        trace_folder = arguments['--trace_folder']
        init_trace_loader(trace_folder, n)
        stub_module(n, strip_defaults=strip_defaults, skip_analysis=skip_analysis, 
                    stub_folder=stub_folder)
      elif arguments['test']:
        print(check_normalizer(arguments["<typestring>"], name))

    