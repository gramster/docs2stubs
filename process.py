__doc__ = f"""
process - simple test driver for docs2stubs for analyzing, stubbing, and augmenting modules
from within VS Code.

Usage:
  process (analyze|stub|augment|all) <package>
  process -h | --help
  process --version

Options:
  <package>             The target package (e.g. matplotlib or sklearn).

The package/module needs to be installed in the environment from which
you are running this script.
"""

__version__ = '0.1'


from docopt import docopt
from docs2stubs.analyzing_transformer import analyze_module
from docs2stubs.stubbing_transformer import stub_module


def main():
    arguments = docopt(__doc__, version=__version__)  # type: ignore
    package = arguments['<package>']
    if arguments['analyze'] or arguments['all']:
        analyze_module(package, output_trivial_types=False)
    if arguments['stub'] or arguments['all']:
        stub_module(package, skip_analysis=True)

if __name__ == '__main__':
    main()
