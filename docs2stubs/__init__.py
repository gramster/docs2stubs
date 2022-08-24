"""
docs2stubs.

Usage:
  docs2stubs <module>
  docs2stubs -h | --help
  docs2stubs --version

Options:
  <module>               The module to generate stubs for (e.g. matplotlib.pyplot).

The package that has the module needs to be installed in the environment from which
you are running docs2stubs.
"""

__version__ = '0.1'

from docopt import docopt, DocoptExit
from .docs2stubs import docs2stubs


def main():
    arguments = docopt(__doc__, version=__version__)
    module = arguments['<package>']
    docs2stubs(module)
    
