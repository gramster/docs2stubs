"""
trace - run a file with monkeytrace. This adds some patching to try to 
    prevent being blocked by plot popups.

Usage:
  trace <file>
  trace -h | --help
  trace --version

Options:
  -h --help    Show this screen.
  --version    Show version.

"""

import sys
import matplotlib
import matplotlib.pyplot as plt
import monkeytype
#import gi
#gi.require_version('Gtk', '3.0')
#from gi.repository import Gtk
from docopt import docopt, DocoptExit


__version__ = '1.0'


def dummy():
  pass


# NOTE: calling exec below can change variables, and so we name
# our variable with a prefix that means they are unlikely to
# be shadowed and messed with.


if __name__ == '__main__':
    arguments = docopt(__doc__, version=__version__)
    __mtt_file = arguments['<file>']
    with monkeytype.trace():
        with open(__mtt_file) as f:
            content = f.read()
        try:
            plt.show = dummy
            #Gtk.main = dummy
            exec(content)
            plt.close('all')
        except Exception as e:
            print(f'{__mtt_file}: {e}')
    sys.exit(0)


