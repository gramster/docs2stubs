#import importlib
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import monkeytype
#import gi
#gi.require_version('Gtk', '3.0')
#from gi.repository import Gtk



def recurse(dir, files):
  for entry in os.listdir(dir):
    path = os.path.join(dir, entry)
    if os.path.isdir(path):
      recurse(path, files)
    elif path.endswith('.py'):
      files.append(path)
  return files


def dummy():
  pass


root = os.path.realpath('submodules/matplotlib/examples')
ex_files = recurse(root, [])
ex_total = len(ex_files)
ex_num = 0
skip = [
  'image_thumbnail_sgskip.py',
  'invert_axes.py',
  'zoom_inset_axes.py',
  'fahrenheit_celsius_scales.py',
  'axes_box_aspect.py',
  'embedding_webagg_sgskip.py',
  'multiprocess_sgskip.py', # This is an evil spawner
]

with monkeytype.trace():
  for ex_path in ex_files:
    ex_num += 1
    if any([ex_path.endswith(s) for s in skip]):
        continue
    with open(ex_path) as f:
      ex_content = f.read()
    print(f'{ex_num}/{ex_total}: {ex_path}')
    try:
      plt.show = dummy
      #Gtk.main = dummy
      exec(ex_content)
      plt.close('all')
    except Exception as e:
      print(f'{ex_path}: {e}')




