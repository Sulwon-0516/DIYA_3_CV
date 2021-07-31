from __future__ import absolute_import
import os 
import sys

#print(os.getcwd())
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir,'..')

if lib_path not in sys.path:
    sys.path.insert(0,lib_path)