import logging
logging.basicConfig(level=logging.DEBUG)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent/".."/"src"/"python"))

from PyCXpress import debug_array

def show(a):
    debug_array(a)