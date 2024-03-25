import logging
logging.basicConfig(level=logging.DEBUG)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent/".."/"src"/"python"))

from PyCXpress import debug_array
from PyCXpress import InputDataSet, OutputDataSet
from contextlib import nullcontext

def show(a):
    debug_array(a)

def init():
    return InputDataSet(), OutputDataSet()

def model(input: InputDataSet, output: OutputDataSet):
    with nullcontext():
        output.output_a = input.input_a + input.input_b