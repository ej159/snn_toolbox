import sys
from os.path import expanduser
import os
import numpy
import matplotlib.pyplot as pyplot

#path to snn_toolbox
home = expanduser("~")
sys.path.append(home +'/git/snn_toolbox/')
import snntoolbox.bin.utils

cwd = os.getcwd()
lenet_5_filename = 'config'
lenet_5_filepath = cwd + '/' + lenet_5_filename

from snntoolbox.bin.utils import update_setup
config = update_setup(lenet_5_filepath)

from snntoolbox.bin.utils import test_full
test_full(config)
