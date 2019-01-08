import sys
from os.path import expanduser
import os

#path to snn_toolbox
home = expanduser("~")
sys.path.append(home +'/git/snn_toolbox/scripts/')
import snntoolbox.bin.utils

#lenet_5_filepath = '/home/edwardjones/git/snn_toolbox/examples/models/lenet5/keras/config_spinnaker'
cwd = os.getcwd()
mobilenet_filename = 'config'
mobilenet_filepath = cwd + '/' + mobilenet_filename

from snntoolbox.bin.utils import update_setup
config = update_setup(mobilenet_filepath)

from snntoolbox.bin.utils import test_full
test_full(config)