import sys
import os
#path to snn_toolbox
sys.path.append('/home/edwardjones/git/snn_toolbox/')
import snntoolbox.bin.utils

#lenet_5_filepath = '/home/edwardjones/git/snn_toolbox/examples/models/lenet5/keras/config_spinnaker'
cwd = os.getcwd()
lenet_5_filename = 'config_retrained'
lenet_5_filepath = cwd + '/' + lenet_5_filename

from snntoolbox.bin.utils import update_setup
config = update_setup(lenet_5_filepath)

from snntoolbox.bin.utils import test_full
test_full(config)
