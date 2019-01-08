import sys
from os.path import expanduser
import os
import numpy
import matplotlib.pyplot as pyplot

#path to snn_toolbox
home = expanduser("~")
sys.path.append(home +'/git/snn_toolbox/scripts/')
import snntoolbox.bin.utils

#lenet_5_filepath = '/home/edwardjones/git/snn_toolbox/examples/models/lenet5/keras/config_spinnaker'
cwd = os.getcwd()
lenet_5_filename = 'config'
lenet_5_filepath = cwd + '/' + lenet_5_filename

from snntoolbox.bin.utils import update_setup
config = update_setup(lenet_5_filepath)

'''
with numpy.load(config.get('paths', 'dataset_path')+"/x_test.npz") as data:
    first = data.f.arr_0
    pyplot.imshow(first[0].reshape(28,28))
    pyplot.show()'''

from snntoolbox.bin.utils import test_full
test_full(config)
