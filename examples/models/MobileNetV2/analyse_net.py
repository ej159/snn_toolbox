import keras
import numpy as np
from keras.applications import mobilenetv2

from os.path import expanduser
home = expanduser("~")

mobilenet_model = mobilenetv2.MobileNetV2()

layer_types = []

for layer in mobilenet_model.layers:
    if type(layer) .__name__ not in layer_types:
        layer_types.append(type(layer).__name__)
        
print(layer_types)
