import keras
import numpy as np
from keras.applications import mobilenet
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
import matplotlib.pyplot as plt


from os.path import expanduser
home = expanduser("~")
keras.backend.set_image_data_format("channels_last")

def get_inbound_layers(layer):
    in_layers = []
    for node in layer._inbound_nodes:
        for in_layer in node.inbound_layers:
            in_layers.append(in_layer)
    return in_layers

#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')


print(get_inbound_layers(mobilenet_model.layers[3]))




