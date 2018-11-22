import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import DepthwiseConv2D
from keras.layers import Input, Dense
import numpy as np

keras.data_format='channel_last'

model1 = Sequential()
model2 = Sequential()

den = Dense(10, input_shape=(10, 10, 3))

layer = DepthwiseConv2D(kernel_size=3, strides=1, kernel_initializer='glorot_normal', bias_initializer='glorot_normal', depth_multiplier=2, use_bias=False)

model1.add(den)
model1.add(layer)


conf = layer.get_config()
'''conf['weights'] = [0]
weights = layer.get_weights()
conf['weights'] = weights'''

print(len(layer.get_weights()))
print(layer.get_weights()[0].shape)

weights = layer.get_weights()

weights.append(np.zeros(20))

conf['weights'] = weights

conf['use_bias'] = True
print(len(weights))

layer2 = DepthwiseConv2D(**conf)

model2.add(den)
model2.add(layer2)

print(layer2.get_weights()[0].shape)

