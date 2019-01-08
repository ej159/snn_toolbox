from __future__ import division
import numpy as np
from keras.models import load_model
from keras.layers import Activation, InputLayer
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import mnist
import keras.utils as utils
from keras.models import model_from_json
from keras.activations import softplus
import keras


K.set_image_dim_ordering('tf')

def noisy_softplus(x, k=0.17, sigma=0.5):
    return sigma*k*K.softplus(x/(sigma*k))

get_custom_objects().update({'noisy_softplus': noisy_softplus})


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_test.shape)

x_train = x_train.reshape((60000, 1, 28, 28))
x_test = x_test.reshape((10000, 1, 28, 28))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255 
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = load_model('lenet5.h5')
'''
score = model.evaluate(x_test, y_test, batch_size = 1, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

new_model = type(model)()

new_model.inputs = model.inputs


ap_counter=1

for layer in model.layers:
    
    layerconf = layer.get_config()
    print(layerconf)
    
    #changing activation
    if hasattr(layer, 'activation') and layer.activation != 'relu':
        layerconf['activation'] = 'relu'
     
    #removing bias
    if hasattr(layer, 'use_bias') and layerconf['use_bias']:
        print("Found biased layer")
        layerconf['use_bias'] = False
        weights, biases = layer.get_weights()
        '''weights[weights<0.0] = 1.0
        
        weights = weights + np.abs(np.amin(weights))'''
        layerconf['weights'] = [weights]
    
    '''#enforcing non-negativity of weights
    if hasattr(layer, 'kernel_constraint'):
        layerconf['kernel_constraint'] = keras.constraints.NonNeg()
     '''
    '''
    #changing maxpooling to avgpooling
    if type(layer).__name__ == 'MaxPooling2D':
        layerconf['name'] = "avg_pooling2d_" + str(ap_counter)
        ap_counter += 1
        layer = keras.layers.AveragePooling2D.from_config(layerconf)
    else:
        layer = type(layer).from_config(layerconf)
    '''
    new_model.add(layer.from_config(layerconf))

model = new_model

model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

for layer in model.layers:
    if hasattr(layer, 'activation'):
        print(layer.activation)

score = model.evaluate(x_test, y_test, batch_size = 1, verbose=1)
print('Pre-retrain Test loss:', score[0])
print('Pre-retrain Test accuracy:', score[1])
'''
epochs = 2
print('Retraining network for %d epochs...' % epochs)



model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=epochs,
                    verbose=1,
validation_data=(x_test, y_test))

print('Evaluating retrained network...')

score = model.evaluate(x_test, y_test, batch_size = 1, verbose=1)
print('Retrained Test loss:', score[0])
print('Retrained Test accuracy:', score[1])
'''
model.save("lenet5_retrained.h5")
