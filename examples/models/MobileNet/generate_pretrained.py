#Adapted from tutorial at https://www.learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/

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

'''from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    mobilenet_model = load_model('mobilenet.h5')'''

#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet') 

 
filename = home + '/git/PyNN8Examples/examples/ANN_Conversion/images/hotairballoon.jpg'
# load an image in PIL format
original = load_img(filename, target_size=(224, 224))
print('PIL image size',original.size)
#plt.imshow(original)
#plt.show()
 
# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
#plt.imshow(np.uint8(numpy_image))
#plt.show()
print('numpy array size',numpy_image.shape)
 
# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
#plt.imshow(np.uint8(image_batch[0]))

# prepare the image for the  model
processed_image = preprocess_input(image_batch.copy())
 
# get the predicted probabilities for each class
predictions = mobilenet_model.predict(processed_image)
print predictions
 
# convert the probabilities to class labels
# We will get top 5 predictions which is the default
label = decode_predictions(predictions)
print label


mobilenet_model.summary()

with open("mobilenet.json", "w") as text_file:
    text_file.write(mobilenet_model.to_json())
mobilenet_model.save_weights('mobilenet.h5')



