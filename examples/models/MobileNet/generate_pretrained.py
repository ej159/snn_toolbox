#Adapted from tutorial at https://www.learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/
import tensorflow
from tensorflow.python import pywrap_tensorflow
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
checkpoint_path = home+'/Downloads/Mobilenet_checkpoint/mobilenet_v1_0.25_128.ckpt'

'''from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'relu6': keras.layers.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    mobilenet_model = load_model('mobilenet.h5')'''

#Load the MobileNet model from tf checkpoint
#Adapted from https://stackoverflow.com/questions/44466066/how-can-i-convert-a-trained-tensorflow-model-to-keras
'''with tf.Session() as sess:

    # import graph
    saver = tf.train.import_meta_graph(checkpoint_path+".meta", clear_devices=True)

    # load weights for graph
    saver.restore(sess, checkpoint_path)

    # get all global variables (including model variables)
    vars_global = tf.global_variables()

    # get their name and value and put them into dictionary
    sess.as_default()
    model_vars = {}
    for var in vars_global:
        try:
            model_vars[var.name] = var.eval()
        except:
            print("For var={}, an exception occurred".format(var.name))
    
    print(model_vars.keys())

#from https://stackoverflow.com/questions/41621071/restore-subset-of-variables-in-tensorflow
def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = get_default_graph().get_tensor_by_name(tensor_name+":0")
            full_var_list.append(tensor_aux)
        except:
            print('Not found: '+tensor_name)
    return full_var_list
    
def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
    varlist=[]
    var_value =[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)  
    
class RestoreCkptCallback(keras.callbacks.Callback):
    
    def __init__(self, pretrained_file):
        self.pretrained_file = pretrained_file
        self.sess = keras.backend.get_session()
        self.all_variables = tensorflow.get_collection_ref(tensorflow.GraphKeys.GLOBAL_VARIABLES)
        self.restorable_variables = get_tensors_in_checkpoint_file(self.pretrained_file)
        self.tensors_to_load = build_tensors_in_checkpoint_file(self.restorable_variables)
        self.loader = tensorflow.train.Saver(self.tensors_to_load)
        
    def on_train_begin(self, logs=None):
        self.sess.run(variables_initializer(all_variables))
        if self.pretrained_file:
            self.loader.restore(self.sess, self.pretrained_file)
            print('load weights: OK.')
'''    
mobilenet_model = keras.applications.mobilenet.MobileNet(alpha=0.25)
mobilenet_model.compile(loss='categorical_crossentropy', optimizer='adam')
mobilenet_model.summary
#restore_ckpt_callback = RestoreCkptCallback(pretrained_file=checkpoint_path) 
#mobilenet_model.fit(steps_per_epoch=0,callbacks=[restore_ckpt_callback])
 
filename = home + '/git/PyNN8Examples/examples/ANN_Conversion/images/hotairballoon.jpg'
# load an image in PIL format
original = load_img(filename, target_size=(224, 224))
#print('PIL image size',original.size)
#plt.imshow(original)
#plt.show()
 
# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
#plt.imshow(np.uint8(numpy_image))
#plt.show()
#print('numpy array size',numpy_image.shape)
 
# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
#print('image batch size', image_batch.shape)
#plt.imshow(np.uint8(image_batch[0]))

# prepare the image for the  model
processed_image = preprocess_input(image_batch.copy())
 
# get the predicted probabilities for each class
predictions = mobilenet_model.predict(processed_image)
#print predictions
 
# convert the probabilities to class labels
# We will get top 5 predictions which is the default
label = decode_predictions(predictions)
print label


mobilenet_model.summary()

with open("mobilenet.json", "w") as text_file:
    text_file.write(mobilenet_model.to_json())
mobilenet_model.save_weights('mobilenet.h5')



