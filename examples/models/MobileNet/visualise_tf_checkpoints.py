import keras
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from os.path import expanduser
home = expanduser("~")
keras.backend.set_image_data_format("channels_last")
checkpoint_path = home+'/Downloads/Mobilenet_checkpoint/mobilenet_v1_0.25_128_eval.pbtxt'



print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False)


import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename = checkpoint_path
    with tf.gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='/tmp/log'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

model = keras.applications.mobilenet.MobileNet()
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()