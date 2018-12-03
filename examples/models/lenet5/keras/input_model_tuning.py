from keras.models import load_model
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

model = load_model('lenet5.h5')


def noisy_softplus(x, k=1, sigma=1):
    return K.multiply(sigma, k, K.softplus(K.div(x, (K.multiply(sigma, k)))))

get_custom_objects().update({'noisy_softplus': Activation(noisy_softplus)})

model.add(Activation(noisy_softplus))