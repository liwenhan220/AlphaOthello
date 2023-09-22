import tensorflow as tf
from cnn import *
import numpy as np

nnet = load_model('nnet')
tf.saved_model.save(nnet, 'trt_init')

