from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
import numpy as np
from Othello import *
import tensorflow as tf

g = Othello()
blocks = 10
nn_size = 128
c = 0.0001

def conv_layer(inp):
    conv1 = Conv2D(nn_size, (g.winNum,g.winNum), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    return relu1

def res_layer(inp):
    shortcut = inp
    
    conv1 = Conv2D(nn_size, (g.winNum,g.winNum), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    conv2 = Conv2D(nn_size, (g.winNum,g.winNum), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(relu1)
    bn2 = BatchNormalization()(conv2)
    # keras.layers.add
    x = bn2 + shortcut
    relu2 = ReLU()(x)

    return relu2

def value_head(inp):
    conv1 = Conv2D(1, (1,1), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    flatten = Flatten()(relu1)
    dense1 = Dense(256, kernel_regularizer=l2(c), bias_regularizer=l2(c))(flatten)
    relu = ReLU()(dense1)
    dense2 = Dense(1, activation='tanh', kernel_regularizer=l2(c), bias_regularizer=l2(c))(relu)

    return dense2

def policy_head(inp, out):
    conv1 = Conv2D(2, (1,1), padding='same', kernel_regularizer=l2(c), bias_regularizer=l2(c))(inp)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    flatten = Flatten()(relu1)
    dense = Dense(out, activation='softmax', kernel_regularizer=l2(c), bias_regularizer=l2(c))(flatten)

    return dense

def create_model(shape, out):
    inputs = keras.Input(shape=shape)
    
    layer = conv_layer(inputs)
    for _ in range(blocks):
        layer = res_layer(layer)
        
    val_outputs = value_head(layer)
    pol_outputs = policy_head(layer,out)
    outputs = [pol_outputs, val_outputs]
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=SGD(learning_rate=0.02, momentum=0.9),metrics=['accuracy'])
    return model
