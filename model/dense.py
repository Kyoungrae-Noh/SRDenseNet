from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.utils.vis_utils import plot_model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import time
import os

def SRDenseNetBlock(input, i, nlayers):
    logits = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu",
                    use_bias=True, name="conv2d_%d_%d" % (i+1, 0+1))(input)

    for j in range(1, nlayers):
        middle = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu",
                        use_bias=True, name="conv2d_%d_%d" %(i+1, j+1))(logits)
        logits = concatenate([logits, middle], name="concatenate_%d_%d" % (i+1, j+1))

        return logits


def SRDenseNet(nblocks=8, nlayers=8):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = Conv2D(filters=16, kernel_size=3, strides=1,
              padding='SAME', activation='relu', use_bias=True)(x)

    G = x

    for i in range(nblocks):
        x = SRDenseNetBlock(x, i, nlayers)
        x = concatenate([x, G])

    x = Conv2D(filters=256, kernel_size=1, padding='SAME',
               activation='relu', use_bias = True)(x)

    x = Conv2DTranspose(filters=256, kernel_size=3, strides=2,
                        padding='SAME', activation='relu', use_bias=True)(x)
    x = Conv2DTranspose(filters=256, kernel_size=3, strides=2,
                        padding='SAME', activation='relu', use_bias=True)(x)
    x = Conv2D(filters=1, kernel_size=3, padding='SAME', use_bias=True)(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="DenseNet")









##################################################################################################################
def Concatenation(layers):
    return tf.concat(layers, axis=3)


def SkipConnect(conv):
    skipconv = list()
    for i in conv:
        x = Concatenation(i)
        skipconv.append(x)
    return skipconv


def dense():
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = desBlock()



def desBlock(desBlock_layer, outlayer, filter_size=3):
    # nextlayer = self.low_conv
    nextlayer = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='SAME')(x)

    conv = list()
    for i in range(1, outlayer+1):
        conv_in = list()
        for j in range(1, desBlock_layer+1):
            # The first conv need connect with low level layer
            if j is 1:
                x = tf.nn.conv2d(nextlayer, self.weight_block['w_H_%d_%d' %(i, j)], strides = [1, 1, 1, 1], padding='SAME') + self.biases_blocks['b_H%d_%d' %(i, j)]
                x = tf.nn.relu(x)
                conv_in.append(x)
            else:
                x = Concatenation(conv_in)
                x = tf.nn.conv2d(x, self.weight_block['w_H_%d_%d' %(i, j)], strides = [1, 1, 1, 1], padding='SAME') + self.biases_block['b_H_%d_%d' %(i, j)]
                x = tf.nn.relu(x)
                conv_in.append(x)
        nextlayer = conv_in[-1]
        conv.append(conv_in)
    return conv