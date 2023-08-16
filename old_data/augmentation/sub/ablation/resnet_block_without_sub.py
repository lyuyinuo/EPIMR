import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
# import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import random
my_seed = 666
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

def bottleneck(index, input, kernel_num, stride, layer_name='bottleneck'):
    k1, k2, k3 = kernel_num
    x = input
    if index == 0:
        conv1 = Conv2D(filters=k1, kernel_size=1, strides=stride, padding='valid',
                       name=layer_name + '_conv1')(input)
    else:
        conv1 = Conv2D(filters=k1, kernel_size=1, strides=1, padding='valid',
                       name=layer_name + '_conv1')(input)
    batn1 = BatchNormalization(name=layer_name + '_bn1')(conv1)
    relu1 = ReLU(name=layer_name + '_relu1')(batn1)

    conv2 = Conv2D(filters=k2, kernel_size=3, strides=1, padding='same',
                   name=layer_name + '_conv2')(relu1)
    batn2 = BatchNormalization(name=layer_name + '_bn2')(conv2)
    relu2 = ReLU(name=layer_name + '_relu2')(batn2)

    conv3 = Conv2D(filters=k3, kernel_size=1, strides=1, padding='valid',
                   name=layer_name + '_conv3')(relu2)
    batn3 = BatchNormalization(name=layer_name + '_bn3')(conv3)

    if index == 0:
        identity = Conv2D(filters=k3, strides=stride, kernel_size=1, padding='valid',
                          name=layer_name + '_shortcut_conv')(x)
        identity = BatchNormalization(name=layer_name + '_shortcut_bn')(identity)
    else:
        identity = x
    shortcut_add = Add(name=layer_name + '_add')([identity, batn3])
    relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)

    return relu3


def basic_block(index, input, kernel_num, stride, layer_name='basic'):  # resnet18/34
    x = input
    if index == 0:
        conv1 = Conv2D(filters=kernel_num, strides=stride, kernel_size=3, padding='same',
                       name=layer_name + '_conv1')(input)
    else:
        conv1 = Conv2D(filters=kernel_num, strides=1, kernel_size=3, padding='same',
                       name=layer_name + '_conv1')(input)
    batn1 = BatchNormalization(name=layer_name + '_bn1')(conv1)
    relu1 = ReLU(name=layer_name + '_relu1')(batn1)

    conv2 = Conv2D(filters=kernel_num, strides=1, kernel_size=3, padding='same',
                   name=layer_name + '_conv2')(relu1)
    batn2 = BatchNormalization(name=layer_name + '_bn2')(conv2)

    if index == 0:
        identity = Conv2D(filters=kernel_num, strides=stride, kernel_size=1, padding='valid',
                          name=layer_name + '_shortcut_conv')(x)
        identity = BatchNormalization(name=layer_name + '_shortcut_bn')(identity)
    else:
        identity = x
    shortcut_add = Add(name=layer_name + '_add')([identity, batn2])
    relu2 = ReLU(name=layer_name + '_relu2')(shortcut_add)
    return relu2


def make_layer(input, stride, block, block_num, kernel_num, layer_name):
    x = input
    for j in range(block_num):
        x = block(index=j, input=x, kernel_num=kernel_num, stride=stride, layer_name=layer_name + str(j))
    return x


def ResNet(input_shape, net_name):  # ='resnet18'
    """
        :param input_shape:
        :param block:
        :return:
    """
    block_setting = {}
    block_setting['resnet18'] = {'block': basic_block, 'block_num': [2, 2, 2, 2], 'kernel_num': [64, 128, 256, 512]}
    block_setting['resnet34'] = {'block': basic_block, 'block_num': [3, 4, 6, 3], 'kernel_num': [64, 128, 256, 512]}
    block_setting['resnet50'] = {'block': bottleneck, 'block_num': [3, 4, 6, 3],
                                 'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet101'] = {'block': bottleneck, 'block_num': [3, 4, 23, 3],
                                  'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                 [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet152'] = {'block': bottleneck, 'block_num': [3, 8, 36, 3],
                                  'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                 [256, 256, 1024], [512, 512, 2048]]}
    # net_name = 'resnet18' if not block_setting.__contains__(net_name) else net_name
    block_num = block_setting[net_name]['block_num']
    kernel_num = block_setting[net_name]['kernel_num']
    block = block_setting[net_name]['block']

    input_X = Input(shape=input_shape)
    zerop_X = ZeroPadding2D((3, 3))(input_X)
    conv1_X = Conv2D(filters=64, strides=2, kernel_size=7, name='first_conv_X')(zerop_X)
    batn1_X = BatchNormalization(name='first_bn_X')(conv1_X)
    pool1_X = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1_X')(batn1_X)

    conv_X = pool1_X
    conv_X0 = make_layer(conv_X, stride=1, block=block, block_num=block_num[0], kernel_num=kernel_num[0],
                         layer_name='layer_X' + str(1))
    conv_X1 = make_layer(conv_X0, stride=2, block=block, block_num=block_num[1], kernel_num=kernel_num[1],
                         layer_name='layer_X' + str(2))
    conv_X2 = make_layer(conv_X1, stride=2, block=block, block_num=block_num[2], kernel_num=kernel_num[2],
                         layer_name='layer_X' + str(3))
    conv_X3 = make_layer(conv_X2, stride=2, block=block, block_num=block_num[3], kernel_num=kernel_num[3],
                         layer_name='layer_X' + str(4))


    pool2_X0 = GlobalAvgPool2D(name='globalavgpool_X0')(conv_X0)
    pool2_X1 = GlobalAvgPool2D(name='globalavgpool_X1')(conv_X1)
    pool2_X2 = GlobalAvgPool2D(name='globalavgpool_X2')(conv_X2)
    pool2_X3 = GlobalAvgPool2D(name='globalavgpool_X3')(conv_X3)

    pool2_X = Concatenate()([pool2_X0, pool2_X1, pool2_X2, pool2_X3])
    pool2_X = Dropout(0.75)(pool2_X)


    input_Y = Input(shape=input_shape)
    zerop_Y = ZeroPadding2D((3, 3))(input_Y)
    conv1_Y = Conv2D(filters=64, strides=2, kernel_size=7, name='first_conv_Y')(zerop_Y)
    batn1_Y = BatchNormalization(name='first_bn_Y')(conv1_Y)
    pool1_Y = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1_Y')(batn1_Y)

    conv_Y = pool1_Y
    conv_Y0 = make_layer(conv_Y, stride=1, block=block, block_num=block_num[0], kernel_num=kernel_num[0],
                        layer_name='layer_Y' + str(1))
    conv_Y1 = make_layer(conv_Y0, stride=2, block=block, block_num=block_num[1], kernel_num=kernel_num[1],
                        layer_name='layer_Y' + str(2))
    conv_Y2 = make_layer(conv_Y1, stride=2, block=block, block_num=block_num[2], kernel_num=kernel_num[2],
                        layer_name='layer_Y' + str(3))
    conv_Y3 = make_layer(conv_Y2, stride=2, block=block, block_num=block_num[3], kernel_num=kernel_num[3],
                        layer_name='layer_Y' + str(4))


    pool2_Y0 = GlobalAvgPool2D(name='globalavgpool_Y0')(conv_Y0)
    pool2_Y1 = GlobalAvgPool2D(name='globalavgpool_Y1')(conv_Y1)
    pool2_Y2 = GlobalAvgPool2D(name='globalavgpool_Y2')(conv_Y2)
    pool2_Y3 = GlobalAvgPool2D(name='globalavgpool_Y3')(conv_Y3)

    pool2_Y = Concatenate()([pool2_Y0, pool2_Y1, pool2_Y2, pool2_Y3])
    pool2_Y = Dropout(0.75)(pool2_Y)


    attention_add = subtract([pool2_X, pool2_Y], name='attention_subtract')
    attention_mul = multiply([pool2_X, pool2_Y], name='attention_mul')

    XY = Concatenate()([pool2_X, pool2_Y, attention_mul])
    XY = BatchNormalization()(XY)
    XY = Dropout(0.75)(XY)

    XY = Dense(1, activation='sigmoid', name='dense',
               kernel_initializer=glorot_uniform(seed=None))(XY)
               # kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l2(0.001))(XY)

    model = Model(inputs=[input_X, input_Y], outputs=XY)

    return model