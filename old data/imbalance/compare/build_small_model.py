import util

# Keras imports
from keras.layers import Input, Conv1D, MaxPooling1D, merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, \
    Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.regularizers import l1, l2

# model parameters
enhancer_length = 3000 # TODO: get this from input
promoter_length = 2000 # TODO: get this from input
n_kernels = 200 # Number of kernels; used to be 1024
filter_length = 40 # Length of each kernel
LSTM_out_dim = 100 # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 800

def build_model(use_JASPAR=True):
    # Convolutional/maxpooling layers to extract prominent motifs
    # Separate identically initialized convolutional layers are trained for
    # enhancers and promoters
    # Define enhancer layers
    enhancer_input = Input(shape=(enhancer_length, 4))
    enhancer_conv_layer = Conv1D(filters=n_kernels,
                                 kernel_size=filter_length,
                                 padding="valid",  # 卷积之后维数变小，output_length = input_length - filter_size + 1
                                 strides=1,
                                 kernel_regularizer=l2(1e-5),
                                 activation='relu')(enhancer_input)

    enhancer_length_slim = enhancer_length + filter_length - 1
    n_kernels_slim = 200
    filter_length_slim = 20
    enhancer_conv_layer_slim = Conv1D(filters=n_kernels_slim,
                                      kernel_size=filter_length_slim,
                                      padding="valid",
                                      strides=1,
                                      kernel_regularizer=l2(1e-5),
                                      activation='relu')(enhancer_conv_layer)
    enhancer_max_pool_layer = MaxPooling1D(pool_size=int(filter_length / 2),
                                           strides=int(filter_length / 2))(enhancer_conv_layer_slim)

    # Define promoter layers branch:
    promoter_input = Input(shape=(promoter_length, 4))
    promoter_conv_layer = Conv1D(filters=n_kernels,
                                 kernel_size=filter_length,
                                 padding="valid",
                                 strides=1,
                                 kernel_regularizer=l2(1e-5),
                                 activation='relu')(promoter_input)

    promoter_length_slim = promoter_length + filter_length - 1
    n_kernels_slim = 200
    filter_length_slim = 20
    promoter_conv_layer_slim = Conv1D(filters=n_kernels_slim,
                                      kernel_size=filter_length_slim,
                                      padding="valid",
                                      strides=1,
                                      kernel_regularizer=l2(1e-5),
                                      activation='relu')(promoter_conv_layer)
    promoter_max_pool_layer = MaxPooling1D(pool_size=int(filter_length / 2),
                                           strides=int(filter_length / 2))(promoter_conv_layer_slim)

    # Define main model layers
    # Concatenate outputs of enhancer and promoter convolutional layers
    merge_layer = merge.concatenate([enhancer_max_pool_layer, promoter_max_pool_layer], axis=1)
    batchnormalization = BatchNormalization()(merge_layer)
    dropout = Dropout(0.25)(batchnormalization)


    # Bidirectional LSTM to extract combinations of motifs
    biLSTM_layer = Bidirectional(LSTM(units=LSTM_out_dim,
                                      return_sequences=True))(dropout)

    # Dense layer to allow nonlinearities
    batchnormalization1 = BatchNormalization()(biLSTM_layer)
    dropout1 = Dropout(0.5)(batchnormalization1)
    flatten = Flatten()(dropout1)

    # Dense layer to allow nonlinearities
    dense_layer = Dense(units=dense_layer_size,
                        kernel_initializer="glorot_uniform",
                        kernel_regularizer=l2(1e-6))(flatten)
    batchnormalization2 = BatchNormalization()(dense_layer)
    activation = Activation('relu')(batchnormalization2)
    dropout2 = Dropout(0.5)(activation)

    # Logistic regression layer to make final binary prediction
    LR_classifier_layer = Dense(units=1)(dropout2)
    batchnormalization3 = BatchNormalization()(LR_classifier_layer)
    activation2 = Activation('sigmoid')(batchnormalization3)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=activation2)

    # Read in and initialize convolutional layers with motifs from JASPAR
    if use_JASPAR:
        util.initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer)

    return model


def build_frozen_model():
    # Freeze all but the dense layers of the network

    enhancer_input = Input(shape=(enhancer_length, 4))
    enhancer_conv_layer = Conv1D(filters=n_kernels,
                                 kernel_size=filter_length,
                                 padding="valid",  # 卷积之后维数变小，output_length = input_length - filter_size + 1
                                 strides=1,
                                 kernel_regularizer=l2(1e-5),
                                 activation='relu',
                                 trainable=False)(enhancer_input)

    enhancer_length_slim = enhancer_length + filter_length - 1
    n_kernels_slim = 200
    filter_length_slim = 20
    enhancer_conv_layer_slim = Conv1D(filters=n_kernels_slim,
                                      kernel_size=filter_length_slim,
                                      padding="valid",
                                      strides=1,
                                      kernel_regularizer=l2(1e-5),
                                      activation='relu',
                                      trainable=False)(enhancer_conv_layer)
    enhancer_max_pool_layer = MaxPooling1D(pool_size=int(filter_length / 2),
                                           strides=int(filter_length / 2),
                                           trainable=False)(enhancer_conv_layer_slim)

    # Define promoter layers branch:
    promoter_input = Input(shape=(promoter_length, 4))
    promoter_conv_layer = Conv1D(filters=n_kernels,
                                 kernel_size=filter_length,
                                 padding="valid",
                                 strides=1,
                                 kernel_regularizer=l2(1e-5),
                                 activation='relu',
                                 trainable=False)(promoter_input)

    promoter_length_slim = promoter_length + filter_length - 1
    n_kernels_slim = 200
    filter_length_slim = 20
    promoter_conv_layer_slim = Conv1D(filters=n_kernels_slim,
                                      kernel_size=filter_length_slim,
                                      padding="valid",
                                      strides=1,
                                      kernel_regularizer=l2(1e-5),
                                      activation='relu',
                                      trainable=False)(promoter_conv_layer)
    promoter_max_pool_layer = MaxPooling1D(pool_size=int(filter_length / 2),
                                           strides=int(filter_length / 2),
                                           trainable=False)(promoter_conv_layer_slim)

    # Define main model layers
    # Concatenate outputs of enhancer and promoter convolutional layers
    merge_layer = merge.concatenate([enhancer_max_pool_layer, promoter_max_pool_layer], axis=1)
    batchnormalization = BatchNormalization()(merge_layer)
    dropout = Dropout(0.25)(batchnormalization)


    # Bidirectional LSTM to extract combinations of motifs
    biLSTM_layer = Bidirectional(LSTM(units=LSTM_out_dim,
                                      return_sequences=True))(dropout)

    # Dense layer to allow nonlinearities
    batchnormalization1 = BatchNormalization()(biLSTM_layer)
    dropout1 = Dropout(0.5)(batchnormalization1)
    flatten = Flatten()(dropout1)

    # Dense layer to allow nonlinearities
    dense_layer = Dense(units=dense_layer_size,
                        kernel_initializer="glorot_uniform",
                        kernel_regularizer=l2(1e-6))(flatten)
    batchnormalization2 = BatchNormalization()(dense_layer)
    activation = Activation('relu')(batchnormalization2)
    dropout2 = Dropout(0.5)(activation)

    # Logistic regression layer to make final binary prediction
    LR_classifier_layer = Dense(units=1)(dropout2)
    batchnormalization3 = BatchNormalization()(LR_classifier_layer)
    activation2 = Activation('sigmoid')(batchnormalization3)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=activation2)

    return model

