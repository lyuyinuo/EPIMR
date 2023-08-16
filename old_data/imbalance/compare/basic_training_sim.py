#数据集0.81:0.09:0.1
from __future__ import division
from __future__ import print_function
# Basic python and data processing imports
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
#import h5py

#import load_data_pairs as ld # my scripts for loading data
import build_sim_model as bm # Keras specification of SPEID model
#import build_sim_multi_model as bm
#import build_module_model as bm
# import matplotlib.pyplot as plt
from datetime import datetime
import util
import tensorflow as tf

# Keras imports
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.metrics import binary_accuracy
from sklearn.metrics import classification_report

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  #指定要使用的GPU序号

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


#cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_lines = ['GM12878']

# Model training parameters
num_epochs = 32
batch_size = 50
kernel_size  = 300
training_frac = 0.9 # fraction of data to use for training

t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
opt = Adam(lr = 1e-5) # opt = RMSprop(lr = 1e-6)

data_path = '/home/puyuqian/lyu/final/old_data/onehot/'
#out_path = data_path
for cell_line in cell_lines:
    #for ro in range(1):
    # np.random.seed(ro+10)
    print('Loading ' + cell_line + ' data')

    X_en_train = np.load(data_path + cell_line + '_enhancers_tr.npy')
    X_pr_train = np.load(data_path + cell_line + '_promoters_tr.npy')
    y_train = np.load(data_path + cell_line + '_labels_tr.npy')

    X_en_valid = np.load(data_path + cell_line + '_enhancers_va.npy')
    X_pr_valid = np.load(data_path + cell_line + '_promoters_va.npy')
    y_valid = np.load(data_path + cell_line + '_labels_va.npy')

    X_en_test = np.load(data_path + cell_line + '_enhancers_ts.npy')
    X_pr_test = np.load(data_path + cell_line + '_promoters_ts.npy')
    y_test = np.load(data_path + cell_line + '_labels_ts.npy')

    model = bm.build_model(use_JASPAR=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])
    model.summary()


    # Define custom callback that prints/plots performance at end of each epoch
    class ConfusionMatrix(Callback):
        def on_train_begin(self, logs={}):
            self.epoch = 0
            self.precisions = []
            self.recalls = []
            self.f1_scores = []
            self.losses = []
            self.training_losses = []
            self.training_accs = []
            self.accs = []
            # plt.ion()

        def on_epoch_end(self, batch, logs={}):
            self.training_losses.append(logs.get('loss'))
            self.training_accs.append(logs.get('acc'))
            self.epoch += 1
            val_predict = model.predict([X_en_train, X_pr_train], batch_size=batch_size, verbose=0)
            val_predict = np.argmax(val_predict, axis=1)
            # util.print_live(self, labels, val_predict, logs)
            # if self.epoch > 1: # need at least two time points to plot
            # util.plot_live(self)


    # print '\nlabels.mean(): ' + str(labels.mean())
    print('Data sizes: ')
    print('train set(num):', len(X_en_train))
    print('test set(num):', len(X_en_test))

    # Instantiate callbacks
    confusionMatrix = ConfusionMatrix()
    checkpoint_path = "./onehot/" + cell_line + "-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    print('Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...')
    model.fit([X_en_train, X_pr_train], y_train,
              validation_data=([X_en_valid, X_pr_valid], y_valid),
              batch_size=batch_size,
              epochs=num_epochs,
              shuffle=True,
              callbacks=[confusionMatrix, checkpointer, earlystopper]
              )

    plotName = cell_line + '_' + t + '.png'
    plt.savefig(plotName)
    print('Saved loss plot to ' + plotName)

    print('Running predictions...')

    # y = labels_ts
    y_score = model.predict([X_en_test, X_pr_test], batch_size=batch_size, verbose=1)
    # np.save(('Basic_lessMax_y_predict_Batch' + str(batch_size) + '_Kernel' + str(kernel_size) + cell_line + '_test' + cell_line + '_R' + str(ro)), y_score)
    np.save(('Batch' + str(batch_size) + '_Kernel' + str(kernel_size) + '_' + cell_line), y_score)


    print('Loading ' + cell_line + ' data')
    util.evaluate(y_test, y_score)
    util.plot_PR_curve(y_test, y_score, cell_line, t)
    util.plot_ROC_curve(y_test, y_score, cell_line, t)
