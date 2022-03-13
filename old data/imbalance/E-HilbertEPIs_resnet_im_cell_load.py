import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, log_loss
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model, load_model
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import tensorflow as tf
import h5py
from datetime import datetime

from resnet_block import ResNet

import random
my_seed = 666
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #指定要使用的GPU序号

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


def shuffle_data(dataset_enhancer, dataset_promoter, labels):
    n = len(labels)
    indices = np.arange(n)
    np.random.shuffle(indices)
    dataset_enhancer = dataset_enhancer[indices]
    dataset_promoter = dataset_promoter[indices]
    labels = labels[indices]

    return dataset_enhancer, dataset_promoter, labels


def print_live(conf_mat_callback, y_val, val_predict, logs):  # y_val实际值，val_predict预测值
    num0, num1 = 0, 0
    for i in y_val:
        if i == 0:
            num0 += 1
        if i == 1:
            num1 += 1
    print('y_val（true value）:', num0, num1)

    num0, num1 = 0, 0
    for i in val_predict:
        if i == 0:
            num0 += 1
        if i == 1:
            num1 += 1
    print('val_predict(predict value):', num0, num1)

    tn, fp, fn, tp = confusion_matrix(y_val, val_predict).ravel()
    print('tn, fp, fn, tp:', tn, fp, fn, tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 / (1 / precision + 1 / recall)
    acc = (tp + tn) / (tp + fn + fp + tn)
    loss = log_loss(y_val, val_predict)

    tpr = recall
    fpr = fp / (tn + fp)

    conf_mat_callback.precisions.append(precision)
    conf_mat_callback.recalls.append(recall)
    conf_mat_callback.f1_scores.append(f1_score)
    conf_mat_callback.losses.append(loss)
    conf_mat_callback.accs.append(acc)

    print('Precision: ' + str(precision) + \
          '  Recall: ' + str(recall) + \
          '  F1: ' + str(f1_score) + \
          '  Accuracy: ' + str(acc) + \
          '  Log Loss: ' + str(loss))

def result_visualization(history):
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()
    
    
cell = "NHEK"  # ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_ts = "K562"  # ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
batch_size = 64
opt = optimizers.Adam(lr=1e-5)
num_epochs = 100
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

data_path = '/home/puyuqian/lyu/old_data/hilbert/'

'''
path = './model/' + cell + '_bestmodel_fold0.h5'  # 文件路径
if os.path.exists(path):
    os.remove(path)

path = './model/' + cell + '_bestfrozenmodel_fold0.h5'  # 文件路径
if os.path.exists(path):
    os.remove(path)
'''


X_en_train = np.load(data_path + cell + '_enhancers.npy')
X_pr_train = np.load(data_path + cell + '_promoters.npy')
y_train = np.load(data_path + cell + '_labels.npy')
X_en_train, X_pr_train, y_train = shuffle_data(X_en_train, X_pr_train, y_train)

X_en_test = np.load(data_path + cell_ts + '_enhancers.npy')
X_pr_test = np.load(data_path + cell_ts + '_promoters.npy')
y_test = np.load(data_path + cell_ts + '_labels.npy')
X_en_test, X_pr_test, y_test = shuffle_data(X_en_test, X_pr_test, y_test)


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
        plt.ion()  # 显示模式转化为交互模式(interactive),即遇到plt.show()，代码还会继续进行

    def on_epoch_end(self, batch, logs={}):
        self.training_losses.append(logs.get('loss'))
        self.training_accs.append(logs.get('acc'))
        self.epoch += 1
        # model.predict 为输入样本生成输出预测
        val_predict = model.predict([X_en_train, X_pr_train], batch_size=batch_size, verbose=0)
        f1.append(metrics.f1_score(y_train, val_predict > 0.5))
        auc.append(metrics.roc_auc_score(y_train, val_predict))
        aupr.append(metrics.average_precision_score(y_train, val_predict))
        acc.append(metrics.accuracy_score(y_train, val_predict > 0.5))

        val_predict = [1 if i > 0.5 else 0 for i in val_predict]
        print_live(self, y_train, val_predict, logs)


print('Running fully trainable model...')

f1 = []
auc = []
aupr = []
acc = []

'''
print('Data sizes: ')
print('train_aug set(num):', len(X_en_train))
print('test set(num):', len(X_en_test))

model = ResNet(input_shape=(64, 64, 4), net_name='resnet34')

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])  # 1e-5 RMSprop

model.summary()

test = False
if not test:
    checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_fold0.h5"
                                            % (cell), verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    print('Training model...')
    m = model.fit([X_en_train, X_pr_train], y_train,
                  epochs=num_epochs,
                  batch_size=batch_size,  # 128
                  shuffle=True,
                  validation_split=0.1,
                  # validation_data=([X_en_valid, X_pr_valid], y_valid),
                  callbacks=[checkpointer, earlystopper],
                  verbose=1)

    result_visualization(m)
'''
model = load_model('./model/' + cell + '_bestmodel_fold0_34_es.h5')
print('Testing model...')
model.load_weights('./model/%s_bestmodel_fold0_34_es.h5' % (cell))
tresults = model.evaluate([X_en_test, X_pr_test], y_test)
print('test_results:', tresults)
y_pred = model.predict([X_en_test, X_pr_test], batch_size, verbose=1)
print('Calculating AUC...')
f1.append(metrics.f1_score(y_test, y_pred > 0.5))
auc.append(metrics.roc_auc_score(y_test, y_pred))
aupr.append(metrics.average_precision_score(y_test, y_pred))
acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))

print(cell, ' train, ' + cell_ts + ' test result:')
print(f1, auc, aupr, acc)
print('---------------------------------')
