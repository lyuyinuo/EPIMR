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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定要使用的GPU序号

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
    
    
cell = "K562"  # ['GM12878', 'HeLa-S3', 'IMR-90', 'K562']
batch_size = 128
opt = optimizers.Adam(lr=1e-5)
num_epochs = 50
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

data_path = '/home/puyuqian/lyu/new_data/hilbert/'

path = './model/' + cell + '_bestmodel_fold0.h5'  # 文件路径
if os.path.exists(path):
    os.remove(path)

path = './model/' + cell + '_bestfrozenmodel_fold0.h5'  # 文件路径
if os.path.exists(path):
    os.remove(path)


X_enhancers = np.load(data_path + cell + '_enhancers.npy')
X_promoters = np.load(data_path + cell + '_promoters.npy')
labels = np.load(data_path + cell + '_labels.npy')
X_enhancers, X_promoters, labels = shuffle_data(X_enhancers, X_promoters, labels)

index = np.arange(X_enhancers.shape[0])
testing_idx = np.random.choice(index, size=int(0.1 * index.shape[0]), replace=False)
X_en_test = X_enhancers[testing_idx, :, :]
X_pr_test = X_promoters[testing_idx, :, :]
y_test = labels[testing_idx]

idx = np.delete(index, testing_idx, axis=0)
X_en_train = X_enhancers[idx, :, :]
X_pr_train = X_promoters[idx, :, :]
y_train = labels[idx]


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

for fold in range(1):  # 10

    print('Data sizes: ')
    print('train set(num):', len(X_en_train))
    print('test set(num):', len(X_en_test))

    model = ResNet(input_shape=(64, 96, 4), net_name='resnet18')

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])  # 1e-5 RMSprop

    model.summary()

    # fraction of samples in each class
    pos_frac = y_train.mean()
    neg_weight = (1 / (1 - pos_frac)) ** (1 / 2)
    pos_weight = (1 / pos_frac) ** (1 / 2)

    print('Positive weight: ' + str(pos_weight) + '  Negative weight: ' + str(neg_weight))

    test = False
    if not test:
        checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_fold%d.h5"
                                                % (cell, fold), verbose=1, save_best_only=True)
        # earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        print('Training model...')
        m = model.fit([X_en_train, X_pr_train], y_train,
                      epochs=num_epochs,
                      batch_size=batch_size,  # 128
                      shuffle=True,
                      validation_split=0.1,
                      # validation_data=([X_en_valid, X_pr_valid], y_valid),
                      callbacks=[checkpointer],  # , earlystopper
                      class_weight={0: 1, 1: 10},
                      # class_weight={0 : neg_weight, 1 : pos_weight},
                      verbose=1)

        result_visualization(m)

    print('Testing model...')
    model.load_weights('./model/%s_bestmodel_fold%d.h5' % (cell, fold))
    tresults = model.evaluate([X_en_test, X_pr_test], y_test)
    print('test_results:', tresults)
    y_pred = model.predict([X_en_test, X_pr_test], batch_size, verbose=1)
    print('Calculating AUC...')
    f1.append(metrics.f1_score(y_test, y_pred > 0.5))
    auc.append(metrics.roc_auc_score(y_test, y_pred))
    aupr.append(metrics.average_precision_score(y_test, y_pred))
    acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))
print(cell, 'result:')
print(f1, auc, aupr, acc)
print('---------------------------------')
