import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, log_loss
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import tensorflow as tf
import h5py
from datetime import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #指定要使用的GPU序号

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


def cnn_model():
    enhancer_input = Input(shape=(64, 64, 4))
    enhancer_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
    enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
    enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)

    promoter_input = Input(shape=(64, 64, 4))
    promoter_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
    promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
    promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)

    branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
    branch_output1 = layers.Dropout(0.5)(branch_output)
    branch_output2 = Dense(128, activation='relu')(branch_output1)
    output = Dense(1, activation='sigmoid')(branch_output2)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
    print(model.summary())

    return model


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
    
    
cell = "GM12878"  # ['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878', 'HeLa-S3']
training_frac = 0.9
batch_size = 128
opt = optimizers.Adam(lr=1e-5)
num_epochs = 32
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

data_path = '/home/puyuqian/lyu/final/old_data/hilbert/'
# data_path = 'D:/python/data_SPEID_npy/split_data/hilbert/'

path = './model/' + cell + '_bestmodel_fold0.h5'  # 文件路径
if os.path.exists(path):
    os.remove(path)

path = './model/' + cell + '_bestfrozenmodel_fold0.h5'  # 文件路径
if os.path.exists(path):
    os.remove(path)


X_en_train = np.load(data_path + cell + '_enhancers_tr.npy')
X_pr_train = np.load(data_path + cell + '_promoters_tr.npy')
y_train = np.load(data_path + cell + '_labels_tr.npy')

X_en_valid = np.load(data_path + cell + '_enhancers_va.npy')
X_pr_valid = np.load(data_path + cell + '_promoters_va.npy')
y_valid = np.load(data_path + cell + '_labels_va.npy')

X_en_test = np.load(data_path + cell + '_enhancers_ts.npy')
X_pr_test = np.load(data_path + cell + '_promoters_ts.npy')
y_test = np.load(data_path + cell + '_labels_ts.npy')


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


model = cnn_model()
model.compile(loss='binary_crossentropy',  # model.compile用于配置训练模型
              optimizer=opt,  # 优化器名or优化实例
              metrics=["accuracy"])
model.summary()

print('Data sizes: ')
print('train set(num):', len(X_en_train))
print('test set(num):', len(X_en_test))

confusionMatrix = ConfusionMatrix()
checkpoint_path = "../DeepInteractions/weights/test-delete-this-" + cell + "-basic-" + t + ".hdf5"
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1)  # 保存最佳模型

print('Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...')

f1 = []
auc = []
aupr = []
acc = []

for fold in range(1):  # 10
    test = False
    if not test:
        checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_fold%d.h5"
                                                % (cell, fold), verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        print('Training model...')
        m = model.fit([X_en_train, X_pr_train], y_train,
                      epochs=100, batch_size=256,  # 128
                      shuffle=True,
                      validation_data=([X_en_valid, X_pr_valid], y_valid),
                      callbacks=[checkpointer, earlystopper], verbose=1)
        # result_visualization(m)

    print('Testing model...')
    model.load_weights('./model/%s_bestmodel_fold%d.h5' % (cell, fold))
    tresults = model.evaluate([X_en_test, X_pr_test], y_test)
    print('test_results:', tresults)
    y_pred = model.predict([X_en_test, X_pr_test], 128, verbose=1)
    print('Calculating AUC...')
    f1.append(metrics.f1_score(y_test, y_pred > 0.5))
    auc.append(metrics.roc_auc_score(y_test, y_pred))
    aupr.append(metrics.average_precision_score(y_test, y_pred))
    acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))

print(cell, 'result:')
print(f1, auc, aupr, acc)
print('---------------------------------')
