import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.python.keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #指定要使用的GPU序号

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

cell_line = 'K562' # 'GM12878', 'HeLa-S3', 'IMR-90', 'K562'

model_path = '/home/puyuqian/lyu/new/new_data/pre-train/model/'
model = load_model(model_path + '6cell_retr_%s_bestmodel_fold0_727.h5' % (cell_line))

subdata_path = '/home/puyuqian/lyu/final/new_data/aug/sub/hilbert/'
e_tr = np.load(subdata_path + cell_line + '_enhancers_sub_tr.npy')
p_tr = np.load(subdata_path + cell_line + '_promoters_sub_tr.npy')
l_tr = np.load(subdata_path + cell_line + '_labels_sub_tr.npy')

e_va = np.load(subdata_path + cell_line + '_enhancers_sub_va.npy')
p_va = np.load(subdata_path + cell_line + '_promoters_sub_va.npy')
l_va = np.load(subdata_path + cell_line + '_labels_sub_va.npy')

e_ts = np.load(subdata_path + cell_line + '_enhancers_sub_ts.npy')
p_ts = np.load(subdata_path + cell_line + '_promoters_sub_ts.npy')
l_ts = np.load(subdata_path + cell_line + '_labels_sub_ts.npy')

sub_e = np.concatenate((e_tr, e_va, e_ts), axis=0)
sub_p = np.concatenate((p_tr, p_va, p_ts), axis=0)
sub_l = np.concatenate((l_tr, l_va, l_ts), axis=0)

tsne_layer = Model(inputs=model.input, outputs=model.get_layer(name='globalavgpool_Y3').output)  # index=141
tsne_output = tsne_layer.predict([sub_e, sub_p])

tsne = TSNE(n_components=2, init='pca')
tsne_results = tsne.fit_transform(tsne_output)

y_test_cat = np_utils.to_categorical(sub_l, num_classes=2)  # 总的类别
color_map = np.argmax(y_test_cat, axis=1)
# plt.figure(figsize=(10, 10))
labelname = ['EPIs', 'non-EPIs']
for cl in range(2):  # 总的类别
    indices = np.where(color_map == cl)
    indices = indices[0]
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=labelname[cl], alpha=0.8)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, loc="upper right")
# plt.show()
plt.savefig(cell_line + '_34_tsne_stage4_p.png')