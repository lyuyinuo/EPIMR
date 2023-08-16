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

model_path = '/home/puyuqian/lyu/final/old_data/aug/sub/model/'
model = load_model(model_path + '%s_bestmodel_fold00.h5' % (cell_line))

subdata_path = '/home/puyuqian/lyu/final/old_data/aug/sub/hilbert/'
sub_e = np.load(subdata_path + cell_line + '_enhancers_sub_tr.npy')
sub_p = np.load(subdata_path + cell_line + '_promoters_sub_tr.npy')
sub_l = np.load(subdata_path + cell_line + '_labels_sub_tr.npy')

tsne_layer = Model(inputs=model.input, outputs=model.get_layer(name='globalavgpool_X0').output)  # index=141
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
plt.legend()
# plt.show()
plt.savefig(cell_line + '_34_tsne_stage1_e.png')