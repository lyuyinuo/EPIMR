import numpy as np

data_path = 'D:/python/new_data/split_data/hilbert/'
out_path = 'D:/python/new_data/split_data//hilbert_half/'

X = np.load(out_path + 'GM12878_enhancers_tr.npy')
print(X.shape)