import numpy as np
from hilbert_curve import hilbert_to_point
data_path = '/home/puyuqian/lyu/new_data/hilbert_all_aug/data_onehot/'
out_path = '/home/puyuqian/lyu/new_data/hilbert_all_aug/data_hilbert/'

cell_lines = ['GM12878', 'HeLa-S3']  # 'GM12878', 'HeLa-S3', 'IMR-90', 'K562'
factors = ['enhancers', 'promoters']

for cell_line in cell_lines:
    print('Loading ' + cell_line + ' data')
    for factor in factors:
        print('Loading ' + factor + ' data')
        X = np.load(data_path + cell_line + '_' + factor + '_aug.npy')
        Y = np.zeros((X.shape[0], 128, 128, 4))
        for i in range(X.shape[0]):
            tmp = X[i, :, :]
            for j in range(tmp.shape[0]):
                (a, b) = hilbert_to_point(j, 7)
                Y[i, a, b, :] = tmp[j, :]
        print(Y.shape)
        Y = Y.astype(np.int16)
        np.save(out_path + cell_line + '_' + factor + '_aug.npy', Y)
