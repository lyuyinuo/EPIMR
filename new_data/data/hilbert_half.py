import numpy as np

data_path = 'D:/python/new_data/split_data/hilbert/'
out_path = 'D:/python/new_data/split_data//hilbert_half/'

cell_lines = ['K562']  # 'GM12878', 'HeLa-S3', 'IMR-90', 'K562'
factors = ['enhancers', 'promoters']  # 'enhancers', 'promoters'

for cell_line in cell_lines:
    print('Loading ' + cell_line + ' data')
    for factor in factors:
        print('Loading ' + factor + ' data')
        X = np.load(data_path + cell_line + '_' + factor + '_ts.npy')
        print(X.shape)
        X = X[:, 0:64, 0:96, :]
        print(X.shape)
        np.save(out_path + cell_line + '_' + factor + '_ts.npy', X)