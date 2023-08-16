import numpy as np
data_path = 'D:/python/data_SPEID_npy/split_data/sub/'
out_path = 'D:/python/data_SPEID_npy/split_data/sub/onehot/'

cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562','NHEK']
# cell_lines = ['NHEK'] # 'GM12878', 'HeLa-S3', 'IMR-90', 'K562'
# factors = ['enhancers', 'promoters']

def shuffle_data(dataset_enhancer, dataset_promoter, labels):
    n = len(labels)
    indices = np.arange(n)
    np.random.shuffle(indices)
    np.save(out_path + cell_line + '_shuffle_indices.npy', indices)
    dataset_enhancer = dataset_enhancer[indices]
    dataset_promoter = dataset_promoter[indices]
    labels = labels[indices]

    return dataset_enhancer, dataset_promoter, labels

for cell_line in cell_lines:
    print('Loading ' + cell_line + ' data')
    X = np.load(data_path + cell_line + '_enhancers_sub.npy')
    Y = np.load(data_path + cell_line + '_promoters_sub.npy')
    # print(X.shape)
    L = np.load(data_path + cell_line + '_labels_sub.npy')
    print(L.shape)
    X_e, X_p, X_l = shuffle_data(X, Y, L)

    X_e_tr = X_e[0:int(0.8 * L.shape[0]), :, :]
    X_p_tr = X_p[0:int(0.8 * L.shape[0]), :, :]
    X_l_tr = X_l[0:int(0.8 * L.shape[0])]
    print(X_l_tr.shape)

    X_e_va = X_e[int(0.8 * L.shape[0]):int(0.9 * L.shape[0]), :, :]
    X_p_va = X_p[int(0.8 * L.shape[0]):int(0.9 * L.shape[0]), :, :]
    X_l_va = X_l[int(0.8 * L.shape[0]):int(0.9 * L.shape[0])]
    print(X_l_va.shape)

    X_e_ts = X_e[int(0.9 * L.shape[0]):, :, :]
    X_p_ts = X_p[int(0.9 * L.shape[0]):, :, :]
    X_l_ts = X_l[int(0.9 * L.shape[0]):]
    print(X_l_ts.shape)

    np.save(out_path + cell_line + '_enhancers_sub_tr.npy', X_e_tr)
    np.save(out_path + cell_line + '_promoters_sub_tr.npy', X_p_tr)
    np.save(out_path + cell_line + '_labels_sub_tr.npy', X_l_tr)

    np.save(out_path + cell_line + '_enhancers_sub_va.npy', X_e_va)
    np.save(out_path + cell_line + '_promoters_sub_va.npy', X_p_va)
    np.save(out_path + cell_line + '_labels_sub_va.npy', X_l_va)

    np.save(out_path + cell_line + '_enhancers_sub_ts.npy', X_e_ts)
    np.save(out_path + cell_line + '_promoters_sub_ts.npy', X_p_ts)
    np.save(out_path + cell_line + '_labels_sub_ts.npy', X_l_ts)


