import numpy as np

def sub_sample(X_en, X_pr, lab):
    idx_pos = np.where(lab==1)
    idx_neg = np.where(lab==0)
    pos_e = np.delete(X_en, idx_neg, axis=0)
    neg_e = np.delete(X_en, idx_pos, axis=0)
    pos_p = np.delete(X_pr, idx_neg, axis=0)
    neg_p = np.delete(X_pr, idx_pos, axis=0)
    pos_l = np.delete(lab, idx_neg, axis=0)
    neg_l = np.delete(lab, idx_pos, axis=0)

    index = np.arange(neg_e.shape[0])
    np.random.shuffle(index)
    neg_e = neg_e[index]
    neg_p = neg_p[index]
    neg_l = neg_l[index]

    neg_e = neg_e[0:pos_e.shape[0], :, :]
    neg_p = neg_p[0:pos_e.shape[0], :, :]
    neg_l = neg_l[0:pos_e.shape[0]]

    X_enhancers = np.append(pos_e, neg_e, axis=0)
    X_promoters = np.append(pos_p, neg_p, axis=0)
    labels = np.append(pos_l, neg_l)

    index = np.arange(X_enhancers.shape[0])
    np.random.shuffle(index)
    X_enhancers = X_enhancers[index]
    X_promoters = X_promoters[index]
    labels = labels[index]

    print('Finish sub-sample')

    return X_enhancers, X_promoters, labels

data_path = 'D:/python/data_SPEID_npy/'
out_path = 'D:/python/data_SPEID_npy/split_data/sub/'
cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
# cell_lines = ['NHEK'] # 'GM12878', 'HeLa-S3', 'IMR-90', 'K562'

for cell_line in cell_lines:
    print('Loading ' + cell_line + ' data')
    X_enhancers = np.load(data_path + cell_line + '_enhancers.npy')
    X_promoters = np.load(data_path + cell_line + '_promoters.npy')
    labels = np.load(data_path + cell_line + '_labels.npy')

    X_enhancers, X_promoters, labels = sub_sample(X_enhancers, X_promoters, labels)

    print('X_enhancers ' + str(X_enhancers.shape))
    print('X_promoters ' + str(X_promoters.shape))
    print('labels ' + str(labels.shape))

    np.save(out_path + cell_line + '_enhancers_sub.npy', X_enhancers)
    np.save(out_path + cell_line + '_promoters_sub.npy', X_promoters)
    np.save(out_path + cell_line + '_labels_sub.npy', labels)