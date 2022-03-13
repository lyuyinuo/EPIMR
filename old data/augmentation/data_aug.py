import numpy as np
data_path = 'D:/python/data_SPEID_npy/'
out_path = 'D:/python/GPU/hilbert_all_aug/data_onehot/'
cell_line = 'NHEK'  # 'GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK'

enhancers = np.load(data_path + cell_line + '_enhancers.npy')
promoters = np.load(data_path + cell_line + '_promoters.npy')
labels = np.load(data_path + cell_line + '_labels.npy')

# 保存正样本（p）和负样本的数量（n）
p = 0
n = 0
for i in labels:
    if i == 0:
        n += 1
    else:
        p += 1
print(p, n)

# 在100以内随机取10个数字，分别用于增强子和启动子，实现正样本扩展
random_num = np.random.randint(1, 100, 10)

# 存储正样本和负样本对应的增强子和启动子
pos_enhancers = enhancers[0:p]
pos_promoters = promoters[0:p]
print(len(pos_enhancers))
print(pos_enhancers.shape)
print(type(pos_enhancers))
print('-----------------')

neg_enhancers = enhancers[-n:]
neg_promoters = promoters[-n:]
print(len(neg_enhancers))
print(neg_enhancers.shape)
print(type(neg_enhancers))
print('-----------------')
print(labels[-n:])

# 扩展正样本(enhancer交换顺序，promoter不变)
exmp = pos_enhancers[0]
aug_enhancers = np.empty([p * 20, 3000, 4])
aug_promoters = np.empty([p * 20, 2000, 4])

i = 0
for num_e in range(len(pos_enhancers)):  # 0-1253
    exmp_e = pos_enhancers[num_e]  # 第num_e个正样本对应的enhancer
    exmp_p = pos_promoters[num_e]  # 第num_e个正样本对应的promoter
    aug_enhancers[i] = exmp_e
    aug_promoters[i] = exmp_p
    i += 1  # 1 + 10 + 9
    #     print(i)
    for r in range(len(random_num)):  # 0-1-10 pro一样
        new_sample = np.concatenate(
            (exmp_e[random_num[r]:], exmp_e[0:random_num[r]]), axis=0)
        aug_enhancers[i] = new_sample
        aug_promoters[i] = exmp_p
        i += 1
    #     print(i)
    for s in range(1, len(random_num)):  # 11-19 enh一样
        new_sample = np.concatenate(
            (exmp_p[random_num[s]:], exmp_p[0:random_num[s]]), axis=0)
        aug_promoters[i] = new_sample
        aug_enhancers[i] = exmp_e
        i += 1
    #     print(i)

print(aug_enhancers.shape)
print(aug_promoters.shape)

# 合并所有数据
len_p = p
len_aug_p = len(aug_enhancers)
len_all = len_aug_p + n
print('原始正样本：', len_p)
print('拓展正样本：', len_aug_p)
print('原始负样本：', n)
print('总样本：', len_all)

aug_enhancers_all = np.empty([len_all, 3000, 4])
aug_promoters_all = np.empty([len_all, 2000, 4])
aug_labels = np.empty([len_all, 1])

number = 0
for i in range(len_aug_p):
    aug_enhancers_all[i] = aug_enhancers[i]
    aug_promoters_all[i] = aug_promoters[i]
    aug_labels[number] = 1
    number += 1
print(number)

for i in range(n):
    aug_enhancers_all[len_aug_p + i] = neg_enhancers[i]
    aug_promoters_all[len_aug_p + i] = neg_promoters[i]
    aug_labels[len_aug_p + i] = 0
    number += 1
print(number)

print(aug_enhancers_all.shape)
print(aug_promoters_all.shape)
aug_labels = aug_labels.squeeze()
print('正样本的EP数据拓展（finish!）')

index = np.arange(aug_labels.shape[0])
np.random.shuffle(index)
aug_enhancers_all = aug_enhancers_all[index]
aug_promoters_all = aug_promoters_all[index]
aug_labels = aug_labels[index]

np.save(out_path + cell_line + '_enhancers_aug.npy', aug_enhancers_all)
np.save(out_path + cell_line + '_promoters_aug.npy', aug_promoters_all)
np.save(out_path + cell_line + '_labels_aug.npy', aug_labels)