import numpy as np

# data: (num_sequences X sequence_length X 4) 3-tensor of num_sequences

#root = '/home/sss1/Desktop/projects/DeepInteractions/data/uniform_len/original/'
#data_path = root + 'all_data.h5'
#filename = 'all_sequence_data.h5'
cell_lines = ['GM12878', 'HeLa-S3', 'IMR-90', 'K562'] # 'GM12878', 'HeLa-S3', 'IMR-90', 'K562'
factors = ['enhancers', 'promoters'] # 'enhancers', 'promoters'

# Map:
#   [1,0,0,0] -> A
#   [0,1,0,0] -> T
#   [0,0,1,0] -> C
#   [0,0,0,1] -> G

def base_to_onehot(str):
    if str == 'A':
        onehot = [1, 0, 0, 0]
    elif str == 'T':
        onehot = [0, 1, 0, 0]
    elif str == 'C':
        onehot = [0, 0, 1, 0]
    elif str == 'G':
        onehot = [0, 0, 0, 1]
    elif str == 'N':
        onehot = [0, 0, 0, 0]
    onehot = np.array(onehot, dtype='int16')  # 转数组
    return onehot


for cell_line in cell_lines:
    for factor in factors:
        with open(cell_line + '_' + factor + '.txt', 'r') as f:
            enhancers_npy = []
            for index, line in enumerate(f):  # 只取序列
                if index % 2 == 1:
                    enhancer = line.strip('\n')
                    onehot = []
                    for base in enhancer:
                        onehot_tmp = base_to_onehot(base)
                        onehot.append(onehot_tmp)
                    enhancers_npy.append(onehot)
            out_filename = cell_line + '_' + factor + '.npy'
            np.save(out_filename, enhancers_npy)