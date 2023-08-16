import numpy as np
import h5py

data_path = 'D:/python/data_SPEID_npy/'
filename = 'D:/python/data_SPEID_npy/all_sequence_data.h5'
cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']

for cell_line in cell_lines:
    print('Loading ' + cell_line + ' data ...')
    X_enhancers = None
    X_promoters = None
    labels = None
    with h5py.File(filename, 'r') as hf:

        # data: (num_sequences X sequence_length X 4) 3-tensor of num_sequences
        #   one-hot encoded nucleotide sequences of equal-length sequence_length
        # name: string label for the data set (e.g., 'K562_enhancers')
        # path: string file path to which to print the data

        # Print enhancer data
        X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
        out_filename = data_path + cell_line + '_enhancers.npy'
        np.save(out_filename, X_enhancers)
  
        # Print promoter data
        X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
        out_filename = data_path + cell_line + '_promoters.npy'
        np.save(out_filename, X_promoters)

        # Print label data
        labels = np.array(hf.get(cell_line + 'labels'))
        out_filename = data_path + cell_line + '_labels.npy'
        np.save(out_filename, labels)
