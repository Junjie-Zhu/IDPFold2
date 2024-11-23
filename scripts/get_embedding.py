import os
import pickle

import numpy as np
import pandas as pd
import torch
import esm

from esm2_extract import calculate_representation, save_representation

# constants
batch_size = 128
metadata_path = '/lustre/home/acct-clschf/clschf/jjzhu/ai2pse/data/dryrun_metadata.csv'
embedding_path = '/lustre/home/acct-clschf/clschf/jjzhu/ai2pse/data/embeddings'

if not os.path.exists(embedding_path):
    os.mkdir(embedding_path)

# dict for restype to index
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V', 'X'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_reverse_order = {i: restype for i, restype in enumerate(restypes)}

# load esm2-650M
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)

# load sequence data
df_metadata = pd.read_csv(metadata_path)
pdb_ids = df_metadata['pdb_name'].tolist()
processed_paths = df_metadata['processed_path'].tolist()


# get sequence embeddings
def get_sequence_dict(id, path_to_pkl):
    seq_dict = []

    for ids, paths in zip(id, path_to_pkl):
        with open(paths, 'rb') as f:
            data_object = pickle.load(f)

            modeled_idx = np.where(data_object['aatype'] != 20)[0]
            min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)
            aatype = [restype_reverse_order[i] for i in data_object['aatype'][min_idx:max_idx + 1]]
            
            aatype_str = ''.join(aatype)
            seq_dict.append((ids, aatype_str))
            
    return seq_dict


for i in range(0, len(pdb_ids), batch_size):
    
    if i + batch_size > len(pdb_ids):
        seq_dict = get_sequence_dict(pdb_ids[i:], processed_paths[i:])
        sequence_labels, sequence_strs, representation = calculate_representation(model, alphabet, seq_dict, device)
    else:
        seq_dict = get_sequence_dict(pdb_ids[i:i + batch_size], processed_paths[i:i + batch_size])
        sequence_labels, sequence_strs, representation = calculate_representation(model, alphabet, seq_dict, device)

    for labels, strs, reps in zip(sequence_labels,sequence_strs, representation):
        save_representation(labels, strs, reps, os.path.join(embedding_path, labels + '.pkl'))

