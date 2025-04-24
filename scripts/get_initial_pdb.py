import os
import pickle

import argparse

import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--input_path", "-i", type=str, help="Input fasta file path")
args.add_argument("--output_path", "-o", type=str, help="Output pdb file path")
args = args.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

ccd_path = '../src/data/components/ccd_atom37.pkl'
with open(ccd_path, 'rb') as f:
    ccd = pickle.load(f)

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

to_process_list = []
with open(args.input_path, 'r') as f:
    lines = f.readlines()

    seq_name, seq = '', ''
    for line in lines:
        if line.startswith('>'):
            seq_name = line[1:].strip()
        else:
            seq = line.strip()
            to_process_list.append((seq_name, seq))

# set a restype dictionary
restype_dict = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
                'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V', 'X'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}

# Create virtual pdb files in pdb_path for inference
for seq_name, seq in to_process_list:
    with open(os.path.join(args.output_path, (seq_name + '.pdb')), 'w') as f:

        for i, aa in enumerate(seq):
            restype_idx = restype_order[aa]
            atom_coords = np.array(ccd[restype_idx]['coord'])
            atom_mask = (np.sum(atom_coords, axis=1) != 0)
            exist_atom_names = np.array(atom_types)[atom_mask]

            for atom_name in exist_atom_names:
                if atom_name == 'OXT':
                    continue
                f.write(
                    f'ATOM  {i + 1:>5} {atom_name:<4} {restype_dict[aa]:>3} A {i + 1:>3}      0.000   0.000   0.000  1.00  0.00           {atom_name[0]}\n')

