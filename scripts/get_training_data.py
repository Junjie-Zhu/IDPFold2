import glob
import pickle
import gzip
import os
import argparse
import multiprocessing as mp
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd
import biotite.structure.io.pdbx as pdbx

import residue_constants as rc

COMPONENT_FILE = './components.cif'
ccd_cif = pdbx.CIFFile.read(COMPONENT_FILE)

element_onehot_dict = {}
for index, key in enumerate(range(32)):
    onehot = [0] * 32
    onehot[index] = 1
    element_onehot_dict[key] = onehot


def convert_atom_id_name(atom_names: str):
    """
        Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
        Each character is encoded as ord(c) - 32
    """
    onehot_dict = {}
    for index, key in enumerate(range(64)):
        onehot = [0] * 64
        onehot[index] = 1
        onehot_dict[key] = onehot

    # [4, 64]
    atom_encode = []
    for name_str in atom_names.ljust(4):
        atom_encode.append(onehot_dict[ord(name_str) - 32])
    onehot_tensor = np.array(atom_encode)
    return onehot_tensor


def process_single_system(data: dict, system_name: str):

    atom_positions = data['atom_positions']  # atom37
    atom_mask = data['atom_mask']  # atom37
    aatype = data['aatype']
    residue_index = data['residue_index']
    chain_index = data['chain_index']
    token_index = data['modeled_idx']

    # information to extract
    flatten_atom_positions = []
    atom_to_token_index = []
    ca_positions = []
    ca_mask = []
    ref_positions = []
    ref_elements = []
    ref_atom_name_chars = []

    chain_start = -1
    seqs_str = []
    for i, (positions, pos_mask,
            aa, res_i, chain_i, token_i) in enumerate(
            zip(atom_positions, atom_mask, aatype, residue_index, chain_index, token_index)):
        if chain_i != chain_start:
            seqs_str.append('')
            chain_start = chain_i

        # process sequence
        aa_str = rc.restype_idx21[aa]
        aa_str3 = rc.restype_1to3[aa_str]
        seqs_str[-1] += aa_str

        # process ccd information
        len_ref_info = len(rc.RES_ATOMS_DICT[aa_str3])
        ref_pos = np.zeros((len_ref_info, 3), dtype=np.float32)
        ref_mask = np.zeros((len_ref_info,), dtype=np.float32)
        element = np.zeros((len_ref_info, 32), dtype=np.int32)
        atom_name_chars = np.zeros((len_ref_info, 4, 64), dtype=np.int32)

        comp = pdbx.get_component(ccd_cif, data_block=aa_str3, use_ideal_coord=True)
        comp = comp[~np.isin(comp.element, ["H", "D"])]
        for atom in comp:
            if atom.atom_name not in rc.RES_ATOMS_DICT[aa_str3]:
                continue
            ref_pos[rc.RES_ATOMS_DICT[aa_str3][atom.atom_name]] = atom.coord
            ref_mask[rc.RES_ATOMS_DICT[aa_str3][atom.atom_name]] = 1.0
            element[rc.RES_ATOMS_DICT[aa_str3][atom.atom_name], :] = element_onehot_dict[rc.ELEMENT_MAPPING[atom.element]]
            atom_name_chars[rc.RES_ATOMS_DICT[aa_str3][atom.atom_name], :] = convert_atom_id_name(atom.atom_name)

        # process atom information
        atom_pos = np.zeros((len_ref_info, 3), dtype=np.float32)
        atom_mask = np.zeros((len_ref_info,), dtype=np.float32)
        ca_pos = np.zeros((1, 3), dtype=np.float32)
        ca_mask_ = 0
        crt_token_index = np.array([token_i] * len_ref_info, dtype=np.int64)

        for atom_idx, (pos, mask) in enumerate(zip(positions, pos_mask)):
            if mask == 0:
                continue
            atom_name = rc.atom_types[atom_idx]
            if atom_name not in rc.RES_ATOMS_DICT[aa_str3]:
                continue
            atom_pos[rc.RES_ATOMS_DICT[aa_str3][atom_name]] = pos
            atom_mask[rc.RES_ATOMS_DICT[aa_str3][atom_name]] = mask
            if atom_idx == 1:
                ca_pos[0] = pos
                ca_mask_ = 1

        # apply mask
        mask = (atom_mask * ref_mask).astype(bool)
        atom_pos = atom_pos[mask]
        crt_token_index = crt_token_index[mask]
        ref_pos = ref_pos[mask]
        element = element[mask]
        atom_name_chars = atom_name_chars[mask]

        # append to lists
        flatten_atom_positions.append(atom_pos)
        atom_to_token_index.append(crt_token_index)
        ca_positions.append(ca_pos)
        ca_mask.append(ca_mask_)
        ref_positions.append(ref_pos)
        ref_elements.append(element)
        ref_atom_name_chars.append(atom_name_chars)

    # get the output data dict
    data_object = {
        'atom_positions': np.concatenate(flatten_atom_positions),
        'atom_to_token_index': np.concatenate(atom_to_token_index),

        'ca_positions': np.concatenate(ca_positions),
        'ca_mask': np.array(ca_mask, dtype=np.float32),

        'aatype': aatype,
        'chain_index': chain_index,
        'residue_index': residue_index,
        'token_index': token_index,

        'ref_positions': np.concatenate(ref_positions),
        'ref_element': np.concatenate(ref_elements),
        'ref_atom_name_chars': np.concatenate(ref_atom_name_chars),
    }
    seqs = {
        f'{system_name}_{i}': seq for i, seq in enumerate(seqs_str)
    }
    return data_object, seqs


def process_fn(input_file, output_dir):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    accession_code = os.path.basename(input_file).replace('.pkl', '')
    data_object, seqs = process_single_system(data, accession_code)

    with gzip.open(f'{output_dir}/{accession_code}.pkl.gz', 'wb') as f:
        pickle.dump(data_object, f)
    with open(f'{output_dir}/all_seqs.fasta', 'a') as f:
        for k, seq in seqs.items():
            f.write(f'>{k}\n{seq}\n')

    # process metadata
    data_info = {
        'accession_code': accession_code,
        'token_num': len(data_object['aatype']),
        'chain_num': len(np.unique(data_object['chain_index'])),
    }

    # some additional checks
    ca_mask = data_object['ca_mask']
    if len(ca_mask) != np.sum(ca_mask):  # check if all CA atoms are present
        return data_info, accession_code

    return data_info, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process training data.')
    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Directory containing input .pkl files.')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Directory to save processed data.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = glob.glob(os.path.join(args.input_dir, 'pdb*', '*.pkl'))

    if args.debug:
        input_files = input_files[:20]

    process_fn_ = partial(process_fn, output_dir=args.output_dir)
    with open(f'{args.output_dir}/all_seqs.fasta', 'w') as f:
        f.write('')  # clear the file if it exists

    all_data_info = {
        'accession_code': [],
        'token_num': [],
        'chain_num': []
    }
    partial_systems = []
    if os.cpu_count() > 1 and not args.debug:
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_fn_, input_files), total=len(input_files)))
        for data_info, accession_code in results:
            for k in data_info:
                all_data_info[k].append(data_info[k])
            if accession_code is not None:
                partial_systems.append(accession_code)
    else:
        for input_file in tqdm(input_files):
            data_info, accession_code = process_fn_(input_file)
            for k in data_info:
                all_data_info[k].append(data_info[k])
            if accession_code is not None:
                partial_systems.append(accession_code)

    all_data_info = pd.DataFrame(all_data_info)
    all_data_info.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)

    if len(partial_systems) > 0:
        with open(os.path.join(args.output_dir, 'partial_systems.txt'), 'w') as f:
            for system in partial_systems:
                f.write(f'{system}\n')



