import pickle
import numpy as np

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

# define the element atomic number for each atom type
element_atomic_number = [
    7, 6, 6, 6, 8, 6, 6, 6, 8, 8, 16, 6,
    6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6,
    7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6,
    6, 7, 8
]


def read_pkl(data_path):
    with open(data_path, 'rb') as f:
        data_object = pickle.load(f)

    return data_object


def convert_atom_id_name(atom_id: str):
  """
    Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
    Each character is encoded as ord(c) - 32
  """
  atom_id_pad = atom_id.ljust(4, ' ')
  assert len(atom_id_pad) == 4
  return [ord(c) - 32 for c in atom_id_pad]


def get_atom_features(data_object):
    # get atom_positions according to atom_mask
    atom_positions = data_object['atom_positions'][data_object['atom_mask']].reshape(-1, 3)

    atom_mask = data_object['atom_mask']
    token2atom_map = np.zeros(atom_mask.sum(), dtype=np.int64)
    atom_elements = np.zeros(atom_mask.sum(), dtype=np.int32)

    index_start, token = 0, 0
    atom_type = []
    for residues in atom_mask:
        length = residues.sum()
        index_end = index_start + length

        token2atom_map[index_start:index_end] += token

        atom_index = np.where(residues)[0]
        atom_elements[index_start:index_end] = [element_atomic_number[i] for i in atom_index]
        atom_type.extend([atom_types[i] for i in atom_index])

        index_start = index_end
        token += 1

    atom_space_uid = token2atom_map
    atom_name_char = np.array([convert_atom_id_name(atom_id) for atom_id in atom_type], dtype=np.int32)

    output_batch = {
        'ref_pos': atom_positions,
        'ref_token2atom_idx': token2atom_map,
        'all_atom_pos_mask': np.ones_like(atom_positions[:, 0]).squeeze(),

        'residue_index': data_object['residue_index'],
        'seq_mask': np.ones_like(data_object['residue_index']),

        'ref_space_uid': atom_space_uid,
        'ref_atom_name_chars': atom_name_char,
        'ref_element': atom_elements,
        'ref_mask': atom_mask,
    }

    return output_batch



