import pickle
import random
from typing import Optional
import tree

import numpy as np
import torch

from src.utils.model_utils import uniform_random_rotation, rot_vec_mul
from src.data.cropping import single_chain_truncate, contiguous_truncate, spatial_truncate

CA_IDX = 1
DTYPE_MAPPING = {
    'atom_positions': torch.float32,
    'atom_mask': torch.long,
    'atom_to_token_index': torch.long,

    'atom_ca': torch.float32,
    'plm_embedding': torch.float32,
    'aatype': torch.int64,
    'moltype': torch.int32,
    'chain_index': torch.int32,
    'residue_index': torch.int32,
    'token_index': torch.int32,

    'ref_positions': torch.float32,
    'ref_mask': torch.long,
    'ref_element': torch.int32,
    'ref_atom_name_chars': torch.int32,
    'ref_space_uid': torch.long,
}
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


def convert_atom_id_name(atom_names: list[str]):
    """
        Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
        Each character is encoded as ord(c) - 32
    """
    onehot_dict = {}
    for index, key in enumerate(range(64)):
        onehot = [0] * 64
        onehot[index] = 1
        onehot_dict[key] = onehot

    mol_encode = []
    for atom_name in atom_names:
        # [4, 64]
        atom_encode = []
        for name_str in atom_name.ljust(4):
            atom_encode.append(onehot_dict[ord(name_str) - 32])
        mol_encode.append(atom_encode)
    onehot_tensor = torch.Tensor(mol_encode)
    return onehot_tensor


def convert_atom_name_id(onehot_tensor: torch.Tensor):
    """
        Converts integer of atom_name to unique atom_id names.
        Each character is encoded as chr(c + 32)
    """
    # Create reverse mapping from one-hot index to characters
    index_to_char = {index: chr(key + 32) for key, index in enumerate(range(64))}

    # Extract atom names from the tensor
    atom_names = []
    for atom_encode in onehot_tensor:
        atom_name = ''
        for char_onehot in atom_encode:
            index = char_onehot.argmax().item()  # Find the index of the maximum value
            atom_name += index_to_char[index]
        atom_names.append(atom_name.strip())  # Remove padding spaces

    return atom_names


def calc_centre_of_mass(coords, atom_mass):
    mass_coords = coords * atom_mass[:, None]
    return torch.sum(mass_coords, dim=0) / torch.sum(atom_mass)


class BioFeatureTransform:
    def __init__(self,
                 truncate_size: Optional[int] = None,
                 recenter_atoms: bool = True,
                 eps: float = 1e-8,
                 training: bool = True,
                 ):

        if truncate_size is not None:
            assert truncate_size > 0, f"Invalid truncate_length: {truncate_size}"
        self.truncate_length = truncate_size

        self.recenter_and_scale = recenter_atoms
        self.eps = eps
        self.training = training

    def __call__(self, data_object):
        atom_object, token_object = self.patch_features(data_object)

        if self.truncate_length is not None:
            atom_object, token_object = self.truncate(atom_object, token_object, truncate_size=self.truncate_length)

        # Recenter and scale atom positions
        if self.recenter_and_scale:
            atom_object = self.recenter_and_scale_coords(atom_object, eps=self.eps)
        data_object = {**atom_object, **token_object}

        data_object = self.map_to_tensors(data_object)
        data_object = self.update_ref_features(data_object, training=self.training)
        return data_object

    @staticmethod
    def patch_features(data_object):
        atom_object = {
            'atom_positions': data_object['atom_positions'],
            'atom_to_token_index': data_object['atom_to_token_index'],

            'ref_positions': data_object['ref_positions'],
            'ref_element': data_object['ref_element'],
            'ref_atom_name_chars': data_object['ref_atom_name_chars'],
        }
        token_object = {
            'atom_ca': data_object['atom_ca'],
            'plm_embedding': data_object['plm_embedding'],

            'aatype': data_object['aatype'],
            'residue_index': data_object['residue_index'],
            'chain_index': data_object['chain_index'],
            'token_index': data_object['token_index'],
        }
        return atom_object, token_object

    @staticmethod
    def map_to_tensors(chain_feats):
        chain_feats = {k: torch.as_tensor(v) for k, v in chain_feats.items()}
        # Alter dtype
        for k, dtype in DTYPE_MAPPING.items():
            if k in chain_feats:
                chain_feats[k] = chain_feats[k].type(dtype)
        return chain_feats

    @staticmethod
    def recenter_and_scale_coords(atom_object, eps=1e-8):
        atom_center = np.sum(atom_object['atom_positions'], axis=0) / (np.sum(atom_object['atom_mask']) + eps)
        atom_object['atom_positions'] -= atom_center[None, :]
        return atom_object

    @staticmethod
    def truncate(atom_object, token_object, truncate_size=384):
        random_state = random.random()
        if random_state < 0.6:
            return single_chain_truncate(atom_object, token_object, truncate_size)
        # elif random_state < 0.6:
        #     return contiguous_truncate(atom_object, token_object, truncate_size)
        else:
            return spatial_truncate(atom_object, token_object, truncate_size)

    @staticmethod
    def update_ref_features(data_object, training=True):
        data_object['ref_space_uid'] = data_object['atom_to_token_index']

        if training:
            ca_distance = (data_object['atom_ca'][:, None, :] - data_object['atom_ca'][None, :, :]).norm(dim=-1)
            data_object['bond_mask'] = ca_distance < 15.0
            data_object['coordinate_mask'] = torch.ones_like(data_object['token_index'])

        return data_object



