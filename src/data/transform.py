import random
from typing import Optional

import numpy as np
import torch

from src.data.cropping import single_chain_truncate, contiguous_truncate, spatial_truncate

CA_IDX = 1
DTYPE_MAPPING = {
    'atom_positions': torch.float32,
    'atom_mask': torch.long,
    'atom_to_token_index': torch.long,

    'ca_positions': torch.float32,
    'coordinate_mask': torch.float32,
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
        atom_object['ref_mask'] = np.ones(atom_object['ref_positions'].shape[0])

        # Recenter and scale atom positions
        if self.recenter_and_scale:
            atom_object, token_object = self.recenter_and_scale_coords(atom_object, token_object, eps=self.eps)
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
            'ca_positions': data_object['ca_positions'],
            'coordinate_mask': data_object['ca_mask'],
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
    def recenter_and_scale_coords(atom_object, token_object, eps=1e-8):
        _center = np.sum(token_object['ca_positions'], axis=0) / (np.sum(token_object['coordinate_mask']) + eps)
        # atom_center = np.sum(atom_object['atom_positions'], axis=0) / (np.sum(atom_object['atom_mask']) + eps)
        atom_object['atom_positions'] -= _center[None, :]
        token_object['ca_positions'] -= _center[None, :]
        return atom_object, token_object

    @staticmethod
    def truncate(atom_object, token_object, truncate_size=384):
        random_state = random.random()
        chain_num = len(np.unique(token_object['chain_index']))
        if chain_num == 1 or random_state < 0.3:
            return single_chain_truncate(atom_object, token_object, truncate_size)
        # elif random_state < 0.6:
        #     return contiguous_truncate(atom_object, token_object, truncate_size)
        else:
            return spatial_truncate(atom_object, token_object, truncate_size)

    @staticmethod
    def update_ref_features(data_object, training=True):
        data_object['ref_space_uid'] = data_object['atom_to_token_index']

        if training:
            ca_distance = (data_object['ca_positions'][:, None, :] - data_object['ca_positions'][None, :, :]).norm(dim=-1)
            lddt_mask = data_object['coordinate_mask'][None, :] * data_object['coordinate_mask'][:, None]
            data_object['lddt_mask'] = (ca_distance < 15.0) & lddt_mask
            data_object['bond_mask'] = data_object['lddt_mask']

        return data_object



