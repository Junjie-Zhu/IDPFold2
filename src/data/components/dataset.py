"""Protein dataset class."""
import os
import pickle
from pathlib import Path
from glob import glob
from random import random
from typing import Optional, Sequence, List, Union
from functools import lru_cache
import tree

import numpy as np
import pandas as pd
import torch

from src.common import residue_constants, data_transforms, rigid_utils, protein


CA_IDX = residue_constants.atom_order['CA']
DTYPE_MAPPING = {
    'aatype': torch.long,
    'atom_positions': torch.double,
    'atom_mask': torch.double,
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


def convert_atom_name_id(atom_name: List[int]):
    """
        Converts integer of atom_name to unique atom_id names.
        Each character is decoded as chr(c + 32)
    """
    atom_name = ''.join([chr(c + 32) for c in atom_name])
    return atom_name.strip()


def calc_centre_of_mass(coords, atom_mass):
    mass_coords = coords * atom_mass[:, None]
    return torch.sum(mass_coords, dim=0) / torch.sum(atom_mass)

def get_atom_features(data_object, ccd_atom14):

    atom_mask = data_object['atom_mask']
    atom_positions = torch.zeros(int(atom_mask.sum()), 3, dtype=torch.float32)
    atom_com = torch.zeros(data_object['residue_index'].shape[0], 3, dtype=torch.float32)

    ref_positions = torch.zeros(int(atom_mask.sum()), 3, dtype=torch.float32)
    ref_com = torch.zeros(data_object['residue_index'].shape[0], 3, dtype=torch.float32)

    token2atom_map = torch.zeros(int(atom_mask.sum()), dtype=torch.int64)
    atom_elements = torch.zeros(int(atom_mask.sum()), dtype=torch.int64)
    atom_charge = torch.zeros(int(atom_mask.sum()), dtype=torch.float32)

    index_start, token = 0, 0
    atom_type = []
    for residues, residue_positions in zip(atom_mask, data_object['atom_positions']):
        length = int(residues.sum())
        index_end = index_start + length

        token2atom_map[index_start:index_end] += token

        atom_index = torch.where(residues)[0]
        atom_positions[index_start:index_end] = residue_positions[atom_index]
        atom_elements[index_start:index_end] = torch.tensor([element_atomic_number[i] for i in atom_index], dtype=torch.int64)
        ref_positions[index_start:index_end] = ccd_atom14['coord'][int(data_object['aatype'][token])][atom_index]

        atom_com[token] = calc_centre_of_mass(atom_positions[index_start:index_end], atom_elements[index_start:index_end] * 2)
        ref_com[token] = calc_centre_of_mass(ref_positions[index_start:index_end], atom_elements[index_start:index_end] * 2)
        # atom_charge[index_start:index_end] = ccd_atom14[data_object['aatype'][token]]['charge'][atom_index]

        atom_type.extend([atom_types[i] for i in atom_index])

        index_start = index_end
        token += 1

    atom_space_uid = token2atom_map
    atom_name_char = convert_atom_id_name(atom_type)

    onehot_dict = {}
    for index, key in enumerate(range(32)):
        onehot = [0] * 32
        onehot[index] = 1
        onehot_dict[key] = onehot

    onehot_encoded_data = [onehot_dict[int(item)] for item in atom_elements]
    atom_elements = torch.Tensor(onehot_encoded_data)

    output_batch = {
        'atom_to_token_idx': token2atom_map,

        'residue_index': data_object['residue_index'],
        'seq_mask': torch.ones_like(data_object['residue_index'], dtype=torch.float32),
        'aatype': data_object['aatype'],
        'seq_emb': data_object['seq_emb'],
        'residue_com': atom_com,
        'ref_com': ref_com,o

        'ref_pos': ref_positions,
        'ref_space_uid': atom_space_uid,
        'ref_atom_name_chars': atom_name_char,
        'ref_element': atom_elements,
        'ref_mask': torch.ones_like(atom_space_uid, dtype=torch.float)
    }

    label_batch = {
        'coordinate': atom_positions,
        'coordinate_mask': torch.ones_like(atom_positions[:, 0]).squeeze(),
    }

    # calculate the distance matrix from coordinates
    distance_matrix = label_batch['coordinate'][:, None, :] - label_batch['coordinate'][None, :, :]
    lddt_mask = distance_matrix.norm(dim=-1) < 15.0
    label_batch['lddt_mask'] = lddt_mask

    return output_batch, label_batch


class ProteinFeatureTransform:
    def __init__(self, 
                 unit: Optional[str] = 'angstrom', 
                 truncate_length: Optional[int] = None,
                 strip_missing_residues: bool = True,
                 recenter_and_scale: bool = True,
                 eps: float = 1e-8,
    ):
        if unit == 'angstrom':
            self.coordinate_scale = 1.0
        elif unit in ('nm', 'nanometer'):
            self.coordiante_scale = 0.1
        else:
            raise ValueError(f"Invalid unit: {unit}")
        
        if truncate_length is not None:
            assert truncate_length > 0, f"Invalid truncate_length: {truncate_length}"
        self.truncate_length = truncate_length
        
        self.strip_missing_residues = strip_missing_residues
        self.recenter_and_scale = recenter_and_scale
        self.eps = eps
        
    def __call__(self, chain_feats, ccd_atom14):
        chain_feats = self.patch_feats(chain_feats)
        
        if self.strip_missing_residues:
            chain_feats = self.strip_ends(chain_feats)
        
        if self.truncate_length is not None:
            chain_feats = self.random_truncate(chain_feats, max_len=self.truncate_length)
        
        # Recenter and scale atom positions
        if self.recenter_and_scale:
            chain_feats = self.recenter_and_scale_coords(chain_feats, coordinate_scale=self.coordinate_scale, eps=self.eps)
        
        # Map to torch Tensor
        chain_feats = self.map_to_tensors(chain_feats)

        # transform to all-atom features
        chain_feats, label_dict = get_atom_features(chain_feats, ccd_atom14)

        # Add extra features from AF2 
        # chain_feats = self.protein_data_transform(chain_feats)
        
        # ** refer to line 170 in pdb_data_loader.py **
        return chain_feats, label_dict
    
    @staticmethod
    def patch_feats(chain_feats):
        seq_mask = chain_feats['atom_mask'][:, CA_IDX]   # a little hack here
        # residue_idx = np.arange(seq_mask.shape[0], dtype=np.int64)
        residue_idx = chain_feats['residue_index'] - np.min(chain_feats['residue_index'])   # start from 0, possibly has chain break
        patch_feats = {
            'seq_mask': seq_mask,
            'residue_mask': seq_mask,
            'residue_idx': residue_idx,
            'fixed_mask': np.zeros_like(seq_mask),
            'sc_ca_t': np.zeros(seq_mask.shape + (3, )),
        }
        chain_feats.update(patch_feats)
        return chain_feats
    
    @staticmethod
    def strip_ends(chain_feats):
        # Strip missing residues on both ends
        modeled_idx = np.where(chain_feats['aatype'] != 20)[0]
        min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)

        seq_emb = chain_feats['seq_emb']
        chain_feats = tree.map_structure(
                lambda x: x[min_idx : (max_idx+1)], chain_feats)
        chain_feats['seq_emb'] = seq_emb
        return chain_feats
    
    @staticmethod
    def random_truncate(chain_feats, max_len):
        L = chain_feats['aatype'].shape[0]
        if L > max_len:
            # Randomly truncate
            start = np.random.randint(0, L - max_len + 1)
            end = start + max_len
            chain_feats = tree.map_structure(
                    lambda x: x[start : end], chain_feats)
        return chain_feats
    
    @staticmethod
    def map_to_tensors(chain_feats):
        chain_feats = {k: torch.as_tensor(v) for k,v in chain_feats.items()}
        # Alter dtype 
        for k, dtype in DTYPE_MAPPING.items():
            if k in chain_feats:
                chain_feats[k] = chain_feats[k].type(dtype)
        return chain_feats
    
    @staticmethod
    def recenter_and_scale_coords(chain_feats, coordinate_scale, eps=1e-8):
        # recenter and scale atom positions
        bb_pos = chain_feats['atom_positions'][:, CA_IDX]
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['seq_mask']) + eps)
        centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
        scaled_pos = centered_pos * coordinate_scale
        chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
        return chain_feats

    @staticmethod
    def protein_data_transform(chain_feats):
        chain_feats.update(
            {
                "all_atom_positions": chain_feats["atom_positions"],
                "all_atom_mask": chain_feats["atom_mask"],
            }
        )
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles("")(chain_feats)
        chain_feats = data_transforms.get_backbone_frames(chain_feats)
        chain_feats = data_transforms.get_chi_angles(chain_feats)
        chain_feats = data_transforms.make_pseudo_beta("")(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        
        # Add convenient key
        chain_feats.pop("all_atom_positions")
        chain_feats.pop("all_atom_mask")
        return chain_feats
    

class MetadataFilter:
    def __init__(self, 
                 min_len: Optional[int] = None,
                 max_len: Optional[int] = None,
                 min_chains: Optional[int] = None,
                 max_chains: Optional[int] = None,
                 min_resolution: Optional[int] = None,
                 max_resolution: Optional[int] = None,
                 include_structure_method: Optional[List[str]] = None,
                 include_oligomeric_detail: Optional[List[str]] = None,
                 **kwargs,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.min_chains = min_chains
        self.max_chains = max_chains
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.include_structure_method = include_structure_method
        self.include_oligomeric_detail = include_oligomeric_detail
    
    def __call__(self, df):
        _pre_filter_len = len(df)
        if self.min_len is not None:
            df = df[df['raw_seq_len'] >= self.min_len]
        if self.max_len is not None:
            df = df[df['raw_seq_len'] <= self.max_len]
        if self.min_chains is not None:
            df = df[df['num_chains'] >= self.min_chains]
        if self.max_chains is not None:
            df = df[df['num_chains'] <= self.max_chains]
        if self.min_resolution is not None:
            df = df[df['resolution'] >= self.min_resolution]
        if self.max_resolution is not None:
            df = df[df['resolution'] <= self.max_resolution]
        if self.include_structure_method is not None:
            df = df[df['include_structure_method'].isin(self.include_structure_method)]
        if self.include_oligomeric_detail is not None:
            df = df[df['include_oligomeric_detail'].isin(self.include_oligomeric_detail)]
        
        print(f">>> Filter out {len(df)} samples out of {_pre_filter_len} by the metadata filter")
        return df


class RandomAccessProteinDataset(torch.utils.data.Dataset):
    """Random access to pickle protein objects of dataset.
    
    dict_keys(['atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors'])
    
    Note that each value is a ndarray in shape (L, *), for example:
        'atom_positions': (L, 37, 3)
    """
    def __init__(self, 
                 path_to_dataset: Union[Path, str],
                 path_to_seq_embedding: Optional[Path] = None,
                 metadata_filter: Optional[MetadataFilter] = None,
                 training: bool = True,
                 transform: Optional[ProteinFeatureTransform] = None, 
                 suffix: Optional[str] = '.pkl',
                 accession_code_fillter: Optional[Sequence[str]] = None,
                 **kwargs,
    ):
        super().__init__()
        path_to_dataset = os.path.expanduser(path_to_dataset)
        suffix = suffix if suffix.startswith('.') else '.' + suffix
        assert suffix in ('.pkl', '.pdb'), f"Invalid suffix: {suffix}"
        
        if os.path.isfile(path_to_dataset): # path to csv file
            assert path_to_dataset.endswith('.csv'), f"Invalid file extension: {path_to_dataset} (have to be .csv)"
            self._df = pd.read_csv(path_to_dataset)
            self._df.sort_values('modeled_seq_len', ascending=False)
            if metadata_filter:
                self._df = metadata_filter(self._df)
            self._data = self._df['processed_path'].tolist()
        elif os.path.isdir(path_to_dataset):  # path to directory
            self._data = sorted(glob(os.path.join(path_to_dataset, '*' + suffix)))
            assert len(self._data) > 0, f"No {suffix} file found in '{path_to_dataset}'"
        else:   # path as glob pattern
            _pattern = path_to_dataset
            self._data = sorted(glob(_pattern))
            assert len(self._data) > 0, f"No files found in '{_pattern}'"
        
        if accession_code_fillter and len(accession_code_fillter) > 0:
            self._data = [p for p in self._data
                if np.isin(os.path.splitext(os.path.basename(p))[0], accession_code_fillter) 
            ]
            
        self.data = np.asarray(self._data)
        self.path_to_seq_embedding = os.path.expanduser(path_to_seq_embedding) \
                if path_to_seq_embedding is not None else None
        self.suffix = suffix
        self.transform = transform
        self.training = training  # not implemented yet

        # get absolute path
        cwd = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(cwd, 'ccd_atom14.pkl'), 'rb') as f:
            self.ccd_atom14 = pickle.load(f)
    
    @property    
    def num_samples(self):
        return len(self.data)
    
    def len(self): 
        return self.__len__()

    def __len__(self):
        return self.num_samples 

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        """return single pyg.Data() instance
        """
        data_path = self.data[idx]
        accession_code = os.path.splitext(os.path.basename(data_path))[0]
        
        if self.suffix == '.pkl':
            # Load pickled protein
            with open(data_path, 'rb') as f:
                data_object = pickle.load(f)
        elif self.suffix == '.pdb':
            # Load pdb file
            with open(data_path, 'r') as f:
                pdb_string = f.read()
            data_object = protein.from_pdb_string(pdb_string).to_dict()
        
        # Get sequence embedding if have
        if self.path_to_seq_embedding is not None:
            with open(os.path.join(self.path_to_seq_embedding, f"{accession_code}.pkl"), 'rb') as f:
                embed_dict = pickle.load(f)
                data_object.update(
                    {
                        'seq_emb': embed_dict['representations'],
                    } # 33 is for ESM650M
                )

        # Apply data transform
        if self.transform is not None:
            data_object, label_object = self.transform(data_object, self.ccd_atom14)
            
            # fixed_mask = torch.zeros_like(data_object['ref_mask'], dtype=torch.float)
            # # randomly mask 20% of the atoms
            # mask_idx = torch.randperm(fixed_mask.shape[0])[:int(fixed_mask.shape[0] * 0.2)]
            # fixed_mask[mask_idx] = 1
            # data_object['fixed_mask'] = fixed_mask

        data_object['accession_code'] =  accession_code

        data_object.update(label_object)
        return data_object  # dict of arrays

    

class PretrainPDBDataset(RandomAccessProteinDataset):
    def __init__(self, 
                 path_to_dataset: str,
                 metadata_filter: MetadataFilter,
                 transform: ProteinFeatureTransform, 
                 **kwargs,
    ):
        super(PretrainPDBDataset, self).__init__(path_to_dataset=path_to_dataset, 
                                                 metadata_filter=metadata_filter,
                                                 transform=transform,
                                                 **kwargs,
        )


class SamplingPDBDataset(RandomAccessProteinDataset):
    def __init__(self, 
                 path_to_dataset: str,
                 path_to_seq_embedding: str,
                 training: bool = False,
                 suffix: str = '.pdb',
                 transform: Optional[ProteinFeatureTransform] = None, 
                 accession_code_fillter: Optional[Sequence[str]] = None,
    ):
        # assert os.path.isdir(path_to_dataset), f"Invalid path (expected to be directory): {path_to_dataset}"
        super(SamplingPDBDataset, self).__init__(path_to_dataset=path_to_dataset, 
                                            path_to_seq_embedding=path_to_seq_embedding,
                                            training=training,
                                            suffix=suffix,
                                            transform=transform,
                                            accession_code_fillter=accession_code_fillter,
                                            metadata_filter=None,
        )
        
