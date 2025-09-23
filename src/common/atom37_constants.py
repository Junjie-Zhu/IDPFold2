# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from typing import Dict

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

ATOM_NUMBERING: Dict[str, int] = {
    atom: i
    for i, atom in enumerate(
        [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "OG",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OD1",
            "ND2",
            "CG1",
            "CG2",
            "CD",
            "CE",
            "NZ",
            "OD2",
            "OE1",
            "NE2",
            "OE2",
            "OH",
            "NE",
            "NH1",
            "NH2",
            "OG1",
            "SD",
            "ND1",
            "SG",
            "NE1",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
            "OXT",
        ]
    )
}
"""Default ordering of atoms in (dimension 1 of) a protein structure tensor."""


# PDB and OpenFold have different atom ordering, these utils convert between the two
# PDB ordering: https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf
# OpenFold ordering: https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L556
# more background: https://kdidi.netlify.app/blog/proteins/2024-02-03-protein-representations/
PDB_TO_OPENFOLD_INDEX_TENSOR = torch.tensor(
    [ATOM_NUMBERING[atom] for atom in atom_types]
)
OPENFOLD_TO_PDB_INDEX_TENSOR = torch.tensor(
    [atom_types.index(atom) for atom in ATOM_NUMBERING]
)

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}


def atom37_to_atom14_indices():
    indices_mapping = torch.zeros((len(restype_name_to_atom14_names), 14)).long()
    mask_mapping = torch.zeros((len(restype_name_to_atom14_names), 14), dtype=torch.bool)
    restype_to_idx = {name: i for i, name in enumerate(restype_name_to_atom14_names.keys())}

    for i, (restype, atom14_names) in enumerate(restype_name_to_atom14_names.items()):
        for j, atom_name in enumerate(atom14_names):
            if atom_name in atom_types:
                indices_mapping[i, j] = atom_types.index(atom_name)
                mask_mapping[i, j] = True
            else:
                indices_mapping[i, j] = -1  # Padding index for missing atoms
    return indices_mapping, mask_mapping, restype_to_idx


ATOM37_TO_ATOM14_INDICES, ATOM14_MASK, RESTYPE_TO_IDX = atom37_to_atom14_indices()
