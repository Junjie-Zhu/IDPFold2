import os
import string
from collections import defaultdict
from typing import Any

import torch
import biotite.structure.io.pdbx as pdbx

import src.common.residue_constants as rc

ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}


def to_pdb_simple(
    input_feature_dict: dict[str, Any],
    atom_positions: torch.Tensor,
    output_dir: str,
):
    # all atoms are CA
    aatype = input_feature_dict['aatype'].squeeze()
    chain_indices = input_feature_dict["chain_index"].squeeze().cpu()
    residue_indices = input_feature_dict["residue_index"].squeeze().cpu()
    atom_positions = atom_positions.squeeze().cpu()
    accession_code = input_feature_dict["accession_code"][0]

    n_samples, n_res, _ = atom_positions.shape

    restype = [rc.IDX_TO_RESIDUE[res.item()] for res in aatype]
    chain_id = [INT_TO_CHAIN[chain_indices[i].item()] for i in range(n_res)]

    with open(os.path.join(output_dir, f"{accession_code}.pdb"), 'w') as f:
        for sample in range(n_samples):
            f.write(f'MODEL {sample + 1}\n')
            for i in range(atom_positions.shape[1]):
                atom_name = 'CA'
                resname = restype[i]
                chain_idx = chain_id[i]
                res_idx = residue_indices[i]

                x, y, z = atom_positions[sample][i]

                if chain_idx != chain_id[i - 1]:
                    f.write("TER\n")
                f.write(
                    f"ATOM  {i + 1:>5} {atom_name:<4} {resname:>3} {chain_idx}{res_idx:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2}\n"
                )

            f.write("ENDMDL\n")
        f.write("END\n")


def to_pdb(
    input_feature_dict: dict[str, Any],
    atom_positions: torch.Tensor,
    output_dir: str,
):
    """save atom_positions as pdb with biotite"""
    aatype = input_feature_dict["aatype"].squeeze()
    atom_to_token_index = input_feature_dict["atom_to_token_index"].squeeze().cpu()
    chain_indices = input_feature_dict["chain_index"].squeeze().cpu()
    ref_atom_names = input_feature_dict["ref_atom_name_chars"].squeeze()
    atom_positions = atom_positions.squeeze().cpu()
    accession_code = input_feature_dict["accession_code"][0]

    restype = [rc.IDX_TO_RESIDUE[res.item()] for res in aatype]
    atom_names = convert_atom_name_id(ref_atom_names)

    residue_indices = atom_to_token_index.tolist()
    chain_id_per_atom = [INT_TO_CHAIN[chain_indices[res_idx].item()]
                         for res_idx in residue_indices]
    resname_per_atom = [restype[res_idx] for res_idx in residue_indices]

    chain_to_res_ids = defaultdict(set)
    for res_idx, chain_id in zip(residue_indices, chain_id_per_atom):
        chain_to_res_ids[chain_id].add(res_idx)

    chain_to_res_map = {
        chain_id: {global_res: local_res+1  # start from 1
                   for local_res, global_res in enumerate(sorted(list(res_ids)))}
        for chain_id, res_ids in chain_to_res_ids.items()
    }

    with open(os.path.join(output_dir, f"{accession_code}.pdb"), 'w') as f:
        for i in range(atom_positions.shape[0]):
            atom_name = atom_names[i]
            resname = resname_per_atom[i]
            chain_id = chain_id_per_atom[i]
            global_res_idx = residue_indices[i]
            local_res_idx = chain_to_res_map[chain_id][global_res_idx]

            x, y, z = atom_positions[i]

            if chain_id != chain_id_per_atom[i - 1]:
                f.write("TER\n")
            f.write(
                f"ATOM  {i + 1:>5} {atom_name:<4} {resname:>3} {chain_id}{local_res_idx:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2}\n"
            )
        f.write("END\n")


def to_mmcif(
    input_feature_dict: dict[str, Any],
    atom_positions: torch.Tensor,
    output_dir: str,
):
    """save atom_positions as pdb with biotite"""
    aatype = input_feature_dict["aatype"].squeeze()
    atom_to_token_index = input_feature_dict["atom_to_token_index"].squeeze().cpu()
    chain_indices = input_feature_dict["chain_index"].squeeze().cpu()
    ref_atom_names = input_feature_dict["ref_atom_name_chars"].squeeze()
    atom_positions = atom_positions.squeeze().cpu()
    accession_code = input_feature_dict["accession_code"][0]

    restype = [rc.IDX_TO_RESIDUE[res.item()] for res in aatype]
    atom_names = convert_atom_name_id(ref_atom_names)

    residue_indices = atom_to_token_index.tolist()
    chain_id_per_atom = [chain_indices[res_idx].item()
                         for res_idx in residue_indices]
    resname_per_atom = [restype[res_idx] for res_idx in residue_indices]

    chain_to_res_ids = defaultdict(set)
    for res_idx, chain_id in zip(residue_indices, chain_id_per_atom):
        chain_to_res_ids[chain_id].add(res_idx)

    chain_to_res_map = {
        chain_id: {global_res: local_res+1  # start from 1
            for local_res, global_res in enumerate(sorted(list(res_ids)))}
        for chain_id, res_ids in chain_to_res_ids.items()
    }

    n_atoms = len(residue_indices)
    cif_block = {
        "group_PDB": ["ATOM"] * n_atoms,
        "type_symbol": [i[0] for i in atom_names],
        "label_atom_id": atom_names,
        "label_comp_id": resname_per_atom,
        "label_asym_id": chain_id_per_atom,
        "label_seq_id": [
            chain_to_res_map[cid][res_idx]
            for res_idx, cid in zip(residue_indices, chain_id_per_atom)
        ],
        "Cartn_x": atom_positions[:, 0].tolist(),
        "Cartn_y": atom_positions[:, 1].tolist(),
        "Cartn_z": atom_positions[:, 2].tolist(),
        "occupancy": [1.0] * n_atoms,
        "B_iso_or_equiv": [0.0] * n_atoms,
        "pdbx_PDB_model_num": [1] * n_atoms,
    }

    # Write mmCIF file
    os.makedirs(output_dir, exist_ok=True)
    cif_file_path = os.path.join(output_dir, f"{accession_code}.cif")
    cif_file = pdbx.PDBxFile()
    cif_file.__setitem__((accession_code, "atom_site"), cif_block)
    with open(cif_file_path, 'w') as f:
        cif_file.write(f)


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

