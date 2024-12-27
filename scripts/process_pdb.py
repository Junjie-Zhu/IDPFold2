import argparse
import collections
import os
import pickle
from functools import partial
from glob import glob
from typing import Dict
import multiprocessing as mp

import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import pandas as pd
from tqdm import tqdm

import residue_constants_ as rc


def write_to_pkl(
        protein_dict: dict,
        output_dir: str,
):
    with open(output_dir, "wb") as f:
        pickle.dump(protein_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_pdb_paths(
        pdb_dir: str,
        target_pdb_ids: list[str] = None,
):
    pdb_dir = os.path.expanduser(pdb_dir)
    all_pdb_paths = glob(os.path.join(pdb_dir, "*.pdb"), recursive=True)

    if target_pdb_ids is not None:
        all_pdb_paths = [
            pdb_path for pdb_path in all_pdb_paths
            if os.path.basename(pdb_path)[:4].upper() in target_pdb_ids
        ]

    return all_pdb_paths


def concat_chain_features(chain_feats: Dict[str, Dict[str, np.ndarray]]):
    """Performs a nested concatenation of feature dicts.

    Args:
        chain_feats: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in chain_feats.values():
        for feat_name, feat_val in chain_dict.items():
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def instantiate_protein(structure: struc.AtomArray) -> dict:
    """
    Instantiate a protein structure.

    Parameters
    ----------
    structure : struc.AtomArray
        The protein structure.

    Returns
    -------
    dict
        A dictionary containing the protein structure.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    for residues in struc.residue_iter(structure):
        residue_name = rc.restype_3to1.get(residues[0].res_name, "X")
        restype_idx = rc.restype_order.get(residue_name, rc.restype_num)
        pos = np.zeros((rc.atom_type_num, 3))
        mask = np.zeros((rc.atom_type_num,))

        for atom in residues:
            if atom.atom_name not in rc.atom_types:
                continue
            pos[rc.atom_order[atom.atom_name]] = atom.coord
            mask[rc.atom_order[atom.atom_name]] = 1.

        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(residues[0].res_id)
        chain_ids.append(residues[0].chain_id)

    # split by chain_ids
    chain_ids = np.array(chain_ids)
    unique_chain_ids = np.unique(chain_ids)

    protein_dicts = {}
    for chain_id in unique_chain_ids:
        chain_mask = chain_ids == chain_id
        protein_dicts[chain_id] = {
            "atom_positions": np.array(atom_positions)[chain_mask],
            "aatype": np.array(aatype)[chain_mask],
            "atom_mask": np.array(atom_mask)[chain_mask],
            "residue_index": np.array(residue_index)[chain_mask],
            "chain_ids": np.array(chain_ids)[chain_mask],
        }
    return protein_dicts


def process_pdb(
        pdb_path: list[str],
        output_dir: str,
        per_chain: bool = True,
):
    """
    process single pdb file
    """
    metadata = {
        "pdb_name": [],
        "processed_path": [],
        "modeled_seq_len": [],
    }
    pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
    structure = strucio.load_structure(pdb_path)

    try:
        struc_depth = structure.stack_depth()
    except AttributeError:
        struc_depth = 1

    if struc_depth > 1:
        for model_idx, model in enumerate(structure):
            model_dict = instantiate_protein(model)

            if per_chain:
                for chain_id, chain_dict in model_dict.items():
                    metadata["pdb_name"].append('.'.join([pdb_name, str(model_idx), str(chain_id)]))

                    output_path = os.path.join(output_dir, f"{metadata['pdb_name'][-1]}.pkl")
                    write_to_pkl(chain_dict, output_path)

                    metadata["processed_path"].append(output_path)
                    metadata["modeled_seq_len"].append(len(chain_dict["aatype"]))

            else:
                metadata["pdb_name"].append('.'.join([pdb_name, str(model_idx), 'all']))

                output_path = os.path.join(output_dir, f"{metadata['pdb_name'][-1]}.pkl")
                write_to_pkl(concat_chain_features(model_dict), output_path)

                metadata["processed_path"].append(output_path)
                metadata["modeled_seq_len"].append(sum([len(chain_dict["aatype"]) for chain_dict in model_dict.values()]))

    elif struc_depth == 1:
        for chain_id, chain_dict in instantiate_protein(structure).items():
            metadata["pdb_name"].append('.'.join([pdb_name, "0", str(chain_id)]))

            output_path = os.path.join(output_dir, f"{metadata['pdb_name'][-1]}.pkl")
            write_to_pkl(chain_dict, output_path)

            metadata["processed_path"].append(output_path)
            metadata["modeled_seq_len"].append(len(chain_dict["aatype"]))

    else:
        raise ValueError("Invalid structure stack depth.")

    return metadata


def main(args):
    pdb_paths = get_pdb_paths(args.input_dir)
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _process_fn = partial(
        process_pdb,
        output_dir=output_dir,
        per_chain=args.per_chain,
    )

    all_metadata = {
        "pdb_name": [],
        "processed_path": [],
        "modeled_seq_len": [],
    }
    if args.num_processes == 1:
        for pdb_path in tqdm.tqdm(pdb_paths):
            crt_metadata = _process_fn(pdb_path)
            all_metadata["pdb_name"].extend(crt_metadata["pdb_name"])
            all_metadata["processed_path"].extend(crt_metadata["processed_path"])
            all_metadata["modeled_seq_len"].extend(crt_metadata["modeled_seq_len"])
    else:
        with mp.Pool() as pool:
            _all_metadata = []
            # Use imap to track progress with tqdm
            for result in tqdm(pool.imap(_process_fn, pdb_paths), total=len(pdb_paths)):
                _all_metadata.append(result)

        # Now extend the fields in all_metadata
        for list_data in _all_metadata:
            all_metadata["pdb_name"].extend(list_data["pdb_name"])
            all_metadata["processed_path"].extend(list_data["processed_path"])
            all_metadata["modeled_seq_len"].extend(list_data["modeled_seq_len"])

    # concat all metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def get_args():
    # Define the parser
    parser = argparse.ArgumentParser(description='mmCIF processing script.')

    parser.add_argument('--input_dir', help='Path to directory with mmcif files.', type=str)
    parser.add_argument('--output_dir', help='Path to write results to.', type=str,
                        default='./data/processed_pdb')
    parser.add_argument('--num_processes', help='Number of processes. (Set to be 1 if serially)', type=int,
                        default=32)
    parser.add_argument('--per_chain', help='Whether to process single chain instead of complex.',
                        action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = get_args()
    main(args)