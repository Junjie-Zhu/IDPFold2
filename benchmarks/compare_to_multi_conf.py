import os
import sys
import shutil
import json
import warnings
from typing import Optional, Union
import multiprocessing as mp

import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.sequence as seq
from biotite.sequence.align import align_optimal, SubstitutionMatrix
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

root_dir = sys.argv[1]
assert os.path.exists(root_dir), f'Root directory does not exist: {root_dir}'
input_dir = os.path.join(root_dir)
processing_dir = os.path.join(root_dir, 'processing')
os.makedirs(processing_dir, exist_ok=True)

ref_cryp = pd.read_csv('./crypticpocket/references.csv')
ref_domi = pd.read_csv('./domainmotion/references.csv')
ref_loca = pd.read_csv('./localunfolding/references.csv')
ref_ood60 = pd.read_csv('./ood60/references.csv')
ref_oodval = pd.read_csv('./oodval/references.csv')

alignment_matrix = SubstitutionMatrix.std_protein_matrix()

RESI_THREE_TO_1 = {
    "3HP": "X",
    "4HP": "X",
    "5HP": "Q",
    "ABA": "A",
    "ACE": "X",
    "AIB": "A",
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASX": "B",
    "AYA": "A",
    "BMT": "T",
    "BOC": "X",
    "CBX": "X",
    "CEA": "C",
    "CGU": "E",
    "CME": "C",
    "CRO": "TYG",
    "CSD": "C",
    "CSO": "C",
    "CSS": "C",
    "CSW": "C",
    "CSX": "C",
    "CXM": "M",
    "CYS": "C",
    "CYX": "C",
    "DAL": "A",
    "DAR": "R",
    "DCY": "C",
    "DGL": "E",
    "DGN": "Q",
    "DHI": "H",
    "DIL": "I",
    "DIV": "V",
    "DLE": "L",
    "DLY": "K",
    "DPN": "F",
    "DPR": "P",
    "DSG": "N",
    "DSN": "S",
    "DSP": "D",
    "DTH": "T",
    "DTR": "W",
    "DTY": "Y",
    "DVA": "V",
    "FME": "M",
    "FOR": "X",
    "GLN": "Q",
    "GLU": "E",
    "GLX": "Z",
    "GLY": "G",
    "HID": "H",  # Different protonation states of HIS
    "HIE": "H",  # Different protonation states of HIS
    "HIP": "H",  # Different protonation states of HIS
    "HIS": "H",
    "HYP": "P",
    "ILE": "I",
    "IVA": "X",
    "KCX": "K",
    "LEU": "L",
    "LLP": "K",
    "LYS": "K",
    "MET": "M",
    "MLE": "L",
    "MSE": "M",
    "MVA": "V",
    "NH2": "X",
    "NLE": "L",
    "NLW": "L",
    "OCS": "C",
    "ORN": "A",
    "PCA": "Q",
    "PHE": "F",
    "PRO": "P",
    "PSW": "U",
    "PTR": "Y",
    "PVL": "X",
    "PYL": "O",
    "SAR": "G",
    "SEC": "U",
    "SEP": "S",
    "SER": "S",
    "STY": "Y",
    "THR": "T",
    "TPO": "T",
    "TPQ": "Y",
    "TRP": "W",
    "TYR": "Y",
    "TYS": "Y",
    "UNK": "X",
    "VAL": "V",
}

def main():
    print('Collecting system names...')
    system_names = [f.replace('.pdb', '') for f in os.listdir(input_dir) if f.endswith('.pdb')]

    # first we have to filter systems for this benchmark
    # test_cases = set(ref_cryp['test_case'].tolist() +
    #                  ref_domi['test_case'].tolist() +
    #                  ref_loca['test_case'].tolist() +
    #                  ref_ood60['test_case'].tolist() +
    #                  ref_oodval['test_case'].tolist())
    test_cases = ref_loca['test_case'].tolist()

    system_names = [name.split(':') for name in system_names]
    filtered_system_names = []
    for name_group in system_names:
        for name in name_group:
            if name in test_cases:
                filtered_system_names.append(name)
                shutil.copy(os.path.join(input_dir, f'{":".join(name_group)}.pdb'),
                            os.path.join(processing_dir, f'{name}.pdb'))
                break
    print(f'Found {len(filtered_system_names)} systems for benchmarking.')

    if os.cpu_count == 1:
        results = []
        for name in tqdm(filtered_system_names):
            result = process_single_prediction(name)
            results.append(result)
    else:
        with mp.Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_single_prediction, filtered_system_names),
                                total=len(filtered_system_names)))

    # consolidate results
    consolidated = {
        'test_case': [],
        'ref': [],
        'local_rmsd': [],
        'global_rmsd': [],
    }
    for res in results:
        consolidated['test_case'].append(res['test_case'])
        consolidated['ref'].append(res['ref'])
        consolidated['local_rmsd'].append(res['local_rmsd'])
        consolidated['global_rmsd'].append(res['global_rmsd'])

    # save results
    with open('./metrics_rmsd.pkl', 'wb') as f:
        import pickle
        pickle.dump(consolidated, f)


def find_reference(name, benchmark):
    ref_local_info = json.load(open(f'./{benchmark}/local_residinfo/{name}.json', 'r')) \
        if os.path.exists(f'./{benchmark}/local_residinfo/{name}.json') else None
    ref_structures = os.listdir(f'./{benchmark}/reference/{name}')
    ref = [strucio.load_structure(os.path.join(f'./{benchmark}/reference/{name}', f)) for f in ref_structures if f.endswith('.pdb')]
    return ref, ref_local_info, ref_structures


def process_single_prediction(name):
    pred_path = os.path.join(processing_dir, f'{name}.pdb')
    assert os.path.exists(pred_path), f'Prediction file not found: {pred_path}'

    # find reference
    if name in ref_cryp['test_case'].tolist():
        ref, ref_local_info, ref_structures = find_reference(name, 'crypticpocket')
    elif name in ref_domi['test_case'].tolist():
        ref, ref_local_info, ref_structures = find_reference(name, 'domainmotion')
    elif name in ref_loca['test_case'].tolist():
        ref, ref_local_info, ref_structures = find_reference(name, 'localunfolding')
    elif name in ref_ood60['test_case'].tolist():
        ref, ref_local_info, ref_structures = find_reference(name, 'ood60')
    elif name in ref_oodval['test_case'].tolist():
        ref, ref_local_info, ref_structures = find_reference(name, 'oodval')
    else:
        raise ValueError(f'No reference found for test case: {name}')
    pred = strucio.load_structure(pred_path)
    local_rmsd, global_rmsd, contacts = align_to_reference(pred, ref, ref_local_info)

    # save contacts info
    np.save(os.path.join(processing_dir, f'{name}_contacts.npy'), contacts)

    return {
        'test_case': name,
        'ref': ref_structures,
        'local_rmsd': local_rmsd,
        'global_rmsd': global_rmsd,
    }


def select_ca(struct: struc.AtomArray) -> struc.AtomArray:
    """Selects the CA atoms from a structure."""
    coords = struct.coord
    coords_shape = coords.shape
    if len(coords_shape) == 3:
        # multiple models
        struct_mask = (struct.atom_name == 'CA').reshape(1, -1).repeat(coords_shape[0], axis=0)
        return coords[struct_mask].reshape(coords_shape)
    else:
        return coords[struct.atom_name == 'CA']


def get_sequence(struct: struc.AtomArray):
    """Extracts the amino acid sequence from a structure."""
    if len(struct.coord.shape) == 3:
        model_0 = struct[0]
    else:
        model_0 = struct
    seq_str = ''
    for residue in struc.residue_iter(model_0):
        res_name = residue[0].res_name
        seq_str += RESI_THREE_TO_1.get(res_name, 'X')
    return seq.ProteinSequence(seq_str)


def align_to_reference(pred: struc.AtomArray,
                       ref: list[struc.AtomArray],
                       ref_local_info: Optional[dict]):
    """Aligns the predicted structure to the reference structures and computes RMSD."""
    seq_pred = get_sequence(pred)
    seq_ref = [get_sequence(r) for r in ref]

    pred = select_ca(pred)
    ref = [select_ca(r) for r in ref]

    local_rmsds = []
    global_rmsds = []
    ranges = ref_local_info.get('alignment_resid_ranges') if ref_local_info else None
    metrics_ranges = ref_local_info.get('metrics_resid_ranges') if ref_local_info else None

    for i, ref_struct in enumerate(ref):
        ref_seq = seq_ref[i]
        alignment = align_optimal(seq_pred, ref_seq, alignment_matrix)[0]
        trace = alignment.trace

        matched_mask = (trace[:, 0] != -1) & (trace[:, 1] != -1)
        pred_indices = trace[matched_mask, 0]
        ref_indices = trace[matched_mask, 1]

        if ranges:
            range_mask = np.zeros(len(pred_indices), dtype=bool)
            for start, end in ranges:
                in_range = (pred_indices >= (start - 1)) & (pred_indices < end)
                range_mask = range_mask | in_range
            anchor_pred_idx = pred_indices[range_mask]
            anchor_ref_idx = ref_indices[range_mask]
        else:
            anchor_pred_idx = pred_indices
            anchor_ref_idx = ref_indices

        if metrics_ranges:
            metric_mask = np.zeros(len(pred_indices), dtype=bool)
            for start, end in metrics_ranges:
                in_range = (pred_indices >= (start - 1)) & (pred_indices < end)
                metric_mask = metric_mask | in_range
            met_pred_indices = pred_indices[metric_mask]
            met_ref_indices = ref_indices[metric_mask]
        else:
            met_pred_indices = pred_indices
            met_ref_indices = ref_indices

        mobile_anchors = pred[:, anchor_pred_idx, :]  # multiple models
        fixed_anchors = ref_struct[anchor_ref_idx]

        met_mobile = pred[:, met_pred_indices, :]
        met_fixed = ref_struct[met_ref_indices]

        # calculate contact fraction for the local regions
        contact_fractions = calculate_contacts(
            struct=met_mobile,
            ref_struct=met_fixed,
            threshold=8.0
        )

        aligned, transform = struc.superimpose(
            fixed=fixed_anchors,
            mobile=mobile_anchors
        )
        rmsd_local = struc.rmsd(
            reference=fixed_anchors,
            subject=aligned
        )
        rmsd_global = struc.rmsd(
            reference=ref_struct[ref_indices],
            subject=transform.apply(pred[:, pred_indices, :])
        )
        local_rmsds.append(rmsd_local)
        global_rmsds.append(rmsd_global)
    return np.array(local_rmsds), np.array(global_rmsds), contact_fractions


def calculate_contacts(struct: Union[struc.AtomArray, np.ndarray],
                       ref_struct: Union[struc.AtomArray, np.ndarray],
                       threshold: float = 8.0) -> np.ndarray:
    '''Calculates the fraction of native contacts between two structures.'''
    if isinstance(struct, struc.AtomArray):
        struct = select_ca(struct)
    if isinstance(ref_struct, struc.AtomArray):
        ref_struct = select_ca(ref_struct)

    struct_contacts = np.linalg.norm(
        struct[:, :, np.newaxis, :] - struct[:, np.newaxis, :, :],
        axis=-1
    )  < threshold  # n_frames, n_res, n_res
    ref_contacts = np.linalg.norm(
        ref_struct[:, np.newaxis, :] - ref_struct[np.newaxis, :, :],
        axis=-1
    ) < threshold  # n_res, n_res

    # remove neighbor contacts
    neighbor_mask = np.ones(ref_contacts.shape)
    for i in range(ref_contacts.shape[0]):
        neighbor_mask[i, max(0, i - 3):min(ref_contacts.shape[0], i + 3)] = 0
    ref_contacts = ref_contacts & neighbor_mask.astype(bool)

    contact_fractions = [
        (struct_contact & ref_contacts).sum() / ref_contacts.sum()
        for struct_contact in struct_contacts
    ]
    return np.array(contact_fractions)


if __name__ == '__main__':
    main()