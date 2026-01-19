import os
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# --- Core RDC Physics Logic ---

def read_calc_RDCs(filename: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse calculated RDC CSV and account for 15N gyromagnetic ratio."""
    df = pd.read_csv(filename)
    residues = df.iloc[:, 0].values.astype(int)
    # Exclude residue column, transpose to (n_frames, n_residues), and multiply by -1
    data = df.iloc[:, 1:].values.T * -1
    return residues, data


def scale_rdcs_to_minimize_q(calc_all_frames: np.ndarray, exp: np.ndarray,
                             weights: np.ndarray, is_prior: bool = True,
                             scale_matching: bool = True) -> Tuple:
    """
    Calculates the scaling factor s and Cornilescu Q-factor.
    Strictly follows your original scaling logic.
    """
    exp = np.array(exp)

    if is_prior:
        # Mask out unphysical conformations (NaN weights)
        mask_nan = ~np.isnan(weights)
        calc_avg = np.average(calc_all_frames[mask_nan, :], axis=0)
    else:
        # Standard reweighting
        w = np.nan_to_num(weights, nan=0)
        if np.sum(w) == 0: return None, None, np.nan, np.nan
        calc_avg = np.average(calc_all_frames, weights=w / np.sum(w), axis=0)

    # Scaling factor calculation
    if scale_matching:
        prod = calc_avg * exp
        keepidxs = np.where(prod > 0)[0]
        if keepidxs.size == 0:
            return None, None, np.nan, np.nan
        calc_filt = calc_avg[keepidxs]
        exp_filt = exp[keepidxs]
        s = np.sum(calc_filt * exp_filt) / np.sum(calc_filt ** 2)
    else:
        s = np.sum(calc_avg * exp) / np.sum(calc_avg ** 2)

    s = max(s, 0)  # Constrain to non-negative
    scaled = s * calc_avg

    # Q-factor calculation
    q = np.sqrt(np.mean((scaled - exp) ** 2)) / np.sqrt(np.mean(exp ** 2))
    q_nonscaled = np.sqrt(np.mean((calc_avg - exp) ** 2)) / np.sqrt(np.mean(exp ** 2))

    return scaled, s, q, q_nonscaled


# --- Main Worker ---

def rdc_worker(protein: str, paths: Dict, info: pd.DataFrame) -> Tuple[str, Optional[Dict]]:
    """Worker function to process RDC data for a single protein."""
    # Define specific file paths
    rew_file = paths['proton_root'] / f"CSrew_{protein}.npy"
    rdc_csv = paths['rdc_root'] / protein / "RDC/RDC.csv"
    exp_file = paths['exp_root'] / protein / "RDC_HN.dat"

    if not rew_file.exists() or not rdc_csv.exists():
        print(1, protein)
        return protein, None

    # 1. Load Data
    calc_res, calc_data = read_calc_RDCs(rdc_csv)
    csrew = np.load(rew_file, allow_pickle=True).item()

    prot_length = info.loc[protein, 'length']

    # Original logic check: skip entries where reweighting failed
    if 'note' in csrew or 'weights' not in csrew:
        print(2, protein)
        return protein, None

    weights = csrew['weights']

    # 2. Load and Filter Experimental Data
    exp_raw = np.loadtxt(exp_file)
    exp_res = exp_raw[:, 0].astype(int)
    exp_val = exp_raw[:, 1]

    # Mask NaNs and remove termini
    mask = ~np.isnan(exp_val)
    exp_res, exp_val = exp_res[mask], exp_val[mask]

    if exp_res[0] == 1:
        exp_res, exp_val = exp_res[1:], exp_val[1:]
    if exp_res[-1] == prot_length:
        exp_res, exp_val = exp_res[:-1], exp_val[:-1]

    # 3. Align simulation residues to experimental residues
    valid_idxs = [list(calc_res).index(r) for r in exp_res if r in calc_res]
    if not valid_idxs:
        print(3, protein)
        return protein, None

    calc_res_aligned = calc_res[valid_idxs]
    calc_data_aligned = calc_data[:, valid_idxs]

    if calc_data_aligned.shape[0] != weights.shape[0]:
        calc_data_aligned = calc_data_aligned[:-1, :]

    # 4. Assessment (Prior vs Post)
    # Using weights_nan for prior to exclude unphysical samples
    res_prior = scale_rdcs_to_minimize_q(calc_data_aligned, exp_val, weights, is_prior=True)
    # Using cleaned weights for post
    res_post = scale_rdcs_to_minimize_q(calc_data_aligned, exp_val, weights, is_prior=False)

    # STRICTLY PRESERVE THE ORIGINAL DICT FORMAT
    return protein, {
        'Prior Q': res_prior[2],
        'Post. Q': res_post[2],
        'Residues': calc_res_aligned,
        'Exp': exp_val,
        'Prior': res_prior[0],
        'Post.': res_post[0]
    }


# --- Main Execution ---

if __name__ == "__main__":
    # === Config Paths ===
    import argparse

    args = argparse.ArgumentParser(description="RDC Analysis Multiprocessing")
    args.add_argument('--proton_root', '-i', type=str, required=True)
    args.add_argument('--exp_root', '-e', type=str, required=True)
    args.add_argument('--rdc_path', '-r', type=str, required=True)
    args.add_argument('--info_path', type=str, required=True)
    parsed_args = args.parse_args()

    PATHS = {
        'proton_root': Path(parsed_args.proton_root),
        'exp_root': Path(parsed_args.exp_root),
        'rdc_root': Path(parsed_args.rdc_path),
        'info_path': Path(parsed_args.info_path)
    }

    # Identify proteins with RDC data
    rdc_proteins = [p for p in os.listdir(PATHS['exp_root']) if (PATHS['exp_root'] / p / "RDC_HN.dat").exists()]

    print(f"Found {len(rdc_proteins)} proteins with RDC data. Starting multiprocessing...")

    rdcdict = {}

    # Process proteins in parallel
    worker_func = partial(rdc_worker, paths=PATHS, info=pd.read_csv(PATHS['info_path'], index_col='label'))
    with mp.Pool(processes=min(len(rdc_proteins) // 3, mp.cpu_count())) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, rdc_proteins), total=len(rdc_proteins)))

    # Collect results into the final dictionary
    for protein, data in results:
        if data is not None:
            rdcdict[protein] = data
            
        out_file = PATHS['rdc_root'] / f"RDC_analysis_{protein}.npy"
        np.save(out_file, data)

    # Final summary
    print(f"Analysis complete. Successfully processed {len(rdcdict)} proteins.")

