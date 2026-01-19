import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# --- Constants ---
K_CONST = 1.23e16  # ?^6 s^-2
TAU_T = 0.5e-9  # s
DELAY_HSQC = 0.010
DELAY_HMQC = 0.01086
R2H_HSQC = 10.0
R2H_HMQC = 50.0
R2MQ_HMQC = 50.0


# --- Utilities ---

def find_files(path, suffix):
    return [f for f in os.listdir(path) if f.endswith(suffix)]


def calc_gamma2(r3, r6, angular, wh_mhz, tau_c, tau_t):
    """Calculates PRE rates (gamma2) from distances and correlation times."""
    s_pre = np.power(r3, 2) / r6 * angular
    wh_rad = 2 * np.pi * 1e6 * wh_mhz

    def spectral_density(w):
        return s_pre * tau_c / (1 + (w * tau_c) ** 2) + (1 - s_pre) * tau_t / (1 + (w * tau_t) ** 2)

    return K_CONST * r6 * (4 * spectral_density(0) + 3 * spectral_density(wh_rad))


def calc_intensity_ratio(gamma2, experiment_type):
    """Converts gamma2 rates to I_para/I_dia ratios based on experiment type."""
    if experiment_type in ['HSQC', 'gamma2']:
        return np.exp(-DELAY_HSQC * gamma2) * (R2H_HSQC / (R2H_HSQC + gamma2))
    elif experiment_type == 'HMQC':
        return np.exp(-DELAY_HMQC * gamma2) * (R2H_HMQC / (R2H_HMQC + gamma2)) * (R2MQ_HMQC / (R2MQ_HMQC + gamma2))
    return None


def calculate_rmse(gen, exp):
    exp = np.minimum(1.0, np.array(exp))  # Cap experimental at 1.0
    return np.sqrt(np.mean((np.array(gen) - exp) ** 2))


# --- Core Analysis Logic ---

def get_ensemble_pre(tau_c, weights, pre_gen_data, wh_exp, exp_type, residue_mask):
    """Calculates the weighted average PRE intensities for a specific tau_c."""
    # 1. Calculate Gamma2 for all frames
    gamma2 = calc_gamma2(pre_gen_data['r3'], pre_gen_data['r6'], pre_gen_data['angular'],
                         wh_mhz=wh_exp, tau_c=tau_c, tau_t=TAU_T)

    # 2. Handle frames where spin label couldn't be fit (NaNs)
    valid_mask = ~np.isnan(gamma2).any(axis=1)
    if not np.any(valid_mask):
        return None

    gamma2_filtered = gamma2[valid_mask]
    w_filtered = weights[valid_mask]

    if np.sum(w_filtered) == 0:  # Avoid division by zero
        return None

    # 3. Ensemble Average
    avg_gamma2 = np.average(gamma2_filtered, weights=w_filtered, axis=0)

    # 4. Convert to Intensities
    i_gen = calc_intensity_ratio(avg_gamma2, exp_type)

    # 5. Filter to match experimental residues
    gen_residues = pre_gen_data['Residue'].astype(int)
    idxs = [list(gen_residues).index(r) for r in residue_mask if r in gen_residues]

    return i_gen[idxs]


def process_protein_pre(protein, input_root, exp_root, output_path, tauc_values, ess_threshold=100):
    """
    Main execution logic for a single protein.
    - input_root: Path to the directory containing SAXSrew.npy and PREdata-*.npy
    - exp_root: Path to the directory containing experimental .dat and info.csv
    """
    exp_path = Path(exp_root) / protein
    output_protein_path = Path(output_path) / protein

    # 1. Load Reweighting Data
    saxs_file = Path(input_root) / f"SAXSrew_{protein}.npy"
    if not saxs_file.exists(): return f"Missing SAXS file for {protein}"

    saxs_data = np.load(saxs_file, allow_pickle=True).item()
    all_weights = saxs_data.get("all_weights", [saxs_data.get("weights")])
    all_ess = saxs_data.get("all_ess", [0])
    prior_weights = np.ones(len(all_weights[0])) / len(all_weights[0])

    # 2. Load Exp Info
    info_df = pd.read_csv(exp_path / "info.csv")
    pre_files = [f for f in find_files(exp_path, ".dat") if "PRE" in f]

    # Pre-load generated data and exp data to save time in tau_c loop
    exp_cache = {}
    for f in pre_files:
        site = int(f.split('-')[-1].split('.')[0])
        data = np.loadtxt(exp_path / f)
        data = data[~np.isnan(data).any(axis=1)]
        data = data[(data[:, 0] != site) & (data[:, 0] != 1)]  # Exclude site and Res 1

        dtype = "HMQC" if "HMQC" in f else ("HSQC" if "PRE-I" in f else "gamma2")
        try:
            wh_exp = info_df[info_df["Experiment"] == f[:-4]]["PRE_MHz"].mean()
        except TypeError as e:
            print(f"Warning: Missing PRE_MHz for {protein} site {site}")
            wh_exp = 700  # Default to 700 MHz

        gen_data = np.load(output_protein_path / f"PREdata-{site}.npy", allow_pickle=True).item()

        exp_cache[site] = {
            'res': data[:, 0].astype(int),
            'exp_i': calc_intensity_ratio(data[:, 1], 'HSQC') if dtype == 'gamma2' else data[:, 1],
            'type': dtype,
            'wh': wh_exp,
            'gen_data': gen_data
        }

    # 3. Tau_c Scanning Function
    def perform_scan(weights_to_use):
        best_rmse = float('inf')
        best_data = {}
        best_tau = 0

        for tc in tauc_values:
            current_intensities = {}
            all_gen, all_exp = [], []

            for site, d in exp_cache.items():
                i_gen = get_ensemble_pre(tc, weights_to_use, d['gen_data'], d['wh'], d['type'], d['res'])
                if i_gen is not None:
                    current_intensities[site] = i_gen
                    all_gen.append(i_gen)
                    all_exp.append(d['exp_i'])

            if not all_gen: continue
            rmse = calculate_rmse(np.concatenate(all_gen), np.concatenate(all_exp))

            if rmse < best_rmse:
                best_rmse = rmse
                best_tau = tc
                best_data = current_intensities
        return best_rmse, best_tau, best_data

    # 4. Run Analysis
    # Prior
    prior_rmse, prior_tau, prior_i = perform_scan(prior_weights)

    # Posterior (iterating through weight sets from SAXS reweighting)
    post_results = []
    for w in all_weights:
        post_results.append(perform_scan(np.nan_to_num(w, nan=0)))

    # Filter by ESS
    valid_idxs = np.where(np.array(all_ess) >= ess_threshold)[0]
    if len(valid_idxs) == 0:
        best_post_idx = 0
    else:
        valid_rmses = [post_results[i][0] for i in valid_idxs]
        best_post_idx = valid_idxs[np.argmin(valid_rmses)]

    post_rmse, post_tau, post_i = post_results[best_post_idx]

    # 5. Final Data Assembly
    results = {
        "protein": protein,
        "Prior_RMSE": prior_rmse,
        "Post_RMSE": post_rmse,
        "Prior_TauC": np.round(prior_tau * 1e9, 2),
        "Post_TauC": np.round(post_tau * 1e9, 2),
        "Final_ESS": all_ess[best_post_idx],
        "Sites": {}
    }

    for site in exp_cache.keys():
        results["Sites"][site] = {
            "Residues": exp_cache[site]['res'].tolist(),
            "I_exp": exp_cache[site]['exp_i'].tolist(),
            "I_prior": prior_i[site].tolist() if site in prior_i else [],
            "I_post": post_i[site].tolist() if site in post_i else []
        }

    # Save to JSON
    out_file = Path(output_path) / f"PRE_analysis_{protein}.json"
    with open(out_file, 'w') as jf:
        json.dump(results, jf, indent=4)

    return f"Completed {protein}: Prior RMSE {prior_rmse:.3f}, Post RMSE {post_rmse:.3f}"


# --- Execution ---

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description="PRE Analysis Multiprocessing")
    args.add_argument('--input_root', '-i', type=str, required=True)
    args.add_argument('--exp_root', '-e', type=str, required=True)
    args.add_argument('--pre_path', '-p', type=str, required=True)
    parsed_args = args.parse_args()


    INPUT_DIR = parsed_args.input_root
    EXP_DIR = parsed_args.exp_root
    output_path = parsed_args.pre_path
    # os.makedirs(output_path, exist_ok=True)

    tauc_scan = np.linspace(1e-9, 20e-9, 20)
    proteins = [d.name for d in Path(EXP_DIR).iterdir() if d.is_dir()]

    worker = partial(process_protein_pre,
                     input_root=INPUT_DIR,
                     exp_root=EXP_DIR,
                     output_path=output_path,
                     tauc_values=tauc_scan)

    worker_num = min(len(proteins), os.cpu_count())
    with mp.Pool(processes=worker_num) as pool:
        results = list(tqdm(pool.imap_unordered(worker, proteins), total=len(proteins)))

    for r in results:
        print(r)
