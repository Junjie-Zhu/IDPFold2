import json
import multiprocessing as mp
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from functools import partial
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize
from tqdm import tqdm

# --- Physics Constants & Uncertainties ---

POTENCI_UNCERTAINTIES = {"C": 0.1861, "CA": 0.1862, "CB": 0.1677, "N": 0.5341, "H": 0.0735, "HA": 0.0319, "HB": 0.0187}
CS_UNCERTAINTIES = {
    "UCBshift": {"C": 1.14, "CA": 1.09, "CB": 1.34, "N": 2.61, "H": 0.45, "HA": 0.26, "HB": 0.170},
    "Sparta+": {"C": 1.25, "CA": 1.16, "CB": 1.36, "N": 2.73, "H": 0.51, "HA": 0.27, "HB": 0.212},
    "ShiftX2": {"C": 1.20, "CA": 1.15, "CB": 1.37, "N": 2.73, "H": 0.49, "HA": 0.27, "HB": 0.212},
}
CS_UNCERTAINTIES["CSpred"] = CS_UNCERTAINTIES["UCBshift"]


# --- Core Optimization Engine ---

def cs_gamma_objective(lmbd: np.ndarray, alpha: float, std_delta_cs: np.ndarray):
    """Dual objective function for CS reweighting."""
    arg = -np.dot(lmbd, std_delta_cs)
    log_z = logsumexp(arg)

    # f = ln(Z) + 0.5 * alpha * ||lmbd||^2
    f = log_z + 0.5 * alpha * np.dot(lmbd, lmbd)

    # grad = - <delta>_rew + alpha * lmbd
    grad = -np.dot(std_delta_cs, softmax(arg)) + alpha * lmbd
    return f, grad


def run_cs_minimization_turbo(alpha_range: np.ndarray, std_delta_cs: np.ndarray) -> np.ndarray:
    """Sequential alpha scan with warm-starting."""
    std_delta_cs = np.ascontiguousarray(std_delta_cs)
    n_obs = std_delta_cs.shape[0]
    last_lmbd = np.zeros(n_obs)

    # Sort alphas largest to smallest (High alpha = easier solution)
    sorted_alphas = np.sort(alpha_range)[::-1]
    alpha_to_lmbd = {}

    for alpha in sorted_alphas:
        res = minimize(
            fun=cs_gamma_objective,
            x0=last_lmbd,
            args=(alpha, std_delta_cs),
            jac=True,
            method='L-BFGS-B',
            options={'gtol': 1e-6, 'ftol': 1e-7, 'maxiter': 500}
        )
        if res.success:
            last_lmbd = res.x
            alpha_to_lmbd[alpha] = res.x
        else:
            alpha_to_lmbd[alpha] = np.full(n_obs, np.nan)

    return np.array([alpha_to_lmbd[a] for a in alpha_range])


# --- Data Handling Utilities ---

def standardize_deltas(gen_df: pd.DataFrame, exp_dict: Dict[Tuple[int, str], float],
                       gscores: Dict[int, float], predictor: str) -> Tuple[np.ndarray, List]:
    """Standardizes chemical shifts using predictor uncertainties and g-scores."""
    unc_map = CS_UNCERTAINTIES.get(predictor, CS_UNCERTAINTIES["UCBshift"])

    # Find common keys between experiment and generated data
    keys = sorted(set(exp_dict.keys()).intersection(set(gen_df.index)))
    if not keys: return np.array([]), []

    # Calculate combined uncertainty per residue/atom
    # Using g-score to scale between POTENCI (intrinsic) and Predictor (error-prone) limits
    std_devs = []
    for r, a in keys:
        g = gscores.get(r, 0.5)
        # Higher g-score = more 'ordered' = less uncertainty penalty
        sigma = POTENCI_UNCERTAINTIES.get(a, 0.2) + \
                (unc_map.get(a, 1.0) - POTENCI_UNCERTAINTIES.get(a, 0.2)) * (1 - g)
        std_devs.append(sigma)

    std_devs = np.array(std_devs).reshape(-1, 1)
    exp_vals = np.array([exp_dict[k] for k in keys]).reshape(-1, 1)
    gen_vals = gen_df.loc[keys].values  # (n_obs, n_frames)

    return (gen_vals - exp_vals) / std_devs, keys


def load_filtered_exp(path: Path, bmrb_stats: pd.DataFrame, sigma_cutoff: float) -> Dict:
    """Loads and filters experimental shifts against BMRB statistical ranges."""
    df = pd.read_csv(path, sep='\s+')
    exp_dict = {}
    for _, row in df.iterrows():
        res, resname, name = int(row['#RESID']), str(row['RESNAME']), str(row.get('HN', row.get('H', '')))
        # Logic to iterate over columns like CA, CB, etc.
        for atom in ["CA", "HA", "CB", "C", "N", "H", "HN"]:
            if atom in row and not pd.isna(row[atom]) and row[atom] > 0:
                val = float(row[atom])
                atom_id = 'H' if atom in ['HN', 'H'] else atom

                # BMRB Filtering
                stat = bmrb_stats[(bmrb_stats["comp_id"] == resname) & (bmrb_stats["atom_id"] == atom_id)]
                if not stat.empty:
                    mu, sd = stat["avg"].values[0], stat["std"].values[0]
                    if np.abs(val - mu) < sd * sigma_cutoff:
                        exp_dict[(res, atom_id)] = val
    return exp_dict


def get_RMSE(std_delta_cs: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Standard RMSE calculation matching your original logic."""
    mask_nan = ~np.isnan(std_delta_cs).any(axis=0)
    if weights is None:
        avg = np.average(std_delta_cs[:, mask_nan], axis=1)
    else:
        # Only use weights for frames that aren't NaN
        m_w = ~np.isnan(weights)
        total_mask = mask_nan & m_w
        if not total_mask.any(): return np.nan
        avg = np.average(std_delta_cs[:, total_mask], weights=weights[total_mask], axis=1)
    return np.linalg.norm(avg) / np.sqrt(len(std_delta_cs))


def get_ESS(weights: np.ndarray) -> float:
    """Kish effective sample size."""
    mask_nan = ~np.isnan(weights)
    if not mask_nan.any(): return np.nan
    w = weights[mask_nan]
    return float(np.sum(w)**2 / np.dot(w, w))


# --- Main Worker ---

def cs_reweight_worker(protein: str, ensemble_root: Path, exp_root: Path,
                       bmrb_path: Path, gscore_dict: Dict, predictor: str = "UCBshift") -> str:
    """Full pipeline for a single protein's chemical shift reweighting."""
    out_file = ensemble_root / f"CSrew_{protein}.npy"
    gen_file = ensemble_root / f"UCBshift-{protein}.csv"
    exp_file = exp_root / protein / "CS.dat"
    gscore_file = exp_root / protein / "info.csv"

    if not gen_file.exists(): return f"Skipped {protein}: No CS predictions."

    # 1. Load Data
    bmrb_stats = pd.read_csv(bmrb_path)
    exp_dict = load_filtered_exp(exp_file, bmrb_stats, sigma_cutoff=3.0)
    gen_df = pd.read_csv(gen_file).set_index(["resSeq", "name"])

    gscores = {i: s for i, s in enumerate(gscore_dict.get(protein, {})) if not np.isnan(s)}
    
    # 2. Standardization
    std_delta_cs, keys = standardize_deltas(gen_df, exp_dict, gscores, predictor)
    if std_delta_cs.size == 0: return f"Error {protein}: No valid data."

    # 3. Handle Physical Filtering and NaNs
    mask_nan = ~np.isnan(std_delta_cs).any(axis=0)
    valid_samples = np.where(mask_nan)[0]  # You can intersect this with physical_traj_indices

    # 4. Optimization (Scanning Alphas)
    alpha_range = np.flip(10 ** np.linspace(-2, 7, 64))
    opt_lmbds = run_cs_minimization_turbo(alpha_range, std_delta_cs[:, valid_samples])

    # 5. Populate wopt_array (matches your original shape requirements)
    wopt_array = []
    for lmbd in opt_lmbds:
        weights = np.full(std_delta_cs.shape[1], np.nan)
        if not np.isnan(lmbd).any():
            w_valid = softmax(-np.dot(lmbd, std_delta_cs[:, valid_samples]))
            weights[valid_samples] = w_valid
        wopt_array.append(weights)

    # 6. Metrics and Selection
    rew_RMSEs = np.array([get_RMSE(std_delta_cs, w) for w in wopt_array])
    ESSs = np.array([get_ESS(w) for w in wopt_array])

    # ESS Thresholding
    n_samples_i = len(valid_samples)
    ess_threshold = min(np.nanmax(ESSs), max(100, 0.1 * n_samples_i))
    valid_indices = np.where(ESSs >= ess_threshold)[0]

    if len(valid_indices) == 0: return f"Error {protein}: ESS threshold not met."

    # Selection Logic
    ess_thresh = max(100, 0.1 * len(valid_samples))
    valid_idx = np.where(ESSs >= ess_thresh)[0]
    sel = valid_idx[-1] if len(valid_idx) > 0 else 0

    # 7. Save
    res = {
        'n_obs': std_delta_cs.shape[0],
        'n_samples': n_samples_i,
        'prior_rmse': get_RMSE(std_delta_cs[:, valid_samples]),
        'alpha': alpha_range[sel],
        'post_rmse': rew_RMSEs[sel],
        'ess': ESSs[sel],
        'weights': wopt_array[sel],
        'all_alphas': alpha_range,
        'all_post_rmse': rew_RMSEs,
        'all_ess': ESSs,
        'all_weights': wopt_array
    }
    np.save(out_file, res)
    return f"Success {protein}: RMSE {rew_RMSEs[sel]:.3f}"


# --- Multiprocessing Entry Point ---

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description="CS Reweighting Multiprocessing")
    args.add_argument('--ensemble_dir', '-i', type=str, required=True)
    args.add_argument('--exp_dir', '-e', type=str, required=True)
    args.add_argument('--bmrb_path', type=str, required=True)
    args.add_argument('--info_path', type=str, required=True)
    parsed_args = args.parse_args()

    ENSEMBLE_DIR = Path(parsed_args.ensemble_dir)
    EXP_DIR = Path(parsed_args.exp_dir)
    BMRB_FILE = Path(parsed_args.bmrb_path)
    GSCORE_FILE = Path(parsed_args.info_path)
    gscore_df = pd.read_csv(GSCORE_FILE, index_col='label')
    gscore_dict = {
        label: np.asarray(json.loads(gscore_df.loc[label, "gscores"]), dtype=float)
        for label in gscore_df.index
    }

    proteins = [d.name for d in EXP_DIR.iterdir() if d.is_dir()]

    print(f"Starting CS reweighting for {len(proteins)} proteins...")

    worker = partial(cs_reweight_worker, ensemble_root=ENSEMBLE_DIR,
                     exp_root=EXP_DIR, bmrb_path=BMRB_FILE, gscore_dict=gscore_dict, predictor="UCBshift")
    num_worker = min(len(proteins) // 3, mp.cpu_count())
    with mp.Pool(num_worker) as pool:
        results = list(tqdm(pool.imap_unordered(worker, proteins), total=len(proteins)))

    for r in results: print(r)
