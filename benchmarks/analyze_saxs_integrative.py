import os
import traceback
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import multiprocessing as mp
from functools import partial

from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from tqdm import tqdm


# --- Core Logic Functions ---

def get_weights(lmbd: np.ndarray, std_delta_cs: np.ndarray) -> np.ndarray:
    """Get normalized weights from lambda. Samples with any NaN cs will have weight 0."""
    mask_nan = ~np.isnan(std_delta_cs).any(axis=0)
    is_flat = lmbd.ndim == 1

    if is_flat:
        lmbd = lmbd.reshape(1, -1)

    weights = np.zeros((lmbd.shape[0], std_delta_cs.shape[1]))
    # Compute softmax only for non-NaN samples
    weights[:, mask_nan] = softmax(-np.dot(lmbd, std_delta_cs[:, mask_nan]), axis=-1)

    return weights.flatten() if is_flat else weights


def get_ESS(weights: np.ndarray) -> float:
    """Kish effective sample size."""
    mask_nan = ~np.isnan(weights)
    if not mask_nan.any():
        return np.nan
    w = weights[mask_nan]
    norm = np.sum(w)
    return float(norm ** 2 / np.dot(w, w))


def get_RMSE(std_delta_cs: np.ndarray, weights: Optional[np.ndarray] = None, order: int = 2) -> float:
    """Root Mean Square Error calculation."""
    mask_nan = ~np.isnan(std_delta_cs).any(axis=0)

    if weights is None:
        avg = np.average(std_delta_cs[:, mask_nan], axis=1)
    else:
        mask_nan_w = ~np.isnan(weights)
        total_mask = mask_nan & mask_nan_w
        if not total_mask.any():
            return np.nan
        avg = np.average(std_delta_cs[:, total_mask], weights=weights[total_mask], axis=1)

    return np.linalg.norm(avg, ord=order) / np.power(len(std_delta_cs), 1 / order)


def parse_gensaxs_dat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors='ignore')
    q = df.columns.values.astype(float)
    I_gen = df.values
    return q, I_gen


def parse_saxs_dat(path: Path) -> pd.DataFrame:
    expt_df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["q", "I(q)", "sigma"], usecols=[0, 1, 2],
        on_bad_lines="skip", encoding="latin1"
    )
    expt_df = expt_df.apply(pd.to_numeric, errors="coerce").dropna()
    if expt_df.empty:
        raise pd.errors.ParserError(f"File {path} is empty or corrupted.")
    return expt_df


def get_std_delta_saxs(I_exp, sigma_exp, I_gen, intensity_scaling=True) -> np.ndarray:
    if intensity_scaling:
        # Svergun et al. (1995) scaling
        num = np.sum(I_exp * I_gen / sigma_exp ** 2, axis=1)
        den = np.sum((I_gen / sigma_exp) ** 2, axis=1)
        c = (num / den).reshape(-1, 1)
    else:
        c = 1.0
    return ((c * I_gen - I_exp) / sigma_exp).T


def gamma(lmbd: np.ndarray, alpha: float, std_delta_saxs: np.ndarray, rebalance: float = 1.0) -> Tuple[
    float, np.ndarray]:
    logw = -np.dot(lmbd, std_delta_saxs)
    g_val = logsumexp(logw) + 0.5 * alpha * np.dot(lmbd, lmbd)
    jac = -np.dot(std_delta_saxs, softmax(logw)) + alpha * lmbd
    return g_val * rebalance, jac * rebalance


def run_gamma_minimization(alpha_range: np.ndarray, std_delta_saxs: np.ndarray) -> np.ndarray:
    mask_nan = ~np.isnan(std_delta_saxs).any(axis=0)
    results = []

    for alpha in alpha_range:
        res = minimize(
            lambda x: gamma(x, alpha, std_delta_saxs[:, mask_nan], min(1, 1 / alpha)),
            np.random.rand(len(std_delta_saxs)),
            jac=True,
            method='L-BFGS-B'  # Generally faster for this type of problem
        )
        if not res.success:
            results.append(np.full(len(std_delta_saxs), np.nan))
        else:
            results.append(res.x)
    return np.array(results)


def gamma_objective(lmbd: np.ndarray, alpha: float, std_delta_saxs: np.ndarray):
    """
    Lagrangian dual objective function for Maximum Entropy reweighting.
    Optimized for L-BFGS-B.
    """
    # Dot product: (n_obs,) . (n_obs, n_samples) -> (n_samples,)
    arg = -np.dot(lmbd, std_delta_saxs)

    # logsumexp handles numerical stability for the partition function
    log_z = logsumexp(arg)

    # Objective: ln(Z) + 0.5 * alpha * ||lmbd||^2
    f = log_z + 0.5 * alpha * np.dot(lmbd, lmbd)

    # Gradient: - <O>_reweighted + alpha * lmbd
    # softmax(arg) gives the weights for the current lmbd
    grad = -np.dot(std_delta_saxs, softmax(arg)) + alpha * lmbd

    return f, grad


def run_gamma_minimization_turbo(alpha_range: np.ndarray, std_delta_saxs: np.ndarray) -> np.ndarray:
    """
    Optimized minimization using:
    1. L-BFGS-B (faster than trust-constr)
    2. Warm-Starting (using previous alpha's result as next x0)
    """
    # Ensure data is contiguous in memory for fast dot products
    std_delta_saxs = np.ascontiguousarray(std_delta_saxs)
    n_obs = std_delta_saxs.shape[0]

    # Initialize multipliers at 0 (corresponds to uniform weights)
    last_lmbd = np.zeros(n_obs)
    results = []

    # Sort alphas from LARGEST to SMALLEST.
    # High alpha = high regularization = solution closer to 0 (easier to solve).
    # We move from the 'easy' uniform distribution toward the 'hard' reweighted one.
    sorted_alphas = np.sort(alpha_range)[::-1]

    # Map to keep track of original order if alpha_range wasn't sorted
    alpha_to_lmbd = {}

    for alpha in tqdm(sorted_alphas):
        # Warm-start: x0 = last_lmbd
        res = minimize(
            fun=gamma_objective,
            x0=last_lmbd,
            args=(alpha, std_delta_saxs),
            jac=True,
            method='L-BFGS-B',
            options={'gtol': 1e-6, 'ftol': 1e-7, 'maxiter': 1000}
        )

        if res.success:
            last_lmbd = res.x
            alpha_to_lmbd[alpha] = res.x
        else:
            # If it fails, try a small jump from the last successful one
            # or keep the last_lmbd but flag it
            alpha_to_lmbd[alpha] = np.full(n_obs, np.nan)

    # Return in the original order requested by the user
    return np.array([alpha_to_lmbd[a] for a in alpha_range])

# --- Main Processing Worker ---

def saxs_reweight_worker(protein: str, ensemble_root: Path, exp_root: Path, n_alphas: int = 64) -> str:
    """
    Worker function to process a single protein.
    Returns a status message string.
    """
    out_file = ensemble_root / f"SAXSrew_{protein}.npy"
    pred_file = ensemble_root / f"Pepsi-{protein}.csv"
    exp_file = exp_root / protein / "SAXS_bift.dat"

    try:
        # Load data
        q, I_gen = parse_gensaxs_dat(pred_file)
        exp_df = parse_saxs_dat(exp_file)
        I_exp = exp_df['I(q)'].values
        sigma_exp = exp_df['sigma'].values

        alpha_range = np.flip(10 ** np.linspace(-2, 8, n_alphas))
        std_delta_saxs = get_std_delta_saxs(I_exp, sigma_exp, I_gen, intensity_scaling=True)

        if std_delta_saxs.size == 0:
            return f"Error {protein}: No data found."

        # Optimization
        opt_lmbd = run_gamma_minimization_turbo(alpha_range, std_delta_saxs)
        weights = get_weights(opt_lmbd, std_delta_saxs)

        if np.isnan(weights).all():
            return f"Warning {protein}: Minimization failed for all alphas."

        # Metrics
        rew_RMSEs = np.array([get_RMSE(std_delta_saxs, w) for w in weights])
        ESSs = np.array([get_ESS(w) for w in weights])

        # Selection logic (ESS thresholds)
        ess_abs_threshold = 100
        ess_rel_threshold = 0.1
        n_samples_i = std_delta_saxs.shape[1]
        ess_threshold = min(np.nanmax(ESSs), max(ess_abs_threshold, ess_rel_threshold * n_samples_i))

        # Find the index of the first alpha that satisfies the ESS threshold
        valid_indices = np.arange(len(ESSs))[ESSs >= ess_threshold]
        if len(valid_indices) == 0:
            return f"Error {protein}: No alpha satisfies ESS threshold."

        sel = valid_indices[-1]

        # Save results
        resdict = {
            'n_obs': len(q),
            'n_samples': n_samples_i,
            'prior_rmse': get_RMSE(std_delta_saxs),
            'alpha': alpha_range[sel],
            'post_rmse': rew_RMSEs[sel],
            'ess': ESSs[sel],
            'weights': weights[sel],
            'all_alphas': alpha_range,
            'all_post_rmse': rew_RMSEs,
            'all_ess': ESSs,
            'all_weights': weights
        }

        np.save(out_file, resdict)
        return f"Success {protein}: Processed."

    except Exception:
        return f"Failed {protein}: {traceback.format_exc()}"


# --- Execution ---

if __name__ == "__main__":
    # Define paths using Path objects
    import argparse

    args = argparse.ArgumentParser(description="SAXS Reweighting Multiprocessing")
    args.add_argument('--ensemble_root', '-i', type=str, required=True)
    args.add_argument('--exp_root', '-e', type=str, required=True)
    parsed_args = args.parse_args()

    ENSEMBLE_ROOT = Path(parsed_args.ensemble_root)
    EXP_ROOT = Path(parsed_args.exp_root)

    # Identify proteins to process
    all_proteins = [d.name for d in EXP_ROOT.iterdir() if d.is_dir()]
    saxs_proteins = [p for p in all_proteins if (EXP_ROOT / p / "SAXS_bift.dat").exists()]

    print(f"Found {len(saxs_proteins)} proteins with SAXS data. Starting multiprocessing...")

    # Configure the number of workers (None defaults to CPU count)
    num_workers = min(len(saxs_proteins), os.cpu_count() or 1)

    # Use a partial to fix the path arguments for the worker
    worker_fn = partial(saxs_reweight_worker, ensemble_root=ENSEMBLE_ROOT, exp_root=EXP_ROOT)

    results = []
    with mp.Pool(num_workers) as executor:
        # Wrap the executor in tqdm for a progress bar
        list_results = list(tqdm(executor.imap_unordered(worker_fn, saxs_proteins), total=len(saxs_proteins)))

    # Summary of run
    print("\nProcessing complete. Summary:")
    for res in list_results:
        if "Success" not in res:
            print(res)
