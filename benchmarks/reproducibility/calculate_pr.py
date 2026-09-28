"""Calculate SAXS P(r) fits for every ``.dat`` profile in a directory.

The pipeline follows ``daily/26-9-18/calculate_pr.py``. It loads each profile with
``saxs_assistant``, selects a Guinier Rg, enumerates P(r) fits, and keeps the
best fit. Run it once on the experimental profiles and once on predicted
intensities saved as ``.dat`` files (columns q, I(q), sigma).
``prepare_saxs_cases.py`` writes those predicted profiles from Pepsi CSV files.

Each profile writes:

- ``{sample}_pr_fits.npy``: every P(r) fit
- ``{sample}_best_pr_fit.csv``: the selected fit, including Rg, Dmax, and P(r)

``plot_ax_case_pr`` reads r, P(r), Rg, and Dmax from those two best-fit tables.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Filled by _bind_saxs() so --help works without saxs_assistant installed.
raw = None
Profile_Loader = None
get_all_pr_results = None
select_best_pr_fit = None
unpack_pr_fits_dict = None
reprocess_sasbdb_q_values = None
rg_method_1 = None
select_best_rg_method = None
select_final_rg_from_candidates = None


def _bind_saxs():
    """Import saxs_assistant, including the modules the original script imported."""
    global raw, Profile_Loader, get_all_pr_results, select_best_pr_fit, unpack_pr_fits_dict
    global reprocess_sasbdb_q_values, rg_method_1, select_best_rg_method, select_final_rg_from_candidates
    from saxs_assistant.ML import (  # noqa: F401
        assign_gmm_clusters,
        compute_franke_features,
        load_model,
        predict_dmax_from_features_only,
    )
    from saxs_assistant.PDDF_tools import (
        get_all_pr_results as _get_all_pr_results,
        select_best_pr_fit as _select_best_pr_fit,
        unpack_pr_fits_dict as _unpack_pr_fits_dict,
    )
    from saxs_assistant.features import get_GPA, get_kratky  # noqa: F401
    from saxs_assistant.rawutils import Profile_Loader as _Profile_Loader
    from saxs_assistant.rawutils import rawapitry as _raw
    from saxs_assistant.rawutils import sasm as SASM  # noqa: F401
    from saxs_assistant.rawutils import sasexceptions as SASEceptions  # noqa: F401
    from saxs_assistant.rg_tools import (
        rg_method_1 as _rg_method_1,
        select_best_rg_method as _select_best_rg_method,
        select_final_rg_from_candidates as _select_final_rg_from_candidates,
    )
    from saxs_assistant.utils.helpers import (
        reprocess_sasbdb_q_values as _reprocess_sasbdb_q_values,
        setup_profile_cache,  # noqa: F401
    )

    raw = _raw
    Profile_Loader = _Profile_Loader
    get_all_pr_results = _get_all_pr_results
    select_best_pr_fit = _select_best_pr_fit
    unpack_pr_fits_dict = _unpack_pr_fits_dict
    reprocess_sasbdb_q_values = _reprocess_sasbdb_q_values
    rg_method_1 = _rg_method_1
    select_best_rg_method = _select_best_rg_method
    select_final_rg_from_candidates = _select_final_rg_from_candidates


def update_murthy_df_row(df, index, updates):
    for key, value in updates.items():
        df.loc[df.index[index], key] = value


def _load_profile(path, angular_unit):
    profile = Profile_Loader.load_profiles(path)[0]
    if angular_unit == "1/A":
        reprocessed = reprocess_sasbdb_q_values(
            profile._q_raw, profile._i_raw, profile._err_raw
        )
    elif angular_unit == "1/nm":
        reprocessed = reprocess_sasbdb_q_values(
            profile._q_raw / 10, profile._i_raw, profile._err_raw
        )
    else:
        raise ValueError(f"Unsupported angular unit: {angular_unit}")

    profile._q_raw = reprocessed["q"]
    profile._i_raw = reprocessed["I"]
    profile._err_raw = reprocessed["error"]
    profile.q = profile._q_raw
    profile.i = profile._i_raw
    profile.err = profile._err_raw

    mask = profile.q < 0.255
    profile.q = profile.q[mask]
    profile.i = profile.i[mask]
    profile.err = profile.err[mask]
    return profile


def process_profile(sample_id, path, angular_unit, output_dir):
    """Fit one SAXS profile. Return False, and print why, when Rg selection fails."""
    profile = _load_profile(path, angular_unit)
    q, intensity, err = profile.q, profile.i, profile.err

    row = pd.DataFrame(
        {
            "file name": [sample_id],
            "path": [os.path.dirname(path) or "."],
            "Angular unit": [angular_unit],
        }
    )

    (
        rg_auto,
        i0_auto,
        rg_err,
        i0_err,
        qmin,
        qmax,
        qrg_min,
        qrg_max,
        idx_min,
        idx_max,
        r_sq,
    ) = raw.auto_guinier(profile)

    update_murthy_df_row(
        row,
        0,
        {
            "AutoRg Rg": rg_auto,
            "AutoRg I0": i0_auto,
            "AutoRg Rg Err": rg_err,
            "AutoRg I0 Err": i0_err,
            "AutoRg qmin": qmin,
            "AutoRg qmax": qmax,
            "AutoRg qRg Min": qrg_min,
            "AutoRg qRg Max": qrg_max,
            "AutoRg Guinier R?": r_sq,
            "AutoRg Rg idx min": idx_min,
            "AutoRg Rg idx max": idx_max,
        },
    )

    pr_fits_dict = get_all_pr_results(profile, q, intensity, err)
    update_murthy_df_row(
        row,
        0,
        {
            "Num P(r) Fits": len(pr_fits_dict),
            "Avg P(r) Rg": np.mean([v["Rg"] for v in pr_fits_dict.values()]),
        },
    )

    pr_rg_list = [fit["Rg"] for fit in pr_fits_dict.values()]
    rg_method1_df, rg_method1_all = rg_method_1(q, intensity, err, pr_rg_list)
    if rg_method1_df is None:
        print(f"Skipping {sample_id}: method-1 Rg candidates are empty.", file=sys.stderr)
        return False

    update_murthy_df_row(
        row,
        0,
        {
            "Method 1 Rg Count": len(rg_method1_df),
            "Method 1 Rg Mean": round(rg_method1_df["Rg"].mean(), 2),
            "Method 1 Best R?": round(rg_method1_df["fit_r2"].max(), 3),
        },
    )

    final_method1_df = select_final_rg_from_candidates(
        rg_method1_all, q, intensity, err, sample_id
    )
    if final_method1_df is None:
        print(f"Skipping {sample_id}: method-1 Rg selection failed.", file=sys.stderr)
        return False

    best_fit = final_method1_df.iloc[0]
    update_murthy_df_row(
        row,
        0,
        {
            "Method 1 Final Rg": best_fit["Rg"],
            "Method 1 Final Rg Err": best_fit["Rg Err"],
            "Method 1 Final I0": best_fit["i0"],
            "Method 1 Final I0 Err": best_fit["I0 Err"],
            "Method 1 Final R?": best_fit["fit_r2"],
            "Method 1 qRgmin": best_fit["qRgmin"],
            "Method 1 qRgmax": best_fit["qRgmax"],
            "Method 1 GPA Peak": best_fit["GPA x peak"],
            "Method 1 Res Width": best_fit["Res window"],
            "Method 1 nmin": best_fit["nmin"],
            "Method 1 nmax": best_fit["nmax"],
            "Method 1 Peak/Xcross": best_fit["peaks/x-cross"],
            "Method 1 GPA x peak": best_fit["GPA x peak"],
            "Method 1 GPA y peak": best_fit["GPA y peak"],
        },
    )

    selection = select_best_rg_method(
        q,
        intensity,
        err,
        rg_auto,
        i0_auto,
        best_fit["Rg"],
        best_fit["i0"],
        rg_auto_err=row.loc[0, "AutoRg Rg Err"],
        i0_auto_err=row.loc[0, "AutoRg I0 Err"],
        rg1_err=best_fit["Rg Err"],
        i01_err=best_fit["I0 Err"],
        r2_auto=row.loc[0, "AutoRg Guinier R?"],
        r2_1=best_fit["fit_r2"],
        nmin_auto=row.loc[0, "AutoRg Rg idx min"],
        nmax_auto=row.loc[0, "AutoRg Rg idx max"],
        m1_nmin=best_fit["nmin"],
        m1_nmax=best_fit["nmax"],
        method1_exists=True,
        sample_id=sample_id,
    )
    if not selection:
        print(f"Skipping {sample_id}: no Guinier Rg method was selected.", file=sys.stderr)
        return False

    update_murthy_df_row(
        row,
        0,
        {
            "Final Rg": selection["Final Rg"],
            "Final qRg min": selection["Final qRg min"],
            "Final qRg max": selection["Final qRg max"],
            "Final I0": selection["Final I0"],
            "Final Rg Err": selection["Final Rg Err"],
            "Final I0 Err": selection["Final I0 Err"],
            "Final Rg R?": selection["Final R?"],
            "Rg Method": selection["Selected Method"],
            "Final Rg Residual Mean": selection["Residual Stats"]["residual_mean"][0],
            "Final Rg nmin": selection["Final nmin"],
            "Final Rg nmax": selection["Final nmax"],
        },
    )
    if selection["Selected Method"] == "Method 1":
        row.loc[0, "Final G qRgmin"] = row.loc[0, "Method 1 qRgmin"]
        row.loc[0, "Final G qRgmax"] = row.loc[0, "Method 1 qRgmax"]
    elif selection["Selected Method"] == "AutoRg":
        row.loc[0, "Final G qRgmin"] = row.loc[0, "AutoRg qRg Min"]
        row.loc[0, "Final G qRgmax"] = row.loc[0, "AutoRg qRg Max"]

        rg_value = selection["Final Rg"]
        r2_value = selection["Final R?"]
        nmin_rg = selection["Final nmin"]
        nmax_rg = selection["Final nmax"]
        num_points = nmax_rg - nmin_rg
        if nmin_rg == 0 and nmax_rg == 6:
            num_points += 1

        flag_reason = None
        if rg_value == -1:
            flag_reason = "Rg is -1"
        elif num_points < 7:
            flag_reason = f"Fewer than 7 points ({num_points})"
        elif r2_value < 0.73:
            flag_reason = f"Low R? ({r2_value:.2f})"

        if flag_reason:
            print(
                f"Skipping {sample_id}: method 1 failed and AutoRg could not meet "
                f"thresholds (R>0.73 and npoints >= 7): {flag_reason}.",
                file=sys.stderr,
            )
            return False

    (
        rg_min,
        pr_rg_err_list,
        i0_p_list,
        pr_i0_err_list,
        chi_sq_list,
        dmax_mins,
        logas,
        dmax_err,
        pr_qmin_list,
        pr_qmax_list,
        pr_list,
        pr_i_orig,
        pr_fit,
        pr_err_orig,
        pr_q_orig,
        pr_qxt,
        nmins,
    ) = unpack_pr_fits_dict(pr_fits_dict)

    fin_pr, _best_idx = select_best_pr_fit(
        rg_min,
        pr_rg_err_list,
        i0_p_list,
        pr_i0_err_list,
        chi_sq_list,
        dmax_mins,
        logas,
        dmax_err,
        pr_qmin_list,
        pr_qmax_list,
        row.loc[0, "Final Rg"],
        row.loc[0, "Final I0"],
        pr_list,
        pr_i_orig,
        pr_fit,
        pr_err_orig,
        pr_q_orig,
        pr_qxt,
        nmins,
        q,
        intensity,
        err,
        sample_id,
        row,
        0,
    )

    np.save(os.path.join(output_dir, f"{sample_id}_pr_fits.npy"), pr_fits_dict)
    fin_pr.to_csv(os.path.join(output_dir, f"{sample_id}_best_pr_fit.csv"), index=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Calculate P(r) fits for SAXS .dat profiles.")
    parser.add_argument("input_dir", help="Directory of .dat profiles (q, I(q), sigma).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for {sample}_pr_fits.npy and {sample}_best_pr_fit.csv. "
        "Defaults to {input_dir}/results.",
    )
    parser.add_argument(
        "--angular-unit",
        choices=("1/A", "1/nm"),
        default="1/A",
        help="q unit of every profile in input_dir (default: 1/A).",
    )
    args = parser.parse_args()

    try:
        _bind_saxs()
    except ImportError as exc:
        print(f"calculate_pr.py requires saxs_assistant: {exc}", file=sys.stderr)
        sys.exit(1)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(input_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    saxs_files = sorted(name for name in os.listdir(input_dir) if name.endswith(".dat"))
    if not saxs_files:
        print(f"No .dat profiles in {input_dir}", file=sys.stderr)
        sys.exit(1)

    n_ok = 0
    for sample_id in tqdm(saxs_files, desc="SAXS P(r)"):
        path = os.path.join(input_dir, sample_id)
        try:
            if process_profile(sample_id, path, args.angular_unit, output_dir):
                n_ok += 1
        except Exception as exc:
            print(f"Skipping {sample_id}: {exc}", file=sys.stderr)

    print(f"Wrote {n_ok}/{len(saxs_files)} P(r) fits to {output_dir}")
    if n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
