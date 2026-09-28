"""Write the SAXS case files that the figure panels and calculate_pr.py read.

Pepsi-SAXS writes ``Pepsi-{name}.csv``: one row per frame, one column per q.
Experimental profiles are whitespace-separated ``q I(q) sigma`` files named
``{name}.dat`` or ``{name}-bift.dat``.

For each matched pair this writes, into ``--output-dir``:

- ``{name}_pred.dat``: ensemble-mean intensity on the Pepsi q grid, columns
  ``q I(q) sigma``. q stays in the Pepsi unit (1/Å). ``calculate_pr.py`` reads
  this file with ``--angular-unit 1/A``. sigma is the standard deviation across
  frames, or 1% of |I| when that deviation is zero.
- ``{name}_intensity.csv``: ``q``, ``Iq_exp``, ``sigma``, ``Iq_pred``. Predicted
  intensity is interpolated onto the experimental q grid. ``q`` is multiplied
  by ``--q-scale`` (default 10) so a 1/Å profile matches the 1/nm figure axis.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def experimental_stem(filename):
    stem = filename[:-4] if filename.endswith(".dat") else filename
    if stem.endswith("-bift"):
        stem = stem[: -len("-bift")]
    return stem


def pepsi_stem(filename):
    stem = filename[:-4] if filename.endswith(".csv") else filename
    if stem.startswith("Pepsi-"):
        stem = stem[len("Pepsi-") :]
    return stem


def read_experimental(path):
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["q", "I", "sigma"],
        usecols=[0, 1, 2],
        comment="#",
        engine="python",
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError(f"No numeric rows in {path}")
    return frame["q"].to_numpy(), frame["I"].to_numpy(), frame["sigma"].to_numpy()


def read_pepsi(path):
    frame = pd.read_csv(path)
    frame = frame.drop(columns=["Unnamed: 0"], errors="ignore")
    q = frame.columns.astype(float).to_numpy()
    intensity = frame.to_numpy(dtype=float)
    if intensity.ndim != 2 or intensity.shape[1] != len(q):
        raise ValueError(f"Expected one intensity column per q in {path}")
    order = np.argsort(q)
    return q[order], intensity[:, order]


def predicted_sigma(intensity, mean):
    if intensity.shape[0] == 1:
        sigma = np.zeros_like(mean)
    else:
        sigma = np.nanstd(intensity, axis=0)
    floor = np.maximum(np.abs(mean) * 0.01, 1e-6)
    return np.where(sigma > 0, sigma, floor)


def write_pred_dat(path, q, mean, sigma):
    stacked = np.column_stack([q, mean, sigma])
    np.savetxt(path, stacked, fmt="%.8g")


def match_pairs(experimental_dir, pepsi_dir):
    experimental = {}
    for name in sorted(os.listdir(experimental_dir)):
        if not name.endswith(".dat"):
            continue
        stem = experimental_stem(name)
        if stem in experimental:
            raise ValueError(
                f"Two experimental profiles map to {stem}: "
                f"{experimental[stem]} and {name}"
            )
        experimental[stem] = name

    pairs = []
    missing = []
    for name in sorted(os.listdir(pepsi_dir)):
        if not name.endswith(".csv") or not name.startswith("Pepsi-"):
            continue
        stem = pepsi_stem(name)
        if stem not in experimental:
            missing.append(name)
            continue
        pairs.append((stem, experimental[stem], name))
    return pairs, missing


def process_pair(stem, experimental_name, pepsi_name, experimental_dir, pepsi_dir, output_dir, q_scale):
    exp_q, exp_i, exp_sigma = read_experimental(os.path.join(experimental_dir, experimental_name))
    pepsi_q, pepsi_i = read_pepsi(os.path.join(pepsi_dir, pepsi_name))
    mean = np.nanmean(pepsi_i, axis=0)
    sigma = predicted_sigma(pepsi_i, mean)

    write_pred_dat(os.path.join(output_dir, f"{stem}_pred.dat"), pepsi_q, mean, sigma)

    iq_pred = np.interp(exp_q, pepsi_q, mean)
    intensity = pd.DataFrame(
        {
            "q": exp_q * q_scale,
            "Iq_exp": exp_i,
            "sigma": exp_sigma,
            "Iq_pred": iq_pred,
        }
    )
    intensity.to_csv(os.path.join(output_dir, f"{stem}_intensity.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Write predicted SAXS .dat files and experimental-vs-predicted intensity tables."
    )
    parser.add_argument(
        "--experimental-dir",
        required=True,
        help="Directory of {name}.dat or {name}-bift.dat profiles (q, I(q), sigma).",
    )
    parser.add_argument(
        "--pepsi-dir",
        required=True,
        help="Directory of Pepsi-{name}.csv files (rows are frames, columns are q).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for {name}_pred.dat and {name}_intensity.csv.",
    )
    parser.add_argument(
        "--q-scale",
        type=float,
        default=10.0,
        help="Factor applied to q in the intensity CSV. Default 10 converts 1/Å to 1/nm.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.experimental_dir):
        print(f"Experimental directory does not exist: {args.experimental_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.pepsi_dir):
        print(f"Pepsi directory does not exist: {args.pepsi_dir}", file=sys.stderr)
        sys.exit(1)

    pairs, missing = match_pairs(args.experimental_dir, args.pepsi_dir)
    for name in missing:
        print(f"Skipping {name}: no matching experimental profile.", file=sys.stderr)
    if not pairs:
        print("No Pepsi CSV matched an experimental profile.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    for stem, experimental_name, pepsi_name in pairs:
        process_pair(
            stem,
            experimental_name,
            pepsi_name,
            args.experimental_dir,
            args.pepsi_dir,
            args.output_dir,
            args.q_scale,
        )
    print(f"Wrote {len(pairs)} SAXS case files to {args.output_dir}")


if __name__ == "__main__":
    main()
