"""Fit a deeptime TICA model on Cα–Cα distances.

The reference can be a multi-model PDB or a topology plus an xtc/dcd trajectory.
A directory is fit one system at a time: each top-level PDB, and each subdirectory
that contains ``topology.pdb`` plus ``traj.xtc``, ``traj.dcd``, or ``samples.xtc``.

The model is written to ``{name}_tica.npz`` with the pickled TICA model and the
reference projection under ``projected_data``. Distances are mdtraj nanometers.
``project_tica.py`` must use the same Cα order.
"""

import argparse
import pickle
import sys
from itertools import combinations
from pathlib import Path

import mdtraj as md
import numpy as np
from deeptime.decomposition import TICA


TRAJ_NAMES = ("traj.xtc", "traj.dcd", "samples.xtc", "traj_no_clash.xtc")


def ca_distances(trajectory):
    ca_indices = trajectory.topology.select("name CA")
    if len(ca_indices) < 2:
        raise ValueError("Need at least two Cα atoms.")
    pairs = np.asarray(list(combinations(ca_indices.tolist(), 2)), dtype=np.int32)
    return md.compute_distances(trajectory, pairs)


def load_trajectory(path, topology=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdb" and topology is None:
        return md.load(str(path))
    if topology is None:
        raise ValueError(f"{path} needs a topology PDB.")
    return md.load(str(path), top=str(topology))


def iter_systems(input_path, topology=None):
    """Yield (name, trajectory) for a file or a directory of systems."""
    input_path = Path(input_path)
    if input_path.is_file():
        yield input_path.stem, load_trajectory(input_path, topology)
        return

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    for pdb_path in sorted(input_path.glob("*.pdb")):
        yield pdb_path.stem, md.load(str(pdb_path))

    for subdirectory in sorted(path for path in input_path.iterdir() if path.is_dir()):
        top_path = subdirectory / "topology.pdb"
        traj_path = next((subdirectory / name for name in TRAJ_NAMES if (subdirectory / name).is_file()), None)
        if top_path.is_file() and traj_path is not None:
            yield subdirectory.name, md.load(str(traj_path), top=str(top_path))


def fit_system(name, trajectory, output_dir, lag_time, dim):
    if trajectory.n_frames <= lag_time:
        raise ValueError(
            f"{name} has {trajectory.n_frames} frames, which is not enough for lag {lag_time}."
        )
    features = ca_distances(trajectory)
    estimator = TICA(lagtime=lag_time, dim=dim)
    estimator.fit(features)
    model = estimator.fetch_model() if hasattr(estimator, "fetch_model") else estimator
    projected = np.asarray(model.transform(features))
    output_path = Path(output_dir) / f"{name}_tica.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        projected_data=projected,
        tica_model=np.asarray(pickle.dumps(model)),
        lag_time=np.int32(lag_time),
        dim=np.int32(dim),
        n_frames=np.int32(projected.shape[0]),
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fit TICA on all Cα–Cα distances.")
    parser.add_argument(
        "input_path",
        help="Reference PDB, trajectory, or a directory of reference systems.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for {name}_tica.npz. Defaults to the input directory, "
        "or the PDB's directory for a single file.",
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="Topology PDB. Required when input_path is an xtc or dcd file.",
    )
    parser.add_argument("--lag-time", type=int, default=10, help="TICA lag in frames (default: 10).")
    parser.add_argument("--dim", type=int, default=5, help="Number of TICA dimensions (default: 5).")
    args = parser.parse_args()

    if args.lag_time < 1 or args.dim < 2:
        print("lag-time must be >= 1 and dim must be >= 2.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input_path)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif input_path.is_dir():
        output_dir = input_path
    else:
        output_dir = input_path.parent

    written = []
    try:
        systems = list(iter_systems(input_path, args.topology))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not systems:
        print(f"No reference trajectories found in {input_path}", file=sys.stderr)
        sys.exit(1)

    for name, trajectory in systems:
        try:
            output_path = fit_system(name, trajectory, output_dir, args.lag_time, args.dim)
        except Exception as exc:
            print(f"Skipping {name}: {exc}", file=sys.stderr)
            continue
        written.append(output_path)
        print(f"{name}: {trajectory.n_frames} frames -> {output_path}")

    if not written:
        sys.exit(1)
    print(f"Fit {len(written)} TICA model(s) in {output_dir}")


if __name__ == "__main__":
    main()
