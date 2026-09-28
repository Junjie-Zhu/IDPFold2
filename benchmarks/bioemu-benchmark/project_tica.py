"""Project ensembles with a fitted TICA model and score the free-energy surface.

``--model-dir`` contains ``{name}_tica.npz`` files from ``fit_tica.py``. For each
model the script looks up a sample with the same name, projects it, and writes:

- ``{name}_ref.npz``: reference ``projected_data`` stored in the fit
- ``{name}_pred.npz``: projection of ``--sample-dir``
- ``{name}_bioemu.npz``: projection of ``--bioemu-dir``, when that flag is set

Samples are either ``{sample_dir}/{name}.pdb`` or
``{sample_dir}/{name}/topology.pdb`` plus ``traj.xtc``, ``traj.dcd``,
``samples.xtc``, or ``traj_no_clash.xtc``. The Cα count and order must match the
reference used to fit the model.

``results_metrics.csv`` compares the sample projection with the reference
(columns ``test_case``, ``mae``, ``rmse``, ``coverage``, plus a ``mean`` row).
When ``--bioemu-dir`` is set, ``results_bioemu_metrics.csv`` is the same
comparison for those trajectories. The metric uses the first two TICA components.
"""

import argparse
import pickle
import sys
from pathlib import Path

import mdtraj as md
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from state_metric import DistributionMetricSettings, compute_state_metrics


TRAJ_NAMES = ("traj.xtc", "traj.dcd", "samples.xtc", "traj_no_clash.xtc")


def load_pickle(blob):
    try:
        return pickle.loads(blob)
    except ModuleNotFoundError as exc:
        if exc.name == "numpy._core":
            import numpy.core as numpy_core
            sys.modules.setdefault("numpy._core", numpy_core)
            return pickle.loads(blob)
        raise


def load_fit(path):
    data = np.load(path, allow_pickle=True)
    blob = data["tica_model"]
    if isinstance(blob, np.ndarray):
        blob = blob.item() if blob.shape == () else blob.tobytes()
    return load_pickle(blob), np.asarray(data["projected_data"])


def ca_distances(trajectory):
    from itertools import combinations

    ca_indices = trajectory.topology.select("name CA")
    if len(ca_indices) < 2:
        raise ValueError("Need at least two Cα atoms.")
    pairs = np.asarray(list(combinations(ca_indices.tolist(), 2)), dtype=np.int32)
    return md.compute_distances(trajectory, pairs)


def resolve_trajectory(root, name):
    root = Path(root)
    pdb_path = root / f"{name}.pdb"
    if pdb_path.is_file():
        return md.load(str(pdb_path))
    subdirectory = root / name
    top_path = subdirectory / "topology.pdb"
    traj_path = next(
        (subdirectory / traj_name for traj_name in TRAJ_NAMES if (subdirectory / traj_name).is_file()),
        None,
    )
    if top_path.is_file() and traj_path is not None:
        return md.load(str(traj_path), top=str(top_path))
    return None


def save_projection(path, projected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, projected_data=np.asarray(projected))


def project_trajectory(model, trajectory):
    return np.asarray(model.transform(ca_distances(trajectory)))


def score_projections(sample_projections, reference_projections, output_path):
    settings = DistributionMetricSettings()
    samples = {name: values[:, :2] for name, values in sample_projections.items()}
    references = {name: reference_projections[name][:, :2] for name in samples}
    metrics = compute_state_metrics(
        sample_projections=samples,
        reference_projections=references,
        temperature_K=300.0,
        random_seed=42,
        n_resample=settings.n_resample,
        sigma_resample=settings.sigma_resample,
        num_bins=settings.num_bins,
        energy_cutoff=settings.energy_cutoff,
        padding=settings.padding,
    )
    metrics.to_csv(output_path)
    mean_row = metrics.loc["mean"]
    print(
        f"Saved {output_path}. "
        f"mean MAE={mean_row['mae']:.4f}, mean coverage={mean_row['coverage']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Project samples with fitted TICA models.")
    parser.add_argument("--model-dir", required=True, help="Directory of {name}_tica.npz fits.")
    parser.add_argument("--sample-dir", required=True, help="Predicted ensembles, named like the fits.")
    parser.add_argument("--output-dir", required=True, help="Directory for projections and metrics.")
    parser.add_argument(
        "--bioemu-dir",
        default=None,
        help="Optional BioEmu ensembles, named like the fits.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_paths = sorted(model_dir.glob("*_tica.npz"))
    if not fit_paths:
        print(f"No *_tica.npz files in {model_dir}", file=sys.stderr)
        sys.exit(1)

    reference_projections = {}
    sample_projections = {}
    bioemu_projections = {}

    for fit_path in fit_paths:
        name = fit_path.name[: -len("_tica.npz")]
        try:
            model, reference = load_fit(fit_path)
        except Exception as exc:
            print(f"Skipping {name}: cannot load {fit_path}: {exc}", file=sys.stderr)
            continue

        save_projection(output_dir / f"{name}_ref.npz", reference)
        reference_projections[name] = reference

        sample = resolve_trajectory(args.sample_dir, name)
        if sample is None:
            print(f"Skipping {name}: no sample trajectory in {args.sample_dir}", file=sys.stderr)
            continue
        try:
            projected = project_trajectory(model, sample)
        except Exception as exc:
            print(f"Skipping {name}: {exc}", file=sys.stderr)
            continue
        save_projection(output_dir / f"{name}_pred.npz", projected)
        sample_projections[name] = projected
        print(f"{name}: projected {sample.n_frames} sample frames")

        if args.bioemu_dir is None:
            continue
        bioemu = resolve_trajectory(args.bioemu_dir, name)
        if bioemu is None:
            print(f"No BioEmu trajectory for {name} in {args.bioemu_dir}", file=sys.stderr)
            continue
        try:
            bioemu_projected = project_trajectory(model, bioemu)
        except Exception as exc:
            print(f"Skipping BioEmu {name}: {exc}", file=sys.stderr)
            continue
        save_projection(output_dir / f"{name}_bioemu.npz", bioemu_projected)
        bioemu_projections[name] = bioemu_projected

    if not sample_projections:
        print("No samples were projected.", file=sys.stderr)
        sys.exit(1)

    score_projections(
        sample_projections,
        reference_projections,
        output_dir / "results_metrics.csv",
    )
    if bioemu_projections:
        score_projections(
            bioemu_projections,
            reference_projections,
            output_dir / "results_bioemu_metrics.csv",
        )


if __name__ == "__main__":
    main()
