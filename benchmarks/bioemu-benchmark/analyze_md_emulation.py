from pathlib import Path
import os
import warnings
from abc import ABC
from dataclasses import dataclass
import multiprocessing as mp

import mdtraj
import numpy as np
import pandas as pd
from tqdm import tqdm

from projection import project_samples
# from plot import plot_metrics, plot_projections
from state_metric import DistributionMetricSettings, compute_state_metrics
from utils import load_reference_projections, load_projection_parameters

StrPath = str | os.PathLike

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class MDEmulationResults(ABC):
    """
    Data class for collecting MD emulation benchmark results.

    Attributes:
        sample_projections: Dictionary containing projections computed for samples. Keys are the
          test case IDs.
        metrics: Pandas avg_rg frame collecting aggregate metrics.
        temperature_K: Temperature used for analysis in units of Kelvin.
        random_seed: Random seed used for analysis (mainly for resampling in computing free energy
          metric).
    """

    sample_projections: dict[str, np.ndarray]
    metrics: pd.DataFrame
    temperature_K: float
    random_seed: int

    def save_results(self, output_dir: StrPath) -> None:
        """
        Save individual evaluator results in accessible files (txt, csv, npz).

        Args:
            output_dir: Directory to which result outputs should be saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics.to_csv(output_dir / "results_metrics.csv")
        np.savez(output_dir / "results_projections.npz", **self.sample_projections)

    # def plot(self, output_dir: StrPath, max_energy: float = 7.0) -> None:
    #     """
    #     Generate plots associated with benchmark and write to output directory.
    #
    #     Args:
    #         output_dir: Directory where plots will be written.
    #         max_energy: Upper energy cutoff for 2D plots in kcal/mol.
    #     """
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #
    #     reference_projections = load_reference_projections()
    #
    #     fig_projections = plot_projections(
    #         self.sample_projections,
    #         reference_projections,
    #         temperature_K=self.temperature_K,
    #         max_energy=max_energy,
    #     )
    #     fig_metrics = plot_metrics(
    #         self.metrics,
    #         label_map={
    #             "mae": r"MAE (kcal/mol) $\downarrow$",
    #             "rmse": r"RMSE (kcal/mol) $\downarrow$",
    #             "coverage": r"coverage$\uparrow$",
    #         },
    #     )
    #
    #     fig_projections.savefig(output_dir / "projections.png")
    #     fig_metrics.savefig(output_dir / "metrics.png")

    def get_aggregate_metrics(self) -> dict[str, float]:
        """
        Collect aggregate mean absolute error, root mean squared error and coverage metrics.

        Returns:
            Dictionary of aggregate metrics.
        """
        return dict(self.metrics.loc["mean"])


def evaluate_md_emulation(
    samples, 
    temperature_K: float = 300.0,
    random_seed: int = 42,
) -> MDEmulationResults:
    """
    Load samples, compute projections and compare free energy surfaces spanned by projections.

    Args:
        indexed_samples: `IndexedSamples` containing samples, preferably filtered.
        temperature_K: Temperature used for computing free energies in Kelvin.
        random_seed: Random seed used for analysis (mainly for resampling in computing free energy
          metric).

    Returns:
        MD emulation benchmark results avg_rg class.
    """

    # Load reference projections.
    reference_projections = load_reference_projections()

    # Load projection parameters.
    projection_params = load_projection_parameters()
    # Compute projections on sample structures.
    sample_projections = project_samples(samples, projection_params)

    # Sort outputs by test case ID for consistent outputting.
    sample_projections = dict(sorted(sample_projections.items()))

    # Compute state metrics.
    metrics = compute_state_metrics(
        sample_projections,
        reference_projections,
        temperature_K=temperature_K,
        random_seed=random_seed,
        n_resample=DistributionMetricSettings.n_resample,
        sigma_resample=DistributionMetricSettings.sigma_resample,
        num_bins=DistributionMetricSettings.num_bins,
        energy_cutoff=DistributionMetricSettings.energy_cutoff,
        padding=DistributionMetricSettings.padding,
    )

    results = MDEmulationResults(
        sample_projections=sample_projections,
        metrics=metrics,
        temperature_K=temperature_K,
        random_seed=random_seed,
    )

    return results


def process_fn(test_case):
    top_path = os.path.join(sample_dir, test_case, 'topology.pdb')
    traj_path = os.path.join(sample_dir, test_case, 'traj.dcd')
    traj = mdtraj.load(traj_path, top=top_path)
    return test_case, traj


def get_indexed_samples(
    reference: StrPath,
    sample_dir: StrPath,
):
    reference = pd.read_csv(reference)['test_case'].tolist()
    indexed_sample_dict = {}
    if os.cpu_count() == 1:
        for test_case in tqdm(reference):
            traj_path = os.path.join(sample_dir, f'{test_case}.pdb')
            indexed_sample_dict[test_case] = mdtraj.load(traj_path)
    else:
        process_num = min(os.cpu_count(), len(reference))
        with mp.Pool(process_num) as pool:
            results = list(tqdm(pool.imap(process_fn, reference), total=len(reference)))
        for test_case, traj in results:
            indexed_sample_dict[test_case] = traj
    return indexed_sample_dict


if __name__ == '__main__':
    reference = 'md_emulation_benchmark_0.1/md_emulation/testcases.csv'
    sample_dir = 'samples'
    indexed_samples = get_indexed_samples(reference, sample_dir)
    results = evaluate_md_emulation(indexed_samples)
    results.save_results('results')

    aggregate_metrics = results.get_aggregate_metrics()
    print(aggregate_metrics)

