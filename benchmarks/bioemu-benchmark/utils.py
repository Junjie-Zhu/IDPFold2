from pathlib import Path
import os

import numpy as np

from projection import ProjectionParameters

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
MD_EMULATION_ASSET_DIR = os.path.join(ROOT_DIR, 'md_emulation_benchmark_0.1')


def load_reference_projections() -> dict[str, np.ndarray]:
    """
    Load reference projections from file.

    Args:

    Returns:
        Dictionary with test case identifiers as keys and the target projections arrays as entries.
    """
    benchmark = 'md_emulation'
    projection_path = (
        Path(MD_EMULATION_ASSET_DIR) / benchmark / "reference_projections.npz"
    )
    projections = np.load(projection_path)
    return dict(projections)


def load_projection_parameters() -> dict[str, ProjectionParameters]:
    """
    Load parameters used for projecting samples.

    Args:
        benchmark: MD emulation benchmark.

    Returns:
        Dictionary of projection parameters using test case identifiers as keys.
    """
    benchmark = 'md_emulation'
    parameter_dir = Path(MD_EMULATION_ASSET_DIR) / benchmark
    projection_sqrt_inv_cov = dict(np.load(parameter_dir / "projections_sqrt_inv_cov.npz"))
    projection_mean = dict(np.load(parameter_dir / "projections_mean.npz"))

    assert set(projection_mean) == set(
        projection_sqrt_inv_cov
    ), "Mismatch in systems found for projection parameters."

    projection_params: dict[str, ProjectionParameters] = {}
    for test_case in projection_sqrt_inv_cov.keys():
        projection_params[test_case] = ProjectionParameters(
            sqrt_inv_cov=projection_sqrt_inv_cov[test_case],
            mean=projection_mean[test_case],
        )
    return projection_params