import os
import sys
import pickle
import multiprocessing as mp
from functools import partial
import warnings

import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
from biotite.structure.io import dcd
import tqdm

warnings.filterwarnings('ignore', category=UserWarning)


def main():
    output_dir = sys.argv[1]
    pred_dir = os.path.join(output_dir, 'samples')

    system_names = [f.replace('.pdb', '') for f in os.listdir(pred_dir) if f.endswith('.pdb')]

    _process_fn = partial(process_fn, pred_dir=pred_dir)
    if os.cpu_count() > 1:
        with mp.Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(_process_fn, system_names), total=len(system_names)))
    else:
        results = []
        for system in tqdm.tqdm(system_names):
            results.append(_process_fn(system))

    results = consolidate_results(results)
    with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(results, f)


def process_fn(system, pred_dir):
    """Process a single protein directory and calculate structural properties."""
    pred_path = os.path.join(pred_dir, f'{system}.pdb')

    # Load structures
    predict = strucio.load_structure(pred_path)

    # Calculate structural properties
    ca_dist_predict, clashes = ca_dist(predict)
    rg_predict = rg(predict)
    re2e_predict = re2e(predict)

    if np.sum(clashes) > 0:
        print(f'Warning: {np.sum(clashes)} clashed models found in {system}')

    # filter clashed models
    ca_dist_predict = ca_dist_predict[~clashes]
    rg_predict = rg_predict[~clashes]
    re2e_predict = re2e_predict[~clashes]

    return {
        'name': system,
        'ca_dist_predict': ca_dist_predict,
        'rg_predict': rg_predict,
        're2e_predict': re2e_predict,
    }


def ca_dist(structures):
    dist = []
    clash = []
    for model in structures:
        model = model[model.atom_name == 'CA']
        coords = model.coord

        # calculate neighbor distances
        coords_diff = coords[1:, :] - coords[:-1, :]
        dists = np.linalg.norm(coords_diff, axis=1)
        dist.append(dists)
        clash.append(any(dists > 12.0))
    return np.stack(dist), np.array(clash)


def rg(structures):
    return struc.gyration_radius(structures)


def re2e(structures):
    re2e = []
    for model in structures:
        model = model[model.atom_name == 'CA']
        coords = model.coord

        coords_diff = coords[0, :] - coords[-1, :]
        re2e.append(np.linalg.norm(coords_diff))

    return np.array(re2e)


def min_rmsd(traj, structures):
    min_rmsd = np.inf
    for model in structures:
        model = model[model.atom_name == 'CA']

        # superimpose traj to model
        superposed_traj, _ = struc.superimpose(model, traj)
        min_rmsd = min(min_rmsd, np.min(struc.rmsd(model, superposed_traj)))
    return min_rmsd


def consolidate_results(results):
    """Consolidate a list of results into a dictionary of lists."""
    keys = results[0].keys() if results else []
    return {key: [r[key] for r in results] for key in keys}


if __name__ == '__main__':
    main()



