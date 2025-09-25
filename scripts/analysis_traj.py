import os
import sys
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
from biotite.structure.io import dcd
import tqdm


def main():
    output_dir = sys.argv[1]
    pred_dir = os.path.join(output_dir, 'samples')
    traj_dir = '/lustre/home/acct-clschf/clschf/jjzhu/datasets/_2024_Cao_CALVADOSCOM_Zenodo/data/IDPs_MDPsCOM_2.2_0.08_2_validate'

    system_names = [f.replace('.pdb', '') for f in os.listdir(pred_dir) if f.endswith('.pdb')]

    _process_fn = partial(process_fn, pred_dir=pred_dir)
    _traj_process_fn = partial(traj_process_fn, traj_dir=traj_dir)
    if os.cpu_count() > 1:
        with mp.Pool(os.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(_traj_process_fn, system_names), total=len(system_names)))
    else:
        results = []
        for system in tqdm.tqdm(system_names):
            results.append(_traj_process_fn(system))

    results = consolidate_results(results)
    with open(os.path.join(output_dir, 'metrics_traj.pkl'), 'wb') as f:
        pickle.dump(results, f)


def process_fn(system, pred_dir):
    """Process a single protein directory and calculate structural properties."""
    pred_path = os.path.join(pred_dir, f'{system}.pdb')

    # Load structures
    predict = strucio.load_structure(pred_path)

    # Calculate structural properties
    return {
        'name': system,
        'ca_dist_predict': ca_dist(predict),
        'rg_predict': rg(predict),
        're2e_predict': re2e(predict),
    }


def traj_process_fn(system, traj_dir):
    top_path = os.path.join(traj_dir, f'{system}', '3', f'{system}_first.pdb')
    traj_path = os.path.join(traj_dir, f'{system}', '3', f'{system}.dcd')

    # Load trajectory
    traj = dcd.DCDFile.read(traj_path).get_structure(strucio.load_structure(top_path))

    # Calculate structural properties
    return {
        'name': system,
        'ca_dist_traj': ca_dist(traj),
        'rg_traj': rg(traj),
        're2e_traj': re2e(traj),
    }


def ca_dist(structures):
    dist = []
    for model in structures:
        model = model[model.atom_name == 'CA']
        coords = model.coord

        # calculate neighbor distances
        coords_diff = coords[1:, :] - coords[:-1, :]
        dist.append(np.linalg.norm(coords_diff, axis=1))

    return np.stack(dist)


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



