import os
import pickle
import multiprocessing as mp

import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
from biotite.structure.io import dcd
import tqdm

max_thread = True
num_workers = 1
test_name = 'wo_top'
input_dir = ('/root/autodl-tmp/test_dataset/'
             '/_2024_Cao_CALVADOSCOM_Zenodo/data/IDPs_MDPsCOM_2.2_0.08_2_validate/')
predict_dir = '/root/autodl-tmp/ai2pse/logs/eval/runs/2024-12-30_19-52-18/samples'
dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

if max_thread:
    num_workers = mp.cpu_count()


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


def dihedrals(structures):
    dihedrals = []
    for model in structures:
        ph, ps, omg = struc.dihedral_backbone(model)
        dihedrals.append(np.stack([ph[1:-1], ps[1:-1]]))
    return np.stack(dihedrals)


def process_fn(protein_dir):
    """Process a single protein directory and calculate structural properties."""
    top_path = os.path.join(input_dir, f"{protein_dir}/3/{protein_dir}_first.pdb")
    traj_path = os.path.join(input_dir, f"{protein_dir}/3/{protein_dir}.dcd")
    predict_path = os.path.join(predict_dir, f"{protein_dir}.pdb")

    # Load structures
    trajectory = dcd.DCDFile.read(traj_path).get_structure(strucio.load_structure(top_path))
    predict = strucio.load_structure(predict_path)

    # Calculate structural properties
    return {
        'name': protein_dir,
        'ca_dist_traj': ca_dist(trajectory),
        'ca_dist_predict': ca_dist(predict),
        'rg_traj': rg(trajectory),
        'rg_predict': rg(predict),
        're2e_traj': re2e(trajectory),
        're2e_predict': re2e(predict),
        # 'min_rmsd': min_rmsd(trajectory, predict),
        'dihedrals': dihedrals(predict)
    }


def process_all(dirs, num_workers=1):
    """Process all protein directories with optional multiprocessing."""
    if num_workers == 1:
        results = [process_fn(d) for d in tqdm.tqdm(dirs)]
    else:
        with mp.Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap(process_fn, dirs), total=len(dirs)))
    return results


def consolidate_results(results):
    """Consolidate a list of results into a dictionary of lists."""
    keys = results[0].keys() if results else []
    return {key: [r[key] for r in results] for key in keys}


all_result = process_all(dirs, num_workers=num_workers)
all_result = consolidate_results(all_result)
with open(f'results_{test_name}.pkl', 'wb') as f:
    pickle.dump(all_result, f)
