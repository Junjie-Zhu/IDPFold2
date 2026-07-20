import os
import shutil
import MDAnalysis
import numpy as np
from DEERPREdict.PRE import PREpredict
import pandas as pd
import pickle as pkl
import gzip
from concurrent.futures import ProcessPoolExecutor
import functools
from tqdm import tqdm


# --- HELPER FUNCTIONS ---
def find_dat_filenames(path_to_dir, suffix=".dat"):
    return [f for f in os.listdir(path_to_dir) if f.endswith(suffix)]


def find_pdb_filenames(path_to_dir, suffix=".pdb"):
    return [f for f in os.listdir(path_to_dir) if f.endswith(suffix)]


# --- WORKER FUNCTION ---
def process_single_frame_pre(frame_idx, protein_path, site, temp, libname):
    """
    Calculates PRE for a single frame.
    Returns data arrays to be aggregated by the main process.
    """
    # Create a unique temp directory for this specific frame/site combination
    # to avoid race conditions with DEER-PREdict's file writing
    worker_tmp = os.path.join(protein_path, f"tmp_site_{site}_frame_{frame_idx}")
    os.makedirs(worker_tmp, exist_ok=True)

    log_path = os.path.join(worker_tmp, "log")
    output_prefix = os.path.join(worker_tmp, "PRE")
    pdb_file = os.path.join(protein_path, f"frame{frame_idx}.pdb")

    # 1. Run PRE Calculation
    u = MDAnalysis.Universe(pdb_file)
    PRE = PREpredict(u, residue=site, libname=libname,
                     tau_t=0.5e-9, log_file=log_path,
                     temperature=temp, z_cutoff=0.05,
                     atom_selection='H', Cbeta=False)

    PRE.run(output_prefix=output_prefix, tau_t=0.5e-9, delay=10e-3,
            tau_c=5e-09, r_2=10, wh=750)

    # 2. Extract data from the generated pickle
    # DEER-PREdict appends -{site}.pkl to the prefix
    pkl_path = f"{output_prefix}-{site}.pkl"
    dat_path = f"{output_prefix}-{site}.dat"

    with gzip.open(pkl_path, "rb") as f:
        data = pkl.load(f)
        tmpr3 = np.array(data['r3'])
        tmpr6 = np.array(data['r6'])
        tmpangular = np.array(data['angular'])

    # Get residue mask (only needed for the first frame usually,
    # but returning it ensures consistency)
    d = np.loadtxt(dat_path)
    residues = d[:, 0].astype(int)
    vals = d[:, 1].astype(float)
    mask_nan = ~np.isnan(vals)

    shutil.rmtree(worker_tmp, ignore_errors=True)

    return {
        'r3': tmpr3, 'r6': tmpr6, 'angular': tmpangular,
        'residues': residues[mask_nan], 'success': True
    }


# === CONFIG ===
import argparse

argparse.add_argument('--input-root', '-i', type=str, required=True)
argparse.add_argument('--exp-root', '-e', type=str, required=True)
argparse.add_argument('--libname', type=str, default='MTSSL MMMx')
argparse.add_argument('--num-cores', type=int, default=os.cpu_count())

args = argparse.parse_args()
input_root = args.input_root
exp_root = args.exp_root
libname = args.libname
NUM_CORES = args.num_cores

# === Load protein list and info (Same as your original logic) ===
proteins = [f for f in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, f))]
PREproteins_info = {}

for protein in proteins:
    datfiles = find_dat_filenames(os.path.join(exp_root, protein))
    if any('PRE' in f for f in datfiles):
        info = pd.read_csv(os.path.join(exp_root, protein, 'info.csv'))
        sites, temps = [], []
        for file in datfiles:
            if 'PRE' in file:
                res = int(file.split('.')[0].split('-')[-1])
                dataset = file[:-4]
                temp = np.mean(info[info['Experiment'] == dataset]['Temp(K)'])
                sites.append(res);
                temps.append(temp)
        PREproteins_info[protein] = {'Sites': sites, 'Temperatures': temps}

# === MAIN LOOP ===
for protein, info in PREproteins_info.items():
    protein_path = os.path.join(input_root, protein)

    for site, temp in zip(info['Sites'], info['Temperatures']):
        final_output = os.path.join(protein_path, f"PREdata-{site}.npy")

        if os.path.exists(final_output):
            print(f"Skipping {protein} Site {site} (Already exists)")
            continue

        nframes = len(find_pdb_filenames(protein_path))
        print(f"Parallelizing {protein} Site {site}: {nframes} frames on {NUM_CORES} cores")

        # Prepare worker
        worker_func = functools.partial(process_single_frame_pre,
                                        protein_path=protein_path,
                                        site=site, temp=temp,
                                        libname=libname)

        # Run Multiprocessing
        if NUM_CORES == 1:
            results = []
            for frame_idx in tqdm(range(1, nframes + 1), desc=f"Site {site}"):
                result = worker_func(frame_idx)
                results.append(result)
        else:
            results = []
            with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
                # We use range(1, nframes + 1) to match your frame naming
                results = list(tqdm(executor.map(worker_func, range(1, nframes + 1)),
                                    total=nframes, desc=f"Site {site}"))

        # Filter failed tasks and aggregate
        valid_results = [r for r in results if r['success']]
        if not valid_results:
            continue

        # Concatenate results
        resdict = {
            'Residue': valid_results[0]['residues'],
            'r3': np.concatenate([r['r3'] for r in valid_results], axis=0),
            'r6': np.concatenate([r['r6'] for r in valid_results], axis=0),
            'angular': np.concatenate([r['angular'] for r in valid_results], axis=0)
        }

        np.save(final_output, resdict)
        print(f"Saved: {final_output}")

print('All done!')
