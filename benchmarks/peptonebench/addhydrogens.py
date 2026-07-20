# Copyright 2025 Peptone Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os
import multiprocessing as mp

import mdtraj as md
import numpy as np
import pandas as pd
from tqdm import tqdm
from pdbfixer import PDBFixer
from openmm.app import PDBFile


def process_single_frame(frame_info, pH, output_path, tmp_path):
    """
    Processes a single frame: adds hydrogens and saves directly to disk.
    """
    frame_idx, traj_subset = frame_info
    f_in = os.path.join(tmp_path, f"frame{frame_idx + 1}.pdb")
    traj_subset.save_pdb(f_in)

    fixer = PDBFixer(filename=f_in)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    frame_filename = os.path.join(output_path, f"frame{frame_idx + 1}.pdb")
    with open(frame_filename, 'w') as out_pdb:
        PDBFile.writeFile(fixer.topology, fixer.positions, out_pdb)
        out_pdb.write("END\n")

# === CONFIG ===
# MAKE SURE TO REPLACE THESE PATHS WITH YOUR PATHS 
import argparse

argparse.add_argument('--input-root', '-i', type=str, required=True)
argparse.add_argument('--output-root', '-o', type=str, required=True)
argparse.add_argument('--exp-root', '-e', type=str, required=True)
args = argparse.parse_args()
input_root = args.input_root
output_root = args.output_root
exp_path = args.exp_path

os.makedirs(output_root, exist_ok=True)

# === Load protein list ===
proteins = [f for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))]

# === Load average pH values ===
pH_dict = {prot: np.mean(pd.read_csv(os.path.join(exp_path, prot, 'info.csv'))['pH']) for prot in proteins}

NUM_CORES = max(os.cpu_count() // 4, 1)
# === Iterate over proteins ===
for protein in proteins:
    input_model = os.path.join(input_root, protein, "topology.pdb")
    traj_path = input_model.replace('topology.pdb', 'traj_no_clash.xtc')
    output_path = os.path.join(output_root, protein)
    tmp_path = os.path.join(output_path, 'tmp')

    if os.path.exists(os.path.join(output_path, "frame1.pdb")):
        print(f"Skipping {protein}: frames already exist.")
        continue

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)

    # Load trajectory
    if not os.path.exists(traj_path):
        continue
    traj = md.load(traj_path, top=input_model)
    pH = pH_dict[protein]

    print(f"Processing {protein} ({traj.n_frames} frames) using {NUM_CORES} cores...")

    # --- MULTIPROCESSING EXECUTION ---
    worker_func = functools.partial(process_single_frame, pH=pH, output_path=output_path, tmp_path=tmp_path)
    if NUM_CORES == 1:
        for i, frame in enumerate(tqdm(traj)):
            worker_func((i, frame))
    else:
        with mp.Pool(NUM_CORES) as executor:
            # We pass individual frames (slices) to the workers
            list(tqdm(executor.imap_unordered(worker_func, enumerate(traj)), total=traj.n_frames))

    print(f"Finished: {protein}")

