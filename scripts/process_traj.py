import multiprocessing as mp
import os

import mdtraj as md
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('uniprot_clean.csv')
system_names = df['uniprot'].tolist()

output_dir = "./MDP_trajectory/frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def process_fn(system_name):
    try:
        traj = md.load(f"./MDP_trajectory/dcd/{system_name}_aa.dcd", top=f"./MDP_trajectory/pdb/{system_name}_aa.pdb")
        for idx, frame in enumerate(traj[::20]):
            frame.save_pdb(f"{output_dir}/{system_name}_{idx}.pdb")
    except Exception as e:
        print(f"Error processing {system_name}: {e}")


cpu_count = os.cpu_count()

if cpu_count > 1:
    with mp.Pool(cpu_count) as pool:
        list(tqdm(pool.imap_unordered(process_fn, system_names), total=len(system_names)))
else:
    for system_name in tqdm(system_names):
        process_fn(system_name)
