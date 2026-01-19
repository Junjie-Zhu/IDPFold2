import os
import argparse
import multiprocessing as mp

import mdtraj as md
from tqdm import tqdm

# command_template = ('convert_cg2all '
#                     '-p processed/10036_1_1_1/topology.pdb '
#                     '-d processed/10036_1_1_1/traj.dcd '
#                     '-o ./test.dcd -opdb ./test.pdb '
#                     '--cg CalphaBasedModel '
#                     '--batch 1 --proc 1')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', type=str, required=True)
parser.add_argument('--output_dir', '-o', type=str, required=True)
parser.add_argument('--model', '-m', type=str, default='CalphaBasedModel')
parser.add_argument('--num_proc', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=500)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
model = args.model
num_proc = args.num_proc
batch_size = args.batch_size

assert os.path.exists(input_dir), f'Input directory {input_dir} does not exist.'
os.makedirs(output_dir, exist_ok=True)

def process_fn(system):
    command = f'convert_cg2all ' \
              f'-p {output_dir}/{system}/topology.pdb ' \
              f'-d {output_dir}/{system}/traj.dcd ' \
              f'-o {output_dir}/{system}/aa_traj.dcd ' \
              f'-opdb {output_dir}/{system}/aa_topology.pdb ' \
              f'--cg {model} ' \
              f'--batch {batch_size} --proc {num_proc}'
    # os.makedirs(f'{output_dir}/{system}', exist_ok=True)
    os.system(command)


def traj_fn(system_name):
    traj = md.load(f'{input_dir}/{system_name}.pdb')

    # save to dcd and topology
    save_dir = f'{output_dir}/{system_name}'
    os.makedirs(save_dir, exist_ok=True)
    traj.save_dcd(f'{save_dir}/traj.dcd')
    traj[0].save_pdb(f'{save_dir}/topology.pdb')

    
if __name__ == '__main__':
    # first convert pdbs into dcd
    existing_systems = [f.replace('.pdb', '') for f in os.listdir(input_dir) if f.endswith('.pdb')]
    if os.cpu_count() == 1:
        for system in existing_systems:
            traj_fn(system)
    else:
        with mp.Pool(processes=os.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(traj_fn, existing_systems), total=len(existing_systems)))

    system_names = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    for system in system_names:
        process_fn(system)
            
            
            
