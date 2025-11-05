import os
import multiprocessing as mp
from functools import partial

import pandas as pd
import torch
# import esm
import mdtraj as md
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 512

# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model = model.to(DEVICE)


def main():
    input_path = './sims_cytosol'

    embedding_path = './embedding'
    structure_path = './pdb'
    seq_path = './'
    os.makedirs(embedding_path, exist_ok=True)
    os.makedirs(structure_path, exist_ok=True)

    # gather all directories
    all_data = [i for i in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, i))]

    get_structure_data_afc(all_data, input_path, structure_path, debug=False)

    # get_structure_data(data, structure_path)
    # seq_data = get_seq_data(data)
    # save_seq_to_fasta(seq_data, seq_path)

    # calculate_representation(model, alphabet, seq_data, DEVICE, output_path)


def get_structure_data(df, structure_path, debug=False):
    process_structure = partial(process_single_structure_idrome, structure_path_=structure_path)

    seq_name = df['seq_name'].tolist()
    if os.cpu_count() == 1 or debug:
        for seq in tqdm(seq_name):
            process_structure(seq)
    else:
        print(f'Using {mp.cpu_count()} CPUs to process structures...')
        with mp.Pool(mp.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_structure, seq_name), total=len(seq_name)))


def get_structure_data_afc(all_data, input_path, output_path, debug=False):
    process_structure = partial(process_single_traj_afc, input_path=input_path, output_path=output_path)

    if os.cpu_count() == 1 or debug:
        system_length = []
        for name in tqdm(all_data):
            system_length.append(process_structure(name))
    else:
        print(f'Using {mp.cpu_count()} CPUs to process structures...')
        with mp.Pool(os.cpu_count()) as pool:
            system_length = list(tqdm(pool.imap_unordered(process_structure, all_data), total=len(all_data)))

    print(f'Minium system length: {min(system_length)}')
    print(f'Maxium system length: {max(system_length)}')


def process_single_traj_afc(name, input_path, output_path):
    traj = md.load(
        os.path.join(input_path, name, f'{name}.dcd'),
        top=os.path.join(input_path, name, f'top.pdb')
    )

    # get 50 frames evenly distributed
    indices = list(range(0, traj.n_frames, traj.n_frames // 50))[:50]
    for i in indices:
        traj[i].save_pdb(os.path.join(output_path, f'{name}_f{i}.pdb'))
    return traj.n_residues


def read_csv(file_path):
    data = pd.read_csv(file_path)
    data = data[data['N'] < MAX_SEQ_LENGTH]
    first_column_name = data.columns[0]
    data.rename(columns={first_column_name: 'seq_name'}, inplace=True)
    return data


def get_seq_data(df):
    seq_data = []
    for idx, row in df.iterrows():
        seq_id = row['seq_name']
        sequence = row['fasta']
        seq_data.append((seq_id, sequence))
    return seq_data


def save_seq_to_fasta(seq_data, output_path):
    with open(os.path.join(output_path, 'seqs.fasta'), 'w') as f:
        for seq_id, sequence in seq_data:
            f.write(f'>{seq_id}\n{sequence}\n')


def process_single_structure_idrome(seq_, structure_path_):
    seq = seq_.split('_')
    seq_location = f'{seq[1]}_{seq[2]}'
    seq_dir ='/'.join([seq[0][i*2:i*2+2] for i in range(3)])
    if len(seq[0]) > 6:
        seq_dir = os.path.join('IDRome_v4', seq_dir, seq[0][6:], seq_location)
    else:
        seq_dir = os.path.join('IDRome_v4', seq_dir, seq_location)
    traj = md.load(os.path.join(seq_dir, 'traj.xtc'), top=os.path.join(seq_dir, 'top.pdb'))

    # get 20 frames evenly distributed
    indices = list(range(0, traj.n_frames, traj.n_frames // 20))[:20]
    for i in indices:
        traj[i].save_pdb(os.path.join(structure_path_, f'{seq_}_f{i}.pdb'))


def calculate_representation(model, alphabet, data, device, output_dir):
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    total_sequences, num_batches = len(data), len(data) // BATCH_SIZE + (len(data) % BATCH_SIZE != 0)

    for batch in tqdm(range(num_batches)):
        start_idx, end_idx = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
        batch_labels, batch_strs, batch_tokens = batch_converter(data[start_idx:end_idx])
        batch_tokens, batch_lens = batch_tokens.to(device), (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            token_representations = model(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33]

        for i, tokens_len in enumerate(batch_lens):
            torch.save(token_representations[i, 1: tokens_len - 1].cpu(), os.path.join(output_dir, f"{batch_labels[i]}.pt"))

        del batch_labels, batch_strs, batch_tokens, token_representations
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()



