import os
import argparse
import gzip
import pickle

import numpy as np
import torch
import esm
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.half().to(device)


def parse_fasta(filename):
    with open(filename, 'r') as f:
        contents = f.read().split('>')[1:]
        data = []
        for entry in contents:
            lines = entry.split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])
            sequence = sequence.replace("*", "") if "*" in sequence else sequence
            data.append((header, sequence))
    return data


def calculate_representation(model, alphabet, data, device, output_dir, BATCH_SIZE=4):
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
            # sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).cpu())
            # save_representation(token_representations[i, 1: tokens_len - 1].cpu().numpy(), os.path.join(output_dir, f'{batch_labels[i]}.pkl.gz'), zipped=True)
            save_representation(token_representations[i, 1: tokens_len - 1].cpu(),
                                os.path.join(output_dir, f'{batch_labels[i]}.pt'), zipped=False)
            # sequence_strs.append(batch_strs[i])

        del batch_labels, batch_strs, batch_tokens, token_representations
        torch.cuda.empty_cache()


def save_representation(sequence_representations, output_file, zipped=False):
    if zipped:
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(sequence_representations, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        torch.save(sequence_representations, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', required=True, type=str, help='input file')
    parser.add_argument('--output_dir', '-o', required=True, type=str, help='output directory')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    args = parser.parse_args()

    data = parse_fasta(args.input_file)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    data = [seq for seq in data if len(seq[1]) <= 1500]

    os.makedirs(args.output_dir, exist_ok=True)
    calculate_representation(model, alphabet, data, device, args.output_dir, BATCH_SIZE=args.batch_size)


