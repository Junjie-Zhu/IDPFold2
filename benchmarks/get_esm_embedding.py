import os

import pandas as pd
import torch
import esm
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 1000

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(DEVICE)


def main():
    csv_path = './all_benchmark.csv'
    output_path = './embedding'
    os.makedirs(output_path, exist_ok=True)

    data = read_csv(csv_path)
    seq_data = get_seq_data(data)
    calculate_representation(model, alphabet, seq_data, DEVICE, output_path)


def read_csv(file_path):
    data = pd.read_csv(file_path)
    data = data[data['length'] < MAX_SEQ_LENGTH]
    return data


def get_seq_data(df):
    seq_data = []
    for idx, row in df.iterrows():
        seq_id = row['test_case']
        sequence = row['sequence']
        seq_data.append((seq_id, sequence))
    return seq_data


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



