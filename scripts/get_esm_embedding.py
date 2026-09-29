import os
import argparse
import shutil

import torch
import esm
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 1000

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(DEVICE)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument(
        '--only_f0',
        action='store_true',
        help='Training FASTA only. IDRome names with a trailing _fN share one embedding, '
             'saved as prefix_f0.pt. Other frames of that system are skipped.',
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    records = read_fasta(args.fasta_path)
    if args.only_f0:
        records = collapse_idrome_frames(records)
    seq_data = filter_and_deduplicate(records)
    calculate_representation(model, alphabet, seq_data, DEVICE, args.output_path)


def read_fasta(file_path):
    records = []
    header = None
    chunks = []

    with open(file_path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.replace(' ', ''))

    if header is not None:
        records.append((header, ''.join(chunks)))
    if not records:
        raise ValueError(f'No FASTA records found in {file_path}')
    return records


def _idrome_frame(token):
    """Map an IDRome multi-frame name to its shared ``prefix_f0`` embedding.

    This follows ``PDBDataset.get_embedding_name``: more than two underscore
    fields, and the last field ``fN``, load ``prefix_f0.pt``.
    """
    parts = token.split('_')
    if len(parts) <= 2:
        return None
    frame = parts[-1]
    if len(frame) < 2 or frame[0] != 'f' or not frame[1:].isdigit():
        return None
    return f"{'_'.join(parts[:-1])}_f0", int(frame[1:])


def collapse_idrome_frames(records):
    """Keep one ``prefix_f0`` record per IDRome system and drop the other frames."""
    chosen = {}
    warned = set()
    n_frames = 0

    for header, sequence in records:
        parsed = _idrome_frame(_header_names(header)[0])
        if parsed is None:
            continue
        f0_name, frame = parsed
        n_frames += 1
        current = chosen.get(f0_name)
        if current is None:
            chosen[f0_name] = (frame, sequence)
            continue
        current_frame, current_sequence = current
        if frame == 0 and current_frame != 0:
            if current_sequence != sequence:
                print(f'Warning: {f0_name} frames differ; keeping f0')
                warned.add(f0_name)
            chosen[f0_name] = (0, sequence)
        elif current_sequence != sequence and f0_name not in warned:
            print(f'Warning: {f0_name} frames differ; keeping the first')
            warned.add(f0_name)

    collapsed = []
    emitted = set()
    for header, sequence in records:
        parsed = _idrome_frame(_header_names(header)[0])
        if parsed is None:
            collapsed.append((header, sequence))
            continue
        f0_name = parsed[0]
        if f0_name in emitted:
            continue
        emitted.add(f0_name)
        collapsed.append((f0_name, chosen[f0_name][1]))

    n_skipped = n_frames - len(chosen)
    print(f'IDRome systems: {len(chosen)} kept as _f0, {n_skipped} other _fN records ignored')
    return collapsed


def _header_names(header):
    parts = header.split()
    if not parts:
        raise ValueError('FASTA record has an empty name')
    token = parts[0]
    full = '_'.join(parts).replace('/', '_').replace('\\', '_')
    return token, full


def filter_and_deduplicate(records):
    """Drop long sequences and group identical sequences under their FASTA names.

    Each unique sequence is extracted once. Every other record with that sequence
    still receives its own embedding file by copying the extracted tensor.
    """
    grouped = {}
    order = []
    token_to_sequence = {}
    seen_headers = set()
    used_files = set()
    skipped_length = 0
    skipped_empty = 0
    skipped_conflict = 0

    for header, sequence in records:
        token, full = _header_names(header)
        if not sequence:
            skipped_empty += 1
            continue
        if len(sequence) >= MAX_SEQ_LENGTH:
            skipped_length += 1
            continue
        if header in seen_headers:
            if token_to_sequence.get(token) not in (None, sequence):
                print(f'Warning: {token} appears with different sequences; keeping the first')
                skipped_conflict += 1
            continue

        previous = token_to_sequence.get(token)
        if previous is None:
            filename = token
        elif previous == sequence:
            # Same sequence already extracted under this id. Keep a separate
            # file when the header still distinguishes the system.
            if full in used_files:
                continue
            filename = full
        else:
            print(f'Warning: {token} appears with different sequences; keeping the first')
            skipped_conflict += 1
            continue

        seen_headers.add(header)
        used_files.add(filename)
        token_to_sequence.setdefault(token, sequence)
        if sequence not in grouped:
            grouped[sequence] = []
            order.append(sequence)
        grouped[sequence].append(filename)

    seq_data = [(grouped[sequence], sequence) for sequence in order]
    n_files = sum(len(names) for names, _ in seq_data)
    n_copies = n_files - len(seq_data)
    print(
        f'{len(records)} FASTA records, '
        f'{skipped_length} skipped by length (>= {MAX_SEQ_LENGTH}), '
        f'{skipped_empty} empty, {skipped_conflict} conflicting names, '
        f'{len(seq_data)} unique sequences -> {n_files} embedding files '
        f'({n_copies} copied)'
    )
    return seq_data


def embedding_path(output_dir, name):
    safe_name = name.replace('/', '_').replace('\\', '_')
    return os.path.join(output_dir, f'{safe_name}.pt')


def calculate_representation(model, alphabet, data, device, output_dir):
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    total_sequences = len(data)
    num_batches = total_sequences // BATCH_SIZE + (total_sequences % BATCH_SIZE != 0)

    for batch in tqdm(range(num_batches)):
        start_idx, end_idx = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
        batch_items = data[start_idx:end_idx]
        converter_input = [(names[0], sequence) for names, sequence in batch_items]
        _, _, batch_tokens = batch_converter(converter_input)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            token_representations = model(batch_tokens, repr_layers=[33], return_contacts=True)['representations'][33]

        for i, tokens_len in enumerate(batch_lens):
            embedding = token_representations[i, 1: tokens_len - 1].cpu()
            names = batch_items[i][0]
            primary_path = embedding_path(output_dir, names[0])
            torch.save(embedding, primary_path)
            for name in names[1:]:
                dest_path = embedding_path(output_dir, name)
                if os.path.abspath(dest_path) != os.path.abspath(primary_path):
                    shutil.copyfile(primary_path, dest_path)

        del batch_tokens, token_representations
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
