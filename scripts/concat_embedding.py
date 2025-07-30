import glob
import os
import shutil
import multiprocessing as mp

from tqdm import tqdm
import pandas as pd
import torch

metadata = pd.read_csv('metadata.csv')
accession_code = metadata['accession_code']
embedding_path = './embedding'
processed_path = './processed'
os.makedirs(processed_path, exist_ok=True)


def process_fn(code):
    embeddings = glob.glob(os.path.join(embedding_path, f'{code}_*.pt'))

    if len(embeddings) == 1:
        shutil.copy(embeddings[0], os.path.join(processed_path, f'{code}.pt'))
    else:
        # sort by number in filename
        embeddings.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        embeddings_list = [torch.load(i) for i in embeddings]
        torch.save(torch.concat(embeddings_list, dim=0),
                   os.path.join(processed_path, f'{code}.pt'))


if os.cpu_count() > 1:
    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_fn, accession_code), total=len(accession_code)))
else:
    for code in tqdm(accession_code):
        process_fn(code)
