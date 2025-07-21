import torch
from faesm.esm import FAEsmForMaskedLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval().to(torch.float16)


def process_one_seq(seq):
    inputs = model.tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs['last_hidden_state']


seqs = []
with open('./pkl/all_seqs.fasta', 'r') as f:
    lines = f.readlines()
    for l in lines:
        if l.startswith('>'):
            seq_name = l.strip()[1:]  # Remove '>'
        else:
            seq = l.strip()
            seqs.append((seq_name, seq))

for (seq_name, seq) in seqs:
    embedding = process_one_seq(seq)
    torch.save(embedding, f'./embeddings/{seq_name}.pt')
