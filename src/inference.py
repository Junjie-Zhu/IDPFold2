import math
import os
import warnings
from typing import List, Optional, Union
import rootutils
import datetime


import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.model.integral import generating_predict
from src.model.protein_transformer import ProteinTransformerAF3
from src.model.flow_matching.r3flow import R3NFlowMatcher
from src.model.components.motif_factory import SingleMotifFactory
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.pdb_utils import to_pdb_simple, to_pdb
from src.utils.cluster_utils import log_info
from src.common.residue_constants import restypes

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


class GenerationDataset(Dataset):
    def __init__(
            self,
            csv_path: str,
            plm_emb_dir: str,
            dt: float = 0.005,
            nsamples: Union[int, List[int]] = 10,
            load_multimer: bool = False,
    ):
        super().__init__()
        df = pd.read_csv(csv_path)

        if not os.path.isdir(plm_emb_dir):
            self.get_esm_embedding(df, plm_emb_dir, load_multimer)

        self.dt = dt
        self.seqs = []
        self.data_paths = []
        if not load_multimer:
            self.seqs = df['sequence'].tolist()
            self.seqs = [get_resid(seq) for seq in self.seqs]
            self.data_paths = [os.path.join(plm_emb_dir, f"{name}.pt") for name in df['test_case'].tolist()]

            # sort by sequence length
            self.seqs, self.data_paths = zip(*sorted(zip(self.seqs, self.data_paths), key=lambda x: x[0].shape[0]))
            self.seqs = list(self.seqs)
            self.data_paths = list(self.data_paths)
        else:
            self.seqs = [
                torch.cat([get_resid(seq) for seq in df.iloc[i]['sequence'].split(':')], dim=0)
                for i in range(len(df))
            ]
            self.data_paths = [[os.path.join(plm_emb_dir, f"{name}.pt")
                                for name in df.iloc[i]['test_case'].split(':')]
                               for i in range(len(df))]

            # sort by sequence length
            self.seqs, self.data_paths = zip(*sorted(zip(self.seqs, self.data_paths), key=lambda x: x[0].shape[0]))
            self.seqs = list(self.seqs)
            self.data_paths = list(self.data_paths)
        self.nsamples = [nsamples] * len(self.data_paths)
        self.load_multimer = load_multimer

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> dict:
        data = {
            "dt": self.dt,
            "nsamples": self.nsamples[idx],
        }

        if not self.load_multimer:
            plm_emb = torch.load(self.data_paths[idx])
            assert plm_emb.shape[0] == len(self.seqs[idx]), f"Sequence length mismatch for {self.data_paths[idx]}"
            data["nres"] = plm_emb.shape[0]
            data["plm_emb"] = plm_emb
            data["name"] = os.path.basename(self.data_paths[idx]).replace('.pt', '')
            data["residue_type"] = self.seqs[idx]
            return data

        log_info(f"Loading multimer PLM embeddings for {self.data_paths[idx]}")
        plm_embs = [torch.load(path) for path in self.data_paths[idx]]
        chains = torch.cat(
            [torch.ones(plm_emb.shape[0], dtype=torch.long) + i for i, plm_emb in enumerate(plm_embs)],
        dim=0)
        residue_idx = torch.cat(
            [torch.arange(plm_emb.shape[0], dtype=torch.long) for plm_emb in plm_embs],
        dim=0)
        plm_embs = torch.cat(plm_embs, dim=0)
        assert plm_embs.shape[0] == len(self.seqs[idx]), f"Sequence length mismatch for {self.data_paths[idx]}"
        data["nres"] = plm_embs.shape[0]
        data["plm_emb"] = plm_embs
        data["name"] = ':'.join([os.path.basename(path).replace('.pt', '') for path in self.data_paths[idx]])
        data["residue_type"] = self.seqs[idx]
        data["residue_idx"] = residue_idx
        data["chains"] = chains
        return data

    @staticmethod
    def get_esm_embedding(df, plm_emb_dir, load_multimer=False):
        log_info(f"PLM embedding directory not found: {plm_emb_dir}, creating embeddings")
        os.makedirs(plm_emb_dir, exist_ok=True)

        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if torch.cuda.device_count() > 0:
            device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
        else:
            device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        BATCH_SIZE = 1

        if not load_multimer:
            seq_data = [(row['test_case'], row['sequence']) for idx, row in df.iterrows()]
        else:
            seq_data = []
            for i, row in df.iterrows():
                names = row['test_case'].split(':')
                sequences = row['sequence'].split(':')
                seq_data.extend([(names[j], sequences[j]) for j in range(len(names))])

        batch_converter = alphabet.get_batch_converter()
        total_sequences, num_batches = len(seq_data), len(seq_data) // BATCH_SIZE + (len(seq_data) % BATCH_SIZE != 0)
        for batch in tqdm(range(num_batches)):
            start_idx, end_idx = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
            batch_labels, batch_strs, batch_tokens = batch_converter(seq_data[start_idx:end_idx])
            batch_tokens, batch_lens = batch_tokens.to(device), (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                token_representations = model(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33]

            for i, tokens_len in enumerate(batch_lens):
                torch.save(token_representations[i, 1: tokens_len - 1].cpu(), os.path.join(plm_emb_dir, f"{batch_labels[i]}.pt"))

            del batch_labels, batch_strs, batch_tokens, token_representations
            torch.cuda.empty_cache()
        log_info(f"Finished creating PLM embeddings in {plm_emb_dir}, total {total_sequences} sequences")


def get_resid(seq: str):
    res_id = torch.tensor(
        [restypes.index(res) for res in seq],
        dtype=torch.long,
    )
    return res_id


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(args: DictConfig):
    logging_dir = os.path.join(args.logging_dir, f"{args.prefix}_INF_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if DIST_WRAPPER.rank == 0:
        # update logging directory with current time
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        os.makedirs(logging_dir)
        os.makedirs(os.path.join(logging_dir, "samples"))  # for saving samples
        os.makedirs(os.path.join(logging_dir, "tmp"))  # for saving raw samples from each rank or batch

        # save current configuration in logging directory
        with open(f"{logging_dir}/config.yaml", "w") as f:
            OmegaConf.save(args, f)

    # check environment
    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        log_info(
            f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if DIST_WRAPPER.world_size > 1:
        log_info(
            f"Using DDP with {DIST_WRAPPER.world_size} processes, rank: {DIST_WRAPPER.rank}"
        )
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    # All ddp process got the same seed
    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    # instantiate dataset
    dataset = GenerationDataset(
        csv_path=args.csv_path,
        plm_emb_dir=args.plm_emb_dir,
        dt=args.dt,
        nsamples=args.nsamples,
        load_multimer=args.load_multimer,
    )
    inference_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
    )  # use plain dataloader, handling DDP inside training loop

    # instantiate model
    model = ProteinTransformerAF3(**args.model).to(device)
    flow_matching = R3NFlowMatcher(zero_com=not args.motif_conditioning, scale_ref=1.0)
    motif_factory = SingleMotifFactory(motif_prob=0 if not args.motif_conditioning else args.motif_prob)

    assert os.path.isfile(args.ckpt_dir), f"Checkpoint file not found: {args.ckpt_dir}"
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    if DIST_WRAPPER.world_size > 1:
        model = DDP(
            model,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if DIST_WRAPPER.rank == 0:
        log_info(f"Loaded checkpoint from {args.ckpt_dir}")
        log_info(f"Model has {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")

    if args.autoguidance_ratio > 0.0 and args.ag_dir is not None:
        model_ag = ProteinTransformerAF3(**args.model).to(device)
        checkpoint_ag = torch.load(args.ag_dir, map_location=device)
        if DIST_WRAPPER.world_size > 1:
            model_ag.module.load_state_dict(checkpoint_ag['model_state_dict'])
        else:
            model_ag.load_state_dict(checkpoint_ag['model_state_dict'])

    torch.cuda.empty_cache()
    model.eval()
    with torch.inference_mode():
        loader_iter = tqdm(
            enumerate(inference_loader),
            disable=(DIST_WRAPPER.rank != 0)) if DIST_WRAPPER.rank == 0 else enumerate(inference_loader)
        for inference_iter, inference_dict in loader_iter:
            torch.cuda.empty_cache()
            inference_dict = to_device(inference_dict, device)

            # assign samples to each rank
            nsamples_per_rank = inference_dict['nsamples'] // DIST_WRAPPER.world_size
            if DIST_WRAPPER.rank < inference_dict['nsamples'] % DIST_WRAPPER.world_size:
                nsamples_per_rank += 1

            # split samples based on memory limit
            nsamples_per_batch = max(1, args.max_batch_length // inference_dict['nres'][0])

            nsamples_generated = 0
            batch_idx = 0
            show_inner_bar = nsamples_per_rank > nsamples_per_batch
            if show_inner_bar:
                total_batches = math.ceil(nsamples_per_rank / nsamples_per_batch)
                pbar_inner = tqdm(total=total_batches,
                                  desc=f"Rank {DIST_WRAPPER.rank}",
                                  position=DIST_WRAPPER.rank,
                                  leave=False)

            while nsamples_generated < nsamples_per_rank:
                current_batch_size = min(nsamples_per_batch, nsamples_per_rank - nsamples_generated)
                inference_dict["nsamples"] = current_batch_size
                pred_structure = generating_predict(
                    batch=inference_dict,
                    flow_matching=flow_matching,
                    model=model,
                    model_ag=model_ag if args.autoguidance_ratio > 0.0 and args.ag_dir is not None else None,
                    motif_factory=motif_factory if args.motif_conditioning else None,
                    target_pred=args.target_pred,
                    guidance_weight=args.guidance_weight,
                    autoguidance_ratio=args.autoguidance_ratio,
                    schedule_args=args.schedule,
                    sampling_args=args.sampling,
                    motif_conditioning=args.motif_conditioning,
                    self_conditioning=args.self_conditioning,
                    device=device,
                )
                pred_structure = pred_structure.detach().cpu()
                if 'chains' not in inference_dict.keys():
                    to_pdb_simple(
                        atom_positions=pred_structure * 10,
                        residue_ids=inference_dict['residue_type'].squeeze(),
                        output_dur=os.path.join(logging_dir, "tmp"),
                        accession_code=f"{inference_dict['name'][0]}_rank_{DIST_WRAPPER.rank}_batch_{batch_idx}",
                    )
                else:
                    to_pdb(
                        atom_positions=pred_structure * 10,
                        residue_ids=inference_dict["residue_type"].squeeze(),
                        chain_ids=inference_dict["chains"].squeeze(),
                        output_dir=os.path.join(logging_dir, "tmp"),
                        accession_code=f"{inference_dict['name'][0]}_rank_{DIST_WRAPPER.rank}_batch_{batch_idx}",
                    )

                nsamples_generated += current_batch_size
                batch_idx += 1
                if show_inner_bar:
                    pbar_inner.update(1)
            if show_inner_bar:
                pbar_inner.close()

            # gather all pdb files to one
            if DIST_WRAPPER.rank == 0:
                log_info(f"Gathering samples for {inference_dict['name'][0]}")
                tmp_files = [i for i in os.listdir(os.path.join(logging_dir, "tmp"))
                             if i.startswith(inference_dict['name'][0])]
                with open(os.path.join(logging_dir, "samples", f"{inference_dict['name'][0]}.pdb"), 'w') as outfile:
                    model_idx = 1
                    for f in tmp_files:
                        with open(os.path.join(logging_dir, "tmp", f), 'r') as infile:
                            for line in infile:
                                if line.startswith("MODEL"):
                                    outfile.write(f"MODEL {model_idx}\n")  # reindex model number
                                    model_idx += 1
                                else:
                                    outfile.write(line)
                        # remove tmp files
                        os.remove(os.path.join(logging_dir, "tmp", f))

    # Clean up process group when finished
    if DIST_WRAPPER.world_size > 1:
        dist.destroy_process_group()


def to_device(obj, device):
    """Move tensor or dict of tensors to device"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        try:
            obj = obj.to(device)
        except:
            raise Exception(f"type {type(obj)} not supported")
    return obj


if __name__ == '__main__':
    main()

