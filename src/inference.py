import logging
import os
import warnings
from typing import List, Optional, Union

import rootutils
import datetime

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.model.integral import generating_predict
from src.model.protein_transformer import ProteinTransformerAF3
from src.model.flow_matching.r3flow import R3NFlowMatcher
from src.model.components.motif_factory import SingleMotifFactory
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.pdb_utils import to_pdb_simple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


class GenerationDataset(Dataset):
    def __init__(
            self,
            dt: float = 0.005,
            nsamples: Union[int, List[int]] = 10,
            plm_emb_dir: Optional[str] = None,
            nres: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dt = dt
        self.use_plm = plm_emb_dir is not None

        if self.use_plm:
            self.data_paths = sorted(
                [os.path.join(plm_emb_dir, f) for f in os.listdir(plm_emb_dir) if f.endswith('.pt')])
            self.nres = [None] * len(self.data_paths)  # Placeholder
        elif nres is not None:
            self.data_paths = [None] * len(nres)  # Placeholder
            self.nres = [int(n) for n in nres]
        else:
            raise ValueError("One of 'plm_emb_dir' or 'nres' must be provided.")

        # Consolidate nsamples handling
        if isinstance(nsamples, int):
            self.nsamples = [nsamples] * len(self.data_paths)
        elif isinstance(nsamples, list):
            if len(nsamples) != len(self.data_paths):
                raise ValueError("Length of 'nsamples' list must match number of data points.")
            self.nsamples = nsamples
        else:
            raise TypeError(f"Unsupported type for 'nsamples': {type(nsamples)}. Expected int or list.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        data = {
            "dt": self.dt,
            "nsamples": self.nsamples[idx],
        }

        if self.use_plm:
            plm_emb = torch.load(self.data_paths[idx])
            data["nres"] = plm_emb.shape[0]
            data["plm_emb"] = plm_emb
        else:
            data["nres"] = self.nres[idx]

        return data


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(args: DictConfig):
    logging_dir = os.path.join(args.logging_dir, f"INF_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if DIST_WRAPPER.rank == 0:
        # update logging directory with current time
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        os.makedirs(logging_dir)
        os.makedirs(os.path.join(logging_dir, "samples"))  # for saving checkpoints

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
        logging.info(
            f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if DIST_WRAPPER.world_size > 1:
        logging.info(
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
        dt=args.dt,
        nsamples=args.nsamples,
        plm_emb_dir=args.ckpt_dir,
        nres=args.nres,
    )
    inference_loader = DataLoader(dataset, batch_size=1)

    # instantiate model
    model = ProteinTransformerAF3(**args.model).to(device)
    flow_matching = R3NFlowMatcher(zero_com=not args.motif_conditioning, scale_ref=1.0)
    motif_factory = SingleMotifFactory(motif_prob=0 if not args.motif_conditioning else args.motif_prob)

    assert os.path.isfile(args.ckpt_dir), f"Checkpoint file not found: {args.ckpt_dir}"
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    if DIST_WRAPPER.world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if DIST_WRAPPER.rank == 0:
        logging.info(f"Loaded checkpoint from {args.ckpt_dir}")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")

    # sanity check
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for inference_iter, inference_dict in tqdm(enumerate(inference_loader)):
            torch.cuda.empty_cache()
            inference_dict = to_device(inference_dict, device)

            pred_structure = generating_predict(
                batch=inference_dict,
                flow_matching=flow_matching,
                model=model,
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

            to_pdb_simple(
                atom_positions=pred_structure * 10,  # nm to Angstrom
                output_dir=os.path.join(logging_dir, "samples"),
            )

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

