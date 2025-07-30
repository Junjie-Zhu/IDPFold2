import logging
import os
import warnings
from random import random

import rootutils
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import BioTrainingDataset
from src.data.dataloader import get_training_dataloader, get_inference_dataloader
from src.data.transform import BioFeatureTransform
from src.model.residue_generate_network import ResGenNet
from src.model.diffusion_sampler import InferenceNoiseScheduler, sample_diffusion
from src.model.loss import AllLosses
from src.model.optimizer import get_optimizer, get_lr_scheduler
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.pdb_utils import to_pdb_simple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


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
    dataset = BioTrainingDataset(
        path_to_dataset=args.data.path_to_dataset,
        transform=BioFeatureTransform(
            recenter_atoms=args.data.recenter_atoms,
            eps=args.data.eps,
        ),
    )
    inference_loader = get_inference_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        distributed=DIST_WRAPPER.world_size > 1,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )

    # instantiate model
    model = ResGenNet(
        s_input=args.model.s_input,
        s_trunk=args.model.s_trunk,
        z_trunk=args.model.z_trunk,
        s_atom=args.model.s_atom,
        z_atom=args.model.z_atom,
        s_noise=args.model.s_noise,
        z_template=args.model.z_template,
        n_layers=args.model.n_layers,
        n_attn_heads=args.model.n_attn_heads,
        sigma_data=args.sigma_data,
    ).to(device)

    assert os.path.isfile(args.ckpt_dir), f"Checkpoint file not found: {args.ckpt_dir}"
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    if DIST_WRAPPER.world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if DIST_WRAPPER.rank == 0:
        logging.info(f"Loaded checkpoint from {args.ckpt_dir}")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters")

    noise_schedule = InferenceNoiseScheduler(
        s_max=args.noise.s_max,
        s_min=args.noise.s_min,
        rho=args.noise.rho,
        sigma_data=args.sigma_data,
    )

    # sanity check
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for inference_iter, inference_dict in tqdm(enumerate(inference_loader)):
            torch.cuda.empty_cache()
            inference_dict = to_device(inference_dict, device)
            s_inputs = inference_dict['plm_embedding']

            batch_num = args.n_samples // args.n_samples_per_batch
            x_denoised = []
            for i in range(batch_num):
                x_denoised.append(sample_diffusion(
                    denoise_net=model,
                    s_inputs=s_inputs,
                    input_feature_dict=inference_dict,
                    noise_schedule=noise_schedule(N_step=200, device=s_inputs.device, dtype=s_inputs.dtype),
                    N_sample=args.n_samples_per_batch
                ))
            x_denoised = torch.cat(x_denoised, dim=1)

            to_pdb_simple(
                inference_dict,
                x_denoised,
                os.path.join(logging_dir, "samples"),
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
        raise Exception(f"type {type(obj)} not supported")
    return obj


if __name__ == '__main__':
    main()

