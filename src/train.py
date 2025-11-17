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

from src.data.dataset import PDBDataModule, PDBDataSelector, PDBDataSplitter
from src.data.transforms import GlobalRotationTransform, ChainBreakPerResidueTransform
from src.model.integral import training_predict, generating_predict
from src.model.protein_transformer import ProteinTransformerAF3
from src.model.ema import EMAWrapper
from src.model.flow_matching.r3flow import R3NFlowMatcher
from src.model.components.motif_factory import SingleMotifFactory
from src.model.optimizer import get_optimizer, get_lr_scheduler
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.cluster_utils import log_info
from src.utils.pdb_utils import to_pdb_simple

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(args: DictConfig):
    logging_dir = os.path.join(args.logging_dir, f"{args.task_prefix}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if DIST_WRAPPER.rank == 0:
        # update logging directory with current time
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        os.makedirs(logging_dir)
        os.makedirs(os.path.join(logging_dir, "checkpoints"))  # for saving checkpoints
        os.makedirs(os.path.join(logging_dir, "samples"))  # for saving samples

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
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if DIST_WRAPPER.world_size > 1:
        if DIST_WRAPPER.rank == 0:
            log_info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            log_info(
                f"Using DDP with {DIST_WRAPPER.world_size} processes, rank: {DIST_WRAPPER.rank}"
            )
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    # else:
    #     log_info(
    #         f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
    #     )

    # All ddp process got the same seed
    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    # dataselector = PDBDataSelector(
    #     data_dir=args.data.data_dir,
    #     fraction=args.data.fraction,
    #     molecule_type=args.data.molecule_type,
    #     experiment_types=args.data.experiment_types,
    #     min_length=args.data.min_length,
    #     max_length=args.data.max_length,
    #     oligomeric_min=args.data.oligomeric_min,
    #     oligomeric_max=args.data.oligomeric_max,
    #     best_resolution=args.data.best_resolution,
    #     worst_resolution= args.data.worst_resolution,
    #     has_ligands=[],
    #     remove_ligands=[],
    #     remove_non_standard_residues=True,
    #     remove_pdb_unavailable=True,
    #     exclude_ids=[]
    # ) if args.data.molecule_type is not None else None

    # instantiate dataset
    data_module = PDBDataModule(
        data_dir=args.data.data_dir,
        dataselector=None,
        datasplitter=PDBDataSplitter(
            data_dir=args.data.data_dir,
            train_val_test=args.data.train_val_test,
            split_type=args.data.split_type,
            split_sequence_similarity=args.data.split_sequence_similarity,
            overwrite_sequence_clusters=False
        ),
        format=args.data.format,
        overwrite=args.data.overwrite,
        batch_padding=args.data.batch_padding,
        sampling_mode=args.data.sampling_mode,
        transforms=[GlobalRotationTransform(), ChainBreakPerResidueTransform()],
        plm_embedding=args.data.plm_emb_dir,
        batch_size=args.batch_size,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
        complex_dir=args.data.complex_dir,
        complex_prop=args.data.complex_prop,
        crop_size=args.data.crop_size,
    )
    # data_module.prepare_data()
    data_module.setup()
    train_loader, val_loader = data_module.get_train_dataloader()

    # instantiate model
    model = ProteinTransformerAF3(**args.model).to(device)
    flow_matching = R3NFlowMatcher(zero_com=not args.motif_conditioning, scale_ref=1.0)
    motif_factory = SingleMotifFactory(motif_prob=0 if not args.motif_conditioning else args.motif_prob)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if DIST_WRAPPER.world_size > 1:
        if DIST_WRAPPER.rank == 0:
            log_info(model)
            log_info(f"Model has {nparam / 1000000:.2f}M parameters")
            log_info("Using DDP")
        model = DDP(
            model,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )
    else:
        log_info(model)
        log_info(f"Model has {nparam / 1000000:.2f}M parameters")

    if args.ema.decay > 0:
        ema_wrapper = EMAWrapper(
            model=model,
            decay=args.ema.decay,
            mutable_param_keywords=args.ema.mutable_param_keywords,
        )
        ema_wrapper.register()
    else:
        ema_wrapper = None

    torch.cuda.empty_cache()
    optimizer = get_optimizer(
        model,
        lr=args.optimizer.lr,
        weight_decay=args.optimizer.weight_decay,
        betas=(args.optimizer.beta1, args.optimizer.beta2),
        use_adamw=args.optimizer.use_adamw
    )
    scheduler = get_lr_scheduler(
        optimizer,
        lr_scheduler=args.optimizer.lr_scheduler,  # by default, use af3 scheduler
        lr=args.optimizer.lr,
        max_steps=args.epochs * len(train_loader) + 100,
        warmup_steps=args.optimizer.warmup_steps,
        decay_every_n_steps=args.optimizer.decay_every_n_steps,
        decay_factor=args.optimizer.decay_factor,
    )

    start_epoch = 1
    if args.resume.ema_dir is not None and args.ema.decay > 0:
        ema_checkpoint = torch.load(args.resume.ema_dir, map_location=device)
        if DIST_WRAPPER.world_size > 1:
            model.module.load_state_dict(ema_checkpoint['model_state_dict'])
        else:
            model.load_state_dict(ema_checkpoint['model_state_dict'])
        ema_wrapper.register()

        # clear checkpoint variables
        del ema_checkpoint

    if args.resume.ckpt_dir is not None:
        checkpoint = torch.load(args.resume.ckpt_dir, map_location=device)
        if DIST_WRAPPER.world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        if not args.resume.load_model_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        del checkpoint

    # sanity check
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for check_iter, check_dict in enumerate(val_loader):
            torch.cuda.empty_cache()
            check_dict = to_device(check_dict, device)

            noise_kwargs = {**args.noise}
            loss, loss_dict = training_predict(
                batch=check_dict,
                flow_matching=flow_matching,
                model=model,
                motif_factory=motif_factory,
                moe_factory=None,
                noise_kwargs=noise_kwargs,
                target_pred=args.target_pred,
                motif_conditioning=args.motif_conditioning,
                moe_conditioning=args.moe_conditioning,
                self_conditioning=args.self_conditioning,
                moe_loss_weight=args.loss.moe_loss_weight,
            )
            if check_iter >= 2:
                break
    log_info(f"Sanity check done")

    if DIST_WRAPPER.rank == 0:
        with open(f"{logging_dir}/loss.csv", "w") as f:
            f.write("Epoch,Loss,Val Loss\n")

    epoch_progress = tqdm(
        total=args.epochs,
        leave=False,
        position=0,
        ncols=100,
    ) if DIST_WRAPPER.rank == 0 else None
    # Main train/eval loop
    for crt_epoch in range(start_epoch, args.epochs + 1):
        epoch_loss, epoch_val_loss = 0, 0
        model.train()

        # Training loop with dynamic progress bar
        train_iter = enumerate(train_loader)
        if DIST_WRAPPER.rank == 0:
            train_iter = tqdm(
                train_iter,
                desc="Step",
                total=len(train_loader),
                leave=True,
                position=1,
                ncols=100,
            )
        crt_step, crt_val_step = 0, 0
        for crt_step, train_dict in train_iter:
            torch.cuda.empty_cache()
            train_dict = to_device(train_dict, device)

            if ema_wrapper is not None:
                ema_wrapper.update()

            noise_kwargs = {**args.noise}
            loss, loss_dict = training_predict(
                batch=train_dict,
                flow_matching=flow_matching,
                model=model,
                motif_factory=motif_factory,
                moe_factory=None,
                noise_kwargs=noise_kwargs,
                target_pred=args.target_pred,
                motif_conditioning=args.motif_conditioning,
                moe_conditioning=args.moe_conditioning,
                self_conditioning=args.self_conditioning,
                moe_loss_weight=args.loss.moe_loss_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_loss = loss.item()
            epoch_loss += step_loss

            # Update the progress bar dynamically
            if DIST_WRAPPER.rank == 0:
                train_iter.set_postfix(step_loss=f"{step_loss:.3f}", **loss_dict)

        # Calculate average epoch loss
        epoch_loss /= (crt_step + 1)

        # Validation loop with dynamic progress bar
        model.eval()
        with torch.no_grad():
            val_iter = enumerate(val_loader)
            if DIST_WRAPPER.rank == 0:
                val_iter = tqdm(
                    val_iter,
                    desc="Validation",
                    total=len(val_loader),
                    leave=True,
                    position=1,
                    ncols=100,
                )

            if ema_wrapper is not None:
                ema_wrapper.apply_shadow()

            for crt_val_step, val_dict in val_iter:
                torch.cuda.empty_cache()
                val_dict = to_device(val_dict, device)

                noise_kwargs = {**args.noise}
                val_loss, val_loss_dict = training_predict(
                    batch=val_dict,
                    flow_matching=flow_matching,
                    model=model,
                    motif_factory=motif_factory,
                    moe_factory=None,
                    noise_kwargs=noise_kwargs,
                    target_pred=args.target_pred,
                    motif_conditioning=args.motif_conditioning,
                    moe_conditioning=args.moe_conditioning,
                    self_conditioning=args.self_conditioning,
                    moe_loss_weight=args.loss.moe_loss_weight,
                    force_moe_capacity=False,  # do not limit capacity during validation
                )

                step_val_loss = val_loss.item()
                epoch_val_loss += step_val_loss

                # Update the validation progress bar dynamically
                if DIST_WRAPPER.rank == 0:
                    val_iter.set_postfix(val_loss=f"{step_val_loss:.3f}", **val_loss_dict)

            if ema_wrapper is not None:
                ema_wrapper.restore()
        # Calculate average validation loss
        epoch_val_loss /= (crt_val_step + 1)

        if DIST_WRAPPER.rank == 0 and epoch_progress is not None:
            epoch_progress.set_postfix(loss=f"{epoch_loss:.3f}", val_loss=f"{epoch_val_loss:.3f}")
            epoch_progress.update()

            # Append loss data to file
            with open(f"{logging_dir}/loss.csv", "a") as f:
                f.write(f"{crt_epoch},{epoch_loss},{epoch_val_loss}\n")

            # Save checkpoint only on master process
            if crt_epoch % args.checkpoint_interval == 0 or crt_epoch == args.epochs:
                checkpoint_path = os.path.join(logging_dir, f"checkpoints/epoch_{crt_epoch}.pth")
                torch.save({
                    'epoch': crt_epoch,
                    'model_state_dict': model.module.state_dict() if DIST_WRAPPER.world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                if ema_wrapper is not None:
                    ema_wrapper.apply_shadow()
                    ema_path = os.path.join(logging_dir, f"checkpoints/_ema_{ema_wrapper.decay}_{crt_epoch}.pth")
                    torch.save({
                        'model_state_dict': model.module.state_dict() if DIST_WRAPPER.world_size > 1 else model.state_dict(),
                    }, ema_path)

                    # test generating
                    inf_dict = {
                        "dt": torch.Tensor([0.005]).to(device),
                        "nsamples": 5,
                        "plm_emb": val_dict["plm_emb"][-1],
                        "nres": val_dict["plm_emb"].shape[1],
                        "residue_type": val_dict["residue_type"][-1],
                        "residue_idx": val_dict["residue_pdb_idx"][-1],
                        "chains": val_dict["chains"][-1],
                        "mask": val_dict["mask"][-1],
                    }
                    pred_structure = generating_predict(
                        batch=inf_dict,
                        flow_matching=flow_matching,
                        model=model,
                        model_ag=None,
                        motif_factory=None,
                        moe_factory=None,
                        target_pred=args.target_pred,
                        guidance_weight=1.0,
                        autoguidance_ratio=0.0,
                        schedule_args={
                            "schedule_mode": "log",
                            "schedule_p": 2.0,
                            },
                        sampling_args={
                            "sampling_mode": "vf",
                            "sc_scale_noise": 0.0,
                            "sc_scale_score": 0.0,
                            "gt_mode": "1/t",
                            "gt_p": 1.0,
                            "gt_clamp_val": None,
                            },
                        motif_conditioning=False,
                        moe_conditioning=False,
                        self_conditioning=False,
                        device=device,
                    )
                    # save pdb
                    to_pdb_simple(
                        atom_positions=pred_structure.squeeze() * 10,
                        residue_ids=inf_dict["residue_type"].squeeze(),
                        chain_ids=inf_dict["chains"].squeeze(),
                        output_dir=os.path.join(logging_dir, "samples"),
                        accession_code=f"val_{crt_epoch}",
                    )

                    ema_wrapper.restore()

        torch.cuda.empty_cache()

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

