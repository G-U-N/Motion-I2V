import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from flowgen.data.dataset import WebVid10M_Multiple_Image
from flowgen.models.unet3d import UNet3DConditionModel

# from animation.pipelines.pipeline_animation import AnimationPipeline
from flowgen.pipelines.pipeline_flow_gen import FlowGenPipeline
from animation.utils.util import save_videos_grid, zero_rank_print
from flowgen.models.controlnet import ControlNetModel


from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import numpy as np
from PIL import Image


def generate_sparse_flow_mask(
    batch_size, channels, flow_height, flow_width, min_block_size=8, sparsity=0.05
):
    """
    Generate a random block-wise mask for a given optical flow tensor.

    Parameters:
    batch_size (int): Batch size of the tensor.
    channels (int): Number of channels in the tensor.
    flow_height (int): Height of the optical flow tensor.
    flow_width (int): Width of the optical flow tensor.
    min_block_size (int): Minimum size of the blocks in the mask.
    sparsity (float): Percentage of sparsity (between 0 and 1).

    Returns:
    torch.Tensor: A binary mask tensor with the same shape as the input optical flow tensor.
    """

    # Initialize a zeros mask
    mask = torch.zeros((batch_size, channels, flow_height, flow_width))

    # Calculate the number of blocks
    num_blocks_height = flow_height // min_block_size
    num_blocks_width = flow_width // min_block_size

    # Calculate the number of blocks to keep based on the sparsity
    total_blocks = num_blocks_height * num_blocks_width
    blocks_to_keep = int(total_blocks * sparsity)
    # print(blocks_to_keep)

    # Randomly select blocks to keep
    keep_blocks = np.random.choice(total_blocks, blocks_to_keep, replace=False)

    # Fill the selected blocks in the mask
    for block in keep_blocks:
        row = (block // num_blocks_width) * min_block_size
        col = (block % num_blocks_width) * min_block_size
        mask[:, :, row : row + min_block_size, col : col + min_block_size] = 1

    return mask


class ForwardWarp(nn.Module):
    """docstring for WarpLayer"""

    def __init__(
        self,
    ):
        super(ForwardWarp, self).__init__()

    def forward(self, img, flo):
        """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy

        """

        # (x1, y1)		(x1, y2)
        # +---------------+
        # |				  |
        # |	o(x, y) 	  |
        # |				  |
        # |				  |
        # |				  |
        # |				  |
        # +---------------+
        # (x2, y1)		(x2, y2)

        N, C, _, _ = img.size()

        # translate start-point optical flow to end-point optical flow
        y = flo[:, 0:1:, :]
        x = flo[:, 1:2, :, :]

        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)

        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get gaussian weights
        w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner
        img11, o11 = self.sample_one(img, x1, y1, w11)
        img12, o12 = self.sample_one(img, x1, y2, w12)
        img21, o21 = self.sample_one(img, x2, y1, w21)
        img22, o22 = self.sample_one(img, x2, y2, w22)

        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22

        return imgw, o

    def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
        w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
        w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
        w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

        return w11, w12, w21, w22

    def sample_one(self, img, shiftx, shifty, weight):
        """
        Input:
                -img (N, C, H, W)
                -shiftx, shifty (N, c, H, W)
        """

        N, C, H, W = img.size()

        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)
        flat_shifty = shifty.view(-1)
        flat_basex = (
            torch.arange(0, H, requires_grad=False)
            .view(-1, 1)[None, None]
            .cuda()
            .long()
            .repeat(N, C, 1, W)
            .view(-1)
        )
        flat_basey = (
            torch.arange(0, W, requires_grad=False)
            .view(1, -1)[None, None]
            .cuda()
            .long()
            .repeat(N, C, H, 1)
            .view(-1)
        )
        flat_weight = weight.view(-1)
        flat_img = img.view(-1)

        # The corresponding positions in I1
        idxn = (
            torch.arange(0, N, requires_grad=False)
            .view(N, 1, 1, 1)
            .long()
            .cuda()
            .repeat(1, C, H, W)
            .view(-1)
        )
        idxc = (
            torch.arange(0, C, requires_grad=False)
            .view(1, C, 1, 1)
            .long()
            .cuda()
            .repeat(N, 1, H, W)
            .view(-1)
        )
        # ttype = flat_basex.type()
        idxx = flat_shiftx.long() + flat_basex
        idxy = flat_shifty.long() + flat_basey

        # recording the inside part the shifted
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

        # Mask off points out of boundaries
        ids = idxn * C * H * W + idxc * H * W + idxx * W + idxy
        ids_mask = torch.masked_select(ids, mask).clone().cuda()

        # (zero part - gt) -> difference
        # difference back propagate -> No influence! Whether we do need mask? mask?
        # put (add) them together
        # Note here! accmulate fla must be true for proper bp
        img_warp = torch.zeros(
            [
                N * C * H * W,
            ]
        ).cuda()
        img_warp.put_(
            ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True
        )

        one_warp = torch.zeros(
            [
                N * C * H * W,
            ]
        ).cuda()
        one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

        return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


def init_dist(launcher="slurm", backend="nccl", port=29501, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        print("[Using port {}]".format(port))
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(
            f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}"
        )

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device)
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def main(
    image_finetune: bool,
    name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,
    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs=None,
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    trainable_modules: Tuple[str] = (None,),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    is_debug: bool = False,
    vae_pretrained_path: str = None,
    resumed_model_path: str = None,
    control_scale: float = 1.0,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = (
        "debug"
        if is_debug
        else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    )
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug):
        print("[Initiate tb at {}]".format(output_dir))
        tb_writer = SummaryWriter(output_dir)
        # run = wandb.init(project="animation", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    vae_img = AutoencoderKL.from_pretrained(
        "models/stage2/StableDiffusion", subfolder="vae"
    )
    import json

    with open("./models/stage1/StableDiffusion-FlowGen/vae/config.json", "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(vae_config)
    print("[Load vae weights from {}]".format(vae_pretrained_path))
    weight = torch.load(vae_pretrained_path, map_location="cpu")
    vae.load_state_dict(weight, strict=True)
    del weight

    if not image_finetune:
        unet = UNet3DConditionModel.from_config_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
        )

    else:
        print("[Not implemented error]")
        exit()
        unet = UNet2DConditionModel.from_config_2d(
            pretrained_model_path, subfolder="unet"
        )

    print("[Load ControlNet weights from pretrained 3D U-Net]")
    controlnet = ControlNetModel.from_unet(unet)
    unet.controlnet = controlnet
    unet.control_scale = 1.0

    print("[Load resumed model weights from {}]".format(resumed_model_path))
    weight = torch.load(resumed_model_path, map_location="cpu")
    weight = weight["state_dict"] if "state_dict" in weight else weight
    m, u = unet.load_state_dict(weight, strict=False)
    del weight

    print("[Constructing Dot Model]")
    from dot.dot.utils.options.demo_options import DemoOptions
    from dot.dot.models import create_model

    dot_args = {
        "model": "dot",
        "height": 320,
        "width": 320,
        "aspect_ratio": 1,
        "batch_size": 1,
        "num_tracks": 2048,
        "sim_tracks": 2048,
        "alpha_thresh": 0.8,
        "is_train": False,
        "worker_idx": 0,
        "num_workers": 2,
        "estimator_config": "dot/configs/raft_patch_8.json",
        "estimator_path": "dot/checkpoints/cvo_raft_patch_8.pth",
        "flow_mode": "direct",
        "refiner_config": "dot/configs/raft_patch_4_alpha.json",
        "refiner_path": "dot/checkpoints/movi_f_raft_patch_4_alpha.pth",
        "tracker_config": "dot/configs/cotracker_patch_4_wind_8.json",
        "tracker_path": "dot/checkpoints/movi_f_cotracker_patch_4_wind_8.pth",
        "sample_mode": "all",
        "interpolation_version": "torch3d",
    }

    class Struct:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    self.__dict__[key] = Struct(**value)
                else:
                    self.__dict__[key] = value

    dot_args = Struct(**dot_args)

    dot_model = create_model(dot_args)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    vae_img.requires_grad_(False)
    text_encoder.requires_grad_(False)
    dot_model.requires_grad_(False)
    dot_model.eval()

    controlnet.requires_grad_(True)
    for param in controlnet.parameters():
        param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    print(f"[trainable params number: {len(trainable_params)}]")
    print(
        f"[trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M]"
    )

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    vae_img.to(local_rank)
    text_encoder.to(local_rank)
    dot_model.to(local_rank)

    # Get the training dataset
    train_dataset = WebVid10M_Multiple_Image(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * num_processes
        )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = FlowGenPipeline(
            vae_img=vae_img,
            vae_flow=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=noise_scheduler,
            safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    control = DDP(controlnet, device_ids=[local_rank], output_device=local_rank)
    # unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps), disable=not is_main_process
    )
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    # grids = coords_grid(15, 320, 512, local_rank)
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch["text"] = [
                    name if random.random() > cfg_random_null_text_ratio else ""
                    for name in batch["text"]
                ]

            ### >>>> Training >>>> ###

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to(local_rank)
            stride = batch["stride"][:, 1:].to(local_rank)
            _b, _f, _c, _h, _w = pixel_values.shape
            video_length = _f - 1
            img1 = pixel_values[:, 0, ...].repeat(video_length, 1, 1, 1)
            img2 = pixel_values[:, 1:, ...].reshape(_b * video_length, _c, _h, _w)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                    flow_pre = []
                    for b_idx in range(_b):
                        traj_pre = dot_model(
                            {"video": pixel_values[b_idx : b_idx + 1]},
                            mode="tracks_from_first_to_every_other_frame",
                            **vars(dot_args),
                        )
                        traj_pre = traj_pre["tracks"]
                        traj_pre = traj_pre.permute(0, 1, 4, 2, 3)[
                            :, 1:, :2, ...
                        ].reshape(video_length, 2, _h, _w)
                        flow_pre.append(traj_pre)

                flow_pre = torch.stack(flow_pre).reshape(_b * video_length, 2, _h, _w)
                flow_pre[:, 0:1, ...] = flow_pre[:, 0:1, ...] / flow_pre.shape[-1]
                flow_pre[:, 1:2, ...] = flow_pre[:, 1:2, ...] / flow_pre.shape[-2]
                flow_pre = (flow_pre + 1) / 2  # (0,1)
                latents = vae.encode(flow_pre).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                # extract spare flow and mask here for controllabel training.

                flow_pre = rearrange(
                    flow_pre, "(b f) c h w -> b c f h w", f=video_length
                )
                b, c, f, h, w = flow_pre.shape

                # set a proper mask ratio. 0.5 for initial training. Smaller value for finetuing.
                mask = repeat(
                    generate_sparse_flow_mask(b, 1, h, w, 8, np.random.rand() * 0.5).to(
                        local_rank
                    ),
                    "b c h w -> b c f h w",
                    f=video_length,
                )  # b x c x f x h x w
                flow_mask = mask * (flow_pre - 1 / 2) + 1 / 2
                flow_mask_latent = rearrange(
                    vae.encode(
                        rearrange(flow_mask, "b c f h w -> (b f) c h w")
                    ).latent_dist.sample(),
                    "(b f) c h w -> b c f h w",
                    f=video_length,
                )
                mask = F.interpolate(mask, scale_factor=(1, 1 / 8, 1 / 8))
                control = torch.cat([flow_mask_latent, mask], dim=1)

                latents_img = vae_img.encode(img1).latent_dist
                latents_img = latents_img.sample()
                latents_img = rearrange(
                    latents_img, "(b f) c h w -> b c f h w", f=video_length
                )

                latents = latents * 0.18215
                latents_img = latents_img * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=flow_pre.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = torch.cat([latents_img, noisy_latents], dim=1)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch["text"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(flow_pre.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    stride,
                    encoder_hidden_states,
                    control=control,
                ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            ### <<<< Training <<<< ###

            # Wandb logging
            if is_main_process and (not is_debug):
                tb_writer.add_scalar("step_loss", loss.item(), global_step)

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.module.state_dict(),
                }

                torch.save(
                    state_dict, os.path.join(save_path, f"checkpoint{global_step}.ckpt")
                )
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (
                global_step % validation_steps == 0
                or global_step in validation_steps_tuple
            ):
                samples = []

                height = (
                    train_data.sample_size[0]
                    if not isinstance(train_data.sample_size, int)
                    else train_data.sample_size
                )
                width = (
                    train_data.sample_size[1]
                    if not isinstance(train_data.sample_size, int)
                    else train_data.sample_size
                )

                prompts = (
                    validation_data.prompts[:2]
                    if global_step < 1000 and (not image_finetune)
                    else validation_data.prompts
                )
                img_paths = (
                    validation_data.img_paths[:2]
                    if global_step < 1000 and (not image_finetune)
                    else validation_data.img_paths
                )
                n_prompts = (
                    validation_data.n_prompts[:2]
                    if global_step < 1000 and (not image_finetune)
                    else validation_data.n_prompts
                )

                for idx, (prompt, img_path, n_prompt) in enumerate(
                    zip(prompts, img_paths, n_prompts)
                ):

                    first_frame_unnorm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )
                    first_frame_norm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                            * 2
                            - 1
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    stride_input = list(range(8, 121, 8))
                    sparse_flow = torch.zeros((1, 2, video_length, height, width))
                    for i in range(video_length):
                        sparse_flow[:, :, i] += (i + 1) / video_length / 2

                    sparse_flow = sparse_flow / 2 + 0.5

                    sample = (sparse_flow.cpu() * 2 - 1).clamp(-1, 1)
                    sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
                    sample[:, 1:2, ...] = sample[:, 1:2, ...] * height
                    this_sample = []
                    this_sample.append(first_frame_unnorm[None].float().cpu())
                    forwardwarpFunc = ForwardWarp()
                    for i in range(sample.shape[2]):
                        warped_img, o = forwardwarpFunc(
                            first_frame_unnorm[None].float().cuda(),
                            sample[:, :, i, ...].float().cuda(),
                        )
                        warped_img = warped_img / (o + 1e-6)
                        this_sample.append(warped_img.cpu())

                    this_sample = torch.stack(this_sample, dim=2)
                    sample = this_sample

                    save_videos_grid(
                        sample,
                        f"{output_dir}/samples/sample-{global_step}/{idx}-gt.gif",
                    )
                    samples.append(sample)

                    sparse_flow = sparse_flow.to(local_rank)
                    sparse_mask = torch.zeros((1, 1, video_length, height, width)).to(
                        local_rank
                    )
                    sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

                    flow_mask_latent = rearrange(
                        vae.encode(
                            rearrange(sparse_flow, "b c f h w -> (b f) c h w")
                        ).latent_dist.sample(),
                        "(b f) c h w -> b c f h w",
                        f=video_length,
                    )

                    sparse_mask = F.interpolate(
                        sparse_mask, scale_factor=(1, 1 / 8, 1 / 8)
                    )
                    control = torch.cat([flow_mask_latent, sparse_mask], dim=1)

                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(global_seed)
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            first_frame=first_frame_unnorm,
                            control=control,
                            stride=torch.tensor([stride_input]).cuda(),
                            negative_prompt=n_prompt,
                            generator=generator,
                            video_length=video_length,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos

                        sample = (sample * 2 - 1).clamp(-1, 1)
                        sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
                        sample[:, 1:2, ...] = sample[:, 1:2, ...] * height

                        this_sample = []
                        this_sample.append(first_frame_unnorm[None].float().cpu())
                        forwardwarpFunc = ForwardWarp()

                        for i in range(sample.shape[2]):
                            warped_img, o = forwardwarpFunc(
                                first_frame_unnorm[None].float().cuda(),
                                sample[:, :, i, ...].float().cuda(),
                            )
                            warped_img = warped_img / (o + 1e-6)
                            this_sample.append(warped_img.cpu())

                        this_sample = torch.stack(this_sample, dim=2)
                        sample = this_sample

                        save_videos_grid(
                            sample,
                            f"{output_dir}/samples/sample-{global_step}/{idx}-zero-mask.gif",
                        )
                        samples.append(sample)

                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator=generator,
                            height=height,
                            width=width,
                            num_inference_steps=validation_data.get(
                                "num_inference_steps", 25
                            ),
                            guidance_scale=validation_data.get("guidance_scale", 8.0),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)

                    first_frame_unnorm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )
                    first_frame_norm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                            * 2
                            - 1
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    stride_input = list(range(8, 121, 8))
                    sparse_flow = torch.zeros((1, 2, video_length, height, width))
                    for i in range(video_length):
                        sparse_flow[:, :, i] += (i + 1) / video_length
                    sparse_flow = sparse_flow / 2 + 0.5
                    sparse_flow = sparse_flow.to(local_rank)
                    sparse_mask = torch.ones((1, 1, video_length, height, width)).to(
                        local_rank
                    )
                    sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

                    flow_mask_latent = rearrange(
                        vae.encode(
                            rearrange(sparse_flow, "b c f h w -> (b f) c h w")
                        ).latent_dist.sample(),
                        "(b f) c h w -> b c f h w",
                        f=video_length,
                    )
                    sparse_mask = F.interpolate(
                        sparse_mask, scale_factor=(1, 1 / 8, 1 / 8)
                    )

                    control = torch.cat([flow_mask_latent, sparse_mask], dim=1)

                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(global_seed)
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            first_frame=first_frame_unnorm,
                            control=control,
                            stride=torch.tensor([stride_input]).cuda(),
                            negative_prompt=n_prompt,
                            generator=generator,
                            video_length=video_length,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos
                        sample = (sample * 2 - 1).clamp(-1, 1)
                        sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
                        sample[:, 1:2, ...] = sample[:, 1:2, ...] * height

                        this_sample = []
                        this_sample.append(first_frame_unnorm[None].float().cpu())
                        forwardwarpFunc = ForwardWarp()

                        for i in range(sample.shape[2]):
                            warped_img, o = forwardwarpFunc(
                                first_frame_unnorm[None].float().cuda(),
                                sample[:, :, i, ...].float().cuda(),
                            )
                            warped_img = warped_img / (o + 1e-6)
                            this_sample.append(warped_img.cpu())

                        this_sample = torch.stack(this_sample, dim=2)
                        sample = this_sample
                        save_videos_grid(
                            sample,
                            f"{output_dir}/samples/sample-{global_step}/{idx}-all-mask.gif",
                        )
                        samples.append(sample)

                    first_frame_unnorm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )
                    first_frame_norm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                            * 2
                            - 1
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    stride_input = list(range(8, 121, 8))
                    sparse_flow = torch.zeros((1, 2, video_length, height, width))
                    for i in range(video_length):
                        sparse_flow[:, :, i] += (i + 1) / video_length
                    sparse_flow = sparse_flow / 2 + 0.5
                    sparse_flow = sparse_flow.to(local_rank)
                    sparse_mask = torch.zeros((1, 1, video_length, height, width)).to(
                        local_rank
                    )
                    sparse_mask[:, :, :, 152:168, 152:168] = 1.0
                    sparse_mask[:, :, :, :16, :16] = 1.0
                    sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

                    flow_mask_latent = rearrange(
                        vae.encode(
                            rearrange(sparse_flow, "b c f h w -> (b f) c h w")
                        ).latent_dist.sample(),
                        "(b f) c h w -> b c f h w",
                        f=video_length,
                    )
                    sparse_mask = F.interpolate(
                        sparse_mask, scale_factor=(1, 1 / 8, 1 / 8)
                    )

                    control = torch.cat([flow_mask_latent, sparse_mask], dim=1)

                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(global_seed)
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            first_frame=first_frame_unnorm,
                            control=control,
                            stride=torch.tensor([stride_input]).cuda(),
                            negative_prompt=n_prompt,
                            generator=generator,
                            video_length=video_length,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos
                        sample = (sample * 2 - 1).clamp(-1, 1)
                        sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
                        sample[:, 1:2, ...] = sample[:, 1:2, ...] * height

                        this_sample = []
                        this_sample.append(first_frame_unnorm[None].float().cpu())
                        forwardwarpFunc = ForwardWarp()

                        for i in range(sample.shape[2]):
                            warped_img, o = forwardwarpFunc(
                                first_frame_unnorm[None].float().cuda(),
                                sample[:, :, i, ...].float().cuda(),
                            )
                            warped_img = warped_img / (o + 1e-6)
                            this_sample.append(warped_img.cpu())

                        this_sample = torch.stack(this_sample, dim=2)
                        sample = this_sample
                        save_videos_grid(
                            sample,
                            f"{output_dir}/samples/sample-{global_step}/{idx}-sparse.gif",
                        )
                        samples.append(sample)

                    first_frame_unnorm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )
                    first_frame_norm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                            * 2
                            - 1
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    stride_input = list(range(8, 121, 8))
                    sparse_flow = torch.zeros((1, 2, video_length, height, width))
                    for i in range(video_length):
                        sparse_flow[:, :, i] += (i + 1) / video_length
                    sparse_flow = sparse_flow / 2 + 0.5
                    sparse_flow = sparse_flow.to(local_rank)
                    sparse_mask = torch.zeros((1, 1, video_length, height, width)).to(
                        local_rank
                    )
                    sparse_mask[
                        :, :, -1, height // 4 : -height // 4, width // 4 : -width // 4
                    ] = 1.0
                    sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

                    flow_mask_latent = rearrange(
                        vae.encode(
                            rearrange(sparse_flow, "b c f h w -> (b f) c h w")
                        ).latent_dist.sample(),
                        "(b f) c h w -> b c f h w",
                        f=video_length,
                    )
                    sparse_mask = F.interpolate(
                        sparse_mask, scale_factor=(1, 1 / 8, 1 / 8)
                    )

                    control = torch.cat([flow_mask_latent, sparse_mask], dim=1)

                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(global_seed)
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            first_frame=first_frame_unnorm,
                            control=control,
                            stride=torch.tensor([stride_input]).cuda(),
                            negative_prompt=n_prompt,
                            generator=generator,
                            video_length=video_length,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos
                        sample = (sample * 2 - 1).clamp(-1, 1)
                        sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
                        sample[:, 1:2, ...] = sample[:, 1:2, ...] * height

                        this_sample = []
                        this_sample.append(first_frame_unnorm[None].float().cpu())
                        forwardwarpFunc = ForwardWarp()

                        for i in range(sample.shape[2]):
                            warped_img, o = forwardwarpFunc(
                                first_frame_unnorm[None].float().cuda(),
                                sample[:, :, i, ...].float().cuda(),
                            )
                            warped_img = warped_img / (o + 1e-6)
                            this_sample.append(warped_img.cpu())

                        this_sample = torch.stack(this_sample, dim=2)
                        sample = this_sample
                        save_videos_grid(
                            sample,
                            f"{output_dir}/samples/sample-{global_step}/{idx}-sparse-last.gif",
                        )
                        samples.append(sample)

                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator=generator,
                            height=height,
                            width=width,
                            num_inference_steps=validation_data.get(
                                "num_inference_steps", 25
                            ),
                            guidance_scale=validation_data.get("guidance_scale", 8.0),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)

                    first_frame_unnorm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )
                    first_frame_norm = (
                        torch.from_numpy(
                            np.array(Image.open(img_path).resize((width, height)))
                            / 255.0
                            * 2
                            - 1
                        )
                        .permute(2, 0, 1)
                        .contiguous()
                    )

                    stride_input = list(range(8, 121, 8))
                    sparse_flow = torch.zeros((1, 2, video_length, height, width))
                    for i in range(video_length):
                        sparse_flow[:, :, i] += (i + 1) / video_length
                    sparse_flow = sparse_flow / 2 + 0.5
                    sparse_flow = sparse_flow.to(local_rank)
                    sparse_mask = torch.zeros((1, 1, video_length, height, width)).to(
                        local_rank
                    )
                    sparse_mask[
                        :, :, :, height // 4 : -height // 4, width // 4 : -width // 4
                    ] = 1.0
                    sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

                    flow_mask_latent = rearrange(
                        vae.encode(
                            rearrange(sparse_flow, "b c f h w -> (b f) c h w")
                        ).latent_dist.sample(),
                        "(b f) c h w -> b c f h w",
                        f=video_length,
                    )
                    sparse_mask = F.interpolate(
                        sparse_mask, scale_factor=(1, 1 / 8, 1 / 8)
                    )
                    control = torch.cat([flow_mask_latent, sparse_mask], dim=1)

                    generator = torch.Generator(device=latents.device)
                    generator.manual_seed(global_seed)
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            first_frame=first_frame_unnorm,
                            control=control,
                            stride=torch.tensor([stride_input]).cuda(),
                            negative_prompt=n_prompt,
                            generator=generator,
                            video_length=video_length,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos
                        sample = (sample * 2 - 1).clamp(-1, 1)
                        sample[:, 0:1, ...] = sample[:, 0:1, ...] * width
                        sample[:, 1:2, ...] = sample[:, 1:2, ...] * height

                        this_sample = []
                        this_sample.append(first_frame_unnorm[None].float().cpu())
                        forwardwarpFunc = ForwardWarp()

                        for i in range(sample.shape[2]):
                            warped_img, o = forwardwarpFunc(
                                first_frame_unnorm[None].float().cuda(),
                                sample[:, :, i, ...].float().cuda(),
                            )
                            warped_img = warped_img / (o + 1e-6)
                            this_sample.append(warped_img.cpu())

                        this_sample = torch.stack(this_sample, dim=2)
                        sample = this_sample
                        save_videos_grid(
                            sample,
                            f"{output_dir}/samples/sample-{global_step}/{idx}-half-mask.gif",
                        )
                        samples.append(sample)

                    else:
                        sample = validation_pipeline(
                            prompt,
                            generator=generator,
                            height=height,
                            width=width,
                            num_inference_steps=validation_data.get(
                                "num_inference_steps", 25
                            ),
                            guidance_scale=validation_data.get("guidance_scale", 8.0),
                        ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=str)
    args = parser.parse_args()

    name = args.task
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
