import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter
import uuid
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision
from flowgen.models.controlnet import ControlNetModel
from scripts.utils import *

import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from flowgen.models.unet3d import UNet3DConditionModel as UNet3DConditionModelFlow
from animation.models.forward_unet import UNet3DConditionModel

from flowgen.pipelines.pipeline_flow_gen import FlowGenPipeline
from animation.pipelines.pipeline_animation import AnimationPipeline


from animation.utils.util import save_videos_grid
from animation.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
import math
from pathlib import Path

from PIL import Image

import numpy as np

import torch.nn as nn

output_dir = "outputs"
ensure_dirname(output_dir)


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


def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def visualize_drag_v2(background_image_path, brush_mask, splited_tracks, width, height):
    trajectory_maps = []

    background_image = Image.open(background_image_path).convert("RGBA")
    background_image = background_image.resize((width, height))
    w, h = background_image.size

    # Create a half-transparent background
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128

    # Create a purple overlay layer
    purple_layer = np.zeros((h, w, 4), dtype=np.uint8)
    purple_layer[:, :, :3] = [128, 0, 128]  # Purple color
    purple_alpha = np.where(brush_mask < 0.5, 64, 0)  # Alpha values based on brush_mask
    purple_layer[:, :, 3] = purple_alpha

    # Convert to PIL image for alpha_composite
    purple_layer = Image.fromarray(purple_layer)
    transparent_background = Image.fromarray(transparent_background)

    # Blend the purple layer with the background
    transparent_background = Image.alpha_composite(transparent_background, purple_layer)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track) - 1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i + 1][0]), int(splited_track[i + 1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track) - 2:
                    cv2.arrowedLine(
                        transparent_layer,
                        start_point,
                        end_point,
                        (255, 0, 0, 192),
                        2,
                        tipLength=8 / arrow_length,
                    )
                else:
                    cv2.line(
                        transparent_layer, start_point, end_point, (255, 0, 0, 192), 2
                    )
        else:
            cv2.circle(
                transparent_layer,
                (int(splited_track[0][0]), int(splited_track[0][1])),
                5,
                (255, 0, 0, 192),
                -1,
            )

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer


class Drag:
    def __init__(
        self,
        device,
        pretrained_model_path,
        inference_config,
        height,
        width,
        model_length,
    ):
        self.device = device

        inference_config = OmegaConf.load(inference_config)
        ### >>> create validation pipeline >>> ###
        print("start loading")
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        # unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        unet = UNet3DConditionModelFlow.from_config_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        vae_img = AutoencoderKL.from_pretrained(
            "models/stage2/StableDiffusion", subfolder="vae"
        )
        import json

        with open("./models/stage1/StableDiffusion-FlowGen/vae/config.json", "r") as f:
            vae_config = json.load(f)
        vae = AutoencoderKL.from_config(vae_config)
        vae_pretrained_path = (
            "models/stage1/StableDiffusion-FlowGen/vae_flow/diffusion_pytorch_model.bin"
        )
        print("[Load vae weights from {}]".format(vae_pretrained_path))
        processed_ckpt = {}
        weight = torch.load(vae_pretrained_path, map_location="cpu")
        vae.load_state_dict(weight, strict=True)
        controlnet = ControlNetModel.from_unet(unet)
        unet.controlnet = controlnet
        unet.control_scale = 1.0

        unet_pretrained_path = (
            "models/stage1/StableDiffusion-FlowGen/unet/diffusion_pytorch_model.bin"
        )
        print("[Load unet weights from {}]".format(unet_pretrained_path))
        weight = torch.load(unet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)

        controlnet_pretrained_path = (
            "models/stage1/StableDiffusion-FlowGen/controlnet/controlnet.bin"
        )
        print("[Load controlnet weights from {}]".format(controlnet_pretrained_path))
        weight = torch.load(controlnet_pretrained_path, map_location="cpu")
        m, u = unet.load_state_dict(weight, strict=False)

        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False
        pipeline = FlowGenPipeline(
            vae_img=vae_img,
            vae_flow=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        )  # .to("cuda")
        pipeline = pipeline.to("cuda")

        self.pipeline = pipeline
        self.height = height
        self.width = width
        self.ouput_prefix = f"flow_debug"
        self.model_length = model_length

        ### >>> create validation pipeline >>> ###
        pretrained_model_path = "models/stage2/StableDiffusion"
        print("start loading")
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                inference_config.unet_additional_kwargs
            ),
        )
        # 3. text_model
        motion_module_path = "models/stage2/Motion_Module/motion_block.bin"
        print("[Loading motion module ckpt from {}]".format(motion_module_path))
        weight = torch.load(motion_module_path, map_location="cpu")
        unet.load_state_dict(weight, strict=False)

        from safetensors import safe_open

        dreambooth_state_dict = {}
        with safe_open(
            "models/stage2/DreamBooth_LoRA/realisticVisionV51_v20Novae.safetensors",
            framework="pt",
            device="cpu",
        ) as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)

        from animation.utils.convert_from_ckpt import (
            convert_ldm_unet_checkpoint,
            convert_ldm_clip_checkpoint,
            convert_ldm_vae_checkpoint,
        )

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            dreambooth_state_dict, vae.config
        )
        vae.load_state_dict(converted_vae_checkpoint)
        personalized_unet_path = "models/stage2/DreamBooth_LoRA/realistic_unet.ckpt"
        print("[Loading personalized unet ckpt from {}]".format(personalized_unet_path))
        unet.load_state_dict(torch.load(personalized_unet_path), strict=False)

        print("finish loading")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False
        pipeline = AnimationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
        )  # .to("cuda")
        pipeline = pipeline.to("cuda")

        self.animate_pipeline = pipeline

    @torch.no_grad()
    def forward_sample(
        self,
        input_drag,
        mask_drag,
        brush_mask,
        input_first_frame,
        prompt,
        n_prompt,
        inference_steps,
        guidance_scale,
        outputs=dict(),
    ):
        device = self.device

        b, l, h, w, c = input_drag.size()
        # drag = torch.cat([torch.zeros_like(drag[:, 0]).unsqueeze(1), drag], dim=1)  # pad the first frame with zero flow
        drag = rearrange(input_drag, "b l h w c -> b c l h w")
        mask = rearrange(mask_drag, "b l h w c -> b c l h w")
        brush_mask = rearrange(brush_mask, "b l h w c -> b c l h w")

        sparse_flow = drag
        sparse_mask = mask

        sparse_flow = (sparse_flow - 1 / 2) * sparse_mask + 1 / 2

        flow_mask_latent = rearrange(
            self.pipeline.vae_flow.encode(
                rearrange(sparse_flow, "b c f h w -> (b f) c h w")
            ).latent_dist.sample(),
            "(b f) c h w -> b c f h w",
            f=l,
        )
        # flow_mask_latent = vae.encode(sparse_flow).latent_dist.sample()*0.18215
        sparse_mask = F.interpolate(sparse_mask, scale_factor=(1, 1 / 8, 1 / 8))
        control = torch.cat([flow_mask_latent, sparse_mask], dim=1)
        # print(drag)
        stride = list(range(8, 121, 8))

        sample = self.pipeline(
            prompt,
            first_frame=input_first_frame.squeeze(0),
            control=control,
            stride=torch.tensor([stride]).cuda(),
            negative_prompt=n_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=w,
            height=h,
            video_length=len(stride),
        ).videos
        sample = (sample * 2 - 1).clamp(-1, 1)
        sample = sample * (1 - brush_mask.to(sample.device))
        sample[:, 0:1, ...] = sample[:, 0:1, ...] * w
        sample[:, 1:2, ...] = sample[:, 1:2, ...] * h

        flow_pre = sample.squeeze(0)
        flow_pre = rearrange(flow_pre, "c f h w -> f c h w")
        flow_pre = torch.cat(
            [torch.zeros(1, 2, h, w).to(flow_pre.device), flow_pre], dim=0
        )
        sample = self.animate_pipeline(
            prompt,
            first_frame=input_first_frame.squeeze(0) * 2 - 1,
            flow_pre=flow_pre,
            brush_mask=brush_mask,
            negative_prompt=n_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=w,
            height=h,
            video_length=self.model_length,
        ).videos

        return sample

    def run(
        self,
        first_frame_path,
        image_brush,
        tracking_points,
        inference_batch_size,
        flow_unit_id,
        prompt,
    ):
        original_width, original_height = 512, 320

        brush_mask = image_brush["mask"]

        brush_mask = (
            cv2.resize(
                brush_mask[:, :, 0],
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            / 255.0
        )

        brush_mask_bool = brush_mask > 0.5
        brush_mask[brush_mask_bool], brush_mask[~brush_mask_bool] = 0, 1

        brush_mask = torch.from_numpy(brush_mask)

        brush_mask = (
            torch.zeros_like(brush_mask) if (brush_mask == 1).all() else brush_mask
        )

        brush_mask = brush_mask.unsqueeze(0).unsqueeze(3)

        input_all_points = tracking_points.constructor_args["value"]
        resized_all_points = [
            tuple(
                [
                    tuple(
                        [
                            int(e1[0] * self.width / original_width),
                            int(e1[1] * self.height / original_height),
                        ]
                    )
                    for e1 in e
                ]
            )
            for e in input_all_points
        ]

        input_drag = torch.zeros(self.model_length - 1, self.height, self.width, 2)
        mask_drag = torch.zeros(self.model_length - 1, self.height, self.width, 1)
        for splited_track in resized_all_points:
            if len(splited_track) == 1:  # stationary point
                displacement_point = tuple(
                    [splited_track[0][0] + 1, splited_track[0][1] + 1]
                )
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[: self.model_length]
            if len(splited_track) < self.model_length:
                splited_track = splited_track + [splited_track[-1]] * (
                    self.model_length - len(splited_track)
                )
            for i in range(self.model_length - 1):
                start_point = splited_track[0]
                end_point = splited_track[i + 1]
                input_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                    0,
                ] = (
                    end_point[0] - start_point[0]
                )
                input_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                    1,
                ] = (
                    end_point[1] - start_point[1]
                )
                mask_drag[
                    i,
                    max(int(start_point[1]) - flow_unit_id, 0) : int(start_point[1])
                    + flow_unit_id,
                    max(int(start_point[0]) - flow_unit_id, 0) : int(
                        start_point[0] + flow_unit_id
                    ),
                ] = 1

        input_drag[..., 0] /= self.width
        input_drag[..., 1] /= self.height

        input_drag = input_drag * (1 - brush_mask)
        mask_drag = torch.where(brush_mask.expand_as(mask_drag) > 0, 1, mask_drag)

        input_drag = (input_drag + 1) / 2
        dir, base, ext = split_filename(first_frame_path)
        id = base.split("_")[-1]

        image_pil = image2pil(first_frame_path)
        image_pil = image_pil.resize((self.width, self.height), Image.BILINEAR).convert(
            "RGB"
        )

        visualized_drag, _ = visualize_drag_v2(
            first_frame_path,
            brush_mask.squeeze(3).squeeze(0).cpu().numpy(),
            resized_all_points,
            self.width,
            self.height,
        )

        first_frames_transform = transforms.Compose(
            [
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
            ]
        )

        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
            if not outputs:

                first_frames = image2arr(first_frame_path)
                Image.fromarray(first_frames).save("./temp.png")
                first_frames = repeat(
                    first_frames_transform(first_frames),
                    "c h w -> b c h w",
                    b=inference_batch_size,
                ).to(self.device)
            else:
                first_frames = outputs[:, -1]

            outputs = self.forward_sample(
                repeat(
                    input_drag[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                repeat(
                    mask_drag[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                repeat(
                    brush_mask[
                        i * (self.model_length - 1) : (i + 1) * (self.model_length - 1)
                    ],
                    "l h w c -> b l h w c",
                    b=inference_batch_size,
                ).to(self.device),
                first_frames,
                prompt,
                "(blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
                25,
                7,
            )
            ouput_video_list.append(outputs)

        outputs_path = f"gradio/samples/output_{id}.gif"
        save_videos_grid(outputs, outputs_path)

        return visualized_drag[0], outputs_path


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">Motion-I2V</h1><br>""")

    gr.Markdown(
        """ Gradio Demo for <a href='https://arxiv.org/abs/2401.15977'><b> [SIGGRAPH 2024] Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling</b></a>.<br>
    The gradio demo is adapted from the gradio demo of DragNuWA. <br>"""
    )

    gr.Image(label="DragNUWA", value="assets/Figure1.gif")

    gr.Markdown(
        """## Usage: <br>
                0. Upload an image via the "Upload Image" button.<br>
                2. Drags & Motion Brush.<br>
                    2.1. Click "Add Drag" when you want to add a control path.<br>
                    2.2. You can click several points which forms a path.<br>
                    2.3. Click "Delete last drag" to delete the whole lastest path.<br>
                    2.4. Click "Delete last step" to delete the lastest clicked control point.<br>
                    2.5 Add mask to select the moveable regions. If no mask, all regions are set to movable.<br>
                3. Animate the image according the path with a click on "Run" button. <br>"""
    )

    DragNUWA_net = Drag(
        "cuda:0",
        "models/stage1/StableDiffusion-FlowGen",
        "configs/configs_flowgen/inference/inference.yaml",
        320,
        512,
        16,
    )
    first_frame_path = gr.State()
    tracking_points = gr.State([])

    def reset_states(first_frame_path, tracking_points):
        first_frame_path = gr.State()
        tracking_points = gr.State([])
        return first_frame_path, tracking_points

    def preprocess_image(image):
        image_pil = image2pil(image.name)
        raw_w, raw_h = image_pil.size
        resize_ratio = max(512 / raw_w, 320 / raw_h)
        image_pil = image_pil.resize(
            (int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR
        )
        image_pil = transforms.CenterCrop((320, 512))(image_pil.convert("RGB"))

        first_frame_path = os.path.join(
            output_dir, f"first_frame_{str(uuid.uuid4())[:4]}.png"
        )
        image_pil.save(first_frame_path)

        return first_frame_path, first_frame_path, first_frame_path, gr.State([])

    def add_drag(tracking_points):
        tracking_points.constructor_args["value"].append([])
        return tracking_points

    def delete_last_drag(tracking_points, first_frame_path):
        tracking_points.constructor_args["value"].pop()
        transparent_background = Image.open(first_frame_path).convert("RGBA")
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args["value"]:
            if len(track) > 1:
                for i in range(len(track) - 1):
                    start_point = track[i]
                    end_point = track[i + 1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track) - 2:
                        cv2.arrowedLine(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                            tipLength=8 / arrow_length,
                        )
                    else:
                        cv2.line(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                        )
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(
            transparent_background, transparent_layer
        )
        return tracking_points, trajectory_map

    def delete_last_step(tracking_points, first_frame_path):
        tracking_points.constructor_args["value"][-1].pop()
        transparent_background = Image.open(first_frame_path).convert("RGBA")
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args["value"]:
            if len(track) > 1:
                for i in range(len(track) - 1):
                    start_point = track[i]
                    end_point = track[i + 1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track) - 2:
                        cv2.arrowedLine(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                            tipLength=8 / arrow_length,
                        )
                    else:
                        cv2.line(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                        )
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(
            transparent_background, transparent_layer
        )
        return tracking_points, trajectory_map

    def add_tracking_points(
        tracking_points, first_frame_path, evt: gr.SelectData
    ):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        tracking_points.constructor_args["value"][-1].append(evt.index)

        transparent_background = Image.open(first_frame_path).convert("RGBA")
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args["value"]:
            if len(track) > 1:
                for i in range(len(track) - 1):
                    start_point = track[i]
                    end_point = track[i + 1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track) - 2:
                        cv2.arrowedLine(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                            tipLength=8 / arrow_length,
                        )
                    else:
                        cv2.line(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                        )
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(
            transparent_background, transparent_layer
        )
        return tracking_points, trajectory_map

    with gr.Row():
        with gr.Column(scale=1):
            image_upload_button = gr.UploadButton(
                label="Upload Image", file_types=["image"]
            )
            add_drag_button = gr.Button(value="Add Drag")
            reset_button = gr.Button(value="Reset")
            run_button = gr.Button(value="Run")
            delete_last_drag_button = gr.Button(value="Delete last drag")
            delete_last_step_button = gr.Button(value="Delete last step")

        with gr.Column(scale=7):
            with gr.Row():
                with gr.Column(scale=6):
                    input_image = gr.Image(
                        label=None,
                        interactive=True,
                        height=320,
                        width=512,
                    )
                with gr.Column(scale=6):
                    output_image = gr.Image(
                        label=None,
                        height=320,
                        width=512,
                    )

    with gr.Row():
        with gr.Column(scale=1):
            inference_batch_size = gr.Slider(
                label="Inference Batch Size", minimum=1, maximum=1, step=1, value=1
            )

            flow_unit_id = gr.Slider(
                label="Flow Unit", minimum=1, maximum=320, step=1, value=64
            )
            prompt = gr.Textbox(label="prompt")

        with gr.Column(scale=5):
            with gr.Row():
                image_brush = gr.Image(
                    label=None,
                    interactive=True,
                    height=320,
                    width=512,
                    type="numpy",
                    tool="sketch",
                )
                output_video = gr.Image(
                    label="Output Video",
                    height=320,
                    width=512,
                )

    with gr.Row():
        gr.Markdown(
            """
            ## Citation
            ```bibtex
            @article{shi2024motion,
            title={Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling},
            author={Shi, Xiaoyu and Huang, Zhaoyang and Wang, Fu-Yun and Bian, Weikang and Li, Dasong and Zhang, Yi and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and others},
            journal={SIGGRAPH 2024},
            year={2024}
            }
            ```
            """
        )

    image_upload_button.upload(
        preprocess_image,
        image_upload_button,
        [input_image, image_brush, first_frame_path, tracking_points],
    )

    add_drag_button.click(add_drag, tracking_points, tracking_points)

    delete_last_drag_button.click(
        delete_last_drag,
        [tracking_points, first_frame_path],
        [tracking_points, input_image],
    )

    delete_last_step_button.click(
        delete_last_step,
        [tracking_points, first_frame_path],
        [tracking_points, input_image],
    )

    reset_button.click(
        reset_states,
        [first_frame_path, tracking_points],
        [first_frame_path, tracking_points],
    )

    input_image.select(
        add_tracking_points,
        [tracking_points, first_frame_path],
        [tracking_points, input_image],
    )

    run_button.click(
        DragNUWA_net.run,
        [
            first_frame_path,
            image_brush,
            tracking_points,
            inference_batch_size,
            flow_unit_id,
            prompt,
        ],
        [output_image, output_video],
    )

gr.close_all()
demo.queue(concurrency_count=3, max_size=20)
demo.launch(share=True, server_name="127.0.0.1")
