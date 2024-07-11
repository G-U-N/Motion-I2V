from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward

from einops import rearrange, repeat
import math

from . import softsplat


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


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def forward(
        self,
        input_tensor,
        temb,
        encoder_hidden_states,
        flow_pre=None,
        attention_mask=None,
        anchor_frame_idx=None,
    ):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, flow_pre=flow_pre
        )

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        flow_pre=None,
    ):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                flow_pre=flow_pre,
                HW=(height, weight),
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=(
                        cross_attention_dim if block_name.endswith("_Cross") else None
                    ),
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    _query_dim=dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        flow_pre=None,
        HW=None,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states
                        if attention_block.is_cross_attention
                        else None
                    ),
                    video_length=video_length,
                    flow_pre=flow_pre,
                    HW=HW,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        _query_dim=-1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and (attention_mode == "Temporal"))
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward_warp_hidden(self, flow, hidden_states):

        batch, video_length, _, H, W = flow.shape  # [1, 6, 2, 40, 40]
        _, _, C = hidden_states.shape
        flow = (
            flow[:, 1:, ...]
            .reshape(batch * (video_length - 1), 2, H, W)
            .contiguous()
            .float()
        )  # [5, 2, 40, 40]
        hidden_states = hidden_states.reshape(
            batch, video_length, H, W, C
        )  # [1, 6, 40, 40, 320]
        hidden_states = hidden_states[:, 0, ...]
        hidden_states = (
            hidden_states.permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .repeat(video_length - 1, 1, 1, 1)
        )
        warped_img = softsplat.softsplat(
            tenIn=hidden_states.detach(), tenFlow=flow, tenMetric=None, strMode="avg"
        )
        return warped_img

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        flow_pre=None,
        HW=None,
    ):

        if self.attention_mode == "Temporal":
            batch_size, sequence_length, C = hidden_states.shape  # [6, 1600, 320]
            H, W = HW
            assert H * W == sequence_length

            B_flow, L_flow, _, H_flow, W_flow = flow_pre.shape
            flow_pre = F.interpolate(
                flow_pre.reshape(B_flow * L_flow, 2, H_flow, W_flow),
                size=(H, W),
                mode="bilinear",
            )
            flow_pre[:, 0:1, ...] = flow_pre[:, 0:1, ...] / (W_flow / W)
            flow_pre[:, 1:2, ...] = flow_pre[:, 1:2, ...] / (H_flow / H)

            flow_pre = flow_pre.reshape(B_flow, L_flow, 2, H, W)  # [1, 6, 2, 40, 40]

            warped_hidden_states = self.forward_warp_hidden(
                flow_pre, hidden_states
            )  # [1, 5, 320, 40, 40]
            warped_hidden_states = warped_hidden_states.reshape(
                B_flow, L_flow - 1, C, H * W
            ).permute(0, 1, 3, 2)

            _tmp = []
            hidden_states = hidden_states.reshape(B_flow, L_flow, H * W, C)
            _tmp.append(hidden_states[:, 0, ...])
            for idx in range(warped_hidden_states.shape[1]):
                _tmp.append(warped_hidden_states[:, idx, ...])
                _tmp.append(hidden_states[:, idx + 1, ...])
            hidden_states = torch.stack(_tmp, dim=1)  # [1, 11, 1600, 320]
            hidden_states = hidden_states.reshape(-1, H * W, C)

            if self.attention_mode == "Temporal":
                d = hidden_states.shape[1]
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> (b d) f c", f=2 * video_length - 1
                )

                if self.pos_encoder is not None:
                    hidden_states = self.pos_encoder(hidden_states)

                encoder_hidden_states = (
                    repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                    if encoder_hidden_states is not None
                    else encoder_hidden_states
                )
            else:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states[:, ::2, :])
            dim = query.shape[-1]
            query = self.reshape_heads_to_batch_dim(query)

            if self.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(
                        attention_mask, (0, target_length), value=0.0
                    )
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(
                    query, key, value, attention_mask
                )
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask)
                else:
                    hidden_states = self._sliced_attention(
                        query, key, value, sequence_length, dim, attention_mask
                    )

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if self.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states
