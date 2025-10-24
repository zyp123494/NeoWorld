from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.spatial import patchify, unpatchify
from ..modules.transformer import (
    AbsolutePositionEmbedder,
    ModulatedTransformerCrossBlockMaskAsCondWeighted,
)
from ..modules.utils import convert_module_to_f16, convert_module_to_f32


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModelMaskAsCondWeighted(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        mask_cond_type="mask_patch",
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(
                *[
                    torch.arange(res, device=self.device)
                    for res in [resolution // patch_size] * 3
                ],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)

        # Pose processing layers
        self.pose_input_mlp = nn.Sequential(
            nn.Linear(6, model_channels // 2),
            nn.SiLU(),
            nn.Linear(model_channels // 2, model_channels),
        )

        self.pose_output_mlp = nn.Sequential(
            nn.Linear(model_channels, model_channels // 2),
            nn.SiLU(),
            nn.Linear(model_channels // 2, 6),
        )

        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlockMaskAsCondWeighted(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

        # Initialize pose MLPs:
        for module in self.pose_input_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Zero-out pose output layer:
        nn.init.constant_(self.pose_output_mlp[-1].weight, 0)
        nn.init.constant_(self.pose_output_mlp[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        pose: torch.Tensor = None,
    ) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], (
            f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        )

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]

        # Add pose token if provided
        if pose is not None:
            pose_token = self.pose_input_mlp(pose).unsqueeze(
                1
            )  # [B, 1, model_channels]
            h = torch.cat([h, pose_token], dim=1)  # Concatenate pose token to sequence

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)

        cond_split = {}
        assert cond.shape[1] == 1374 + 37 * 37 + 37 * 37, "cond shape mismatch"
        cond_split["cond"] = cond[:, :1374, :].type(self.dtype)
        cond_split["mask"] = cond[:, 1374 : 1374 + 37 * 37, 0:1].type(self.dtype)
        cond_split["mask_occ"] = cond[:, 1374 + 37 * 37 :, :].type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond_split)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])

        # Separate pose token and main tokens
        if pose is not None:
            main_tokens = h[:, :-1, :]  # All tokens except the last (pose token)
            pose_token = h[:, -1, :]  # Last token (pose token)

            # Process main tokens
            main_output = self.out_layer(main_tokens)
            main_output = main_output.permute(0, 2, 1).view(
                main_output.shape[0],
                main_output.shape[2],
                *[self.resolution // self.patch_size] * 3,
            )
            main_output = unpatchify(main_output, self.patch_size).contiguous()

            # Process pose token
            pose_output = self.pose_output_mlp(pose_token)

            return main_output, pose_output
        else:
            # Original behavior when no pose token
            h = self.out_layer(h)
            h = h.permute(0, 2, 1).view(
                h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3
            )
            h = unpatchify(h, self.patch_size).contiguous()
            return h
