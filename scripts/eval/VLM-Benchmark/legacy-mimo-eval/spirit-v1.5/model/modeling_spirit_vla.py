# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

"""
SpiritVLA — reference implementation.

Developed and open-sourced by Spirit AI Team.
See the `LICENSE` for details.
"""

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
from PIL import Image
from safetensors.torch import load_file as safe_load_file
from torch import nn
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
)

from utils import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    build_norm_state,
    get_rope_index_3,
    pad_and_cat,
    pad_vector,
    preprocess_qwen_visual,
    sample_noise,
    sample_time,
    no_stats_error_str,
    get_user_prompt,
)

try:
    # transformers >= 4.57.0
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "Qwen3VLForConditionalGeneration is not available in your transformers version. "
        "Please upgrade transformers to >=4.57.0."
    ) from e

# ----------------------------------------------------------------------------------------------------------------------
# Public constants (batch keys)
# ----------------------------------------------------------------------------------------------------------------------

OBS_ROBOT = "observation.state"
ACTION = "action"

# ----------------------------------------------------------------------------------------------------------------------
# Normalization (buffered stats + (un)normalize modules)
# ----------------------------------------------------------------------------------------------------------------------


class Normalize(nn.Module):
    def __init__(
        self,
        features: Dict[str, PolicyFeature],
        norm_map: Dict[str, NormalizationMode],
        stats: Optional[Dict[str, dict[str, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map, stats_buffers = build_norm_state(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue
            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue
            if norm_mode is not NormalizationMode.MIN_MAX:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            min_v = buffer["min"]
            max_v = buffer["max"]
            assert not torch.isinf(min_v).any(), no_stats_error_str("min")
            assert not torch.isinf(max_v).any(), no_stats_error_str("max")
            batch[key] = (batch[key] - min_v) / (max_v - min_v + 1e-8)
            batch[key] = batch[key] * 2 - 1
        return batch


class Unnormalize(nn.Module):
    def __init__(
        self,
        features: Dict[str, PolicyFeature],
        norm_map: Dict[str, NormalizationMode],
        stats: Optional[dict[str, dict[str, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.features = features
        self.norm_map, stats_buffers = build_norm_state(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            if key not in batch:
                continue
            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue
            if norm_mode is not NormalizationMode.MIN_MAX:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            min_v = buffer["min"]
            max_v = buffer["max"]
            assert not torch.isinf(min_v).any(), no_stats_error_str("min")
            assert not torch.isinf(max_v).any(), no_stats_error_str("max")
            batch[key] = (batch[key] + 1) / 2
            batch[key] = batch[key] * (max_v - min_v) + min_v
        return batch


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class SpiritVLAConfig:
    backbone: str = "Qwen/Qwen3-VL-4B-Instruct"
    attention_implementation: str = "eager"
    dit_hidden_size: int = 1024
    dit_num_heads: int = 8
    dit_num_layers: int = 18
    dit_interleave_self_attention: bool = False
    dit_cross_attention_dim: Optional[int] = None
    num_noise_per_sample: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    device: str = "cuda"
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    proj_width: int = 1024
    num_steps: int = 10
    action_dim: int = 14
    action_horizon: int = 60
    dit_dropout: float = 0.0

    def __post_init__(self):
        self.normalization_mapping = {
            k: (v if isinstance(v, NormalizationMode) else NormalizationMode(v))
            for k, v in self.normalization_mapping.items()
        }
        self.input_features = self._convert_features(self.input_features)
        self.output_features = self._convert_features(self.output_features)
        self._ensure_default_features()

    @staticmethod
    def _convert_features(features_dict: dict[str, Any]) -> dict[str, PolicyFeature]:
        out: dict[str, PolicyFeature] = {}
        for k, v in features_dict.items():
            if isinstance(v, PolicyFeature):
                out[k] = v
            else:
                out[k] = PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        return out

    @property
    def action_feature(self):
        if "action" in self.output_features:
            shape = self.output_features["action"].shape
            return torch.zeros(shape)
        return torch.zeros((self.max_action_dim,))

    def _ensure_default_features(self) -> None:
        defaults = {
            "observation.images.cam_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
            "observation.images.cam_left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
            "observation.images.cam_right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        }
        for k, v in defaults.items():
            self.input_features.setdefault(k, v)
        if not self.output_features:
            self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))}


# ----------------------------------------------------------------------------------------------------------------------
# Diffusion Transformer (DiT)
# ----------------------------------------------------------------------------------------------------------------------


class BaseDiTConfig(PretrainedConfig):
    model_type = "BaseDiT"

    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        num_layers: int = 12,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        positional_embeddings: str  = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_vlm_last_embd = 1
        self.param_dict = {
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "num_layers": num_layers,
            "attention_bias": attention_bias,
            "activation_fn": activation_fn,
            "num_embeds_ada_norm": num_embeds_ada_norm,
            "upcast_attention": upcast_attention,
            "norm_type": norm_type,
            "norm_elementwise_affine": norm_elementwise_affine,
            "norm_eps": norm_eps,
            "max_num_positional_embeddings": max_num_positional_embeddings,
            "compute_dtype": compute_dtype,
            "positional_embeddings": positional_embeddings,
            "interleave_self_attention": interleave_self_attention,
            "cross_attention_dim": cross_attention_dim,
            "dropout": dropout,
        }


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        positional_embeddings: Optional[str] =  None,
        num_positional_embeddings:Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type
        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.attn_final_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        attn_output = self.attn_final_dropout(attn_output)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class BaseDiT(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 12,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        positional_embeddings: str = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=self.config.compute_dtype)
        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    cross_attention_dim=curr_cross_attention_dim,
                    dropout=self.config.dropout,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class SpiritVLAPolicy(nn.Module):
    """Spirit VLA policy model - main entry point for training and inference.

    Input:
    - `batch` is a dict containing images, `robot_type`, task string, and state/action tensors.
    Inference Output:
    - unnormalized action trajectories (shape: B x T x action_dim).
    Training Output:
    - loss dict containing flow matching mse.
    """

    config_class = SpiritVLAConfig
    name = "spirit_vla"

    def __init__(
        self,
        config: SpiritVLAConfig,
    ):
        super().__init__()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, None)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, None)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, None)
        assert self.config.backbone is not None, "Specify a backbone."
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            self.config.backbone,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False,
        )
        # Backbone + DiT head (flattened; state_dict prefix: model.*)
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            config.backbone,
            attn_implementation=config.attention_implementation,
            dtype=torch.float32,
        )
        self.image_processor = AutoProcessor.from_pretrained(self.config.backbone).image_processor

        dit_config = BaseDiTConfig(
            num_attention_heads=config.dit_num_heads,
            attention_head_dim=config.dit_hidden_size // config.dit_num_heads,
            num_layers=config.dit_num_layers,
            interleave_self_attention=config.dit_interleave_self_attention,
            cross_attention_dim=config.dit_cross_attention_dim,
            dropout=config.dit_dropout,
        )
        self.dit = BaseDiT(**dit_config.param_dict)
        self.num_vlm_last_embd = dit_config.num_vlm_last_embd

        vlm_hidden_size = self.qwen.config.text_config.hidden_size
        self.proj_vlm_output = (
            nn.Identity()
            if vlm_hidden_size == config.dit_hidden_size or config.dit_cross_attention_dim is not None
            else nn.Linear(vlm_hidden_size, config.dit_hidden_size)
        )
        self.state_proj = nn.Linear(self.config.max_state_dim, config.dit_hidden_size)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, config.dit_hidden_size)
        self.action_out_proj = nn.Linear(config.dit_hidden_size, self.config.max_action_dim)
        
        self.num_noise_per_sample = self.config.num_noise_per_sample
        self.vlm_hidden_size = vlm_hidden_size
        self.dit_hidden_size = config.dit_hidden_size
        self.config.proj_width = self.qwen.language_model.embed_tokens.embedding_dim

    # ----------------------------- Backbone helpers -----------------------------
    def _encode_vision(self, batch: dict) -> torch.Tensor:
        device = batch[OBS_ROBOT].device
        pixel_values_list, image_grid_thw_list, input_ids_list, position_ids_list = self.preprocess_rb_batch(batch)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(device).contiguous()
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0).to(device).contiguous()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True,
            padding_value=self.language_tokenizer.pad_token_id
        ).to(device).contiguous()
        model_max_length = self.language_tokenizer.model_max_length
        input_ids = input_ids[:, :model_max_length]
        position_ids = pad_and_cat(position_ids_list)[:, :, :model_max_length].to(device).contiguous()
        attention_mask = input_ids.ne(self.language_tokenizer.pad_token_id).to(device)
        
        vlm_outputs = self.qwen.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            labels=None,
            output_hidden_states=True,
        )

        num_vlm_last_embd = min(self.num_vlm_last_embd, len(vlm_outputs.hidden_states))
        vlm_last_embed = torch.cat(vlm_outputs.hidden_states[-num_vlm_last_embd:], dim=1)
        return vlm_last_embed

    # ----------------------------- Flow matching core -----------------------------
    def _embed_suffix(self, state, noisy_actions, mask_state=True):
        """Embed state and noisy actions. mask_state=False for training, True for inference."""
        embs = []
        pad_masks = []
        att_masks = []
        if mask_state:
            state[:, :, [2, 9]] = 0
        state_emb = self.state_proj(state)
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        nstates = state_emb.shape[1]
        device = state_emb.device
        state_mask = torch.ones(bsize, nstates, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] + [0] * (nstates - 1)
        action_emb = self.action_in_proj(noisy_actions)
        embs.append(action_emb)
        bsize, action_dim = action_emb.shape[:2]
        action_mask = torch.ones(bsize, action_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_mask)
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def _denoise_step(
        self,
        state,
        vlm_last_embed,
        x_t,
        timestep,
    ):
        suffix_embs, _, _ = self._embed_suffix(state, x_t, mask_state=False)
        suffix_out = self.dit(
            hidden_states=suffix_embs,
            encoder_hidden_states=vlm_last_embed,
            timestep=timestep,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def _sample_actions_unified(
        self,
        state,
        vlm_last_embed,
        noise=None,
    ) -> torch.Tensor:
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = sample_noise(actions_shape, device)
        vlm_last_embed = self.proj_vlm_output(vlm_last_embed)
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step(
                state,
                vlm_last_embed,
                x_t,
                expanded_time,
            )
            x_t += dt * v_t
            time += dt
        return x_t

    # ----------------------------- Data preparation helpers -----------------------------
    def preprocess_rb_batch(self, batch):
        image_keys = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        num_images = len(image_keys)
        ref_key = None
        for key in image_keys:
            if key in batch:
                ref_key = key
                break
        if ref_key is None:
            raise ValueError("No images found in batch.")
        bz = batch[ref_key].shape[0]
        device = batch[ref_key].device
        ref_shape = batch[ref_key].shape
        all_images = []
        for key in image_keys:
            if key in batch:
                all_images.append(batch[key])
            else:
                placeholder = torch.zeros(
                    ref_shape[0],
                    ref_shape[1],
                    ref_shape[2],
                    ref_shape[3],
                    device=device,
                    dtype=batch[ref_key].dtype,
                )
                all_images.append(placeholder)
        pixel_values_list = []
        image_grid_thw_list = []
        input_ids_list = []
        position_ids_list = []
        merge_size = self.image_processor.merge_size
        for i in range(bz):
            sample_pixel_values = []
            sample_grid_thws = []
            for img_batch in all_images:
                img_torch = img_batch[i]
                img_np = (img_torch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                processed_result = self.image_processor.preprocess(img_pil, return_tensors="pt")
                sample_pixel_values.append(processed_result["pixel_values"].to(device).contiguous())
                sample_grid_thws.append(processed_result["image_grid_thw"][0].to(device))
            sample_pixel_values_cat = torch.cat(sample_pixel_values, dim=0)
            pixel_values_list.append(sample_pixel_values_cat)
            sample_grid_thw_stacked = torch.stack(sample_grid_thws, dim=0)
            image_grid_thw_list.append(sample_grid_thw_stacked)
            grid_thw_merged = [thw.prod().item() // (merge_size**2) for thw in sample_grid_thws]
            task = batch["task"][i]
            image_placeholders = " ".join(["<image>"] * num_images)
            robot_type = batch["robot_type"][i]
            user_prompt = get_user_prompt(image_placeholders, robot_type)
            input_id_dict = preprocess_qwen_visual(
                [
                    [
                        {"from": "human", "value": user_prompt},
                        {"from": "gpt", "value": task},
                    ]
                ],
                self.language_tokenizer,
                grid_thw_image=grid_thw_merged,
            )
            input_ids = input_id_dict["input_ids"]
            input_ids_list.append(input_ids.squeeze(0))
            position_ids, _ = get_rope_index_3(
                merge_size,
                input_ids,
                image_grid_thw=sample_grid_thw_stacked,
            )
            position_ids_list.append(position_ids)
        return pixel_values_list, image_grid_thw_list, input_ids_list, position_ids_list

    def prepare_state(self, batch):
        state = batch[OBS_ROBOT]
        if state.ndim == 2:
            state = state[:, None, :]
        state = pad_vector(state, self.config.max_state_dim)
        return state
    
    def prepare_actions(self, batch):
        actions = batch[ACTION]
        actions = pad_vector(actions, self.config.max_action_dim)
        action_mask = pad_vector(
            batch["action_mask"].float(), self.config.max_action_dim
        ).bool()
        return actions, action_mask

    # ----------------------------- Training -----------------------------
    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        return self.compute_loss(batch)

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """model training

        Args:
            batch: {
                "observation.images.cam_high": [B, 3, H, W],
                "observation.images.cam_left_wrist": [B, 3, H, W],
                "observation.images.cam_right_wrist": [B, 3, H, W],
                "observation.state": [B, 1, action_dim] or [B, action_dim],
                "action": [B, action_horizon, action_dim],
                "action_mask": [B, action_horizon, action_dim],  # bool
                "action_is_pad": [B, action_horizon],            # bool
                "task": List[str],
                "robot_type": List[str],
            }

        Returns:
            loss: float
            log_dict: {"loss/flow_mse": float, "loss/total": float}
        """
        device = batch[OBS_ROBOT].device
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        state = self.prepare_state(batch)
        actions, action_mask = self.prepare_actions(batch)

        vlm_last_embed = self._encode_vision(batch)
        vlm_last_embed = self.proj_vlm_output(vlm_last_embed)

        bsize = actions.shape[0]
        noise = sample_noise(actions.shape, device)
        t = sample_time(bsize, device)
        t_expand = t.view(-1, 1, 1)
        noisy_actions = actions + t_expand * (noise - actions)

        suffix_embs, _, _ = self._embed_suffix(state, noisy_actions, mask_state=False)

        v_t = self.dit(
            hidden_states=suffix_embs,
            encoder_hidden_states=vlm_last_embed,
            timestep=t,
        )
        v_t = v_t[:, -self.config.n_action_steps:, :]
        v_t = v_t.to(dtype=torch.float32)
        v_t = self.action_out_proj(v_t)
        u_t = noise - actions
        flow_mse = F.mse_loss(v_t, u_t, reduction="none")

        if "action_is_pad" in batch:
            in_episode = (~batch["action_is_pad"]).float()
            flow_mse = flow_mse * in_episode.unsqueeze(-1)

        valid_mask = action_mask.float()
        num_valid = valid_mask.sum().clamp_min(1.0)
        loss = (flow_mse * valid_mask).sum() / num_valid

        log_dict = {
            "loss/flow_mse": loss.item(),
            "loss/total": loss.item(),
        }
        return loss, log_dict

    # ----------------------------- Inference -----------------------------
    def select_action(
        self,
        batch: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        state = self.prepare_state(batch)

        vlm_last_embed = self._encode_vision(batch)
        actions = self._sample_actions_unified(
            state,
            vlm_last_embed,
            noise=noise,
        )
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        actions = self.unnormalize_outputs({"action": actions})["action"]
        return actions

    # ------------------------------------------------------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, ckpt_path: str, strict: bool = True, train: bool = False):
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
        else:
            raise FileNotFoundError(f"No config.json found in {ckpt_path}")

        def _filter(cfg_cls, data):
            valid_keys = {f.name for f in fields(cfg_cls)}
            return {k: v for k, v in data.items() if k in valid_keys}

        filtered_cfg = _filter(SpiritVLAConfig, cfg_dict)
        config = SpiritVLAConfig(**filtered_cfg)
        model = cls(config)
        weight_path = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"model.safetensors not found in {ckpt_path}")
        # If user requests CUDA and it's available, load weights directly on GPU to avoid an extra copy.
        load_device = "cpu"
        if not train:
            try:
                if isinstance(config.device, str) and config.device.startswith("cuda") and torch.cuda.is_available():
                    load_device = config.device
            except Exception:
                load_device = "cpu"
        # Auto-enable FA2 if available for better memory efficiency and training speed
        if train:
            try:
                from flash_attn import flash_attn_func
                if config.attention_implementation == "eager":
                    print("FA2 is available, enabling FA2 automatically")
                config.attention_implementation = "flash_attention_2"
            except ImportError:
                if "flash_attention" in config.attention_implementation:
                    raise RuntimeError("[ERROR]: Current environment does not support Flash_Attention; please check the config.json configuration!")
        
        state_dict = safe_load_file(weight_path, device=load_device)
        model.load_state_dict(state_dict, strict=strict)
        return model
