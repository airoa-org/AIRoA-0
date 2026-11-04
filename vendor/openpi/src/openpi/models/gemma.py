# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


_ATTENTION_DIAGNOSTIC_BASE_NAMES = (
    "biased_query_head_count",
    "visual_mass_without_bias",
    "visual_mass_with_bias",
    "visual_mass_ratio",
    "low_prior_mass_without_bias",
    "low_prior_mass_with_bias",
    "low_prior_mass_ratio",
    "mid_prior_mass_without_bias",
    "mid_prior_mass_with_bias",
    "mid_prior_mass_ratio",
    "high_prior_mass_without_bias",
    "high_prior_mass_with_bias",
    "high_prior_mass_ratio",
    "unaffected_mass_without_bias",
    "unaffected_mass_with_bias",
    "total_variation",
    "kl_with_bias_vs_without_bias",
    "entropy_without_bias",
    "entropy_with_bias",
    "max_abs_probability_delta",
    "mean_abs_visual_logit_bias",
)
_ATTENTION_DIAGNOSTIC_NAMES = _ATTENTION_DIAGNOSTIC_BASE_NAMES + tuple(
    f"mask_decile_{decile}_{suffix}"
    for decile in range(10)
    for suffix in ("mass_without_bias", "mass_with_bias", "mass_ratio")
)
_attention_diagnostics_state = {"enabled": False}
_attention_diagnostic_records: list[dict[str, float]] = []
_attention_diagnostic_lock = threading.Lock()


def set_attention_diagnostics_enabled(*, enabled: bool) -> None:
    """Enable the opt-in post-softmax audit before tracing a model forward pass."""

    _attention_diagnostics_state["enabled"] = bool(enabled)


def reset_attention_diagnostic_records() -> None:
    with _attention_diagnostic_lock:
        _attention_diagnostic_records.clear()


def attention_diagnostic_records() -> tuple[dict[str, float], ...]:
    with _attention_diagnostic_lock:
        return tuple(dict(record) for record in _attention_diagnostic_records)


def _record_attention_diagnostics(values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size != len(_ATTENTION_DIAGNOSTIC_NAMES):
        raise ValueError(
            f"Attention diagnostic callback received {flat.size} values; expected {len(_ATTENTION_DIAGNOSTIC_NAMES)}."
        )
    record = {
        name: float(value)
        for name, value in zip(_ATTENTION_DIAGNOSTIC_NAMES, flat, strict=True)
    }
    with _attention_diagnostic_lock:
        record["callback_ordinal"] = float(len(_attention_diagnostic_records))
        _attention_diagnostic_records.append(record)


def _post_softmax_attention_diagnostics(
    probs_with_bias: jax.Array,
    probs_without_bias: jax.Array,
    attn_mask: jax.Array,
    attn_logit_bias: jax.Array,
) -> jax.Array:
    """Summarize the causal effect of an additive key prior after softmax.

    The returned statistics average over only query/head rows that can see at
    least one negatively biased key. Mask-score deciles are ordered from the
    most-negative prior (decile 0) to the least-negative prior (decile 9).
    """

    probs_with_bias = jnp.asarray(probs_with_bias, dtype=jnp.float32)
    probs_without_bias = jnp.asarray(probs_without_bias, dtype=jnp.float32)
    if probs_with_bias.shape != probs_without_bias.shape or probs_with_bias.ndim != 5:
        raise ValueError(
            "Attention diagnostic probabilities must share [B,K,G,T,S] shape, got "
            f"{probs_with_bias.shape} and {probs_without_bias.shape}."
        )
    if attn_mask.shape != (
        probs_with_bias.shape[0],
        1,
        probs_with_bias.shape[3],
        probs_with_bias.shape[4],
    ):
        raise ValueError(
            f"Attention diagnostic mask shape mismatch: {attn_mask.shape}."
        )
    if attn_logit_bias.shape != attn_mask.shape:
        raise ValueError(
            f"Attention diagnostic bias shape {attn_logit_bias.shape} does not match mask {attn_mask.shape}."
        )

    allowed = jnp.asarray(attn_mask, dtype=jnp.bool_)[:, :, None, :, :]
    bias = jnp.asarray(attn_logit_bias, dtype=jnp.float32)[:, :, None, :, :]
    visual = jnp.logical_and(allowed, bias < -1e-7)
    query_rows = jnp.any(visual, axis=-1)
    query_rows = jnp.broadcast_to(query_rows, probs_with_bias.shape[:-1])
    query_count = jnp.sum(query_rows.astype(jnp.float32))
    safe_query_count = jnp.maximum(query_count, 1.0)

    def row_mean(values: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(query_rows, values, 0.0)) / safe_query_count

    def key_mass(probs: jax.Array, key_mask: jax.Array) -> jax.Array:
        return row_mean(jnp.sum(jnp.where(key_mask, probs, 0.0), axis=-1))

    def safe_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
        return numerator / jnp.maximum(denominator, 1e-12)

    low_prior = jnp.logical_and(visual, bias <= -3.0)
    mid_prior = jnp.logical_and(visual, jnp.logical_and(bias > -3.0, bias <= -1.0))
    high_prior = jnp.logical_and(visual, bias > -1.0)
    unaffected = jnp.logical_and(allowed, jnp.abs(bias) <= 1e-7)

    visual_without = key_mass(probs_without_bias, visual)
    visual_with = key_mass(probs_with_bias, visual)
    low_without = key_mass(probs_without_bias, low_prior)
    low_with = key_mass(probs_with_bias, low_prior)
    mid_without = key_mass(probs_without_bias, mid_prior)
    mid_with = key_mass(probs_with_bias, mid_prior)
    high_without = key_mass(probs_without_bias, high_prior)
    high_with = key_mass(probs_with_bias, high_prior)
    unaffected_without = key_mass(probs_without_bias, unaffected)
    unaffected_with = key_mass(probs_with_bias, unaffected)

    eps = jnp.asarray(1e-12, dtype=jnp.float32)
    allowed_f = allowed.astype(jnp.float32)
    p = jnp.clip(probs_with_bias, eps, 1.0)
    q = jnp.clip(probs_without_bias, eps, 1.0)
    total_variation = row_mean(
        0.5 * jnp.sum(jnp.abs(probs_with_bias - probs_without_bias), axis=-1)
    )
    kl = row_mean(
        jnp.sum(allowed_f * probs_with_bias * (jnp.log(p) - jnp.log(q)), axis=-1)
    )
    entropy_without = row_mean(
        -jnp.sum(allowed_f * probs_without_bias * jnp.log(q), axis=-1)
    )
    entropy_with = row_mean(-jnp.sum(allowed_f * probs_with_bias * jnp.log(p), axis=-1))
    max_abs_delta = jnp.max(
        jnp.where(
            query_rows[..., None], jnp.abs(probs_with_bias - probs_without_bias), 0.0
        )
    )
    visual_count = jnp.sum(visual.astype(jnp.float32))
    mean_abs_bias = jnp.sum(jnp.where(visual, jnp.abs(bias), 0.0)) / jnp.maximum(
        visual_count, 1.0
    )

    base_values = [
        query_count,
        visual_without,
        visual_with,
        safe_ratio(visual_with, visual_without),
        low_without,
        low_with,
        safe_ratio(low_with, low_without),
        mid_without,
        mid_with,
        safe_ratio(mid_with, mid_without),
        high_without,
        high_with,
        safe_ratio(high_with, high_without),
        unaffected_without,
        unaffected_with,
        total_variation,
        kl,
        entropy_without,
        entropy_with,
        max_abs_delta,
        mean_abs_bias,
    ]

    # Bias is zero on nonvisual keys and strictly negative on visual keys, so
    # ascending bias order ranks only visual keys before unaffected keys.
    order = jnp.argsort(bias, axis=-1)
    rank = jnp.argsort(order, axis=-1)
    visual_keys_per_row = jnp.sum(visual.astype(jnp.int32), axis=-1, keepdims=True)
    decile_index = jnp.minimum(
        (10 * rank) // jnp.maximum(visual_keys_per_row, 1),
        9,
    )
    decile_values = []
    for decile in range(10):
        decile_mask = jnp.logical_and(visual, decile_index == decile)
        mass_without = key_mass(probs_without_bias, decile_mask)
        mass_with = key_mass(probs_with_bias, decile_mask)
        decile_values.extend(
            (mass_without, mass_with, safe_ratio(mass_with, mass_without))
        )

    return jnp.stack((*base_values, *decile_values), axis=0)


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=16, alpha=16.0),
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0),
            },
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(
            jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True
        )  # compute variance in float32
        normed_inputs = jnp.asarray(
            x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        )  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(
            x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype
        )(cond)
        # Legacy sample-wide conditioning retains its exact broadcast path.
        # TT-RTC supplies a separate local flow time for each action token.
        if cond.ndim == 2:
            modulation = modulation[:, None, :]
        elif cond.ndim != 3 or cond.shape[:2] != x.shape[:2]:
            raise ValueError("Adaptive RMSNorm condition must be [B,D] or [B,T,D]")
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = (
            normed_inputs * (1 + scale) + shift
        )  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, attn_logit_bias, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(
            config.head_dim == self.configs[0].head_dim for config in self.configs
        )
        assert all(
            config.num_heads == self.configs[0].num_heads for config in self.configs
        )
        assert all(
            config.num_kv_heads == self.configs[0].num_kv_heads
            for config in self.configs
        )

        dtype = next(
            x.dtype for x in xs if x is not None
        )  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0, 1)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0,)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(
                        in_axis=-2, out_axis=-1, batch_axis=(0, 1)
                    ),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(
            q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads
        )
        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )
        content_logits = logits

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )
        if attn_logit_bias is not None:
            if attn_logit_bias.shape != attn_mask.shape:
                raise ValueError(
                    "Attention logit bias with shape "
                    f"{attn_logit_bias.shape} but expected {attn_mask.shape} for q and k shapes {q.shape} and {k.shape}"
                )
            # Attention logits are accumulated in float32. Keep the additive
            # prior in the same dtype so very small mask probabilities remain
            # finite and lambda=0 is an exact no-op at this interface.
            logits = logits + attn_logit_bias[:, :, None, :, :].astype(jnp.float32)

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        if _attention_diagnostics_state["enabled"] and attn_logit_bias is not None:
            unbiased_masked_logits = jnp.where(
                attn_mask[:, :, None, :, :], content_logits, big_neg
            )
            probs_without_bias = jax.nn.softmax(unbiased_masked_logits, axis=-1).astype(
                dtype
            )
            diagnostics = _post_softmax_attention_diagnostics(
                probs,
                probs_without_bias,
                attn_mask,
                attn_logit_bias,
            )
            jax.debug.callback(_record_attention_diagnostics, diagnostics, ordered=True)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        attn_logit_bias,
        adarms_cond,
        deterministic=True,  # noqa: FBT002
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = (
            nn.Dropout(self.dropout, self.dropout_bdims)
            if self.dropout
            else lambda x, _: x
        )

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(
            pre_attn, positions, attn_mask, attn_logit_bias, kv_cache
        )
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]
]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(6,),  # Excludes self; arg 6 is deterministic.
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=logit_bias, 4=adarms_cond, 5=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def decode(self, hidden):
        return self.embedder.decode(hidden)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[
            at.Float[at.Array, "b _d"] | at.Float[at.Array, "b _t _d"] | None
        ]
        | None = None,
        *,
        attn_logit_bias: at.Float[at.Array, "b t s"] | None = None,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if attn_logit_bias is not None:
            attn_logit_bias = jnp.asarray(attn_logit_bias, dtype=jnp.float32)[
                :, None, :, :
            ]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            attn_logit_bias,
            adarms_cond,
            deterministic,
        )

        assert all(
            e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None
        )

        return [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(
        x.shape[-1] // 2, dtype=jnp.float32
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
