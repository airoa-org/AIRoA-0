# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
import dataclasses
import logging
import math
import os
from typing import Literal
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.training.sharding as sharding

logger = logging.getLogger("openpi")
_SIGMOID_STRENGTH_EPS = 0.0001
_MEAN_PRESERVING_GATE_EPS = 1e-06
_BUDGET_S2_BISECTION_STEPS = 25


def _train_prefix_kv_cache_enabled() -> bool:
    value = os.environ.get("OPENPI_TRAIN_PREFIX_KV_CACHE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.asarray(mask_ar)
    if mask_ar.ndim == 1:
        cumsum = jnp.cumsum(mask_ar, axis=0)
        attn_mask = cumsum[None, None, :] <= cumsum[None, :, None]
    elif mask_ar.ndim == 2 and mask_ar.shape[0] == 1:
        cumsum = jnp.cumsum(mask_ar[0], axis=0)
        attn_mask = cumsum[None, None, :] <= cumsum[None, :, None]
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        cumsum = jnp.cumsum(mask_ar, axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def _one_sided_mask_budget_projection(
    logits: jax.Array,
    valid_mask: jax.Array,
    *,
    tau: float | jax.Array,
    keep_ratio: float,
    iterations: int = _BUDGET_S2_BISECTION_STEPS,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Cap each sample's mean soft mask with a non-negative logit shift.

    The shift is exactly zero for an already-under-budget sample. Otherwise a
    fixed-iteration FP32 bisection finds ``nu`` such that the valid-token mean
    of ``sigmoid((logits - nu) / tau)`` equals ``keep_ratio``. This is a soft
    gate constraint, not a hard token-count or compute constraint.
    """
    logits = jnp.asarray(logits, dtype=jnp.float32)
    valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if logits.shape != valid_mask.shape:
        raise ValueError(
            f"Budget-S2 logits/mask shape mismatch: {logits.shape} != {valid_mask.shape}."
        )
    if logits.ndim != 2:
        raise ValueError(
            f"Budget-S2 expects rank-2 [batch, token] logits, got {logits.shape}."
        )
    if not 0.0 < float(keep_ratio) < 1.0:
        raise ValueError(f"Budget-S2 keep_ratio must be in (0, 1), got {keep_ratio}.")
    if iterations <= 0:
        raise ValueError(
            f"Budget-S2 bisection iterations must be positive, got {iterations}."
        )
    tau = jnp.asarray(tau, dtype=jnp.float32)
    ratio = jnp.asarray(keep_ratio, dtype=jnp.float32)
    valid_f = valid_mask.astype(jnp.float32)
    valid_count = jnp.sum(valid_f, axis=-1)
    safe_logits = jnp.where(valid_mask, logits, jnp.asarray(0.0, dtype=jnp.float32))

    def _probability_and_mean(nu: jax.Array) -> tuple[jax.Array, jax.Array]:
        probability = jax.nn.sigmoid((safe_logits - nu[:, None]) / tau)
        probability = jnp.where(
            valid_mask, probability, jnp.asarray(0.0, dtype=jnp.float32)
        )
        mean = jnp.sum(probability, axis=-1) / jnp.maximum(valid_count, 1.0)
        return (probability, mean)

    zero_shift = jnp.zeros((logits.shape[0],), dtype=jnp.float32)
    (raw_probability, raw_mean) = _probability_and_mean(zero_shift)
    binding = jnp.logical_and(valid_count > 0.0, raw_mean > ratio)
    max_valid_logit = jnp.max(
        jnp.where(valid_mask, logits, jnp.asarray(-jnp.inf, dtype=jnp.float32)), axis=-1
    )
    max_valid_logit = jnp.where(
        valid_count > 0.0, max_valid_logit, jnp.asarray(0.0, dtype=jnp.float32)
    )
    ratio_logit = jnp.log(ratio) - jnp.log1p(-ratio)
    upper = jnp.maximum(max_valid_logit - tau * ratio_logit, 0.0)
    lower = jnp.zeros_like(upper)

    def _bisect(_, bounds):
        (lower_bound, upper_bound) = bounds
        midpoint = (lower_bound + upper_bound) * jnp.asarray(0.5, dtype=jnp.float32)
        (_, midpoint_mean) = _probability_and_mean(midpoint)
        midpoint_is_too_dense = midpoint_mean > ratio
        next_lower = jnp.where(midpoint_is_too_dense, midpoint, lower_bound)
        next_upper = jnp.where(midpoint_is_too_dense, upper_bound, midpoint)
        return (
            jnp.where(binding, next_lower, zero_shift),
            jnp.where(binding, next_upper, zero_shift),
        )

    (_, upper) = jax.lax.fori_loop(0, int(iterations), _bisect, (lower, upper))
    nu = jnp.where(binding, upper, zero_shift)
    (projected_probability, projected_mean) = _probability_and_mean(nu)
    return (projected_probability, nu, raw_mean, projected_mean, binding)


def _budget_s2_alpha(train_step: jax.Array | int, *, warmup_steps: int) -> jax.Array:
    """Deterministic zero-to-one Budget-S2 strength schedule."""
    if warmup_steps <= 0:
        raise ValueError(
            f"Budget-S2 alpha warmup must be positive, got {warmup_steps}."
        )
    step = jnp.asarray(train_step, dtype=jnp.float32)
    return jnp.clip(step / jnp.asarray(warmup_steps, dtype=jnp.float32), 0.0, 1.0)


def _suffix_mpa_attention_bias(
    prefix_key_bias: jax.Array, suffix_len: int
) -> jax.Array:
    """Broadcast a prefix-key prior over suffix queries, leaving suffix keys unbiased."""
    prefix_key_bias = jnp.asarray(prefix_key_bias, dtype=jnp.float32)
    if prefix_key_bias.ndim != 2:
        raise ValueError(
            f"S2-MPA prefix key bias must have rank 2, got {prefix_key_bias.shape}."
        )
    suffix_zeros = jnp.zeros(
        (prefix_key_bias.shape[0], int(suffix_len)), dtype=jnp.float32
    )
    all_key_bias = jnp.concatenate([prefix_key_bias, suffix_zeros], axis=-1)
    return jnp.broadcast_to(
        all_key_bias[:, None, :],
        (prefix_key_bias.shape[0], int(suffix_len), all_key_bias.shape[-1]),
    )


def _monolithic_mpa_attention_bias(
    prefix_key_bias: jax.Array, suffix_len: int
) -> jax.Array:
    """Build full attention bias with nonzero entries only for suffix->prefix keys."""
    suffix_bias = _suffix_mpa_attention_bias(prefix_key_bias, suffix_len)
    prefix_query_zeros = jnp.zeros(
        (prefix_key_bias.shape[0], prefix_key_bias.shape[1], suffix_bias.shape[-1]),
        dtype=jnp.float32,
    )
    return jnp.concatenate([prefix_query_zeros, suffix_bias], axis=1)


def _action_dim_reduced_mse(
    prediction: at.ArrayLike,
    target: at.ArrayLike,
    action_dim_mask: at.ArrayLike | None = None,
    action_time_dim_valid_mask: at.ArrayLike | None = None,
) -> at.Float[at.Array, "*b ah"]:
    squared_error = jnp.square(prediction - target)
    if action_dim_mask is None and action_time_dim_valid_mask is None:
        return jnp.mean(squared_error, axis=-1)
    combined = jnp.ones(squared_error.shape, dtype=jnp.float32)
    if action_dim_mask is not None:
        action_dim_mask = jnp.asarray(action_dim_mask, dtype=jnp.float32)
        if action_dim_mask.shape[-1] != int(squared_error.shape[-1]):
            raise ValueError(
                f"action_dim_mask shape mismatch: expected last dim {squared_error.shape[-1]}, got {action_dim_mask.shape[-1]}"
            )
        combined = combined * action_dim_mask.reshape(
            (1,) * (squared_error.ndim - 1) + (int(squared_error.shape[-1]),)
        )
    if action_time_dim_valid_mask is not None:
        dynamic = jnp.asarray(action_time_dim_valid_mask, dtype=jnp.float32)
        if dynamic.shape[:-1] != squared_error.shape[:-1]:
            raise ValueError(
                f"action_time_dim_valid_mask prefix shape mismatch: expected {squared_error.shape[:-1]}, got {dynamic.shape[:-1]}."
            )
        if dynamic.shape[-1] > squared_error.shape[-1]:
            raise ValueError(
                f"action_time_dim_valid_mask has more action dims than the model: {dynamic.shape[-1]} > {squared_error.shape[-1]}."
            )
        dynamic = jnp.pad(
            dynamic,
            ((0, 0),) * (dynamic.ndim - 1)
            + ((0, squared_error.shape[-1] - dynamic.shape[-1]),),
        )
        combined = combined * dynamic
    masked_error = squared_error * combined
    denom = jnp.maximum(jnp.sum(combined, axis=-1), 1.0)
    return jnp.sum(masked_error, axis=-1) / denom


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class VisualBudgetGateHead(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        image_tokens: at.Float[at.Array, "b n d"],
        prompt_summary: at.Float[at.Array, "b d"],
        token_prior: at.Float[at.Array, "b n"],
    ) -> at.Float[at.Array, "b n"]:
        prompt_features = jnp.broadcast_to(
            prompt_summary[:, None, :], image_tokens.shape
        )
        features = jnp.concatenate(
            [
                image_tokens,
                prompt_features,
                image_tokens * prompt_features,
                token_prior[..., None].astype(image_tokens.dtype),
            ],
            axis=-1,
        )
        hidden = self.in_proj(features)
        hidden = nnx.swish(hidden)
        return self.out_proj(hidden)[..., 0]


class TemporalMemoryCompressor(nnx.Module):
    def __init__(
        self,
        token_dim: int,
        num_slots: int,
        rngs: nnx.Rngs,
        *,
        hidden_dim: int | None = None,
        context_dim: int | None = None,
        out_dim: int | None = None,
    ):
        hidden_dim = token_dim if hidden_dim is None else hidden_dim
        context_dim = token_dim if context_dim is None else context_dim
        out_dim = token_dim if out_dim is None else out_dim
        slot_rng = rngs.params()
        self.slot_embeddings = nnx.Param(
            jax.random.normal(slot_rng, (num_slots, hidden_dim), dtype=jnp.float32)
            * 0.02
        )
        self.context_proj = nnx.Linear(context_dim, hidden_dim, rngs=rngs)
        self.query_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.key_proj = nnx.Linear(token_dim, hidden_dim, rngs=rngs)
        self.value_proj = nnx.Linear(token_dim, hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(
        self,
        token_bank: at.Float[at.Array, "b n d"],
        token_mask: at.Bool[at.Array, "b n"],
        context: at.Float[at.Array, "b d"],
    ) -> at.Float[at.Array, "b s d"]:
        token_bank = sharding.batch_activation_sharding_constraint(token_bank)
        context = sharding.batch_activation_sharding_constraint(context)
        queries = (
            self.slot_embeddings.value[None, :, :]
            + self.context_proj(context)[:, None, :]
        )
        queries = sharding.batch_activation_sharding_constraint(queries)
        q = sharding.batch_activation_sharding_constraint(self.query_proj(queries))
        k = sharding.batch_activation_sharding_constraint(self.key_proj(token_bank))
        v = sharding.batch_activation_sharding_constraint(self.value_proj(token_bank))
        logits = jnp.einsum("bsd,bnd->bsn", q, k) / math.sqrt(q.shape[-1])
        large_neg = jnp.asarray(-1e9, dtype=logits.dtype)
        masked_logits = jnp.where(token_mask[:, None, :], logits, large_neg)
        attn = jax.nn.softmax(masked_logits, axis=-1)
        has_valid = jnp.any(token_mask, axis=-1, keepdims=True)
        attn = jnp.where(has_valid[:, :, None], attn, jnp.zeros_like(attn))
        pooled = jnp.einsum("bsn,bnd->bsd", attn, v)
        pooled = jnp.where(has_valid[:, :, None], pooled, jnp.zeros_like(pooled))
        pooled = sharding.batch_activation_sharding_constraint(pooled)
        return sharding.batch_activation_sharding_constraint(self.out_proj(pooled))


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.image_baked_padding_by_view = _model.normalize_image_baked_padding_by_view(
            getattr(config, "image_baked_padding_by_view", None)
        )
        self.base_view_names = tuple(config.base_view_names)
        self.wrist_view_names = tuple(config.wrist_view_names)
        self.image_view_names = tuple(config.image_view_names)
        self.tt_rtc_enabled = config.tt_rtc_enabled
        self.flash_enabled = config.flash_enabled
        self.flash_chunk_size = config.flash_chunk_size
        self.flash_buffer_size = config.flash_buffer_size
        self.flash_flow_times = config.flash_flow_times
        self.flash_flow_substeps = config.flash_flow_substeps
        self.pa_aux_loss_weight = config.pa_aux_loss_weight
        self.pa_aux_max_token_len = config.pa_aux_max_token_len
        self.pa_aux_logit_chunk_size = config.pa_aux_logit_chunk_size
        self.stage_num_classes = config.stage_num_classes
        self.stage_loss_weight = config.stage_loss_weight
        self.stage_pa_conditioning = config.stage_pa_conditioning
        self.stage_pa_token_ids = config.stage_pa_token_ids
        self.eval_photometric_enabled = config.eval_photometric_enabled
        if self.stage_pa_conditioning:
            if (
                config.stage_action_conditioning
                or config.fine_progress_enabled
                or len(config.stage_pa_token_ids) != config.stage_num_classes
            ):
                raise ValueError(
                    "PA conditioning requires one canonical PA per stage, no parallel stage/progress embedding"
                )
        self.use_temporal_proprio_memory = config.use_temporal_proprio_memory
        self.temporal_proprio_history_lags = config.temporal_proprio_history_lags
        self.temporal_proprio_use_delta = config.temporal_proprio_use_delta
        self.temporal_proprio_tokens_in_prefix = (
            config.temporal_proprio_tokens_in_prefix
        )
        self.temporal_proprio_prefix_token_scale = (
            config.temporal_proprio_prefix_token_scale
        )
        self.temporal_proprio_prefix_gate_warmup_steps = (
            config.temporal_proprio_prefix_gate_warmup_steps
        )
        self.proprio_frequency_enabled = config.proprio_frequency_enabled
        self.white_balance_strength = config.white_balance_strength
        self.white_balance_probability = config.white_balance_probability
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        temporal_hidden_dim = int(
            getattr(config, "temporal_hidden_dim", 0) or paligemma_config.width
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=getattr(config, "siglip_variant", "So400m/14"),
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(
            next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
        )
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(
            config.action_dim, action_expert_config.width, rngs=rngs
        )
        self.image_prior_floor = float(getattr(config, "image_prior_floor", 0.25))
        self.use_visual_budget_gate = bool(
            getattr(config, "use_visual_budget_gate", False)
        )
        self.visual_budget_gate_floor = float(
            getattr(config, "visual_budget_gate_floor", 0.1)
        )
        visual_budget_gate_floor_end = getattr(
            config, "visual_budget_gate_floor_end", None
        )
        self.visual_budget_gate_floor_end = (
            self.visual_budget_gate_floor
            if visual_budget_gate_floor_end is None
            else float(visual_budget_gate_floor_end)
        )
        self.visual_budget_gate_keep_ratio_base = float(
            getattr(config, "visual_budget_gate_keep_ratio_base", 0.4)
        )
        self.visual_budget_gate_keep_ratio_wrist = float(
            getattr(config, "visual_budget_gate_keep_ratio_wrist", 0.6)
        )
        self.visual_budget_gate_lambda_gated = float(
            getattr(config, "visual_budget_gate_lambda_gated", 1.0)
        )
        self.visual_budget_gate_lambda_budget_base = float(
            getattr(config, "visual_budget_gate_lambda_budget_base", 0.1)
        )
        self.visual_budget_gate_lambda_budget_wrist = float(
            getattr(config, "visual_budget_gate_lambda_budget_wrist", 0.1)
        )
        self.visual_budget_gate_lambda_anchor_start = float(
            getattr(config, "visual_budget_gate_lambda_anchor_start", 0.05)
        )
        self.visual_budget_gate_lambda_anchor_end = float(
            getattr(config, "visual_budget_gate_lambda_anchor_end", 0.0)
        )
        self.visual_budget_gate_prior_bias_start = float(
            getattr(config, "visual_budget_gate_prior_bias_start", 0.5)
        )
        self.visual_budget_gate_prior_bias_end = float(
            getattr(config, "visual_budget_gate_prior_bias_end", 0.0)
        )
        self.visual_budget_gate_tau_start = float(
            getattr(config, "visual_budget_gate_tau_start", 1.0)
        )
        self.visual_budget_gate_tau_end = float(
            getattr(config, "visual_budget_gate_tau_end", 0.7)
        )
        self.visual_budget_gate_anchor_delta = float(
            getattr(config, "visual_budget_gate_anchor_delta", 0.2)
        )
        self.visual_budget_mpa_enabled = bool(
            getattr(config, "visual_budget_mpa_enabled", False)
        )
        visual_budget_mpa_tau = getattr(config, "visual_budget_mpa_tau", None)
        self.visual_budget_mpa_tau = (
            None if visual_budget_mpa_tau is None else float(visual_budget_mpa_tau)
        )
        self.visual_budget_mpa_log_eps = float(
            getattr(config, "visual_budget_mpa_log_eps", 1e-06)
        )
        self.visual_budget_mpa_logit_clip = float(
            getattr(config, "visual_budget_mpa_logit_clip", 4.0)
        )
        visual_budget_s2_alpha_warmup_steps = getattr(
            config, "visual_budget_s2_alpha_warmup_steps", None
        )
        self.visual_budget_s2_alpha_warmup_steps = (
            None
            if visual_budget_s2_alpha_warmup_steps is None
            else int(visual_budget_s2_alpha_warmup_steps)
        )
        self.visual_budget_objective_mode = str(
            getattr(config, "visual_budget_objective_mode", "combined_two_pass")
        )
        self.visual_budget_gate_heads = None
        gate_width = int(getattr(config, "visual_budget_gate_hidden_dim", 256))
        input_dim = paligemma_config.width * 3 + 1
        self.visual_budget_gate_heads = nnx.Dict(
            base=VisualBudgetGateHead(input_dim, gate_width, rngs=rngs),
            wrist=VisualBudgetGateHead(input_dim, gate_width, rngs=rngs),
        )
        self.temporal_prompt_proj = None
        self.proprio_memory_in_proj = None
        self.proprio_memory_context_proj = None
        self.proprio_memory_compressor = None
        self.temporal_proprio_prefix_gate_raw = None
        if self.use_temporal_proprio_memory and self.temporal_proprio_history_lags:
            if temporal_hidden_dim != paligemma_config.width:
                self.temporal_prompt_proj = nnx.Linear(
                    paligemma_config.width, temporal_hidden_dim, rngs=rngs
                )
            proprio_input_dim = config.action_dim * (
                2 if self.temporal_proprio_use_delta else 1
            )
            self.proprio_memory_in_proj = nnx.Linear(
                proprio_input_dim, temporal_hidden_dim, rngs=rngs
            )
            self.proprio_memory_context_proj = nnx.Linear(
                config.action_dim, temporal_hidden_dim, rngs=rngs
            )
            self.proprio_memory_compressor = TemporalMemoryCompressor(
                temporal_hidden_dim,
                config.temporal_proprio_memory_tokens,
                rngs=rngs,
                hidden_dim=temporal_hidden_dim,
                context_dim=temporal_hidden_dim,
                out_dim=paligemma_config.width,
            )
            if self.temporal_proprio_tokens_in_prefix:
                self.temporal_proprio_prefix_gate_raw = nnx.Param(
                    jnp.array(0.0, dtype=jnp.float32)
                )
        self.time_mlp_in = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.time_mlp_out = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_out_proj = nnx.Linear(
            action_expert_config.width, config.action_dim, rngs=rngs
        )
        self.stage_classifier = None
        if self.stage_num_classes:
            self.stage_classifier = nnx.Linear(
                paligemma_config.width + config.action_dim,
                self.stage_num_classes,
                dtype=jnp.float32,
                rngs=rngs,
            )

    def _stage_logits(self, prefix_hidden, prefix_mask, state):
        """Observation-only classifier: no PA decoder, action target, or latch input."""
        if self.stage_classifier is None:
            raise ValueError("Stage classifier is disabled")
        weights = prefix_mask.astype(jnp.float32)[..., None]
        pooled = (prefix_hidden.astype(jnp.float32) * weights).sum(1) / jnp.maximum(
            weights.sum(1), 1
        )
        if self.stage_pa_conditioning:
            last = jnp.max(
                jnp.where(prefix_mask, jnp.arange(prefix_mask.shape[1])[None, :], 0),
                axis=1,
            )
            pooled = prefix_hidden[jnp.arange(prefix_hidden.shape[0]), last].astype(
                jnp.float32
            )
        return self.stage_classifier(
            jnp.concatenate([pooled, state.astype(jnp.float32)], axis=-1)
        )

    def _stage_pa_token_count(self):
        count = len(self.stage_pa_token_ids[0])
        return count

    def _append_stage_pa(self, prefix_tokens, prefix_mask, kv_cache, stage_ids):
        """Encode only the selected canonical PA; the shared prefix never sees it."""
        ids = jnp.asarray(self.stage_pa_token_ids, dtype=jnp.int32)[stage_ids]
        valid = ids != 0
        tokens = self.PaliGemma.llm(ids, method="embed")
        local = valid[:, :, None] & valid[:, None, :]
        cross = valid[:, :, None] & prefix_mask[:, None, :]
        mask = jnp.concatenate([cross, local], axis=-1)
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(valid, axis=-1) - 1
        )
        ((_, _), cache) = self.PaliGemma.llm(
            [tokens, None], mask=mask, positions=positions, kv_cache=kv_cache
        )
        return (
            jnp.concatenate([prefix_tokens, tokens], axis=1),
            jnp.concatenate([prefix_mask, valid], axis=1),
            cache,
        )

    def _add_progress_condition(self, condition, progress):
        return condition

    def _add_stage_condition(self, adarms_cond, stage_ids):
        return adarms_cond

    @staticmethod
    def _resize_prior_to_token_grid(
        prior_map: jax.Array, token_count: int
    ) -> jax.Array:
        if prior_map.ndim == 4 and prior_map.shape[-1] == 1:
            prior_map = prior_map[..., 0]
        if prior_map.ndim != 3:
            raise ValueError(
                f"image prior must have shape [batch, h, w], got {prior_map.shape}"
            )
        grid_size = int(math.isqrt(token_count))
        if grid_size * grid_size != token_count:
            raise ValueError(
                f"Expected square image token grid, got token_count={token_count}."
            )
        resized = jax.image.resize(
            prior_map[..., None].astype(jnp.float32),
            (prior_map.shape[0], grid_size, grid_size, 1),
            method=jax.image.ResizeMethod.LINEAR,
        )
        return jnp.clip(resized[..., 0], 0.0, 1.0).reshape(
            prior_map.shape[0], token_count
        )

    @staticmethod
    def _resize_valid_mask_to_token_grid(
        valid_mask: jax.Array, token_count: int
    ) -> jax.Array:
        if valid_mask.ndim == 4 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask[..., 0]
        if valid_mask.ndim != 3:
            raise ValueError(
                f"image valid mask must have shape [batch, h, w], got {valid_mask.shape}"
            )
        grid_size = int(math.isqrt(token_count))
        if grid_size * grid_size != token_count:
            raise ValueError(
                f"Expected square image token grid, got token_count={token_count}."
            )
        resized = jax.image.resize(
            valid_mask[..., None].astype(jnp.float32),
            (valid_mask.shape[0], grid_size, grid_size, 1),
            method=jax.image.ResizeMethod.LINEAR,
        )
        return (resized[..., 0] >= 0.5).reshape(valid_mask.shape[0], token_count)

    def _resolve_image_prior_strength(
        self, image_name: str, dtype: jnp.dtype
    ) -> jax.Array | None:
        return None

    def _compute_image_prior_gate(self, token_prior: jax.Array) -> jax.Array:
        gate = self.image_prior_floor + (1.0 - self.image_prior_floor) * token_prior
        return gate

    def _apply_image_prior(
        self,
        image_name: str,
        image_tokens: jax.Array,
        prior_map: jax.Array | None,
        *,
        train: bool = False,
        prior_dropout_rng: jax.Array | None = None,
    ) -> jax.Array:
        if prior_map is None:
            return image_tokens
        token_prior = self._resize_prior_to_token_grid(
            prior_map, image_tokens.shape[1]
        ).astype(image_tokens.dtype)
        gate_strength = self._resolve_image_prior_strength(
            image_name, image_tokens.dtype
        )
        if gate_strength is None:
            return image_tokens
        gate = self._compute_image_prior_gate(token_prior).astype(image_tokens.dtype)
        return image_tokens * (1.0 + gate_strength * (gate[..., None] - 1.0))

    @staticmethod
    def _interpolate_schedule_value(
        start: float,
        end: float,
        *,
        step: at.Int[at.ArrayLike, ""] | int,
        total_steps: at.Int[at.ArrayLike, ""] | int,
        dtype: jnp.dtype,
    ) -> jax.Array:
        step_value = jnp.asarray(step, dtype=dtype)
        total_steps_value = jnp.asarray(total_steps, dtype=dtype)
        denom = jnp.maximum(
            total_steps_value - jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        )
        progress = jnp.where(
            total_steps_value <= jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
            jnp.clip(step_value / denom, 0.0, 1.0),
        )
        start_value = jnp.asarray(start, dtype=dtype)
        end_value = jnp.asarray(end, dtype=dtype)
        return start_value + (end_value - start_value) * progress

    def _visual_budget_view_kind(self, image_name: str) -> str:
        if image_name in self.base_view_names:
            return "base"
        if image_name in self.wrist_view_names:
            return "wrist"
        raise KeyError(
            f"Unknown image view {image_name!r}; configured views are {self.image_view_names!r}."
        )

    def _visual_budget_keep_ratio_for_kind(
        self, view_kind: str, dtype: jnp.dtype
    ) -> jax.Array:
        value = (
            self.visual_budget_gate_keep_ratio_base
            if view_kind == "base"
            else self.visual_budget_gate_keep_ratio_wrist
        )
        return jnp.asarray(value, dtype=dtype)

    def _visual_budget_keep_ratio(self, image_name: str, dtype: jnp.dtype) -> jax.Array:
        return self._visual_budget_keep_ratio_for_kind(
            self._visual_budget_view_kind(image_name), dtype
        )

    def _visual_budget_budget_weight_for_kind(
        self, view_kind: str, dtype: jnp.dtype
    ) -> jax.Array:
        value = (
            self.visual_budget_gate_lambda_budget_base
            if view_kind == "base"
            else self.visual_budget_gate_lambda_budget_wrist
        )
        return jnp.asarray(value, dtype=dtype)

    def _visual_budget_gate_head(self, image_name: str) -> VisualBudgetGateHead:
        if self.visual_budget_gate_heads is None:
            raise ValueError(
                "visual budget gate heads are missing while use_visual_budget_gate=True."
            )
        return self.visual_budget_gate_heads[self._visual_budget_view_kind(image_name)]

    def _embed_prompt_tokens(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b l emb"] | None:
        if obs.tokenized_prompt is None:
            return None
        return self.PaliGemma.llm(obs.tokenized_prompt, method="embed")

    def _pool_prompt_summary(
        self,
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
        prompt_mask: at.Bool[at.Array, "b l"] | None,
        *,
        batch_size: int,
        dtype: jnp.dtype,
        emb_dim: int,
    ) -> at.Float[at.Array, "b emb"]:
        if prompt_tokens is None or prompt_mask is None:
            return jnp.zeros((batch_size, emb_dim), dtype=dtype)
        mask = prompt_mask.astype(prompt_tokens.dtype)
        denom = jnp.maximum(
            jnp.sum(mask, axis=1, keepdims=True),
            jnp.array(1.0, dtype=prompt_tokens.dtype),
        )
        pooled = jnp.sum(prompt_tokens * mask[..., None], axis=1) / denom
        return sharding.batch_activation_sharding_constraint(pooled.astype(dtype))

    def _project_temporal_prompt_summary(
        self,
        prompt_summary: at.Float[at.Array, "b d"],
    ) -> at.Float[at.Array, "b d"]:
        if self.temporal_prompt_proj is None:
            return prompt_summary
        prompt_summary = sharding.batch_activation_sharding_constraint(prompt_summary)
        projected = self.temporal_prompt_proj(prompt_summary)
        return sharding.batch_activation_sharding_constraint(projected)

    @staticmethod
    def _history_age_embeddings(
        lags: tuple[int, ...],
        emb_dim: int,
        dtype: jnp.dtype,
    ) -> jax.Array:
        if not lags:
            return jnp.zeros((0, emb_dim), dtype=dtype)
        max_lag = float(max(lags))
        age_emb = posemb_sincos(
            jnp.asarray(lags, dtype=jnp.float32),
            emb_dim,
            min_period=1.0,
            max_period=max(2.0, max_lag + 1.0),
        )
        return age_emb.astype(dtype)

    def _temporal_proprio_prefix_gate_multiplier(
        self,
        dtype: jnp.dtype,
        train_step: at.Int[at.ArrayLike, ""] | int | None = None,
    ) -> jax.Array:
        if self.temporal_proprio_prefix_gate_warmup_steps <= 0 or train_step is None:
            return jnp.asarray(1.0, dtype=dtype)
        train_step = jnp.asarray(train_step, dtype=jnp.float32)
        warmup_steps = jnp.asarray(
            float(self.temporal_proprio_prefix_gate_warmup_steps), dtype=jnp.float32
        )
        scale = jnp.clip((train_step + 1.0) / jnp.maximum(warmup_steps, 1.0), 0.0, 1.0)
        return scale.astype(dtype)

    @staticmethod
    def _masked_scalar_mean(
        values: jax.Array, mask: jax.Array, *, dtype: jnp.dtype | None = None
    ) -> jax.Array:
        dtype = values.dtype if dtype is None else dtype
        mask_f = mask.astype(dtype)
        denom = jnp.maximum(jnp.sum(mask_f), jnp.asarray(1.0, dtype=dtype))
        return jnp.sum(values.astype(dtype) * mask_f) / denom

    def _embed_history_image_tokens_by_name(
        self, obs: _model.Observation
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        return ({}, {})

    def _embed_image_tokens_by_name(
        self,
        obs: _model.Observation,
        *,
        train: bool = False,
        prior_dropout_rng: jax.Array | None = None,
    ) -> dict[str, at.Float[at.Array, "b n emb"]]:
        image_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]] = {}
        for image_idx, name in enumerate(obs.images):
            (image_tokens, _) = self.PaliGemma.img(obs.images[name], train=False)
            prior_map = None if obs.image_priors is None else obs.image_priors.get(name)
            image_prior_rng = None
            if prior_dropout_rng is not None:
                image_prior_rng = jax.random.fold_in(prior_dropout_rng, image_idx)
            image_tokens_by_name[name] = self._apply_image_prior(
                name,
                image_tokens,
                prior_map,
                train=train,
                prior_dropout_rng=image_prior_rng,
            )
        return image_tokens_by_name

    def _build_rgb_memory_tokens(
        self,
        obs: _model.Observation,
        image_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]],
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
        *,
        history_tokens_by_name: dict[str, jax.Array] | None = None,
        history_token_masks_by_name: dict[str, jax.Array] | None = None,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        return (None, None)

    def _build_proprio_memory_tokens(
        self,
        obs: _model.Observation,
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        if (
            not self.use_temporal_proprio_memory
            or self.proprio_memory_in_proj is None
            or self.proprio_memory_context_proj is None
            or self.proprio_memory_compressor is None
            or obs.state_history is None
        ):
            return None, None

        dtype = obs.state.dtype
        history_mask = (
            obs.state_history_mask
            if obs.state_history_mask is not None
            else jnp.ones(obs.state_history.shape[:2], dtype=jnp.bool_)
        )
        if self.temporal_proprio_use_delta:
            delta_state = obs.state[:, None, :] - obs.state_history
            history_features = jnp.concatenate(
                [obs.state_history, delta_state], axis=-1
            )
        else:
            history_features = obs.state_history
        history_tokens = sharding.batch_activation_sharding_constraint(
            self.proprio_memory_in_proj(history_features)
        )
        if self.proprio_frequency_enabled:
            if obs.state_history_age is None:
                raise ValueError("Frequency proprio requires actual elapsed seconds")
            from openpi.shared.proprio_frequency import time_encoding

            age_emb = time_encoding(
                obs.state_history_age, history_tokens.shape[-1]
            ).astype(history_tokens.dtype)
            history_tokens = jnp.where(
                history_mask[..., None], history_tokens + age_emb, 0
            )
        else:
            age_emb = self._history_age_embeddings(
                self.temporal_proprio_history_lags,
                history_tokens.shape[-1],
                history_tokens.dtype,
            )
            history_tokens = history_tokens + age_emb[None, :, :]
        history_tokens = sharding.batch_activation_sharding_constraint(history_tokens)
        prompt_summary = self._pool_prompt_summary(
            prompt_tokens,
            obs.tokenized_prompt_mask,
            batch_size=history_tokens.shape[0],
            dtype=obs.state.dtype,
            emb_dim=self.action_in_proj.out_features,
        )
        temporal_prompt_summary = self._project_temporal_prompt_summary(
            prompt_summary.astype(history_tokens.dtype)
        )
        context = sharding.batch_activation_sharding_constraint(
            temporal_prompt_summary + self.proprio_memory_context_proj(obs.state)
        )
        memory_tokens = self.proprio_memory_compressor(
            history_tokens, history_mask, context
        )
        memory_mask = jnp.broadcast_to(
            jnp.any(history_mask, axis=1, keepdims=True), memory_tokens.shape[:2]
        )
        return memory_tokens.astype(dtype), memory_mask

    def _apply_temporal_proprio_prefix_gate(
        self,
        proprio_memory_tokens: jax.Array | None,
        proprio_memory_mask: jax.Array | None,
        *,
        dtype: jnp.dtype,
        train_step: at.Int[at.ArrayLike, ""] | int | None = None,
        keep_when_disabled: bool = True,
    ) -> tuple[jax.Array | None, jax.Array | None, dict[str, jax.Array]]:
        zero_aux = {
            "temporal_proprio_prefix_gate": jnp.asarray(0.0, dtype=dtype),
            "temporal_proprio_prefix_gate_warmup": jnp.asarray(0.0, dtype=dtype),
        }
        if not self.temporal_proprio_tokens_in_prefix:
            if keep_when_disabled:
                return proprio_memory_tokens, proprio_memory_mask, zero_aux
            return None, None, zero_aux
        if (
            proprio_memory_tokens is None
            or proprio_memory_mask is None
            or self.temporal_proprio_prefix_gate_raw is None
        ):
            return None, None, zero_aux

        gate = jnp.tanh(self.temporal_proprio_prefix_gate_raw.value.astype(dtype))
        warmup_scale = self._temporal_proprio_prefix_gate_multiplier(dtype, train_step)
        gate = gate * jnp.asarray(self.temporal_proprio_prefix_token_scale, dtype=dtype)
        gate = gate * warmup_scale
        proprio_memory_tokens = (
            jnp.tanh(proprio_memory_tokens.astype(dtype)) * gate
        ).astype(proprio_memory_tokens.dtype)
        return (
            proprio_memory_tokens,
            proprio_memory_mask,
            {
                "temporal_proprio_prefix_gate": gate,
                "temporal_proprio_prefix_gate_warmup": warmup_scale,
            },
        )

    def _compose_prefix_tokens(
        self,
        obs: _model.Observation,
        image_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]],
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
        rgb_memory_tokens: jax.Array | None = None,
        rgb_memory_mask: jax.Array | None = None,
        proprio_memory_tokens: jax.Array | None = None,
        proprio_memory_mask: jax.Array | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens = image_tokens_by_name[name]
            tokens.append(image_tokens)
            token_input_mask = einops.repeat(
                obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]
            )
            if obs.image_valid_masks is not None and name in obs.image_valid_masks:
                token_valid_mask = self._resize_valid_mask_to_token_grid(
                    obs.image_valid_masks[name], image_tokens.shape[1]
                )
                token_input_mask = jnp.logical_and(token_input_mask, token_valid_mask)
            input_mask.append(token_input_mask)
            ar_mask += [False] * image_tokens.shape[1]
        if rgb_memory_tokens is not None and rgb_memory_mask is not None:
            tokens.append(rgb_memory_tokens)
            input_mask.append(rgb_memory_mask)
            ar_mask += [False] * rgb_memory_tokens.shape[1]
        if proprio_memory_tokens is not None and proprio_memory_mask is not None:
            tokens.append(proprio_memory_tokens)
            input_mask.append(proprio_memory_mask)
            ar_mask += [False] * proprio_memory_tokens.shape[1]
        if prompt_tokens is not None and obs.tokenized_prompt_mask is not None:
            tokens.append(prompt_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return (tokens, input_mask, ar_mask)

    def _run_prefix_suffix_pass(
        self,
        prefix_tokens: at.Float[at.Array, "b sp emb"],
        prefix_mask: at.Bool[at.Array, "b sp"],
        prefix_ar_mask: at.Bool[at.Array, " sp"],
        suffix_tokens: at.Float[at.Array, "b ss emb"],
        suffix_mask: at.Bool[at.Array, "b ss"],
        suffix_ar_mask: at.Bool[at.Array, " ss"],
        adarms_cond: at.Float[at.Array, "b emb"] | at.Float[at.Array, "b s emb"] | None,
        *,
        prefix_key_log_prior: at.Float[at.Array, "b sp"] | None = None,
        prefix_key_log_prior_strength: at.Float[at.Array, ""] | float | None = None,
        prefix_kv_cache=None,
    ) -> at.Float[at.Array, "b ah ad"]:
        scaled_prefix_key_bias = None
        if prefix_key_log_prior is not None:
            if prefix_key_log_prior.shape != prefix_mask.shape:
                raise ValueError(
                    f"S2-MPA prefix prior shape {prefix_key_log_prior.shape} does not match {prefix_mask.shape}."
                )
            if prefix_key_log_prior_strength is None:
                raise ValueError(
                    "Budget-S2 prefix prior requires an explicit deterministic alpha strength."
                )
            prior_strength = jnp.asarray(
                prefix_key_log_prior_strength, dtype=jnp.float32
            )
            scaled_prefix_key_bias = prior_strength * jnp.asarray(
                prefix_key_log_prior, dtype=jnp.float32
            )
        flash_packed = (
            self.flash_enabled
            and suffix_tokens.shape[1] == self.flash_buffer_size * self.action_horizon
        )
        if (
            prefix_kv_cache is not None
            or _train_prefix_kv_cache_enabled()
            or flash_packed
        ):
            kv_cache = prefix_kv_cache
            if kv_cache is None:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
                prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
                (_, kv_cache) = self.PaliGemma.llm(
                    [prefix_tokens, None],
                    mask=prefix_attn_mask,
                    positions=prefix_positions,
                )
            prefix_to_suffix_mask = jnp.broadcast_to(
                prefix_mask[:, None, :],
                (prefix_mask.shape[0], suffix_tokens.shape[1], prefix_tokens.shape[1]),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            if flash_packed:
                from openpi.models import flash

                (suffix_attn_mask, local_positions) = flash.attention(
                    suffix_mask,
                    chunk_size=self.flash_chunk_size,
                    buffer_size=self.flash_buffer_size,
                )
            full_attn_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attn_mask], axis=-1
            )
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            if flash_packed:
                suffix_positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None] + local_positions
                )
            suffix_logit_bias = (
                None
                if scaled_prefix_key_bias is None
                else _suffix_mpa_attention_bias(
                    scaled_prefix_key_bias, suffix_tokens.shape[1]
                )
            )
            ((prefix_out, suffix_out), _) = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                attn_logit_bias=suffix_logit_bias,
            )
            assert prefix_out is None
            return self.action_out_proj(
                suffix_out if flash_packed else suffix_out[:, -self.action_horizon :]
            )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        monolithic_logit_bias = (
            None
            if scaled_prefix_key_bias is None
            else _monolithic_mpa_attention_bias(
                scaled_prefix_key_bias, suffix_tokens.shape[1]
            )
        )
        ((_, suffix_out), _) = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            attn_logit_bias=monolithic_logit_bias,
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def _visual_budget_token_prior_inputs(
        self,
        obs: _model.Observation,
        name: str,
        image_tokens: at.Float[at.Array, "b n emb"],
        *,
        routed_anchor_info: dict[str, dict[str, jax.Array]] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return baked-padding validity, weak prior, and temporal-change prior for one view."""
        batch_size = image_tokens.shape[0]
        dtype = image_tokens.dtype
        if obs.image_valid_masks is not None and name in obs.image_valid_masks:
            token_valid_mask = self._resize_valid_mask_to_token_grid(
                obs.image_valid_masks[name], image_tokens.shape[1]
            )
        else:
            token_valid_mask = jnp.ones(
                (batch_size, image_tokens.shape[1]), dtype=jnp.bool_
            )
        token_valid_mask_f = token_valid_mask.astype(dtype)
        prior_map = None
        if prior_map is None:
            token_prior = jnp.zeros((batch_size, image_tokens.shape[1]), dtype=dtype)
        else:
            token_prior = self._resize_prior_to_token_grid(
                prior_map, image_tokens.shape[1]
            ).astype(dtype)
        token_prior = token_prior * token_valid_mask_f
        change_prior = jnp.zeros_like(token_prior)
        return (token_valid_mask, token_prior, change_prior)

    def _compute_visual_budget_mask_logits(
        self,
        obs: _model.Observation,
        image_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]],
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
        *,
        beta: jax.Array,
        temporal_context: jax.Array | None = None,
        routed_anchor_info: dict[str, dict[str, jax.Array]] | None = None,
    ) -> dict[str, jax.Array]:
        """Evaluate the trainable mask head exactly once per view.

        visual budget feeds these same logits to its embedding and attention
        consumers.  Keeping this as an explicit producer prevents the two
        interfaces from silently observing different mask predictions.
        """
        if not self.use_visual_budget_gate or self.visual_budget_gate_heads is None:
            raise ValueError(
                "visual budget mask logits require pretrained visual-budget heads."
            )
        first_tokens = next(iter(image_tokens_by_name.values()))
        (batch_size, _, emb_dim) = first_tokens.shape
        dtype = first_tokens.dtype
        prompt_summary = self._pool_prompt_summary(
            prompt_tokens,
            obs.tokenized_prompt_mask,
            batch_size=batch_size,
            dtype=dtype,
            emb_dim=emb_dim,
        )
        if temporal_context is not None:
            prompt_summary = prompt_summary + temporal_context.astype(dtype)
        beta = jnp.asarray(beta, dtype=dtype)
        logits_by_name: dict[str, jax.Array] = {}
        for name, image_tokens in image_tokens_by_name.items():
            (_, token_prior, _) = self._visual_budget_token_prior_inputs(
                obs, name, image_tokens, routed_anchor_info=routed_anchor_info
            )
            logits = self._visual_budget_gate_head(name)(
                image_tokens, prompt_summary, token_prior
            )
            logits = logits + beta * token_prior
            logits_by_name[name] = logits
        return logits_by_name

    def _compute_visual_budget_mpa_log_priors(
        self,
        obs: _model.Observation,
        image_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]],
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
        *,
        temporal_context: jax.Array | None = None,
        mask_logits_by_name: dict[str, jax.Array] | None = None,
        return_projected_probabilities: bool = False,
    ) -> (
        tuple[dict[str, jax.Array], dict[str, jax.Array]]
        | tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]
    ):
        """Compute attention priors from detached visual budget gate logits.

        Supplied logits share the embedding gate's predictions. The attention
        prior uses a stop-gradient copy.
        """
        if self.visual_budget_mpa_tau is None:
            raise ValueError("S2-MPA requires an explicit seed-checkpoint tau.")
        if self.visual_budget_gate_heads is None:
            raise ValueError(
                "S2 mask-prior attention requires pretrained visual-budget heads."
            )
        if self.visual_budget_s2_alpha_warmup_steps is None:
            raise ValueError("Budget-S2 requires an explicit alpha warmup.")
        if mask_logits_by_name is not None and set(mask_logits_by_name) != set(
            image_tokens_by_name
        ):
            raise ValueError(
                f"visual budget precomputed mask-logit views must exactly match image views: {sorted(mask_logits_by_name)} != {sorted(image_tokens_by_name)}."
            )
        first_tokens = next(iter(image_tokens_by_name.values()))
        (batch_size, _, emb_dim) = first_tokens.shape
        prompt_summary = None
        if mask_logits_by_name is None:
            prompt_summary = self._pool_prompt_summary(
                prompt_tokens,
                obs.tokenized_prompt_mask,
                batch_size=batch_size,
                dtype=first_tokens.dtype,
                emb_dim=emb_dim,
            )
            if temporal_context is not None:
                prompt_summary = prompt_summary + temporal_context.astype(
                    prompt_summary.dtype
                )
        priors_by_name: dict[str, jax.Array] = {}
        projected_probabilities_by_name: dict[str, jax.Array] = {}
        probability_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        prior_abs_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        valid_den = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        raw_view_mean_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        projected_view_mean_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        nu_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        nu_max = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        binding_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        valid_view_den = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        entropy_num = {
            "base": jnp.asarray(0.0, jnp.float32),
            "wrist": jnp.asarray(0.0, jnp.float32),
        }
        for name, image_tokens in image_tokens_by_name.items():
            if obs.image_valid_masks is not None and name in obs.image_valid_masks:
                token_valid_mask = self._resize_valid_mask_to_token_grid(
                    obs.image_valid_masks[name], image_tokens.shape[1]
                )
            else:
                token_valid_mask = jnp.ones(
                    (batch_size, image_tokens.shape[1]), dtype=jnp.bool_
                )
            view_valid_mask = jnp.logical_and(
                token_valid_mask, obs.image_masks[name][:, None]
            )
            if mask_logits_by_name is not None:
                logits = mask_logits_by_name[name]
                if logits.shape != view_valid_mask.shape:
                    raise ValueError(
                        f"visual budget mask logits for {name!r} have shape {logits.shape}; expected {view_valid_mask.shape}."
                    )
            else:
                prior_map = None
                if prior_map is None:
                    token_prior = jnp.zeros(
                        (batch_size, image_tokens.shape[1]), dtype=image_tokens.dtype
                    )
                else:
                    token_prior = self._resize_prior_to_token_grid(
                        prior_map, image_tokens.shape[1]
                    ).astype(image_tokens.dtype)
                token_prior = jnp.where(
                    view_valid_mask,
                    token_prior,
                    jnp.asarray(0.0, dtype=image_tokens.dtype),
                )
                assert prompt_summary is not None
                logits = self._visual_budget_gate_head(name)(
                    image_tokens, prompt_summary, token_prior
                )
            logits = jax.lax.stop_gradient(logits)
            kind = self._visual_budget_view_kind(name)
            keep_ratio = (
                self.visual_budget_gate_keep_ratio_base
                if kind == "base"
                else self.visual_budget_gate_keep_ratio_wrist
            )
            (probability, nu, raw_view_mean, projected_view_mean, binding) = (
                _one_sided_mask_budget_projection(
                    logits,
                    view_valid_mask,
                    tau=self.visual_budget_mpa_tau,
                    keep_ratio=keep_ratio,
                )
            )
            gate_floor = jnp.exp(
                -jnp.asarray(self.visual_budget_mpa_logit_clip, dtype=jnp.float32)
            )
            effective_gate = gate_floor + (1.0 - gate_floor) * probability
            log_prior = jnp.clip(
                jnp.log(effective_gate), -self.visual_budget_mpa_logit_clip, 0.0
            )
            log_prior = jnp.where(
                view_valid_mask, log_prior, jnp.asarray(0.0, dtype=jnp.float32)
            )
            priors_by_name[name] = jax.lax.stop_gradient(log_prior)
            probability = jax.lax.stop_gradient(probability)
            projected_probabilities_by_name[name] = probability
            kind = self._visual_budget_view_kind(name)
            valid_f = view_valid_mask.astype(jnp.float32)
            valid_view = jnp.any(view_valid_mask, axis=-1)
            valid_view_f = valid_view.astype(jnp.float32)
            probability_num[kind] = probability_num[kind] + jnp.sum(
                probability * valid_f
            )
            prior_abs_num[kind] = prior_abs_num[kind] + jnp.sum(
                jnp.abs(log_prior) * valid_f
            )
            valid_den[kind] = valid_den[kind] + jnp.sum(valid_f)
            raw_view_mean_num[kind] = raw_view_mean_num[kind] + jnp.sum(
                raw_view_mean * valid_view_f
            )
            projected_view_mean_num[kind] = projected_view_mean_num[kind] + jnp.sum(
                projected_view_mean * valid_view_f
            )
            nu_num[kind] = nu_num[kind] + jnp.sum(nu * valid_view_f)
            nu_max[kind] = jnp.maximum(
                nu_max[kind], jnp.max(jnp.where(valid_view, nu, 0.0))
            )
            binding_num[kind] = binding_num[kind] + jnp.sum(
                binding.astype(jnp.float32) * valid_view_f
            )
            valid_view_den[kind] = valid_view_den[kind] + jnp.sum(valid_view_f)
            probability_for_entropy = jnp.clip(
                probability,
                self.visual_budget_mpa_log_eps,
                1.0 - self.visual_budget_mpa_log_eps,
            )
            token_entropy = -probability_for_entropy * jnp.log(
                probability_for_entropy
            ) - (1.0 - probability_for_entropy) * jnp.log1p(-probability_for_entropy)
            entropy_num[kind] = entropy_num[kind] + jnp.sum(token_entropy * valid_f)
        aux = {
            "visual_budget_mpa_tau": jnp.asarray(
                self.visual_budget_mpa_tau, dtype=jnp.float32
            ),
            "visual_budget_mpa_probability_mean_base": probability_num["base"]
            / jnp.maximum(valid_den["base"], 1.0),
            "visual_budget_mpa_probability_mean_wrist": probability_num["wrist"]
            / jnp.maximum(valid_den["wrist"], 1.0),
            "visual_budget_mpa_prior_abs_mean_base": prior_abs_num["base"]
            / jnp.maximum(valid_den["base"], 1.0),
            "visual_budget_mpa_prior_abs_mean_wrist": prior_abs_num["wrist"]
            / jnp.maximum(valid_den["wrist"], 1.0),
        }
        aux["visual_budget_s2_keep_ratio_base"] = jnp.asarray(
            self.visual_budget_gate_keep_ratio_base, dtype=jnp.float32
        )
        aux["visual_budget_s2_keep_ratio_wrist"] = jnp.asarray(
            self.visual_budget_gate_keep_ratio_wrist, dtype=jnp.float32
        )
        aux["visual_budget_s2_gate_floor"] = jnp.exp(
            -jnp.asarray(self.visual_budget_mpa_logit_clip, dtype=jnp.float32)
        )
        for kind in ("base", "wrist"):
            aux[f"visual_budget_s2_raw_probability_mean_{kind}"] = raw_view_mean_num[
                kind
            ] / jnp.maximum(valid_view_den[kind], 1.0)
            aux[f"visual_budget_s2_projected_probability_mean_{kind}"] = (
                projected_view_mean_num[kind] / jnp.maximum(valid_view_den[kind], 1.0)
            )
            aux[f"visual_budget_s2_nu_mean_{kind}"] = nu_num[kind] / jnp.maximum(
                valid_view_den[kind], 1.0
            )
            aux[f"visual_budget_s2_nu_max_{kind}"] = nu_max[kind]
            aux[f"visual_budget_s2_binding_fraction_{kind}"] = binding_num[
                kind
            ] / jnp.maximum(valid_view_den[kind], 1.0)
            aux[f"visual_budget_s2_probability_entropy_{kind}"] = entropy_num[
                kind
            ] / jnp.maximum(valid_den[kind], 1.0)
        if return_projected_probabilities:
            return (priors_by_name, aux, projected_probabilities_by_name)
        return (priors_by_name, aux)

    def _compose_prefix_mpa_log_prior(
        self,
        obs: _model.Observation,
        image_log_priors_by_name: dict[str, jax.Array],
        *,
        prompt_tokens: jax.Array | None,
        rgb_memory_tokens: jax.Array | None = None,
        proprio_memory_tokens: jax.Array | None = None,
    ) -> jax.Array:
        """Align current-image priors to the exact prefix token layout."""
        parts = [image_log_priors_by_name[name] for name in obs.images]
        batch_size = next(iter(image_log_priors_by_name.values())).shape[0]
        if rgb_memory_tokens is not None:
            parts.append(
                jnp.zeros((batch_size, rgb_memory_tokens.shape[1]), dtype=jnp.float32)
            )
        if proprio_memory_tokens is not None:
            parts.append(
                jnp.zeros(
                    (batch_size, proprio_memory_tokens.shape[1]), dtype=jnp.float32
                )
            )
        if prompt_tokens is not None and obs.tokenized_prompt_mask is not None:
            parts.append(
                jnp.zeros((batch_size, prompt_tokens.shape[1]), dtype=jnp.float32)
            )
        return jnp.concatenate(parts, axis=1)

    def _apply_visual_budget_soft_gate(
        self,
        obs: _model.Observation,
        image_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]],
        prompt_tokens: at.Float[at.Array, "b l emb"] | None,
        *,
        beta: jax.Array,
        tau: jax.Array,
        gate_floor: jax.Array | None = None,
        lambda_anchor: jax.Array | None = None,
        temporal_context: jax.Array | None = None,
        routed_anchor_info: dict[str, dict[str, jax.Array]] | None = None,
        compute_aux: bool = False,
        mask_logits_by_name: dict[str, jax.Array] | None = None,
    ) -> tuple[dict[str, at.Float[at.Array, "b n emb"]], dict[str, jax.Array]]:
        first_tokens = next(iter(image_tokens_by_name.values()))
        dtype = first_tokens.dtype
        (batch_size, _, emb_dim) = first_tokens.shape
        if mask_logits_by_name is not None and set(mask_logits_by_name) != set(
            image_tokens_by_name
        ):
            raise ValueError(
                f"visual budget precomputed mask-logit views must exactly match image views: {sorted(mask_logits_by_name)} != {sorted(image_tokens_by_name)}."
            )
        prompt_summary = None
        if mask_logits_by_name is None:
            prompt_summary = self._pool_prompt_summary(
                prompt_tokens,
                obs.tokenized_prompt_mask,
                batch_size=batch_size,
                dtype=dtype,
                emb_dim=emb_dim,
            )
            if temporal_context is not None:
                prompt_summary = prompt_summary + temporal_context.astype(dtype)
        gated_tokens_by_name: dict[str, at.Float[at.Array, "b n emb"]] = {}
        aux: dict[str, jax.Array] = {}
        if not compute_aux:
            compute_aux = False
        eps = jnp.asarray(1e-06, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        kind_stats: dict[str, dict[str, jax.Array]] = {
            "base": {
                "budget_num": zero,
                "budget_den": zero,
                "anchor_num": zero,
                "anchor_den": zero,
                "gate_mean_num": zero,
                "gate_mean_den": zero,
                "prior_mean_num": zero,
                "prior_mean_den": zero,
                "change_prior_num": zero,
                "change_prior_den": zero,
            },
            "wrist": {
                "budget_num": zero,
                "budget_den": zero,
                "anchor_num": zero,
                "anchor_den": zero,
                "gate_mean_num": zero,
                "gate_mean_den": zero,
                "prior_mean_num": zero,
                "prior_mean_den": zero,
                "change_prior_num": zero,
                "change_prior_den": zero,
            },
        }
        gate_floor = jnp.asarray(
            self.visual_budget_gate_floor_end if gate_floor is None else gate_floor,
            dtype=dtype,
        )
        beta = jnp.asarray(beta, dtype=dtype)
        tau = jnp.maximum(jnp.asarray(tau, dtype=dtype), eps)
        anchor_delta = jnp.asarray(self.visual_budget_gate_anchor_delta, dtype=dtype)
        for name, image_tokens in image_tokens_by_name.items():
            (token_valid_mask, token_prior, change_prior) = (
                self._visual_budget_token_prior_inputs(
                    obs, name, image_tokens, routed_anchor_info=routed_anchor_info
                )
            )
            token_valid_mask_f = token_valid_mask.astype(dtype)
            if mask_logits_by_name is None:
                assert prompt_summary is not None
                logits = self._visual_budget_gate_head(name)(
                    image_tokens, prompt_summary, token_prior
                )
                logits = logits + beta * token_prior
            else:
                logits = mask_logits_by_name[name]
                if logits.shape != token_valid_mask.shape:
                    raise ValueError(
                        f"visual budget mask logits for {name!r} have shape {logits.shape}; expected {token_valid_mask.shape}."
                    )
            gates = jnp.where(
                token_valid_mask, jax.nn.sigmoid(logits / tau).astype(dtype), zero
            )
            effective_gates = jnp.where(
                token_valid_mask, gate_floor + (1.0 - gate_floor) * gates, zero
            )
            gated_tokens_by_name[name] = image_tokens * effective_gates[..., None]
            if not compute_aux:
                continue
            view_kind = self._visual_budget_view_kind(name)
            view_mask = obs.image_masks[name].astype(dtype)
            token_weight = view_mask[:, None] * token_valid_mask_f
            valid_count_per_sample = jnp.sum(token_weight, axis=1)
            sample_has_valid = (valid_count_per_sample > 0).astype(dtype)
            valid_count = jnp.sum(sample_has_valid)
            mean_gate_per_sample = jnp.sum(gates * token_weight, axis=1) / jnp.maximum(
                valid_count_per_sample, eps
            )
            mean_prior_per_sample = jnp.sum(
                token_prior * token_weight, axis=1
            ) / jnp.maximum(valid_count_per_sample, eps)
            mean_change_prior_per_sample = jnp.sum(
                change_prior * token_weight, axis=1
            ) / jnp.maximum(valid_count_per_sample, eps)
            kind_stats[view_kind]["gate_mean_num"] = kind_stats[view_kind][
                "gate_mean_num"
            ] + jnp.sum(sample_has_valid * mean_gate_per_sample)
            kind_stats[view_kind]["gate_mean_den"] = (
                kind_stats[view_kind]["gate_mean_den"] + valid_count
            )
            kind_stats[view_kind]["prior_mean_num"] = kind_stats[view_kind][
                "prior_mean_num"
            ] + jnp.sum(sample_has_valid * mean_prior_per_sample)
            kind_stats[view_kind]["prior_mean_den"] = (
                kind_stats[view_kind]["prior_mean_den"] + valid_count
            )
            kind_stats[view_kind]["change_prior_num"] = kind_stats[view_kind][
                "change_prior_num"
            ] + jnp.sum(sample_has_valid * mean_change_prior_per_sample)
            kind_stats[view_kind]["change_prior_den"] = (
                kind_stats[view_kind]["change_prior_den"] + valid_count
            )
            keep_ratio = self._visual_budget_keep_ratio(name, dtype)
            budget_error = jnp.square(mean_gate_per_sample - keep_ratio)
            kind_stats[view_kind]["budget_num"] = kind_stats[view_kind][
                "budget_num"
            ] + jnp.sum(sample_has_valid * budget_error)
            kind_stats[view_kind]["budget_den"] = (
                kind_stats[view_kind]["budget_den"] + valid_count
            )
            prior_mass = jnp.sum(token_prior * token_weight, axis=1)
            anchor_valid = sample_has_valid * (prior_mass > 0).astype(dtype)
            anchor_value = jnp.sum(
                token_prior * jax.nn.relu(anchor_delta - gates) * token_weight, axis=1
            ) / jnp.maximum(prior_mass, eps)
            kind_stats[view_kind]["anchor_num"] = kind_stats[view_kind][
                "anchor_num"
            ] + jnp.sum(anchor_valid * anchor_value)
            kind_stats[view_kind]["anchor_den"] = kind_stats[view_kind][
                "anchor_den"
            ] + jnp.sum(anchor_valid)
        if compute_aux:
            budget_base = kind_stats["base"]["budget_num"] / jnp.maximum(
                kind_stats["base"]["budget_den"], eps
            )
            budget_wrist = kind_stats["wrist"]["budget_num"] / jnp.maximum(
                kind_stats["wrist"]["budget_den"], eps
            )
            anchor_base = kind_stats["base"]["anchor_num"] / jnp.maximum(
                kind_stats["base"]["anchor_den"], eps
            )
            anchor_wrist = kind_stats["wrist"]["anchor_num"] / jnp.maximum(
                kind_stats["wrist"]["anchor_den"], eps
            )
            aux = {
                "visual_budget_budget_loss_base": budget_base,
                "visual_budget_budget_loss_wrist": budget_wrist,
                "visual_budget_anchor_loss_base": anchor_base,
                "visual_budget_anchor_loss_wrist": anchor_wrist,
                "visual_budget_gate_mean_base": kind_stats["base"]["gate_mean_num"]
                / jnp.maximum(kind_stats["base"]["gate_mean_den"], eps),
                "visual_budget_gate_mean_wrist": kind_stats["wrist"]["gate_mean_num"]
                / jnp.maximum(kind_stats["wrist"]["gate_mean_den"], eps),
                "visual_budget_prior_mean_base": kind_stats["base"]["prior_mean_num"]
                / jnp.maximum(kind_stats["base"]["prior_mean_den"], eps),
                "visual_budget_prior_mean_wrist": kind_stats["wrist"]["prior_mean_num"]
                / jnp.maximum(kind_stats["wrist"]["prior_mean_den"], eps),
                "visual_budget_change_prior_mean_base": kind_stats["base"][
                    "change_prior_num"
                ]
                / jnp.maximum(kind_stats["base"]["change_prior_den"], eps),
                "visual_budget_change_prior_mean_wrist": kind_stats["wrist"][
                    "change_prior_num"
                ]
                / jnp.maximum(kind_stats["wrist"]["change_prior_den"], eps),
                "visual_budget_beta": beta,
                "visual_budget_tau": tau,
                "visual_budget_gate_floor": gate_floor,
                "visual_budget_lambda_anchor": zero
                if lambda_anchor is None
                else jnp.asarray(lambda_anchor, dtype=dtype),
                "visual_budget_keep_ratio_base": self._visual_budget_keep_ratio_for_kind(
                    "base", dtype
                ),
                "visual_budget_keep_ratio_wrist": self._visual_budget_keep_ratio_for_kind(
                    "wrist", dtype
                ),
            }
        return (gated_tokens_by_name, aux)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b s"],
        *,
        action_chunk_cond: jax.Array | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b s emb"] | None,
    ]:
        action_len = int(noisy_actions.shape[-2])
        input_mask = []
        ar_mask = []
        tokens = []
        action_tokens = self.action_in_proj(noisy_actions)
        action_tokens = self._condition_action_tokens_on_previous_plan(
            obs, action_tokens
        )
        if action_chunk_cond is not None:
            if action_chunk_cond.shape != action_tokens.shape:
                raise ValueError(
                    f"action_chunk_cond shape {action_chunk_cond.shape} must match action_tokens shape {action_tokens.shape}."
                )
            action_tokens = action_tokens + action_chunk_cond.astype(
                action_tokens.dtype
            )
        if timestep.ndim == 2:
            if (
                not (self.tt_rtc_enabled or self.flash_enabled)
                or timestep.shape != noisy_actions.shape[:2]
            ):
                raise ValueError(
                    "Per-action time requires TT-RTC/Flash and one time per action token"
                )
            time_emb = posemb_sincos(
                timestep.reshape(-1),
                self.action_in_proj.out_features,
                min_period=0.004,
                max_period=4.0,
            ).reshape((*timestep.shape, self.action_in_proj.out_features))
        else:
            time_emb = posemb_sincos(
                timestep,
                self.action_in_proj.out_features,
                min_period=0.004,
                max_period=4.0,
            )
        time_emb = sharding.batch_activation_sharding_constraint(time_emb)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = sharding.batch_activation_sharding_constraint(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        if self.flash_enabled and timestep.ndim == 2:
            ar_mask += [i % self.flash_chunk_size == 0 for i in range(action_len)]
        else:
            ar_mask += [True] + [False] * (action_len - 1)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return (tokens, input_mask, ar_mask, adarms_cond)

    def _condition_action_tokens_on_previous_plan(
        self, obs: _model.Observation, action_tokens: jax.Array
    ) -> jax.Array:
        """Add a masked TCR residual without ever committing output actions."""
        fields = (
            obs.previous_plan_actions,
            obs.previous_plan_mask,
            obs.previous_plan_confidence,
        )
        if any((field is not None for field in fields)):
            raise ValueError(
                "Previous-plan tensors were supplied to a model with TCR disabled."
            )
        return action_tokens

    @at.typecheck
    def compute_training_objective(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train_step: at.Int[at.ArrayLike, ""] | int,
        total_train_steps: at.Int[at.ArrayLike, ""] | int,
        action_dim_mask: at.ArrayLike | None = None,
        objective_mode: Literal["visual_budget"] | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if objective_mode not in (None, "visual_budget"):
            raise ValueError(
                "Only the AIRoA-0 post-training objective is supported"
            )
        objective_mode = (
            self.visual_budget_objective_mode
            if objective_mode is None
            else objective_mode
        )
        (preprocess_rng, noise_rng, time_rng, objective_rng) = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=True,
            skip_color_jitter=self.eval_photometric_enabled,
            image_keys=self.image_view_names,
            image_baked_padding_by_view=self.image_baked_padding_by_view,
        )
        previous_plan_aux: dict[str, jax.Array] = {}
        from openpi.models import white_balance

        (observation, white_balance_aux) = white_balance.augment(
            observation,
            jax.random.fold_in(preprocess_rng, 22338),
            strength=self.white_balance_strength,
            probability=self.white_balance_probability,
        )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        objective_mode = objective_mode
        flash_aux = {}
        flash_valid = None
        if self.flash_enabled:
            from openpi.models import flash

            (x_t, u_t, time, flash_valid, packed_dimensions) = flash.training_inputs(
                jax.random.fold_in(time_rng, 1179402579),
                actions,
                chunk_size=self.flash_chunk_size,
                buffer_size=self.flash_buffer_size,
                temporal=observation.action_time_valid_mask,
                dimensions=observation.action_time_dim_valid_mask,
                levels=self.flash_flow_times,
                **{}
                if self.flash_flow_substeps is None
                else {"substeps": self.flash_flow_substeps},
            )
            actions = flash.pack(
                actions,
                chunk_size=self.flash_chunk_size,
                buffer_size=self.flash_buffer_size,
            )
            observation = dataclasses.replace(
                observation,
                action_time_valid_mask=flash_valid,
                action_time_dim_valid_mask=packed_dimensions,
            )
            flash_aux = {
                "flash_enabled": jnp.asarray(1.0, jnp.float32),
                "flash_supervised_fraction": jnp.mean(flash_valid.astype(jnp.float32)),
            }
        tt_rtc_mask = None
        tt_rtc_aux = {}
        expected_mask_prior_objective = self.visual_budget_objective_mode
        if (
            self.visual_budget_mpa_enabled
            and objective_mode != expected_mask_prior_objective
        ):
            raise ValueError(
                f"S2 mask-prior model received objective_mode={objective_mode!r}; expected {expected_mask_prior_objective!r} to avoid silent fallback."
            )
        image_tokens_by_name = self._embed_image_tokens_by_name(observation, train=True)
        prompt_tokens = self._embed_prompt_tokens(observation)
        pa_loss = None
        pa_metrics = {}
        if self.pa_aux_loss_weight:
            from openpi.models import pa_auxiliary

            (pa_prefix, pa_prefix_mask, _) = self._compose_prefix_tokens(
                observation, image_tokens_by_name, prompt_tokens
            )
            (inputs, targets, mask) = (
                observation.pa_input_tokens,
                observation.pa_target_tokens,
                observation.pa_loss_mask,
            )
            expected = (actions.shape[0], self.pa_aux_max_token_len)
            if any((x is None or x.shape != expected for x in (inputs, targets, mask))):
                raise ValueError("Missing or malformed training PA fields")
            decoder = self.PaliGemma.llm(inputs, method="embed")
            tokens = jnp.concatenate([pa_prefix, decoder], axis=1)
            valid = jnp.concatenate([pa_prefix_mask, mask], axis=1)
            ((hidden, _), _) = self.PaliGemma.llm(
                [tokens, None],
                positions=jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0),
                mask=pa_auxiliary.attention_mask(pa_prefix_mask, mask),
            )
            per_sample = pa_auxiliary.token_cross_entropy(
                lambda h: self.PaliGemma.llm(h, method="decode"),
                hidden[:, -inputs.shape[1] :],
                targets,
                mask,
                chunk_size=self.pa_aux_logit_chunk_size,
            )
            active = jnp.any(mask, axis=-1)
            ce = per_sample.sum() / jnp.maximum(active.sum(), 1)
            pa_loss = self.pa_aux_loss_weight * ce
            pa_metrics = {
                "pa_aux_ce": ce,
                "pa_aux_loss": pa_loss,
                "pa_aux_sht_fraction": jnp.mean(active.astype(jnp.float32)),
            }
        image_dtype = next(iter(image_tokens_by_name.values())).dtype
        dual_beta = None
        dual_tau = None
        dual_lambda_anchor = None
        dual_gate_floor = None
        dual_gate_aux: dict[str, jax.Array] = {}
        mask_logits_by_name: dict[str, jax.Array] | None = None
        dual_beta = self._interpolate_schedule_value(
            self.visual_budget_gate_prior_bias_start,
            self.visual_budget_gate_prior_bias_end,
            step=train_step,
            total_steps=total_train_steps,
            dtype=image_dtype,
        )
        dual_tau = self._interpolate_schedule_value(
            self.visual_budget_gate_tau_start,
            self.visual_budget_gate_tau_end,
            step=train_step,
            total_steps=total_train_steps,
            dtype=image_dtype,
        )
        dual_lambda_anchor = self._interpolate_schedule_value(
            self.visual_budget_gate_lambda_anchor_start,
            self.visual_budget_gate_lambda_anchor_end,
            step=train_step,
            total_steps=total_train_steps,
            dtype=image_dtype,
        )
        dual_gate_floor = self._interpolate_schedule_value(
            self.visual_budget_gate_floor,
            self.visual_budget_gate_floor_end,
            step=train_step,
            total_steps=total_train_steps,
            dtype=image_dtype,
        )
        temporal_context = None
        routed_aux: dict[str, jax.Array | dict[str, jax.Array]] | None = None
        proprio_prefix_aux: dict[str, jax.Array] | None = None
        prefix_key_log_prior = None
        mpa_aux: dict[str, jax.Array] = {}
        (history_tokens_by_name, history_token_masks_by_name) = (
            self._embed_history_image_tokens_by_name(observation)
        )
        prefix_image_tokens_by_name = image_tokens_by_name
        assert dual_beta is not None
        assert dual_tau is not None
        assert dual_lambda_anchor is not None
        assert dual_gate_floor is not None
        mask_logits_by_name = self._compute_visual_budget_mask_logits(
            observation, image_tokens_by_name, prompt_tokens, beta=dual_beta
        )
        (prefix_image_tokens_by_name, dual_gate_aux) = (
            self._apply_visual_budget_soft_gate(
                observation,
                image_tokens_by_name,
                prompt_tokens,
                beta=dual_beta,
                tau=dual_tau,
                gate_floor=dual_gate_floor,
                lambda_anchor=dual_lambda_anchor,
                compute_aux=True,
                mask_logits_by_name=mask_logits_by_name,
            )
        )
        (mpa_priors_by_name, mpa_aux) = self._compute_visual_budget_mpa_log_priors(
            observation,
            image_tokens_by_name,
            prompt_tokens,
            mask_logits_by_name=mask_logits_by_name,
        )
        (rgb_memory_tokens, rgb_memory_mask) = self._build_rgb_memory_tokens(
            observation,
            prefix_image_tokens_by_name,
            prompt_tokens,
            history_tokens_by_name=history_tokens_by_name,
            history_token_masks_by_name=history_token_masks_by_name,
        )
        (proprio_memory_tokens, proprio_memory_mask) = (
            self._build_proprio_memory_tokens(observation, prompt_tokens)
        )
        (proprio_memory_tokens, proprio_memory_mask, proprio_prefix_aux) = (
            self._apply_temporal_proprio_prefix_gate(
                proprio_memory_tokens,
                proprio_memory_mask,
                dtype=prompt_tokens.dtype
                if prompt_tokens is not None
                else observation.state.dtype,
                train_step=train_step,
            )
        )
        (prefix_tokens, prefix_mask, prefix_ar_mask) = self._compose_prefix_tokens(
            observation,
            prefix_image_tokens_by_name,
            prompt_tokens,
            rgb_memory_tokens=rgb_memory_tokens,
            rgb_memory_mask=rgb_memory_mask,
            proprio_memory_tokens=proprio_memory_tokens,
            proprio_memory_mask=proprio_memory_mask,
        )
        prefix_key_log_prior = self._compose_prefix_mpa_log_prior(
            observation,
            mpa_priors_by_name,
            prompt_tokens=prompt_tokens,
            rgb_memory_tokens=rgb_memory_tokens,
            proprio_memory_tokens=proprio_memory_tokens,
        )
        progress_loss = None
        progress_metrics = {}
        stage_loss = None
        stage_metrics = {}
        stage_kv_cache = None
        if self.stage_num_classes:
            from openpi.models import discrete_stage

            labels = observation.stage_label
            if labels is None or labels.shape != (actions.shape[0],):
                raise ValueError("Stage training requires one stage_label per anchor")
            ((stage_hidden, _), stage_kv_cache) = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=make_attn_mask(prefix_mask, prefix_ar_mask),
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
            )
            logits = self._stage_logits(stage_hidden, prefix_mask, observation.state)
            (stage_loss, stage_metrics) = discrete_stage.supervised_loss(
                logits, labels, self.stage_loss_weight
            )
        if self.stage_pa_conditioning:
            (prefix_tokens, prefix_mask, stage_kv_cache) = self._append_stage_pa(
                prefix_tokens, prefix_mask, stage_kv_cache, labels
            )
            count = self._stage_pa_token_count()
            prefix_ar_mask = jnp.concatenate(
                [prefix_ar_mask, jnp.array([True] + [False] * (count - 1))]
            )
            if prefix_key_log_prior is not None:
                prefix_key_log_prior = jnp.pad(
                    prefix_key_log_prior, ((0, 0), (0, count))
                )
        (suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond) = self.embed_suffix(
            observation, x_t, time
        )
        adarms_cond = self._add_stage_condition(adarms_cond, observation.stage_label)
        adarms_cond = self._add_progress_condition(
            adarms_cond, observation.progress_target
        )
        if flash_valid is not None:
            suffix_mask = flash_valid
            suffix_tokens = jnp.where(flash_valid[..., None], suffix_tokens, 0)
            adarms_cond = jnp.where(flash_valid[..., None], adarms_cond, 0)
        dtype = suffix_tokens.dtype
        aux: dict[str, jax.Array] = {
            **previous_plan_aux,
            **white_balance_aux,
            **tt_rtc_aux,
            **flash_aux,
        }
        total_loss = jnp.asarray(0.0, dtype=dtype)
        batch_size = int(actions.shape[0])
        dataset_root_index = observation.dataset_root_index
        if dataset_root_index is not None and dataset_root_index.shape != (batch_size,):
            raise ValueError(
                f"dataset_root_index must have shape {(batch_size,)}, got {dataset_root_index.shape}."
            )
        action_time_valid_mask = observation.action_time_valid_mask
        action_time_dim_valid_mask = observation.action_time_dim_valid_mask
        if tt_rtc_mask is not None:
            suffix_mask = ~tt_rtc_mask
            action_time_valid_mask = (
                suffix_mask
                if action_time_valid_mask is None
                else jnp.logical_and(action_time_valid_mask, suffix_mask)
            )
            aux["tt_rtc_suffix_fraction"] = jnp.mean(suffix_mask.astype(dtype))
        if action_time_dim_valid_mask is not None:
            expected_prefix = tuple(actions.shape[:-1])
            if tuple(action_time_dim_valid_mask.shape[:-1]) != expected_prefix:
                raise ValueError(
                    f"action_time_dim_valid_mask must match action batch/time shape {expected_prefix}, got {action_time_dim_valid_mask.shape}."
                )
            if int(action_time_dim_valid_mask.shape[-1]) > int(actions.shape[-1]):
                raise ValueError(
                    "action_time_dim_valid_mask exceeds the model action dimension."
                )
            action_time_dim_valid_mask = jnp.asarray(
                action_time_dim_valid_mask, dtype=jnp.bool_
            )
            dynamic_time_valid = jnp.any(action_time_dim_valid_mask, axis=-1)
            action_time_valid_mask = (
                dynamic_time_valid
                if action_time_valid_mask is None
                else jnp.logical_and(action_time_valid_mask, dynamic_time_valid)
            )
            aux["action_time_dim_valid_fraction"] = jnp.mean(
                action_time_dim_valid_mask.astype(dtype)
            )
        if action_time_valid_mask is not None:
            expected_shape = tuple(actions.shape[:-1])
            if tuple(action_time_valid_mask.shape) != expected_shape:
                raise ValueError(
                    f"action_time_valid_mask must have shape {expected_shape}, got {action_time_valid_mask.shape}."
                )
            action_time_valid_mask = jnp.asarray(
                action_time_valid_mask, dtype=jnp.bool_
            )
            sample_has_valid_action = jnp.any(action_time_valid_mask, axis=-1)
            aux["action_time_valid_fraction"] = jnp.mean(
                action_time_valid_mask.astype(dtype)
            )
            aux["action_time_all_invalid_sample_fraction"] = jnp.mean(
                jnp.logical_not(sample_has_valid_action).astype(dtype)
            )
        else:
            sample_has_valid_action = jnp.ones((batch_size,), dtype=jnp.bool_)

        def _time_mask_for_samples(sample_mask: jax.Array) -> jax.Array:
            selected = jnp.broadcast_to(
                jnp.asarray(sample_mask, dtype=jnp.bool_)[:, None],
                tuple(actions.shape[:-1]),
            )
            if action_time_valid_mask is not None:
                selected = jnp.logical_and(selected, action_time_valid_mask)
            return selected

        anchor_info = None if routed_aux is None else routed_aux.get("anchor_info")
        anchor_valid_mask = None
        if isinstance(anchor_info, dict):
            anchor_valid_mask = jnp.zeros((batch_size,), dtype=jnp.bool_)
            for valid in anchor_info["valid_by_name"].values():
                anchor_valid_mask = jnp.logical_or(
                    anchor_valid_mask, valid.astype(jnp.bool_)
                )

        def _add_temporal_diagnostics(
            aux_dict: dict[str, jax.Array],
            *,
            full_loss_per_sample: jax.Array | None = None,
            selected_loss_per_sample: jax.Array | None = None,
        ) -> None:
            return

        def _add_dataset_root_diagnostics(
            aux_dict: dict[str, jax.Array],
            *,
            full_loss_per_sample: jax.Array | None = None,
            selected_loss_per_sample: jax.Array | None = None,
        ) -> None:
            if dataset_root_index is None:
                return
            root_indices = jnp.asarray(dataset_root_index, dtype=jnp.int32)
            for root_index in (0, 1):
                mask = root_indices == root_index
                aux_dict[f"dataset_root_{root_index}_fraction"] = jnp.mean(
                    mask.astype(dtype)
                )
                if full_loss_per_sample is not None:
                    aux_dict[f"loss_full_root_{root_index}"] = self._masked_scalar_mean(
                        full_loss_per_sample,
                        jnp.logical_and(mask, sample_has_valid_action),
                        dtype=dtype,
                    )
                if selected_loss_per_sample is not None:
                    aux_dict[f"loss_selected_root_{root_index}"] = (
                        self._masked_scalar_mean(
                            selected_loss_per_sample,
                            jnp.logical_and(mask, sample_has_valid_action),
                            dtype=dtype,
                        )
                    )

        full_loss_per_sample = None
        if prefix_key_log_prior is None:
            raise ValueError(
                "S2 mask-prior objective is missing its aligned prefix log-prior."
            )
        prior_strength = None
        if self.visual_budget_s2_alpha_warmup_steps is None:
            raise ValueError(
                "visual budget objective is missing its alpha warmup provenance."
            )
        prior_strength = _budget_s2_alpha(
            train_step, warmup_steps=self.visual_budget_s2_alpha_warmup_steps
        )
        mpa_aux["visual_budget_s2_alpha"] = prior_strength
        mpa_pred = self._run_prefix_suffix_pass(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
            prefix_key_log_prior=prefix_key_log_prior,
            prefix_key_log_prior_strength=prior_strength,
            prefix_kv_cache=stage_kv_cache,
        )
        mpa_chunked_loss = _action_dim_reduced_mse(
            mpa_pred, u_t, action_dim_mask, action_time_dim_valid_mask
        )
        if action_time_valid_mask is None:
            mask_prior_loss_per_sample = jnp.mean(mpa_chunked_loss, axis=-1)
            loss_mpa = jnp.mean(mpa_chunked_loss)
        else:
            mask_prior_loss_per_sample = _model.action_time_loss_per_sample(
                mpa_chunked_loss, action_time_valid_mask
            )
            loss_mpa = _model.action_time_loss_mean(
                mpa_chunked_loss, action_time_valid_mask
            )
        if dual_lambda_anchor is None or not dual_gate_aux:
            raise ValueError(
                "visual budget training failed closed because its embedding interface is inactive."
            )
        loss_budget_base = (
            self._visual_budget_budget_weight_for_kind("base", dtype)
            * dual_gate_aux["visual_budget_budget_loss_base"]
        )
        loss_budget_wrist = (
            self._visual_budget_budget_weight_for_kind("wrist", dtype)
            * dual_gate_aux["visual_budget_budget_loss_wrist"]
        )
        loss_budget = loss_budget_base + loss_budget_wrist
        loss_anchor = dual_lambda_anchor.astype(dtype) * (
            dual_gate_aux["visual_budget_anchor_loss_base"]
            + dual_gate_aux["visual_budget_anchor_loss_wrist"]
        )
        loss_gated_weighted = (
            jnp.asarray(self.visual_budget_gate_lambda_gated, dtype=dtype) * loss_mpa
        )
        total_loss = loss_gated_weighted + loss_budget + loss_anchor
        aux.update(
            {
                **mpa_aux,
                **dual_gate_aux,
                "loss_full": jnp.asarray(0.0, dtype=dtype),
                "loss_gated": loss_mpa,
                "loss_gated_weighted": loss_gated_weighted,
                "loss_main": loss_gated_weighted,
                "loss_budget": loss_budget,
                "loss_budget_base_weighted": loss_budget_base,
                "loss_budget_wrist_weighted": loss_budget_wrist,
                "loss_anchor": loss_anchor,
                "visual_budget_objective_gated_frac": jnp.asarray(1.0, dtype=dtype),
                "visual_budget_lambda_gated": jnp.asarray(
                    self.visual_budget_gate_lambda_gated, dtype=dtype
                ),
                "visual_budget_lambda_budget_base": self._visual_budget_budget_weight_for_kind(
                    "base", dtype
                ),
                "visual_budget_lambda_budget_wrist": self._visual_budget_budget_weight_for_kind(
                    "wrist", dtype
                ),
                "visual_budget_embedding_gate_active": jnp.asarray(1.0, dtype=jnp.float32),
                "visual_budget_attention_prior_active": jnp.asarray(1.0, dtype=jnp.float32),
            }
        )
        _add_temporal_diagnostics(
            aux, selected_loss_per_sample=mask_prior_loss_per_sample
        )
        _add_dataset_root_diagnostics(
            aux, selected_loss_per_sample=mask_prior_loss_per_sample
        )
        if stage_loss is not None:
            aux["action_objective_before_stage"] = total_loss
            total_loss = total_loss + stage_loss
            aux.update(stage_metrics)
        if pa_loss is not None:
            aux["action_objective_loss"] = aux.get(
                "action_objective_before_stage", total_loss
            )
            total_loss = total_loss + pa_loss
            aux.update(pa_metrics)
        return (total_loss, aux)

    def compute_loss(self, *args, **kwargs):
        raise ValueError(
            "Use compute_training_objective with an explicit training step"
        )

    def sample_actions(self, *args, **kwargs):
        raise NotImplementedError("This runtime supports post-training only")
