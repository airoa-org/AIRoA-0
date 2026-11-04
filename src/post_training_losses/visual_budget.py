# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Visual budget gate statistics and the read-only attention-prior interface."""

import jax
import jax.numpy as jnp


def linear_schedule(start, end, *, step, total_steps, dtype=jnp.float32):
    """Interpolate over update indices 0..total_steps-1, clamping outside the range."""
    step_value = jnp.asarray(step, dtype=dtype)
    total_steps_value = jnp.asarray(total_steps, dtype=dtype)
    denom = jnp.maximum(total_steps_value - jnp.asarray(1.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    progress = jnp.where(
        total_steps_value <= jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        jnp.clip(step_value / denom, 0.0, 1.0),
    )
    start_value = jnp.asarray(start, dtype=dtype)
    end_value = jnp.asarray(end, dtype=dtype)
    return start_value + (end_value - start_value) * progress


def embedding_gates(logits, token_valid, *, tau, gate_floor, dtype=None):
    """Differentiable raw sigmoid and floored gates for the embedding interface.

    Attention projection is deliberately separate. The caller multiplies its
    image embeddings by ``effective_gates[..., None]`` before its action forward.
    """
    dtype = logits.dtype if dtype is None else dtype
    if logits.shape != token_valid.shape:
        raise ValueError("Gate logits and token-valid mask shapes must agree")
    zero = jnp.asarray(0.0, dtype=dtype)
    eps = jnp.asarray(1e-6, dtype=dtype)
    tau = jnp.maximum(jnp.asarray(tau, dtype=dtype), eps)
    floor = jnp.asarray(gate_floor, dtype=dtype)
    gates = jnp.where(token_valid, jax.nn.sigmoid(logits / tau).astype(dtype), zero)
    effective = jnp.where(token_valid, floor + (1.0 - floor) * gates, zero)
    return gates, effective


def project_attention_probabilities(logits, valid_mask, *, tau, keep_ratio, iterations=25):
    """Cap each sample's mean probability by a non-negative FP32 logit shift."""
    logits = jnp.asarray(logits, dtype=jnp.float32)
    valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if logits.shape != valid_mask.shape or logits.ndim != 2:
        raise ValueError("Projection requires matching [batch, token] logits and validity")
    if not 0.0 < float(keep_ratio) < 1.0 or iterations <= 0:
        raise ValueError("keep_ratio must be in (0, 1) and iterations must be positive")
    tau = jnp.asarray(tau, dtype=jnp.float32)
    ratio = jnp.asarray(keep_ratio, dtype=jnp.float32)
    valid_f = valid_mask.astype(jnp.float32)
    valid_count = jnp.sum(valid_f, axis=-1)
    safe_logits = jnp.where(valid_mask, logits, jnp.asarray(0.0, dtype=jnp.float32))

    def probability_and_mean(nu):
        probability = jax.nn.sigmoid((safe_logits - nu[:, None]) / tau)
        probability = jnp.where(valid_mask, probability, jnp.asarray(0.0, dtype=jnp.float32))
        mean = jnp.sum(probability, axis=-1) / jnp.maximum(valid_count, 1.0)
        return probability, mean

    zero_shift = jnp.zeros((logits.shape[0],), dtype=jnp.float32)
    raw_probability, raw_mean = probability_and_mean(zero_shift)
    binding = jnp.logical_and(valid_count > 0.0, raw_mean > ratio)
    max_valid_logit = jnp.max(jnp.where(valid_mask, logits, jnp.asarray(-jnp.inf, dtype=jnp.float32)), axis=-1)
    max_valid_logit = jnp.where(valid_count > 0.0, max_valid_logit, jnp.asarray(0.0, dtype=jnp.float32))
    ratio_logit = jnp.log(ratio) - jnp.log1p(-ratio)
    upper = jnp.maximum(max_valid_logit - tau * ratio_logit, 0.0)
    lower = jnp.zeros_like(upper)

    def bisect(_, bounds):
        lower_bound, upper_bound = bounds
        midpoint = (lower_bound + upper_bound) * jnp.asarray(0.5, dtype=jnp.float32)
        _, midpoint_mean = probability_and_mean(midpoint)
        too_dense = midpoint_mean > ratio
        next_lower = jnp.where(too_dense, midpoint, lower_bound)
        next_upper = jnp.where(too_dense, upper_bound, midpoint)
        return jnp.where(binding, next_lower, zero_shift), jnp.where(binding, next_upper, zero_shift)

    _, upper = jax.lax.fori_loop(0, int(iterations), bisect, (lower, upper))
    nu = jnp.where(binding, upper, zero_shift)
    probability, projected_mean = probability_and_mean(nu)
    return probability, nu, raw_mean, projected_mean, binding


def attention_log_prior(logits, valid_mask, *, tau, keep_ratio, logit_clip=4.0):
    """Return the detached, non-centered budget prior and projected probabilities.

    Add the prior to attention key logits with the chosen warmup strength. This
    interface has zero gradient to gate logits; the embedding interface does not.
    ``valid_mask`` includes both token padding and whole-view validity.
    """
    logits = jax.lax.stop_gradient(logits)
    probability, *_ = project_attention_probabilities(logits, valid_mask, tau=tau, keep_ratio=keep_ratio)
    floor = jnp.exp(-jnp.asarray(logit_clip, dtype=jnp.float32))
    effective = floor + (1.0 - floor) * probability
    prior = jnp.clip(jnp.log(effective), -logit_clip, 0.0)
    prior = jnp.where(valid_mask, prior, jnp.asarray(0.0, dtype=jnp.float32))
    return jax.lax.stop_gradient(prior), jax.lax.stop_gradient(probability)


def attention_strength(step, *, warmup_steps=500):
    if warmup_steps <= 0:
        raise ValueError("Attention-prior warmup must be positive")
    return jnp.clip(jnp.asarray(step, jnp.float32) / jnp.asarray(warmup_steps, jnp.float32), 0.0, 1.0)


def visual_budget_statistics(
    logits_by_view,
    token_valid_by_view,
    token_prior_by_view,
    *,
    view_kinds,
    tau,
    keep_ratio_base=0.9,
    keep_ratio_wrist=0.9,
    anchor_delta=0.2,
    view_valid_by_view=None,
    dtype=None,
):
    """Unweighted budget/anchor terms, aggregated by valid sample-view per kind.

    Logits are the same predictions used by both visual budget interfaces, including
    any prior bias already applied by the caller. Priors must be finite token
    priors on the matching grid. Disabled priors are represented by zeros.
    """
    keys = set(logits_by_view)
    if not keys or keys != set(token_valid_by_view) or keys != set(token_prior_by_view) or keys != set(view_kinds):
        raise ValueError("All nonempty per-view mappings must have identical keys")
    if view_valid_by_view is not None and keys != set(view_valid_by_view):
        raise ValueError("Whole-view validity must match all views")
    dtype = next(iter(logits_by_view.values())).dtype if dtype is None else dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    eps = jnp.asarray(1e-6, dtype=dtype)
    stats = {kind: {key: zero for key in ("budget_num", "budget_den", "anchor_num", "anchor_den")}
             for kind in ("base", "wrist")}
    delta = jnp.asarray(anchor_delta, dtype=dtype)
    for name, logits in logits_by_view.items():
        kind = view_kinds[name]
        if kind not in stats:
            raise ValueError("Each view kind must be 'base' or 'wrist'")
        valid = jnp.asarray(token_valid_by_view[name], bool)
        prior = jnp.asarray(token_prior_by_view[name], dtype=dtype)
        if logits.ndim != 2 or logits.shape != valid.shape or prior.shape != valid.shape:
            raise ValueError("Each view needs matching [batch, token] logits, mask, and prior")
        view_mask = jnp.ones((logits.shape[0],), dtype=dtype)
        if view_valid_by_view is not None:
            view_mask = jnp.asarray(view_valid_by_view[name], dtype=dtype)
            if view_mask.shape != (logits.shape[0],):
                raise ValueError("Whole-view validity must have shape [batch]")
        gates, _ = embedding_gates(logits, valid, tau=tau, gate_floor=0.0, dtype=dtype)
        token_weight = view_mask[:, None] * valid.astype(dtype)
        count_per_sample = jnp.sum(token_weight, axis=1)
        sample_valid = (count_per_sample > 0).astype(dtype)
        count = jnp.sum(sample_valid)
        mean_gate = jnp.sum(gates * token_weight, axis=1) / jnp.maximum(count_per_sample, eps)
        ratio = jnp.asarray(keep_ratio_base if kind == "base" else keep_ratio_wrist, dtype=dtype)
        budget_error = jnp.square(mean_gate - ratio)
        stats[kind]["budget_num"] += jnp.sum(sample_valid * budget_error)
        stats[kind]["budget_den"] += count
        prior_mass = jnp.sum(prior * token_weight, axis=1)
        anchor_valid = sample_valid * (prior_mass > 0).astype(dtype)
        anchor_value = jnp.sum(prior * jax.nn.relu(delta - gates) * token_weight, axis=1) / jnp.maximum(prior_mass, eps)
        stats[kind]["anchor_num"] += jnp.sum(anchor_valid * anchor_value)
        stats[kind]["anchor_den"] += jnp.sum(anchor_valid)
    return {f"{term}_{kind}": stats[kind][f"{term}_num"] / jnp.maximum(stats[kind][f"{term}_den"], eps)
            for term in ("budget", "anchor") for kind in ("base", "wrist")}
