# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Action flow-matching targets and source-preserving masked MSE reductions.

Predictions and velocity targets must be finite before reduction. Masking uses
multiplication, matching the training objective: zero times NaN remains NaN.
These reductions deliberately do not add nan_to_num or change that behavior.
Action normalization belongs to the caller; no normalization is fitted here.
"""

import jax.numpy as jnp


def exact_action_dim_reduced_mse(prediction, target, action_dim_mask=None, action_time_dim_valid_mask=None):
    squared_error = jnp.square(prediction - target)
    if action_dim_mask is None and action_time_dim_valid_mask is None:
        return jnp.mean(squared_error, axis=-1)
    combined = jnp.ones(squared_error.shape, dtype=jnp.float32)
    if action_dim_mask is not None:
        action_dim_mask = jnp.asarray(action_dim_mask, dtype=jnp.float32)
        if action_dim_mask.shape[-1] != int(squared_error.shape[-1]):
            raise ValueError(f'action_dim_mask shape mismatch: expected last dim {squared_error.shape[-1]}, got {action_dim_mask.shape[-1]}')
        combined = combined * action_dim_mask.reshape((1,) * (squared_error.ndim - 1) + (int(squared_error.shape[-1]),))
    if action_time_dim_valid_mask is not None:
        dynamic = jnp.asarray(action_time_dim_valid_mask, dtype=jnp.float32)
        if dynamic.shape[:-1] != squared_error.shape[:-1]:
            raise ValueError(f'action_time_dim_valid_mask prefix shape mismatch: expected {squared_error.shape[:-1]}, got {dynamic.shape[:-1]}.')
        if dynamic.shape[-1] > squared_error.shape[-1]:
            raise ValueError(f'action_time_dim_valid_mask has more action dims than the model: {dynamic.shape[-1]} > {squared_error.shape[-1]}.')
        dynamic = jnp.pad(dynamic, ((0, 0),) * (dynamic.ndim - 1) + ((0, squared_error.shape[-1] - dynamic.shape[-1]),))
        combined = combined * dynamic
    masked_error = squared_error * combined
    denom = jnp.maximum(jnp.sum(combined, axis=-1), 1.0)
    return jnp.sum(masked_error, axis=-1) / denom


def action_time_loss_mean(loss, valid_mask=None, *, sample_weights=None):
    loss = jnp.asarray(loss)
    if loss.ndim < 1:
        raise ValueError(f'Action-time loss must have at least one dimension, got {loss.shape}.')
    weights = None
    if sample_weights is not None:
        weights = jnp.asarray(sample_weights, dtype=loss.dtype)
        if weights.shape != loss.shape[:-1]:
            raise ValueError(f'sample_weights must have shape {loss.shape[:-1]}, got {weights.shape}.')
        weights = weights[..., None]
    if valid_mask is None:
        return jnp.mean(loss if weights is None else loss * weights)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if valid.shape != loss.shape:
        raise ValueError(f'action_time_valid_mask must have shape {loss.shape}, got {valid.shape}.')
    valid_weight = valid.astype(loss.dtype)
    numerator = loss * valid_weight
    if weights is not None:
        numerator = numerator * weights
    denominator = jnp.maximum(jnp.sum(valid_weight), jnp.asarray(1.0, dtype=loss.dtype))
    return jnp.sum(numerator) / denominator


def action_time_loss_per_sample(loss, valid_mask=None):
    loss = jnp.asarray(loss)
    if loss.ndim < 1:
        raise ValueError(f'Action-time loss must have at least one dimension, got {loss.shape}.')
    if valid_mask is None:
        return jnp.mean(loss, axis=-1)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if valid.shape != loss.shape:
        raise ValueError(f'action_time_valid_mask must have shape {loss.shape}, got {valid.shape}.')
    valid_weight = valid.astype(loss.dtype)
    denominator = jnp.maximum(jnp.sum(valid_weight, axis=-1), jnp.asarray(1.0, dtype=loss.dtype))
    return jnp.sum(loss * valid_weight, axis=-1) / denominator


def flow_matching_inputs(actions, noise, time):
    """Interpolate normalized clean actions at time 0 and noise at time 1.

    ``time`` has the leading batch shape of ``actions[..., horizon, dim]``.
    The caller supplies noise and time draws, so model RNG splitting is explicit.
    Inputs must be finite; this ordinary-flow path does not sanitize targets.
    """
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    return x_t, u_t


def flow_matching_loss(prediction, target, *, action_dim_mask=None,
                       action_time_dim_valid_mask=None,
                       action_time_valid_mask=None, sample_weights=None):
    """Return scalar MSE, reducing action dimensions before valid action times.

    Physical dimension masks may be narrower than model width: absent trailing
    dimensions receive zero validity. Sample weights scale the numerator only;
    the denominator counts valid action-time entries, without weight scaling.
    Finite all-invalid inputs return zero. Nonfinite masked operands retain
    the source multiplication behavior described in this module's docstring.
    """
    loss = exact_action_dim_reduced_mse(
        prediction, target, action_dim_mask, action_time_dim_valid_mask)
    return action_time_loss_mean(
        loss, action_time_valid_mask, sample_weights=sample_weights)
