# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""NUF training buffer packing, flow-time sampling and isolated attention.

For N slots of C actions, input length is N*C and packed length is N*N*C.
Each packed state right-aligns an increasingly long prefix of clean chunks.
Consequently chunk exposures are N, N-1, ..., 1; do not multiply them again.
The caller keeps the native action normalization and target validity masks.
Invalid temporal and physical source targets are zeroed before interpolation
exactly as in training; valid NaN targets remain invalid numerical inputs.
Clock arguments are static configuration values, not traced JAX arrays.
"""

import itertools
import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np

from .flow import flow_matching_loss


def flow_levels(levels, *, buffer_size):
    if levels is None:
        return None
    if isinstance(levels, str | bytes) or not hasattr(levels, '__len__'):
        raise ValueError('NUF flow times must be a sequence of N pre-step levels')
    values = tuple(levels)
    if len(values) != buffer_size:
        raise ValueError('NUF flow times need exactly one level per buffer slot')
    if any((isinstance(v, bool | np.bool_) or not isinstance(v, numbers.Real) for v in values)):
        raise ValueError('NUF flow times must be real numbers, not booleans')
    values = tuple((float(v) for v in values))
    if not all((math.isfinite(v) and v > 0 for v in values)):
        raise ValueError('NUF flow times must be finite and positive')
    single = np.asarray(values, np.float32)
    if any((b <= a for (a, b) in itertools.pairwise(values))) or (np.diff(single) <= 0).any():
        raise ValueError('NUF flow times must be strictly ascending (also in float32)')
    if values[-1] != 1.0:
        raise ValueError('NUF flow times must end exactly at noise level 1')
    return values


def post_levels(levels):
    return (0.0, *levels[:-1])


def flow_substeps(paths, *, levels, buffer_size, max_clock_points=8):
    if paths is None:
        return None
    levels = flow_levels(levels, buffer_size=buffer_size)
    if levels is None or isinstance(paths, (str, bytes)) or len(paths) != buffer_size:
        raise ValueError('NUF substeps require one path per nonuniform slot')
    result = []
    for (path, start, end) in zip(paths, levels, post_levels(levels), strict=True):
        if not isinstance(path, (tuple, list)) or not 2 <= len(path) <= max_clock_points:
            raise ValueError(f'NUF slot paths need 2..{max_clock_points} clock points')
        if any((isinstance(v, bool | np.bool_) or not isinstance(v, numbers.Real) for v in path)):
            raise ValueError('NUF substep times must be finite real numbers')
        values = tuple((float(v) for v in path))
        if not all((math.isfinite(v) for v in values)) or values[0] != start or values[-1] != end or any((b >= a for (a, b) in itertools.pairwise(values))) or (np.diff(np.asarray(values, np.float32)) >= 0).any():
            raise ValueError('NUF slot path must decrease exactly from its pre to post level')
        result.append(values)
    return tuple(result)


def pack(values, *, chunk_size, buffer_size, padding=0):
    horizon = chunk_size * buffer_size
    if values.ndim < 2 or values.shape[1] != horizon:
        raise ValueError('NUF packing requires [B,N*C,...] targets')
    states = []
    for count in range(1, buffer_size + 1):
        pads = [(0, 0), ((buffer_size - count) * chunk_size, 0)] + [(0, 0)] * (values.ndim - 2)
        states.append(jnp.pad(values[:, :count * chunk_size], pads, constant_values=padding))
    return jnp.concatenate(states, axis=1)


def training_inputs(rng, actions, *, chunk_size, buffer_size, temporal=None, dimensions=None, levels=None, substeps=None):
    (b, h, a) = actions.shape
    if h != chunk_size * buffer_size:
        raise ValueError('NUF action horizon must equal buffer_size * chunk_size')
    temporal = jnp.ones((b, h), bool) if temporal is None else jnp.asarray(temporal, bool)
    if temporal.shape != (b, h):
        raise ValueError('NUF temporal mask differs from targets')
    if dimensions is not None:
        if dimensions.ndim != 3 or dimensions.shape[:2] != (b, h) or dimensions.shape[-1] > a:
            raise ValueError('NUF physical dimension mask differs from targets')
        dimensions = jnp.asarray(dimensions, bool)
        temporal = temporal & jnp.any(dimensions, axis=-1)
    full_dimensions = jnp.broadcast_to(temporal[..., None], actions.shape)
    if dimensions is not None:
        full_dimensions = full_dimensions.at[..., :dimensions.shape[-1]].set(temporal[..., None] & dimensions)
    clean = pack(jnp.where(full_dimensions, actions, 0), chunk_size=chunk_size, buffer_size=buffer_size)
    valid = pack(temporal, chunk_size=chunk_size, buffer_size=buffer_size, padding=False)
    dim_valid = None if dimensions is None else pack(dimensions, chunk_size=chunk_size, buffer_size=buffer_size, padding=False)
    (time_rng, noise_rng) = jax.random.split(rng)
    u = jax.random.beta(time_rng, 1.5, 1.0, (b, buffer_size, buffer_size))
    if substeps is not None:
        paths = flow_substeps(substeps, levels=levels, buffer_size=buffer_size)
        choices = jax.random.uniform(jax.random.fold_in(time_rng, 1398096467), u.shape)
        slot_values = []
        for (slot, path) in enumerate(paths):
            index = jnp.minimum((choices[:, :, slot] * (len(path) - 1)).astype(jnp.int32), len(path) - 2)
            high = jnp.asarray(path[:-1], jnp.float32)[index]
            low = jnp.maximum(jnp.asarray(path[1:], jnp.float32)[index], 0.001 * high)
            slot_values.append(jnp.minimum(low + (high - low) * u[:, :, slot], high))
        times = jnp.stack(slot_values, axis=-1)
    elif levels is None:
        slots = jnp.arange(buffer_size, dtype=jnp.float32)
        times = 0.001 + 0.999 * (slots[None, None, :] + u) / buffer_size
    else:
        levels = flow_levels(levels, buffer_size=buffer_size)
        high = np.asarray(levels, np.float64)
        low = np.maximum(np.asarray(post_levels(levels), np.float64), 0.001 * high)
        (low, span) = (jnp.asarray(low, jnp.float32), jnp.asarray(high - low, jnp.float32))
        times = jnp.minimum(low + span * u, jnp.asarray(high, jnp.float32))
    times = jnp.repeat(times, chunk_size, axis=-1).reshape(b, buffer_size * h)
    noise = jax.random.normal(noise_rng, clean.shape)
    x = jnp.where(valid[..., None], (1 - times[..., None]) * clean + times[..., None] * noise, 0)
    target = jnp.where(valid[..., None], noise - clean, 0)
    return (x, target, times, valid, dim_valid)


def packed_causal_attention(valid, *, chunk_size, buffer_size):
    (b, length) = valid.shape
    horizon = chunk_size * buffer_size
    if length not in (horizon, buffer_size * horizon):
        raise ValueError('NUF suffix must be one buffer or all packed cold-start buffers')
    index = jnp.arange(length)
    same_buffer = index[:, None] // horizon == index[None, :] // horizon
    chunks = index % horizon // chunk_size
    causal = chunks[None, :] <= chunks[:, None]
    mask = same_buffer & causal
    mask = mask[None] & valid[:, :, None] & valid[:, None, :]
    local = valid.reshape(b, -1, horizon)
    positions = (jnp.cumsum(local, axis=-1) - 1).reshape(b, length)
    return (mask, positions)


def nuf_loss(prediction, target, valid, *, dimensions=None,
             action_dim_mask=None, sample_weights=None):
    """Reduce packed velocity errors with the original validity denominator.

    ``target``, ``valid`` and ``dimensions`` come from ``training_inputs``.
    This reduction preserves repeated-prefix exposure instead of reweighting
    each original action equally. Predictions/targets must be finite.
    """
    return flow_matching_loss(
        prediction, target, action_dim_mask=action_dim_mask,
        action_time_dim_valid_mask=dimensions,
        action_time_valid_mask=valid, sample_weights=sample_weights)
