# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FlashVLA buffer algebra (local flow time: clean=0, noise=1).

Independent implementation of arXiv:2608.27384, sections 3 and A.4. No
checkpoint parameters are introduced. Packed cold-start states share the
observation but never attend to each other. The caller retains native QC masks.

``levels=None`` is the historical uniform clock (slot s refines (s+1)/N -> s/N).
Optional nonuniform pre-step levels l_0<...<l_{N-1}=1 refine slot s from l_s to
l_{s-1} (l_{-1}=0); every call site keeps the legacy branch unchanged.
"""

import itertools
import math
import numbers
import jax
import jax.numpy as jnp
import numpy as np


def flow_levels(levels, *, buffer_size):
    """Validate and normalize nonuniform pre-step levels; None stays legacy."""
    if levels is None:
        return None
    if isinstance(levels, str | bytes) or not hasattr(levels, "__len__"):
        raise ValueError("Flash flow times must be a sequence of N pre-step levels")
    values = tuple(levels)
    if len(values) != buffer_size:
        raise ValueError("Flash flow times need exactly one level per buffer slot")
    if any(
        (
            isinstance(v, bool | np.bool_) or not isinstance(v, numbers.Real)
            for v in values
        )
    ):
        raise ValueError("Flash flow times must be real numbers, not booleans")
    values = tuple((float(v) for v in values))
    if not all((math.isfinite(v) and v > 0 for v in values)):
        raise ValueError("Flash flow times must be finite and positive")
    single = np.asarray(values, np.float32)
    if (
        any((b <= a for (a, b) in itertools.pairwise(values)))
        or (np.diff(single) <= 0).any()
    ):
        raise ValueError(
            "Flash flow times must be strictly ascending (also in float32)"
        )
    if values[-1] != 1.0:
        raise ValueError("Flash flow times must end exactly at noise level 1")
    return values


def post_levels(levels):
    """Post-refinement slot clock: slot s ends at the previous slot's level."""
    return (0.0, *levels[:-1])


def flow_substeps(paths, *, levels, buffer_size, max_clock_points=8):
    """Validate opt-in per-slot solver paths; None preserves every old branch.

    Slots finish on the existing shifted clock. Shorter paths wait before their
    update: N2's near slot stays noisy while the far slot takes its first step.
    The last shared evaluation therefore conditions both slots on the same
    current observation and on the near slot at its trained pre-step time.
    """
    if paths is None:
        return None
    levels = flow_levels(levels, buffer_size=buffer_size)
    if levels is None or isinstance(paths, (str, bytes)) or len(paths) != buffer_size:
        raise ValueError("Flash substeps require one path per nonuniform slot")
    result = []
    for path, start, end in zip(paths, levels, post_levels(levels), strict=True):
        if (
            not isinstance(path, (tuple, list))
            or not 2 <= len(path) <= max_clock_points
        ):
            raise ValueError(
                f"Flash slot paths need 2..{max_clock_points} clock points"
            )
        if any(
            (
                isinstance(v, bool | np.bool_) or not isinstance(v, numbers.Real)
                for v in path
            )
        ):
            raise ValueError("Flash substep times must be finite real numbers")
        values = tuple((float(v) for v in path))
        if (
            not all((math.isfinite(v) for v in values))
            or values[0] != start
            or values[-1] != end
            or any((b >= a for (a, b) in itertools.pairwise(values)))
            or (np.diff(np.asarray(values, np.float32)) >= 0).any()
        ):
            raise ValueError(
                "Flash slot path must decrease exactly from its pre to post level"
            )
        result.append(values)
    return tuple(result)


MAX_INFERENCE_STEPS = 128


def pack(values, *, chunk_size, buffer_size, padding=0):
    """Right-align the first j GT chunks in each of N isolated buffer states."""
    horizon = chunk_size * buffer_size
    if values.ndim < 2 or values.shape[1] != horizon:
        raise ValueError("Flash packing requires [B,N*C,...] targets")
    states = []
    for count in range(1, buffer_size + 1):
        pads = [(0, 0), ((buffer_size - count) * chunk_size, 0)] + [(0, 0)] * (
            values.ndim - 2
        )
        states.append(
            jnp.pad(values[:, : count * chunk_size], pads, constant_values=padding)
        )
    return jnp.concatenate(states, axis=1)


def training_inputs(
    rng,
    actions,
    *,
    chunk_size,
    buffer_size,
    temporal=None,
    dimensions=None,
    levels=None,
    substeps=None,
):
    """Construct all cold-start states with stratified per-slot flow times.

    Nonuniform levels reuse the same Beta(1.5,1) draw inside [l_{s-1}, l_s].
    The lower bound is max(l_{s-1}, 0.001*l_s): never exactly clean, always inside
    the slot interval, and the maximum is the slot's inference level.
    """
    (b, h, a) = actions.shape
    if h != chunk_size * buffer_size:
        raise ValueError("Flash action horizon must equal buffer_size * chunk_size")
    temporal = (
        jnp.ones((b, h), bool) if temporal is None else jnp.asarray(temporal, bool)
    )
    if temporal.shape != (b, h):
        raise ValueError("Flash temporal mask differs from targets")
    if dimensions is not None:
        if (
            dimensions.ndim != 3
            or dimensions.shape[:2] != (b, h)
            or dimensions.shape[-1] > a
        ):
            raise ValueError("Flash physical dimension mask differs from targets")
        dimensions = jnp.asarray(dimensions, bool)
        temporal = temporal & jnp.any(dimensions, axis=-1)
    full_dimensions = jnp.broadcast_to(temporal[..., None], actions.shape)
    if dimensions is not None:
        full_dimensions = full_dimensions.at[..., : dimensions.shape[-1]].set(
            temporal[..., None] & dimensions
        )
    clean = pack(
        jnp.where(full_dimensions, actions, 0),
        chunk_size=chunk_size,
        buffer_size=buffer_size,
    )
    valid = pack(
        temporal, chunk_size=chunk_size, buffer_size=buffer_size, padding=False
    )
    dim_valid = (
        None
        if dimensions is None
        else pack(
            dimensions, chunk_size=chunk_size, buffer_size=buffer_size, padding=False
        )
    )
    (time_rng, noise_rng) = jax.random.split(rng)
    u = jax.random.beta(time_rng, 1.5, 1.0, (b, buffer_size, buffer_size))
    if substeps is not None:
        paths = flow_substeps(substeps, levels=levels, buffer_size=buffer_size)
        choices = jax.random.uniform(jax.random.fold_in(time_rng, 1398096467), u.shape)
        slot_values = []
        for slot, path in enumerate(paths):
            index = jnp.minimum(
                (choices[:, :, slot] * (len(path) - 1)).astype(jnp.int32), len(path) - 2
            )
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
        (low, span) = (
            jnp.asarray(low, jnp.float32),
            jnp.asarray(high - low, jnp.float32),
        )
        times = jnp.minimum(low + span * u, jnp.asarray(high, jnp.float32))
    times = jnp.repeat(times, chunk_size, axis=-1).reshape(b, buffer_size * h)
    noise = jax.random.normal(noise_rng, clean.shape)
    x = jnp.where(
        valid[..., None], (1 - times[..., None]) * clean + times[..., None] * noise, 0
    )
    target = jnp.where(valid[..., None], noise - clean, 0)
    return (x, target, times, valid, dim_valid)


def attention(valid, *, chunk_size, buffer_size):
    """Block-causal attention inside each buffer, zero cross-buffer attention."""
    (b, length) = valid.shape
    horizon = chunk_size * buffer_size
    if length not in (horizon, buffer_size * horizon):
        raise ValueError(
            "Flash suffix must be one buffer or all packed cold-start buffers"
        )
    index = jnp.arange(length)
    same_buffer = index[:, None] // horizon == index[None, :] // horizon
    chunks = index % horizon // chunk_size
    causal = chunks[None, :] <= chunks[:, None]
    mask = same_buffer & causal
    mask = mask[None] & valid[:, :, None] & valid[:, None, :]
    local = valid.reshape(b, -1, horizon)
    positions = (jnp.cumsum(local, axis=-1) - 1).reshape(b, length)
    return (mask, positions)
