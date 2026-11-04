# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Causal training-history sampling and elapsed-time encoding."""

import hashlib
import numpy as np


def time_encoding(age_seconds, width):
    """Seconds, not ordinal frame positions; periods 0.1 to 100 seconds.

    Fixed frequencies shared across all samples and deployment. Also preserves
    differences between irregularly observed timestamps without index rescaling.
    """
    import jax.numpy as jnp

    if width % 2:
        raise ValueError("Time encoding width must be even")
    periods = jnp.geomspace(0.1, 100.0, width // 2)
    phase = jnp.asarray(age_seconds, jnp.float32)[..., None] * (2 * jnp.pi / periods)
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def sample_history_schedule(*, slots, rates_hz, dropout, seed, episode, frame):
    digest = hashlib.sha256(
        f"proprio-frequency-v2:{seed}:{episode}:{frame}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    rate = float(rng.choice(rates_hz))
    count = 0 if rng.random() < dropout else int(rng.integers(1, slots + 1))
    return rate, count


def resampling_plan(past_times, now, *, interval, slots, count, interpolate):
    """No extrapolation; brackets use strictly past observations only.

    If a target lacks a right bracket, retain its recent measured left sample
    and its TRUE age. Do not pretend sample-and-hold is a fresh measurement.
    """
    times = np.asarray(past_times, np.float64)
    if (
        not np.isfinite(times).all()
        or not np.isfinite(now)
        or np.any(np.diff(times) <= 0)
        or np.any(times >= now)
        or interval <= 0
        or not 0 <= count <= slots
    ):
        raise ValueError("Invalid causal history time grid")
    result = []
    used = set()
    for i in range(slots):
        target = now - (i + 1) * interval
        lo = int(np.searchsorted(times, target + 1e-7, side="right")) - 1
        if i >= count or lo < 0 or target - times[lo] > interval + 1e-7:
            result.append(None)
            continue
        hi = lo
        fraction = 0.0
        if (
            interpolate
            and lo + 1 < len(times)
            and times[lo] < target - 1e-7
            and times[lo + 1] - times[lo] <= 2 * interval + 1e-7
        ):
            hi = lo + 1
            fraction = float((target - times[lo]) / (times[hi] - times[lo]))
        if hi == lo and lo in used:
            result.append(None)
            continue
        if hi == lo:
            used.add(lo)
        age = now - (target if hi != lo else times[lo])
        result.append((lo, hi, fraction, float(age)))
    return result
