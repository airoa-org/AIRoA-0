# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Training-only PA teacher forcing; no inference/generation implementation."""

import jax
import jax.numpy as jnp


def attention_mask(prefix_mask, target_mask):
    """Prefix never reads answers; decoder reads prefix and its causal history.

    Decoder input t is BOS for t=0, otherwise target[t-1]. Thus allowing its
    diagonal does not reveal target[t]. Both key and query padding are masked.
    """
    prefix_len, target_len = prefix_mask.shape[1], target_mask.shape[1]
    block = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    jnp.ones((prefix_len, prefix_len), bool),
                    jnp.zeros((prefix_len, target_len), bool),
                ],
                1,
            ),
            jnp.concatenate(
                [
                    jnp.ones((target_len, prefix_len), bool),
                    jnp.tril(jnp.ones((target_len, target_len), bool)),
                ],
                1,
            ),
        ],
        0,
    )
    valid = jnp.concatenate([prefix_mask, target_mask], 1).astype(bool)
    return block[None] & valid[:, :, None] & valid[:, None, :]


def token_cross_entropy(decode, hidden, targets, loss_mask, *, chunk_size=8):
    """Exact full-vocabulary CE, bounded logits memory, normalized per example.

    Rematerialize sequence chunks during backward instead of retaining all
    [batch, PA_length, vocabulary] logits. Keep batch inside each chunk so SPMD
    can preserve data sharding. Scanning flattened batch/token chunks gathers
    the global batch and repeats the full scan on every data-parallel replica.
    EOS participates; padding never does.
    """
    if hidden.shape[:2] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError("PA hidden/target/loss-mask shapes disagree")
    batch, length = targets.shape
    if chunk_size < 1:
        raise ValueError("PA token chunk size must be positive")
    padding = (-length) % chunk_size
    chunks = (length + padding) // chunk_size
    chunk_hidden = (
        jnp.pad(hidden, ((0, 0), (0, padding), (0, 0)))
        .reshape(batch, chunks, chunk_size, hidden.shape[-1])
        .swapaxes(0, 1)
    )
    chunk_targets = (
        jnp.pad(targets, ((0, 0), (0, padding)))
        .reshape(batch, chunks, chunk_size)
        .swapaxes(0, 1)
    )
    chunk_mask = (
        jnp.pad(loss_mask, ((0, 0), (0, padding)))
        .reshape(batch, chunks, chunk_size)
        .swapaxes(0, 1)
    )

    @jax.checkpoint
    def chunk(args):
        values, labels, mask = args
        logits = decode(values).astype(jnp.float32)
        labels = jnp.where(mask, labels, 0)
        selected = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        ce = jax.nn.logsumexp(logits, axis=-1) - selected
        return jnp.where(mask, ce, 0.0)

    losses = (
        jax.lax.map(
            chunk,
            (
                chunk_hidden,
                chunk_targets,
                chunk_mask,
            ),
        )
        .swapaxes(0, 1)
        .reshape(batch, length + padding)[:, :length]
    )
    return losses.sum(-1) / jnp.maximum(loss_mask.sum(-1), 1)
