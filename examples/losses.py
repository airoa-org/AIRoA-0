# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Flow-matching and auxiliary-loss examples using synthetic tensors."""

import jax
import jax.numpy as jnp

from post_training_losses import (
    attention_log_prior,
    attention_strength,
    visual_budget_statistics,
    embedding_gates,
    linear_schedule,
    nuf_loss,
    pa_loss,
    packed_causal_attention,
    post_training_loss,
    stage_loss,
    training_inputs,
)


def main():
    actions = jnp.ones((2, 4, 3), jnp.float32)
    noisy, target, times, valid, dimensions = training_inputs(
        jax.random.key(0), actions, chunk_size=2, buffer_size=2, levels=(0.1, 1.0)
    )
    suffix_mask, positions = packed_causal_attention(valid, chunk_size=2, buffer_size=2)

    gate_logits = jnp.zeros((2, 4), jnp.float32)
    token_valid = jnp.ones_like(gate_logits, dtype=bool)
    floor = linear_schedule(0.1, 0.02, step=1000, total_steps=20000)
    _, effective_gates = embedding_gates(gate_logits, token_valid, tau=0.805, gate_floor=floor)
    log_prior, _ = attention_log_prior(gate_logits, token_valid, tau=0.805, keep_ratio=0.9)
    scaled_log_prior = attention_strength(1000) * log_prior
    # A real action model consumes gated image embeddings, the attention prior,
    # noisy actions, times, suffix_mask, and positions. This example has no model.
    prediction = jnp.zeros_like(target)
    action = nuf_loss(prediction, target, valid, dimensions=dimensions)
    terms = visual_budget_statistics(
        {"wrist": gate_logits}, {"wrist": token_valid}, {"wrist": jnp.zeros_like(gate_logits)},
        view_kinds={"wrist": "wrist"}, tau=0.805,
    )

    pa_hidden = jnp.ones((2, 3, 4), jnp.float32)
    decoder_weights = jnp.ones((4, 8), jnp.float32)
    pa, _ = pa_loss(lambda h: h @ decoder_weights, pa_hidden,
                    jnp.zeros((2, 3), jnp.int32), jnp.ones((2, 3), bool))
    stage, _ = stage_loss(jnp.zeros((2, 3)), jnp.asarray([0, 2], jnp.int32))
    total, metrics = post_training_loss(action, visual_budget_terms=terms, pa_loss=pa, stage_loss=stage)
    print({"total": float(total), "metrics": {k: float(v) for k, v in metrics.items()},
           "packed_shape": noisy.shape, "time_shape": times.shape,
           "suffix_attention_shape": suffix_mask.shape, "position_shape": positions.shape,
           "embedding_gate_shape": effective_gates.shape,
           "attention_prior_shape": scaled_log_prior.shape})


if __name__ == "__main__":
    main()
