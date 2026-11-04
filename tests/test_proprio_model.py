# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Native elapsed-time proprio memory and its actual training-objective path."""

import dataclasses
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0, TemporalMemoryCompressor, make_attn_mask
from openpi.models.pi0_config import Pi0Config


class MemoryHarness(nnx.Module):
    """Exercise the unmodified model methods with small real memory parameters."""

    _build_proprio_memory_tokens = Pi0._build_proprio_memory_tokens
    _pool_prompt_summary = Pi0._pool_prompt_summary
    _project_temporal_prompt_summary = Pi0._project_temporal_prompt_summary
    _apply_temporal_proprio_prefix_gate = Pi0._apply_temporal_proprio_prefix_gate
    _temporal_proprio_prefix_gate_multiplier = (
        Pi0._temporal_proprio_prefix_gate_multiplier
    )

    def __init__(self, *, enabled=True):
        rngs = nnx.Rngs(17)
        self.use_temporal_proprio_memory = enabled
        self.temporal_proprio_use_delta = False
        self.temporal_proprio_tokens_in_prefix = True
        self.temporal_proprio_prefix_token_scale = 0.1
        self.temporal_proprio_prefix_gate_warmup_steps = 0
        self.proprio_frequency_enabled = True
        self.temporal_proprio_history_lags = tuple(range(1, 11))
        self.action_in_proj = SimpleNamespace(out_features=16)
        self.temporal_prompt_proj = nnx.Linear(16, 8, rngs=rngs)
        self.proprio_memory_in_proj = nnx.Linear(32, 8, rngs=rngs)
        self.proprio_memory_context_proj = nnx.Linear(32, 8, rngs=rngs)
        self.proprio_memory_compressor = TemporalMemoryCompressor(
            8, 4, rngs, hidden_dim=8, context_dim=8, out_dim=16
        )
        self.temporal_proprio_prefix_gate_raw = nnx.Param(jnp.array(0.0))


def observation():
    return model_lib.Observation(
        images={},
        image_masks={},
        state=jnp.ones((1, 32)),
        tokenized_prompt_mask=jnp.array([[True, True, False]]),
        state_history=jnp.arange(320, dtype=jnp.float32).reshape(1, 10, 32) / 320,
        state_history_mask=jnp.array([[True] * 7 + [False] * 3]),
        state_history_age=jnp.arange(1, 11, dtype=jnp.float32)[None] / 10,
    )


def gated_memory(module, obs):
    tokens, mask = module._build_proprio_memory_tokens(obs, jnp.ones((1, 3, 16)))
    return module._apply_temporal_proprio_prefix_gate(tokens, mask, dtype=jnp.float32)[
        0
    ]


def test_history_disabled_ignores_supplied_history():
    module = MemoryHarness(enabled=False)
    assert module._build_proprio_memory_tokens(observation(), None) == (None, None)
    assert module._build_proprio_memory_tokens(
        dataclasses.replace(observation(), state_history=None), None
    ) == (None, None)
    spec, _ = Pi0Config().inputs_spec()
    assert (
        spec.state_history is spec.state_history_mask is spec.state_history_age is None
    )


def test_h10_spec_and_past_only_configuration():
    config = Pi0Config(
        use_temporal_proprio_memory=True,
        temporal_proprio_history_lags=tuple(range(1, 11)),
        temporal_hidden_dim=256,
        proprio_frequency_enabled=True,
    )
    spec, _ = config.inputs_spec(batch_size=2)
    assert spec.state_history.shape == (2, 10, 32)
    assert spec.state_history_mask.shape == spec.state_history_age.shape == (2, 10)
    with pytest.raises(ValueError, match="past-only"):
        dataclasses.replace(config, temporal_proprio_history_lags=(0, 1))
    with pytest.raises(ValueError, match="elapsed-time"):
        dataclasses.replace(config, proprio_frequency_enabled=False)


def test_actual_elapsed_seconds_are_required_and_change_memory():
    module, obs = MemoryHarness(), observation()
    with pytest.raises(ValueError, match="actual elapsed seconds"):
        module._build_proprio_memory_tokens(
            dataclasses.replace(obs, state_history_age=None), None
        )
    baseline, _ = module._build_proprio_memory_tokens(obs, None)
    changed, _ = module._build_proprio_memory_tokens(
        dataclasses.replace(obs, state_history_age=obs.state_history_age * 2), None
    )
    assert not np.array_equal(baseline, changed)


def test_padding_does_not_change_memory_and_all_invalid_slots_are_masked():
    module, obs = MemoryHarness(), observation()
    reference, _ = module._build_proprio_memory_tokens(obs, None)
    changed = obs.state_history.at[:, 7:].set(12345)
    actual, _ = module._build_proprio_memory_tokens(
        dataclasses.replace(obs, state_history=changed), None
    )
    np.testing.assert_array_equal(actual, reference)
    tokens, mask = module._build_proprio_memory_tokens(
        dataclasses.replace(obs, state_history_mask=jnp.zeros((1, 10), bool)), None
    )
    assert np.isfinite(np.asarray(tokens)).all()
    assert not np.asarray(mask).any()
    # Prefix queries cannot attend invalid memory slots; memory cannot see suffix actions.
    valid = jnp.concatenate((mask, jnp.ones((1, 2), bool)), axis=1)
    attention = make_attn_mask(valid, jnp.array([False] * 4 + [True, False]))
    assert not np.asarray(attention[:, :, :4]).any()
    assert not np.asarray(attention[:, :4, 4:]).any()


def test_native_zero_gate_then_open_gate_and_history_parameter_gradients():
    module, obs = MemoryHarness(), observation()
    np.testing.assert_array_equal(gated_memory(module, obs), np.zeros((1, 4, 16)))
    zero_grad = nnx.grad(lambda m: jnp.sum(gated_memory(m, obs)))(module)
    assert float(zero_grad.temporal_proprio_prefix_gate_raw.value) != 0
    for value in jax.tree.leaves(zero_grad.proprio_memory_in_proj):
        np.testing.assert_array_equal(value, np.zeros_like(value))
    module.temporal_proprio_prefix_gate_raw.value = jnp.array(0.5)
    changed_obs = dataclasses.replace(obs, state_history=obs.state_history + 0.7)
    assert not np.array_equal(
        gated_memory(module, obs), gated_memory(module, changed_obs)
    )
    gradient = nnx.grad(lambda m: jnp.sum(gated_memory(m, obs) ** 2))(module)
    for branch in (gradient.proprio_memory_in_proj, gradient.proprio_memory_compressor):
        leaves = jax.tree.leaves(branch)
        assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
        assert any(np.any(np.asarray(x) != 0) for x in leaves)


def tiny_config(*, enabled=True, nuf=False):
    return Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        siglip_variant="mu/14",
        dtype="float32",
        action_horizon=4,
        max_token_len=4,
        base_view_names=(),
        wrist_view_names=("left_wrist_0_rgb",),
        visual_budget_gate_hidden_dim=8,
        visual_budget_gate_tau_start=0.805,
        visual_budget_gate_tau_end=0.805,
        visual_budget_mpa_tau=0.805,
        visual_budget_gate_keep_ratio_base=0.9,
        visual_budget_gate_keep_ratio_wrist=0.9,
        visual_budget_s2_alpha_warmup_steps=500,
        use_temporal_proprio_memory=enabled,
        temporal_proprio_history_lags=tuple(range(1, 11)) if enabled else (),
        temporal_hidden_dim=8 if enabled else None,
        proprio_frequency_enabled=enabled,
        flash_enabled=nuf,
        flash_chunk_size=2,
        flash_buffer_size=2,
        flash_flow_times=(0.1, 1.0) if nuf else None,
    )


def test_history_zero_actual_model_has_no_history_parameters_and_exact_loss_parity():
    config = tiny_config(enabled=False)
    module = config.create(jax.random.key(21))
    obs = dataclasses.replace(
        config.fake_obs(),
        tokenized_prompt=jnp.array([[1, 2, 0, 0]]),
        tokenized_prompt_mask=jnp.array([[True, True, False, False]]),
    )
    extra = observation()
    supplied = dataclasses.replace(
        obs,
        state_history=extra.state_history,
        state_history_mask=extra.state_history_mask,
        state_history_age=extra.state_history_age,
    )
    assert not any(
        key.startswith(("proprio_memory", "temporal_prompt", "temporal_proprio"))
        for key in nnx.state(module, nnx.Param).to_pure_dict()
    )

    def objective(record):
        return module.compute_training_objective(
            jax.random.key(2),
            record,
            jnp.zeros((1, 4, 32)),
            train_step=0,
            total_train_steps=20000,
            action_dim_mask=jnp.arange(32) < 20,
        )[0]

    np.testing.assert_array_equal(objective(obs), objective(supplied))
    jax.clear_caches()


def open_tiny_action_attention_gate(module):
    """Emulate a learned checkpoint gate; random AdaRMS gates start at zero."""
    state = nnx.state(module, nnx.Param)
    flat = state.flat_state()
    changed = []
    for path, value in flat.items():
        if tuple(path[-3:]) != ("pre_attention_norm_1", "Dense_0", "bias"):
            continue
        assert path[:2] == ("PaliGemma", "llm")
        bias = value.value
        assert bias.shape[-1] % 3 == 0
        np.testing.assert_array_equal(bias, np.zeros_like(bias))
        width = bias.shape[-1] // 3
        flat[path] = value.replace(bias.at[..., 2 * width :].set(0.5))
        changed.append(path)
    assert len(changed) == 1, changed
    nnx.update(module, nnx.State.from_flat_path(flat))


@pytest.mark.parametrize("nuf", [False, True])
def test_h10_changes_actual_anchor_and_nuf_action_loss_and_receives_gradient(nuf):
    config = tiny_config(nuf=nuf)
    module = config.create(jax.random.key(23))
    open_tiny_action_attention_gate(module)
    module.temporal_proprio_prefix_gate_raw.value = jnp.array(0.5)
    extra = observation()
    obs = dataclasses.replace(
        config.fake_obs(),
        tokenized_prompt=jnp.array([[1, 2, 0, 0]]),
        tokenized_prompt_mask=jnp.array([[True, True, False, False]]),
        state_history=extra.state_history,
        state_history_mask=extra.state_history_mask,
        state_history_age=extra.state_history_age,
    )

    def objective(m, record):
        return m.compute_training_objective(
            jax.random.key(2),
            record,
            jnp.zeros((1, 4, 32)),
            train_step=0,
            total_train_steps=20000,
            action_dim_mask=jnp.arange(32) < 20,
        )[0]

    loss, gradient = nnx.value_and_grad(objective)(module, obs)
    changed = objective(
        module, dataclasses.replace(obs, state_history=obs.state_history + 3)
    )
    assert np.isfinite(np.asarray(loss)).all()
    assert not np.array_equal(np.asarray(loss), np.asarray(changed))
    leaves = jax.tree.leaves(gradient.proprio_memory_in_proj)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert any(np.any(np.asarray(x) != 0) for x in leaves)
    jax.clear_caches()
