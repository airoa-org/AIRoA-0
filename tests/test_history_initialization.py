# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""History is an explicit task addition; pretrained core/visual-gate remain required."""

from pathlib import Path

from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import numpy as np
import pytest

from post_training import configuration
from post_training.settings import Settings


@pytest.fixture
def weights(tmp_path, monkeypatch):
    core = {("PaliGemma", "kernel"): np.ones((2, 2), np.float32)}
    for view in ("base", "wrist"):
        for layer in ("in_proj", "out_proj"):
            for name in ("kernel", "bias"):
                core[("visual_budget_gate_heads", view, layer, name)] = np.ones(
                    (2,), np.float32
                )
    additions = {
        ("proprio_memory_in_proj", "kernel"): jax.ShapeDtypeStruct((2, 2), np.float32),
        ("proprio_memory_compressor", "slot_embeddings"): jax.ShapeDtypeStruct(
            (4, 2), np.float32
        ),
        ("temporal_proprio_prefix_gate_raw",): jax.ShapeDtypeStruct((), np.float32),
    }
    expected = unflatten_dict({**core, **additions})
    actual = dict(core)
    path = tmp_path / "params"
    path.mkdir()
    (path / "_METADATA").write_text("{}")
    monkeypatch.setattr(
        configuration.model_lib,
        "restore_params",
        lambda *a, **k: unflatten_dict(actual),
    )
    return path, actual, expected, additions


def test_pretrained_core_is_restored_and_complete_new_history_branch_can_initialize(
    weights,
):
    path, actual, expected, additions = weights
    restored = flatten_dict(
        configuration.CoreWeights(str(path), allow_new_history=True).load(expected)
    )
    for key, value in actual.items():
        np.testing.assert_array_equal(restored[key], value)
    for key, reference in additions.items():
        assert restored[key] is reference


def test_new_history_requires_explicit_initialization(weights):
    path, _, expected, _ = weights
    with pytest.raises(ValueError, match="explicit initialization"):
        configuration.CoreWeights(str(path)).load(expected)


def test_partial_history_checkpoint_is_not_silently_completed(weights):
    path, actual, expected, _ = weights
    actual[("proprio_memory_in_proj", "kernel")] = np.ones((2, 2), np.float32)
    with pytest.raises(ValueError, match="incomplete proprio history"):
        configuration.CoreWeights(str(path), allow_new_history=True).load(expected)


def test_incompatible_existing_history_is_not_treated_as_a_new_branch(weights):
    path, actual, expected, _ = weights
    actual[("proprio_memory_compressor", "old_projection")] = np.ones((2,), np.float32)
    with pytest.raises(ValueError, match="incomplete proprio history"):
        configuration.CoreWeights(str(path), allow_new_history=True).load(expected)


def test_complete_existing_history_is_restored(weights):
    path, actual, expected, additions = weights
    actual.update(
        {key: np.ones(ref.shape, ref.dtype) for key, ref in additions.items()}
    )
    restored = flatten_dict(
        configuration.CoreWeights(str(path), allow_new_history=True).load(expected)
    )
    for key, value in actual.items():
        np.testing.assert_array_equal(restored[key], value)


@pytest.mark.parametrize("family", ["PaliGemma", "visual_budget_gate_heads"])
def test_history_initialization_never_relaxes_core_requirements(weights, family):
    path, actual, expected, _ = weights
    del actual[next(key for key in actual if key[0] == family)]
    with pytest.raises(ValueError):
        configuration.CoreWeights(str(path), allow_new_history=True).load(expected)


@pytest.mark.parametrize("enabled", [True, False])
def test_public_configuration_binds_h10_and_explicit_off(
    tmp_path, monkeypatch, enabled
):
    source = Path(__file__).resolve().parents[1] / "configs/post_training.toml"
    text = source.read_text()
    if not enabled:
        text = text.replace("slots = 10", "slots = 0")
    filename = tmp_path / "task.toml"
    filename.write_text(text)
    settings = Settings.read(filename)
    monkeypatch.setattr(configuration, "tokenizer_for", lambda _: object())
    monkeypatch.setattr(
        configuration,
        "stage_tokens",
        lambda tokenizer, texts, limit: tuple((1, 2, 0) for _ in texts),
    )
    config = configuration.build(settings, "history_test")
    assert config.model.use_temporal_proprio_memory is enabled
    assert config.weight_loader.allow_new_history is enabled
    if enabled:
        assert config.model.temporal_proprio_history_lags == tuple(range(1, 11))
        assert config.model.temporal_proprio_memory_tokens == 4
        assert config.model.temporal_hidden_dim == 256
        assert config.model.proprio_frequency_rates_hz == (5, 8, 10, 12, 15)
        assert config.model.proprio_frequency_dropout == 0.25
        assert config.model.temporal_proprio_prefix_token_scale == 0.1
        assert not config.model.proprio_frequency_interpolate
    assert config.data.adapter.history_slots == (10 if enabled else 0)
