# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Prepared-data, initialization and configuration contracts for post-training."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict

from post_training import configuration, language
from post_training.data import Inputs, ResizeWithMask, fit_normalization
from post_training.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parameter_fixture(tmp_path, monkeypatch):
    flat = {("PaliGemma", "kernel"): np.ones((2, 3), np.float32)}
    for view in ("base", "wrist"):
        for layer in ("in_proj", "out_proj"):
            for name in ("kernel", "bias"):
                flat[("visual_budget_gate_heads", view, layer, name)] = np.ones(
                    (2,), np.float32
                )
    expected = {**flat, ("stage_classifier", "kernel"): np.zeros((3, 2), np.float32)}
    params = tmp_path / "airoa-0_pretrained" / "params"
    params.mkdir(parents=True)
    (params / "_METADATA").write_text("{}")
    loaded = {"flat": copy.deepcopy(flat)}
    monkeypatch.setattr(
        configuration.model_lib,
        "restore_params",
        lambda *a, **k: unflatten_dict(loaded["flat"]),
    )
    return params, loaded, expected


def test_strict_loader_restores_all_core_and_s2_and_only_initializes_new_stage(
    parameter_fixture,
):
    path, loaded, expected = parameter_fixture
    result = flatten_dict(
        configuration.CoreWeights(str(path.parent)).load(unflatten_dict(expected))
    )
    assert set(result) == set(expected)
    for key, value in loaded["flat"].items():
        np.testing.assert_array_equal(result[key], value)
    np.testing.assert_array_equal(
        result[("stage_classifier", "kernel")], expected[("stage_classifier", "kernel")]
    )


@pytest.mark.parametrize("failure", ["core_missing", "s2_missing", "shape", "nan"])
def test_strict_loader_rejects_incomplete_or_invalid_initialization(
    parameter_fixture, failure
):
    path, loaded, expected = parameter_fixture
    core = ("PaliGemma", "kernel")
    if failure == "core_missing":
        del loaded["flat"][core]
    elif failure == "s2_missing":
        del loaded["flat"][("visual_budget_gate_heads", "wrist", "out_proj", "bias")]
    elif failure == "shape":
        loaded["flat"][core] = np.ones((1,), np.float32)
    else:
        loaded["flat"][core][0, 0] = np.nan
    with pytest.raises(ValueError):
        configuration.CoreWeights(str(path)).load(unflatten_dict(expected))


def read_settings(tmp_path, *, nuf=False):
    text = (ROOT / "configs/post_training.toml").read_text()
    if nuf:
        text = text.replace("action_horizon = 20", "action_horizon = 16")
        text = text.replace("nuf_chunk_size = 0", "nuf_chunk_size = 8")
        text = text.replace("nuf_buffer_size = 0", "nuf_buffer_size = 2")
    path = tmp_path / "settings.toml"
    path.write_text(text)
    return Settings.read(path)


@pytest.mark.parametrize("budget", [0.0, 1.0])
def test_s2_budget_requires_the_open_interval_supported_by_projection(tmp_path, budget):
    text = (ROOT / "configs/post_training.toml").read_text()
    text = text.replace("budget = 0.9", f"budget = {budget}")
    path = tmp_path / "settings.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="Invalid visual-budget ratio"):
        Settings.read(path)


@pytest.mark.parametrize("nuf", [False, True])
def test_anchor_and_nuf_bind_the_actual_model_config(tmp_path, monkeypatch, nuf):
    settings = read_settings(tmp_path, nuf=nuf)
    monkeypatch.setattr(configuration, "tokenizer_for", lambda length: object())
    monkeypatch.setattr(
        configuration,
        "stage_tokens",
        lambda tokenizer, texts, limit: tuple((1, 2, 0) for _ in texts),
    )
    result = configuration.build(settings, "contract_test")
    assert result.model.use_visual_budget_gate
    assert result.model.visual_budget_objective_mode == "visual_budget"
    assert result.model.action_horizon == (16 if nuf else 20)
    assert result.model.flash_enabled == nuf
    if nuf:
        assert (result.model.flash_chunk_size, result.model.flash_buffer_size) == (8, 2)
    assert result.ema_decay == 0.99
    assert result.num_workers == 16
    assert result.num_train_steps == 20000


class Tokenizer:
    def encode(self, text):
        return [ord(c) + 3 for c in text]

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2


def test_pa_targets_are_shifted_and_do_not_enter_the_shared_task_prefix(monkeypatch):
    tokenizer = Tokenizer()
    monkeypatch.setattr(
        language, "tokenizer_for", lambda length: SimpleNamespace(_tokenizer=tokenizer)
    )
    labels = language.Labels(
        "Complete the whole task.", (("Pick.", 0),), True, True, 16
    )
    result = labels({"prompt": "Pick.", "actions": np.ones((2, 2))})
    assert result["prompt"] == "Complete the whole task."
    assert result["stage_label"] == 0
    active = result["pa_loss_mask"]
    targets = result["pa_target_tokens"][active]
    inputs = result["pa_input_tokens"][active]
    assert targets[-1] == tokenizer.eos_id()
    assert inputs[0] == tokenizer.bos_id()
    np.testing.assert_array_equal(inputs[1:], targets[:-1])
    with pytest.raises(ValueError, match="No reviewed PA label"):
        labels({"prompt": "Unknown.", "actions": np.ones((2, 2))})


def adapter():
    return Inputs(
        "state", "action", 2, 2, (("left_wrist_0_rgb", "camera"),), "task", ""
    )


def sample():
    return {
        "state": np.array([1.0, 2.0]),
        "action": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "camera": np.ones((4, 6, 3), np.uint8),
        "prompt": "Pick.",
    }


def test_prepared_masks_and_endpoint_padding_exclude_only_invalid_targets():
    record = sample()
    record["action_time_valid_mask"] = np.array([True, False, True])
    record["action_time_dim_valid_mask"] = np.array(
        [[True, False], [True, True], [True, True]]
    )
    record["action_is_pad"] = np.array([False, False, True])
    record["action"][0, 1] = np.nan
    record["action"][1] = np.nan
    result = adapter()(record)
    np.testing.assert_array_equal(
        result["action_time_valid_mask"], [True, False, False]
    )
    np.testing.assert_array_equal(
        result["action_time_dim_valid_mask"],
        [[True, False], [False, False], [False, False]],
    )
    np.testing.assert_array_equal(
        result["actions"], [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    record["action"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Eligible action targets"):
        adapter()(record)


def test_mask_shapes_are_checked_and_supplied_image_validity_is_retained():
    record = sample()
    record["action_time_valid_mask"] = np.ones(2, bool)
    with pytest.raises(ValueError, match="action_time_valid_mask"):
        adapter()(record)
    record = sample()
    record["image_mask"] = {"left_wrist_0_rgb": np.bool_(False)}
    record["image_valid_mask"] = {"left_wrist_0_rgb": np.zeros((4, 6), bool)}
    result = ResizeWithMask()(adapter()(record))
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert not result["image_valid_mask"]["left_wrist_0_rgb"].any()


def test_normalization_uses_each_dimensions_eligible_values_only():
    batch = {
        "state": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "actions": np.array(
            [[[2.0, 10.0], [4.0, 999.0]], [[6.0, 30.0], [8.0, -999.0]]]
        ),
        "action_time_valid_mask": np.array([[True, False], [True, True]]),
        "action_time_dim_valid_mask": np.array(
            [[[True, True], [True, True]], [[True, True], [True, False]]]
        ),
    }
    stats, counts = fit_normalization([batch], state_dim=2, action_dim=2)
    np.testing.assert_allclose(stats["actions"].mean, [(2.0 + 6.0 + 8.0) / 3.0, 20.0])
    assert counts["eligible_action_values_per_dimension"] == [3, 2]
    np.testing.assert_allclose(stats["state"].mean, [2.0, 3.0])


def test_norm_reuse_refuses_a_changed_dataset_or_config_contract(tmp_path, monkeypatch):
    path = ROOT / "scripts/post_train.py"
    spec = importlib.util.spec_from_file_location("training_cli_contract_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "dataset_info", lambda *_: {"schema": "prepared-v1"})
    settings = SimpleNamespace(data_signature="original", stats_dir=tmp_path)
    contract = cli.stats_contract(settings, object())
    import json

    (tmp_path / "data_contract.json").write_text(json.dumps({"contract": contract}))
    (tmp_path / "norm_stats.json").write_text("{}")
    cli.verify_stats(settings, object())
    settings.data_signature = "changed"
    with pytest.raises(ValueError, match="Dataset schema/config changed"):
        cli.verify_stats(settings, object())


@pytest.mark.parametrize("horizon,chunk,buffer", [(12, 0, 0), (8, 4, 2)])
def test_positive_anchor_horizons_and_consistent_nuf_chunks_are_configurable(
    tmp_path, horizon, chunk, buffer
):
    text = (ROOT / "configs/post_training.toml").read_text()
    text = text.replace("action_horizon = 20", f"action_horizon = {horizon}")
    text = text.replace("nuf_chunk_size = 0", f"nuf_chunk_size = {chunk}")
    text = text.replace("nuf_buffer_size = 0", f"nuf_buffer_size = {buffer}")
    path = tmp_path / "settings.toml"
    path.write_text(text)
    settings = Settings.read(path)
    assert settings.raw["model"]["action_horizon"] == horizon


def test_nuf_mismatched_horizon_is_rejected(tmp_path):
    text = (ROOT / "configs/post_training.toml").read_text()
    text = text.replace("nuf_chunk_size = 0", "nuf_chunk_size = 4")
    text = text.replace("nuf_buffer_size = 0", "nuf_buffer_size = 2")
    path = tmp_path / "settings.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="horizon=chunk"):
        Settings.read(path)
