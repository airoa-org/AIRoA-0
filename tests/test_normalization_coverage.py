# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Normalization covers every selected anchor without changing training batches."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training.data_loader import TorchDataLoader
from post_training.data import fit_normalization


@pytest.fixture
def norm_helper():
    path = Path(__file__).resolve().parents[1] / "vendor/openpi/scripts/compute_norm_stats.py"
    spec = importlib.util.spec_from_file_location("normalization_coverage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def samples(count):
    return [
        {
            "state": np.array([index], np.float32),
            "actions": np.array([[index]], np.float32),
            "action_time_valid_mask": np.array([True]),
            "action_time_dim_valid_mask": np.array([[True]]),
            "prompt": "removed before collation",
        }
        for index in range(count)
    ]


def loader(norm_helper, monkeypatch, *, count, batch_size, max_frames=None):
    monkeypatch.setattr(norm_helper._data_loader, "create_torch_dataset", lambda *args: samples(count))
    config = SimpleNamespace(
        repo_id="test/task",
        repack_transforms=SimpleNamespace(inputs=[]),
        data_transforms=SimpleNamespace(inputs=[]),
    )
    return norm_helper.create_torch_dataloader(config, 1, batch_size, None, 0, max_frames)


@pytest.mark.parametrize("count,batch_size,expected_sizes", [(5, 4, [4, 1]), (2, 8, [2])])
def test_full_normalization_includes_tail_and_small_datasets(
    norm_helper, monkeypatch, count, batch_size, expected_sizes
):
    data, batches = loader(norm_helper, monkeypatch, count=count, batch_size=batch_size)
    collected = list(data)
    assert batches == len(expected_sizes)
    assert [len(batch["state"]) for batch in collected] == expected_sizes
    stats, counts = fit_normalization(collected, state_dim=1, action_dim=1)
    assert counts["sampled_anchors"] == count
    assert counts["eligible_action_values_per_dimension"] == [count]
    np.testing.assert_allclose(stats["state"].mean, [(count - 1) / 2])
    np.testing.assert_allclose(stats["actions"].mean, [(count - 1) / 2])


@pytest.mark.parametrize("limit,expected_sizes", [(1, [1]), (3, [3]), (5, [4, 1]), (9, [4, 1])])
def test_max_frames_is_an_exact_anchor_limit(norm_helper, monkeypatch, limit, expected_sizes):
    data, batches = loader(norm_helper, monkeypatch, count=5, batch_size=4, max_frames=limit)
    collected = list(data)
    assert batches == len(expected_sizes)
    assert [len(batch["state"]) for batch in collected] == expected_sizes
    selected = np.concatenate([np.asarray(batch["state"])[:, 0] for batch in collected])
    assert len(selected) == min(limit, 5)
    assert len(set(selected)) == len(selected)


def test_training_loader_still_drops_partial_batches():
    records = [{key: value for key, value in row.items() if key != "prompt"} for row in samples(5)]
    data = TorchDataLoader(records, 4, num_batches=1, framework="pytorch")
    assert data.torch_loader.drop_last is True
    assert len(next(iter(data))["state"]) == 4
    with pytest.raises(ValueError, match="larger than the dataset"):
        TorchDataLoader(samples(2), 4, num_batches=1, framework="pytorch")
