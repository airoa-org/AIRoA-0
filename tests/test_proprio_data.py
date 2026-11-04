# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Measured-state history stays causal, source-bound and image-free."""

import numpy as np
import pytest
from datasets import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi import transforms
from openpi.shared.normalize import NormStats
from openpi.shared.proprio_frequency import resampling_plan, sample_history_schedule
from openpi.training.data_loader import ConfiguredCameraDataset
from post_training.data import Inputs, rgb_uint8


@pytest.fixture
def history_root(tmp_path):
    root = tmp_path / "history"
    writer = LeRobotDataset.create(
        repo_id="tests/proprio",
        root=root,
        fps=20,
        robot_type="synthetic",
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["x", "y"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "clock": {"dtype": "float64", "shape": (1,), "names": ["seconds"]},
            "state_valid": {"dtype": "bool", "shape": (1,), "names": ["valid"]},
            "camera": {
                "dtype": "image",
                "shape": (4, 6, 3),
                "names": ["height", "width", "channel"],
            },
        },
    )
    for episode in range(2):
        for frame in range(24):
            writer.add_frame(
                {
                    "observation.state": np.array(
                        [episode * 100 + frame, -frame], np.float32
                    ),
                    "action": np.array([10000 + frame, 20000 + frame], np.float32),
                    "clock": np.array(
                        [10 + frame * 0.1 + (1 if frame >= 18 else 0)], np.float64
                    ),
                    "state_valid": np.array([frame != 9], bool),
                    "camera": np.full((4, 6, 3), frame, np.uint8),
                    "task": "Move the object.",
                }
            )
        writer.save_episode()
    return root


def dataset(root, **kwargs):
    return ConfiguredCameraDataset(
        "tests/proprio",
        root=root,
        source_cameras=("camera",),
        delta_timestamps={"action": [0, 0.05]},
        history_slots=10,
        history_time_key="clock",
        history_rates_hz=(5.0,),
        history_dropout=0,
        **kwargs,
    )


def test_real_history_uses_measured_state_clock_and_decodes_only_current_rgb(
    history_root, monkeypatch
):
    ds = dataset(history_root)
    calls = []
    original = Image.decode_example

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Image, "decode_example", counted)
    sample = ds[8]
    assert len(calls) == 1
    rate, count = sample_history_schedule(
        slots=10, rates_hz=(5.0,), dropout=0, seed=42, episode=0, frame=8
    )
    plan = resampling_plan(
        ds._history_times[:8],
        ds._history_times[8],
        interval=1 / rate,
        slots=10,
        count=count,
        interpolate=False,
    )
    for slot, entry in enumerate(plan):
        assert bool(sample["state_history_mask"][slot]) == (entry is not None)
        if entry is not None:
            row = entry[0]
            np.testing.assert_array_equal(sample["state_history"][slot], [row, -row])
            stored_age = np.float32(ds._history_times[8] - ds._history_times[row])
            assert sample["state_history_age"][slot] == stored_age
            assert abs(float(stored_age) - 0.1 * (8 - row)) < 1e-6
            assert row < 8
    np.testing.assert_array_equal(
        rgb_uint8(sample["camera"]), np.full((4, 6, 3), 8, np.uint8)
    )
    np.testing.assert_array_equal(
        sample["action"].numpy(), [[10008, 20008], [10009, 20009]]
    )


def test_episode_qc_and_clock_gaps_never_bridge(history_root):
    ds = dataset(history_root)
    for index in (0, 9, 10, 18, 24):
        sample = ds[index]
        assert not sample["state_history_mask"].any(), index
        assert not sample["state_history"].any()
    # After QC rejection at frame9, only frame>=10 can support later history.
    sample = ds[17]
    assert np.all(sample["state_history"][sample["state_history_mask"], 0] >= 10)
    assert ds[24]["episode_index"].item() == 1


def test_missing_or_invalid_history_metadata_fails(history_root):
    with pytest.raises(ValueError, match="Missing proprio"):
        dataset(history_root, history_valid_key="missing_qc")
    ds = dataset(history_root)
    # A malformed configured clock cannot silently fall back to native FPS.
    bad = ds.hf_dataset.remove_columns("clock").add_column("clock", [np.nan] * len(ds))
    ds.hf_dataset = bad
    with pytest.raises(ValueError, match="finite"):
        ds._init_history("observation.state", "clock", "", (5.0,), 0, 42)


def test_history_uses_current_state_norm_and_zero_padding():
    data = {
        "state": np.array([3.0, 8.0], np.float32),
        "state_history": np.array([[1.0, 4.0], [0.0, 0.0]], np.float32),
        "state_history_mask": np.array([True, False]),
        "state_history_age": np.array([0.2, 0], np.float32),
    }
    stats = NormStats(
        mean=np.array([1.0, 2.0]),
        std=np.array([2.0, 3.0]),
        q01=np.array([-1.0, 0.0]),
        q99=np.array([5.0, 12.0]),
    )
    for quantiles in (False, True):
        norm = transforms.Normalize({"state": stats}, use_quantiles=quantiles)
        result = norm(dict(data))
        expected = norm({"state": data["state_history"][0]})["state"]
        np.testing.assert_array_equal(result["state_history"][0], expected)
        np.testing.assert_array_equal(result["state_history"][1], 0)
        padded = transforms.PadStatesAndActions(32)(result)
        assert padded["state_history"].shape == (2, 32)
        assert not padded["state_history"][:, 2:].any()


def adapter():
    return Inputs(
        "observation.state",
        "action",
        2,
        2,
        (("base_0_rgb", "camera"),),
        "fixed",
        "Move.",
        history_slots=2,
    )


def test_adapter_requires_masks_and_sanitizes_only_invalid_history():
    record = {
        "observation.state": np.array([1.0, 2.0]),
        "action": np.zeros((2, 2)),
        "camera": np.zeros((4, 6, 3), np.uint8),
        "state_history": np.array([[0.0, 1.0], [np.nan, np.nan]]),
        "state_history_mask": np.array([True, False]),
        "state_history_age": np.array([0.2, 0]),
    }
    sample = adapter()(record)
    np.testing.assert_array_equal(sample["state_history"][1], 0)
    for key, value in (
        ("state_history_mask", np.array([1, 0])),
        ("state_history_age", np.array([0.0, 0.0])),
    ):
        with pytest.raises(ValueError):
            adapter()({**record, key: value})
    with pytest.raises(KeyError, match="state_history_mask"):
        adapter()({k: v for k, v in record.items() if k != "state_history_mask"})


def test_disabled_history_preserves_original_dataset(history_root):
    options = dict(
        repo_id="tests/proprio",
        root=history_root,
        delta_timestamps={"action": [0, 0.05]},
    )
    base = LeRobotDataset(**options)
    actual = ConfiguredCameraDataset(**options, source_cameras=("camera",))
    expected, result = base[8], actual[8]
    assert set(result) == set(expected)
    for key in result:
        np.testing.assert_array_equal(result[key], expected[key])


def test_nonconsecutive_frames_reset_support_and_nonboolean_qc_fails(history_root):
    ds = dataset(history_root)
    frames = np.asarray(ds.hf_dataset.with_format("numpy")["frame_index"]).copy()
    frames[5:24] += 1
    ds.hf_dataset = ds.hf_dataset.remove_columns("frame_index").add_column(
        "frame_index", frames
    )
    ds._init_history("observation.state", "clock", "", (5.0,), 0, 42)
    assert not ds[5]["state_history_mask"].any()
    bad = ds.hf_dataset.remove_columns("state_valid").add_column(
        "state_valid", [1] * len(ds)
    )
    ds.hf_dataset = bad
    with pytest.raises(ValueError, match="boolean"):
        ds._init_history("observation.state", "clock", "", (5.0,), 0, 42)


def test_frequency_schedule_is_deterministic_and_ordered():
    schedule = sample_history_schedule(
        slots=10,
        rates_hz=(5, 8, 10, 12, 15),
        dropout=0.25,
        seed=42,
        episode=3,
        frame=50,
    )
    assert schedule == sample_history_schedule(
        slots=10,
        rates_hz=(5, 8, 10, 12, 15),
        dropout=0.25,
        seed=42,
        episode=3,
        frame=50,
    )
    rate, count = schedule
    assert rate in (5, 8, 10, 12, 15) and 0 <= count <= 10
    times = np.arange(40) / 30
    plan = resampling_plan(
        times, 40 / 30, interval=1 / rate, slots=10, count=count, interpolate=False
    )
    selected = [p for p in plan if p is not None]
    assert all(
        lo == hi and fraction == 0 and age > 0 for lo, hi, fraction, age in selected
    )
    assert len({p[0] for p in selected}) == len(selected)
    assert all(a[3] < b[3] for a, b in zip(selected, selected[1:]))


def test_model_and_data_history_configuration_must_agree():
    from types import SimpleNamespace
    from openpi.training.data_loader import configured_proprio_history

    def data(slots):
        return SimpleNamespace(
            data_transforms=SimpleNamespace(
                inputs=[
                    SimpleNamespace(
                        history_slots=slots,
                        state_key="observation.state",
                        history_time_key="timestamp",
                        history_valid_key="",
                    )
                ]
            )
        )

    model = SimpleNamespace(
        use_temporal_proprio_memory=True,
        temporal_proprio_history_lags=(1, 2),
        proprio_frequency_rates_hz=(5.0,),
        proprio_frequency_dropout=0.25,
        proprio_frequency_seed=42,
    )
    with pytest.raises(ValueError, match="positive history_slots"):
        configured_proprio_history(data(0), model)
    with pytest.raises(ValueError, match="positive history_slots"):
        configured_proprio_history(
            SimpleNamespace(data_transforms=SimpleNamespace(inputs=[])), model
        )
    with pytest.raises(ValueError, match="model proprio memory"):
        configured_proprio_history(
            data(2), SimpleNamespace(use_temporal_proprio_memory=False)
        )
    with pytest.raises(ValueError, match="slots differ"):
        configured_proprio_history(data(3), model)
    assert configured_proprio_history(data(2), model)["history_slots"] == 2
    assert (
        configured_proprio_history(
            data(0), SimpleNamespace(use_temporal_proprio_memory=False)
        )
        == {}
    )
