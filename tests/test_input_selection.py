# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Configured-camera decoding preserves the original reader's selected samples."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.training.data_loader import (
    ConfiguredCameraDataset,
    configured_source_cameras,
)


@pytest.fixture
def two_image_dataset(tmp_path):
    root = tmp_path / "two_camera_data"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "action": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    }
    for key in ("observation.images.first", "observation.images.second"):
        features[key] = {
            "dtype": "image",
            "shape": (8, 10, 3),
            "names": ["height", "width", "channel"],
        }
    writer = LeRobotDataset.create(
        repo_id="tests/input_selection",
        root=root,
        fps=20,
        robot_type="synthetic",
        use_videos=False,
        features=features,
    )
    pixels = np.arange(240, dtype=np.uint8).reshape(8, 10, 3)
    for episode in range(2):
        for frame in range(4):
            writer.add_frame(
                {
                    "observation.state": np.array([episode, frame, 0.5], np.float32),
                    "action": np.array([frame + 1, episode + 0.25, -frame], np.float32),
                    "observation.images.first": pixels.copy(),
                    "observation.images.second": np.flip(pixels, axis=1).copy(),
                    "task": "Complete this task.",
                }
            )
        writer.save_episode()
    return root


def test_real_image_reader_exact_samples_and_only_configured_image_is_decoded(
    two_image_dataset, monkeypatch
):
    selected_key = "observation.images.second"
    excluded_key = "observation.images.first"
    options = dict(
        repo_id="tests/input_selection",
        root=two_image_dataset,
        delta_timestamps={"action": [0.0, 0.05, 0.1]},
    )
    original = LeRobotDataset(**options)
    mapping = SimpleNamespace(
        data_transforms=SimpleNamespace(
            inputs=[SimpleNamespace(cameras=(("left_wrist_0_rgb", selected_key),))]
        )
    )
    selected = ConfiguredCameraDataset(
        **options, source_cameras=configured_source_cameras(mapping)
    )
    assert selected_key in selected.hf_dataset.column_names
    assert excluded_key not in selected.hf_dataset.column_names
    assert len(original) == len(selected) == 8
    calls = []
    decode_image = Image.decode_example

    def counted_decode(self, value, *args, **kwargs):
        calls.append(value.get("path"))
        return decode_image(self, value, *args, **kwargs)

    monkeypatch.setattr(Image, "decode_example", counted_decode)
    for index in range(len(original)):
        calls.clear()
        baseline = original[index]
        assert len(calls) == 2
        calls.clear()
        actual = selected[index]
        assert len(calls) == 1
        assert excluded_key not in actual
        for key, value in actual.items():
            if isinstance(value, str):
                assert value == baseline[key]
            else:
                np.testing.assert_array_equal(
                    np.asarray(value), np.asarray(baseline[key]), err_msg=key
                )
        for key in (
            selected_key,
            "action",
            "action_is_pad",
            "observation.state",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
            "index",
        ):
            np.testing.assert_array_equal(
                np.asarray(actual[key]), np.asarray(baseline[key]), err_msg=key
            )
    # Episode-tail clamping and the endpoint padding flags stay identical.
    np.testing.assert_array_equal(selected[3]["action_is_pad"], [False, True, True])
    np.testing.assert_array_equal(selected[7]["action_is_pad"], [False, True, True])


def test_only_configured_cameras_are_decoded_with_original_pixels_and_timestamps(
    tmp_path, monkeypatch
):
    import torch
    from openpi.training import data_loader

    video = torch.arange(18, dtype=torch.uint8).reshape(1, 3, 2, 3)
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    decoded_columns, decoder_calls = [], []

    class FrameTable:
        def __init__(self, columns):
            self.column_names = columns

        def remove_columns(self, columns):
            return FrameTable([key for key in self.column_names if key not in columns])

        def __getitem__(self, index):
            assert index == 0
            decoded_columns.extend(
                key for key in self.column_names if key.endswith("_image")
            )
            row = {
                "episode_index": torch.tensor(0),
                "index": torch.tensor(0),
                "timestamp": torch.tensor(0.25),
                "task_index": torch.tensor(0),
                "keep_image": image.copy(),
            }
            return {
                key: value for key, value in row.items() if key in self.column_names
            }

    def initialize(dataset, *args, **kwargs):
        import pandas as pd

        # LeRobot >= 0.4 (dataset v3.0): episodes share concatenated video files,
        # so each episode records its start offset; tasks are a DataFrame.
        dataset.meta = SimpleNamespace(
            camera_keys=("keep_video", "unused_video", "keep_image", "unused_image"),
            image_keys=("keep_image", "unused_image"),
            video_keys=("keep_video", "unused_video"),
            get_video_file_path=lambda episode, key: key + ".mp4",
            episodes=[
                {
                    "videos/keep_video/from_timestamp": 0.0,
                    "videos/unused_video/from_timestamp": 0.0,
                }
            ],
            tasks=pd.DataFrame({"task_index": [0]}, index=["Task."]),
            features={},
            subtasks=None,
        )
        dataset._lazy_loading = False
        dataset.hf_dataset = FrameTable(
            [
                "episode_index",
                "index",
                "timestamp",
                "task_index",
                "keep_image",
                "unused_image",
            ]
        )
        dataset.image_transforms = None
        dataset.delta_indices = None
        dataset.root = tmp_path
        dataset.tolerance_s = 0.0001
        dataset.video_backend = "pyav"

    def decode(path, timestamps, tolerance, backend):
        decoder_calls.append((Path(path).name, timestamps, tolerance, backend))
        return video.clone()

    monkeypatch.setattr(
        data_loader.lerobot_dataset.LeRobotDataset, "__init__", initialize
    )
    monkeypatch.setattr(data_loader.lerobot_dataset, "decode_video_frames", decode)
    cameras = (("left_wrist_0_rgb", "keep_video"), ("right_wrist_0_rgb", "keep_image"))
    mapped = SimpleNamespace(
        data_transforms=SimpleNamespace(inputs=[SimpleNamespace(cameras=cameras)])
    )
    sources = data_loader.configured_source_cameras(mapped)
    dataset = data_loader.ConfiguredCameraDataset(
        "fixture/task", source_cameras=sources
    )
    result = dataset[0]
    assert decoded_columns == ["keep_image"]
    assert decoder_calls == [("keep_video.mp4", [0.25], 0.0001, "pyav")]
    np.testing.assert_array_equal(
        result["keep_video"].numpy(), video.squeeze(0).numpy()
    )
    np.testing.assert_array_equal(result["keep_image"], image)


def test_cache_check_ignores_missing_videos_of_unused_cameras(tmp_path):
    """LeRobot >= 0.4 re-downloads when any listed video is missing; only decoded cameras matter."""
    from openpi.training import data_loader

    class Episodes:
        def unique(self, key):
            assert key == "episode_index"
            return [0, 1]

        def __len__(self):
            return 2

    dataset = data_loader.ConfiguredCameraDataset.__new__(
        data_loader.ConfiguredCameraDataset
    )
    dataset.hf_dataset = Episodes()
    dataset.episodes = None
    dataset.root = tmp_path
    dataset.source_cameras = ("observation.images.used",)
    dataset.meta = SimpleNamespace(
        total_episodes=2,
        video_keys=["observation.images.used", "observation.images.unused"],
        get_video_file_path=lambda episode, key: Path("videos") / key / f"{episode}.mp4",
    )
    for episode in (0, 1):
        path = tmp_path / "videos" / "observation.images.used" / f"{episode}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    assert dataset._check_cached_episodes_sufficient()
    (tmp_path / "videos" / "observation.images.used" / "1.mp4").unlink()
    assert not dataset._check_cached_episodes_sufficient()


def test_video_backend_falls_back_to_pyav_when_torchcodec_cannot_load(monkeypatch):
    import builtins

    from openpi.training import data_loader

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("torchcodec"):
            raise RuntimeError("Could not load libtorchcodec")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert data_loader.lerobot_video_backend() == "pyav"


def test_task_mapping_accepts_lerobot_v3_dataframe_and_legacy_dict():
    import pandas as pd

    from openpi import transforms
    from openpi.training import data_loader

    table = pd.DataFrame({"task_index": [0, 1]}, index=["Pick.", "Place."])
    assert data_loader.lerobot_task_mapping(SimpleNamespace(tasks=table)) == {
        0: "Pick.",
        1: "Place.",
    }
    assert data_loader.lerobot_task_mapping({0: "Pick.", 1: "Place."}) == {
        0: "Pick.",
        1: "Place.",
    }
    prompt = transforms.PromptFromLeRobotTask(table)({"task_index": 1})["prompt"]
    assert prompt == "Place."


def _write_v30_dataset(root, *, with_overhead):
    """Write a small LeRobot v3.0 dataset with two wrist cameras (and optionally an overhead one)."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.left_wrist": {"dtype": "image", "shape": (8, 10, 3), "names": ["height", "width", "channel"]},
        "observation.images.right_wrist": {"dtype": "image", "shape": (8, 10, 3), "names": ["height", "width", "channel"]},
    }
    if with_overhead:
        features["observation.images.overhead"] = dict(features["observation.images.left_wrist"])
    dataset = LeRobotDataset.create("local/wrist_only", fps=10, root=root, features=features, use_videos=False)
    for episode, task in enumerate(("Pick.", "Place.")):
        for frame in range(4):
            row = {
                "observation.state": np.full(3, 10 * episode + frame, np.float32),
                "action": np.full(2, 100 * episode + frame, np.float32),
                "observation.images.left_wrist": np.full((8, 10, 3), 10 * episode + frame, np.uint8),
                "observation.images.right_wrist": np.full((8, 10, 3), 50 + 10 * episode + frame, np.uint8),
                "task": task,
            }
            if with_overhead:
                row["observation.images.overhead"] = np.full((8, 10, 3), 200, np.uint8)
            dataset.add_frame(row)
        dataset.save_episode()
    dataset.finalize()
    return dataset.meta


@pytest.mark.parametrize("with_overhead", [False, True])
def test_lerobot_v30_dataset_reads_wrist_cameras_without_overhead(tmp_path, with_overhead):
    from openpi.training import data_loader

    root = tmp_path / "wrist_only"
    meta = _write_v30_dataset(root, with_overhead=with_overhead)
    assert meta.info["codebase_version"] == "v3.0"
    assert ("observation.images.overhead" in meta.camera_keys) is with_overhead
    wrists = ("observation.images.left_wrist", "observation.images.right_wrist")
    dataset = data_loader.ConfiguredCameraDataset(
        "local/wrist_only",
        root=root,
        delta_timestamps={"action": [t / 10 for t in range(3)]},
        source_cameras=wrists,
    )
    assert len(dataset) == 8
    assert data_loader.lerobot_task_mapping(dataset.meta) == {0: "Pick.", 1: "Place."}
    sample = dataset[5]  # episode 1, frame 1
    assert "observation.images.overhead" not in sample
    assert int(sample["episode_index"]) == 1 and int(sample["task_index"]) == 1
    np.testing.assert_array_equal(np.asarray(sample["observation.state"]), np.full(3, 11, np.float32))
    np.testing.assert_array_equal(
        np.asarray(sample["action"]), np.array([[101, 101], [102, 102], [103, 103]], np.float32)
    )
    assert not np.asarray(sample["action_is_pad"]).any()
    left, right = (np.asarray(sample[key]) for key in wrists)
    assert left.shape == right.shape == (3, 8, 10)
    np.testing.assert_allclose(left, 11 / 255, atol=1e-6)
    np.testing.assert_allclose(right, 61 / 255, atol=1e-6)
    end = dataset[7]  # last frame: action chunk pads past the episode end
    np.testing.assert_array_equal(np.asarray(end["action_is_pad"]), [False, True, True])
    if not with_overhead:
        with pytest.raises(ValueError, match="Configured source cameras"):
            data_loader.ConfiguredCameraDataset(
                "local/wrist_only",
                root=root,
                source_cameras=(*wrists, "observation.images.overhead"),
            )
