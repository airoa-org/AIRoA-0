# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Run reservation, retries and checkpoint continuation."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def entrypoint():
    path = Path(__file__).resolve().parents[1] / "scripts/post_train.py"
    spec = importlib.util.spec_from_file_location("post_train_preflight_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reservation(tmp_path):
    out = tmp_path / "checkpoints/run01"
    record = tmp_path / "records/run01.json"
    resolved = {
        "config": {"training": {"steps": 20}, "data": {"repo_id": "test/task"}},
        "data_signature": "dataset-signature",
        "checkpoint_dir": str(out),
        "init_params": "/separate/pretrained/params",
        "steps": 20,
        "retry_lock_contract": "local_flock_v1",
    }
    return record, out, resolved


def test_invalid_wandb_mode_fails_before_any_run_reservation(entrypoint, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    config = SimpleNamespace(wandb_enabled=True)
    with pytest.raises(ValueError, match="WANDB_MODE"):
        entrypoint.train(object(), config)
    assert list(tmp_path.iterdir()) == []


def test_lock_covers_the_trainer_and_releases_after_failure(entrypoint, monkeypatch, tmp_path):
    config = SimpleNamespace(
        wandb_enabled=False, checkpoint_base_dir=str(tmp_path), name="task", exp_name="run01"
    )
    record = tmp_path / "_run_records/task/run01.json"

    def fail_while_locked(settings, config):
        with pytest.raises(FileExistsError, match="Another invocation"):
            with entrypoint.run_lock(record):
                pytest.fail("Concurrent invocation acquired the run")
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(entrypoint, "train_locked", fail_while_locked)
    with pytest.raises(RuntimeError, match="initialization failed"):
        entrypoint.train(object(), config)
    with entrypoint.run_lock(record):
        pass


@pytest.mark.parametrize("output_exists", [False, True])
def test_retry_retains_original_record_and_wandb_evidence(entrypoint, reservation, output_exists):
    record, out, resolved = reservation
    with entrypoint.run_lock(record):
        assert not entrypoint.prepare_run_record(record, out, resolved, resume=False)
    original = record.read_bytes()
    if output_exists:
        out.mkdir(parents=True)
        (out / "wandb_id.txt").write_text("failed-attempt-id")
    with entrypoint.run_lock(record):
        assert entrypoint.prepare_run_record(record, out, resolved, resume=False) == output_exists
    assert record.read_bytes() == original
    receipts = list(record.parent.glob("run01.retry-*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["original_record_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["previous_wandb_id"] == ("failed-attempt-id" if output_exists else None)
    assert receipt["retry"] == resolved


@pytest.mark.parametrize("directory", ["0", "17", "17.orbax-checkpoint-tmp"])
def test_retry_never_reuses_numeric_checkpoint_directories(entrypoint, reservation, directory):
    record, out, resolved = reservation
    entrypoint.prepare_run_record(record, out, resolved, resume=False)
    checkpoint = out / directory
    checkpoint.mkdir(parents=True)
    (checkpoint / "evidence.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="checkpoint progress"):
        entrypoint.prepare_run_record(record, out, resolved, resume=False)
    assert (checkpoint / "evidence.txt").read_text() == "preserve"
    assert not list(record.parent.glob("run01.retry-*.json"))


def test_retry_rejects_changed_config_legacy_records_and_orphan_outputs(entrypoint, reservation):
    record, out, resolved = reservation
    entrypoint.prepare_run_record(record, out, resolved, resume=False)
    changed = copy.deepcopy(resolved)
    changed["config"]["training"]["steps"] += 1
    with pytest.raises(ValueError, match="Retry config changed"):
        entrypoint.prepare_run_record(record, out, changed, resume=False)
    legacy = copy.deepcopy(resolved)
    legacy.pop("retry_lock_contract")
    record.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="Legacy run ownership"):
        entrypoint.prepare_run_record(record, out, resolved, resume=False)
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="no original record"):
        entrypoint.prepare_run_record(record.with_name("other.json"), out, resolved, resume=False)


def test_completed_run_cannot_be_retried_without_its_checkpoints(entrypoint, reservation):
    record, out, resolved = reservation
    entrypoint.prepare_run_record(record, out, resolved, resume=False)
    record.with_name("run01.complete.json").write_text("{}")
    with pytest.raises(FileExistsError, match="checkpoint progress"):
        entrypoint.prepare_run_record(record, out, resolved, resume=False)


def test_committed_resume_still_requires_only_a_stop_step_extension(entrypoint, reservation):
    record, out, resolved = reservation
    entrypoint.prepare_run_record(record, out, resolved, resume=False)
    committed = out / "19"
    committed.mkdir(parents=True)
    (committed / "_CHECKPOINT_METADATA").write_text("{}")
    with pytest.raises(ValueError, match="already reached"):
        entrypoint.prepare_run_record(record, out, resolved, resume=True)
    extended = copy.deepcopy(resolved)
    extended["steps"] = extended["config"]["training"]["steps"] = 30
    assert not entrypoint.prepare_run_record(record, out, extended, resume=True)
    extended["config"]["data"]["repo_id"] = "different/task"
    with pytest.raises(ValueError, match="only training.steps"):
        entrypoint.prepare_run_record(record, out, extended, resume=True)


def test_actual_checkpoint_manager_initializes_an_empty_retry_directory(tmp_path):
    from openpi.training import checkpoints

    out = tmp_path / "failed_initialization"
    out.mkdir()
    (out / "wandb_id.txt").write_text("prior-attempt-id")
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        out, keep_period=None, overwrite=False, resume=True
    )
    try:
        assert resuming is False
        assert tuple(manager.all_steps()) == ()
        assert (out / "wandb_id.txt").read_text() == "prior-attempt-id"
    finally:
        manager.close()


def test_actual_checkpoint_manager_restores_committed_step_zero(tmp_path):
    import numpy as np
    from openpi.training import checkpoints

    out = tmp_path / "one_update"
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        out, keep_period=None, overwrite=False, resume=False
    )
    saved = {
        "train_state": {"step": np.asarray(1, np.int32)},
        "params": {"params": {"probe": np.asarray([2.5], np.float32)}},
    }
    try:
        assert resuming is False
        manager.save(0, items=saved)
        manager.wait_until_finished()
        assert tuple(manager.all_steps()) == (0,)
    finally:
        manager.close()

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        out, keep_period=None, overwrite=False, resume=True
    )
    try:
        assert resuming is True
        restored = manager.restore(0, items=saved)
        np.testing.assert_array_equal(restored["train_state"]["step"], saved["train_state"]["step"])
        np.testing.assert_array_equal(
            restored["params"]["params"]["probe"], saved["params"]["params"]["probe"]
        )
    finally:
        manager.close()
