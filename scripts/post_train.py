#!/usr/bin/env python3
# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Run with ./run.sh; imports are delayed so --help needs no GPU runtime."""

from pathlib import Path
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
import time

if sys.version_info < (3, 11):
    raise SystemExit(
        "Python 3.11+ is required. Use ./run.sh or the pinned environment."
    )

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/openpi"
sys.path[:0] = [
    str(ROOT / "src"),
    str(VENDOR / "src"),
    str(VENDOR / "packages/openpi-client/src"),
]
from post_training.settings import Settings  # noqa: E402


def upstream_script(name):
    spec = importlib.util.spec_from_file_location(
        "s2_upstream_" + name, VENDOR / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataset_info(settings, config):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from openpi.training.data_loader import lerobot_task_mapping

    meta = LeRobotDatasetMetadata(config.data.repo_id)
    data = settings.raw["data"]
    required = [data["state_key"], data["action_key"], *data["cameras"].values()]
    if settings.raw.get("proprio_history", {}).get("slots", 0):
        required.append(data.get("history_time_key", "timestamp"))
        if data.get("history_valid_key"):
            required.append(data["history_valid_key"])
    missing = set(required) - set(meta.features)
    if missing:
        raise ValueError(f"Missing dataset features: {sorted(missing)}")
    for key, dimension in (
        (data["state_key"], data["state_dim"]),
        (data["action_key"], data["action_dim"]),
    ):
        if tuple(meta.features[key]["shape"]) != (dimension,):
            raise ValueError(
                f"{key}: metadata shape does not match configured dimension {dimension}"
            )
    task_strings = set(lerobot_task_mapping(meta).values())
    if settings.raw["labels"]["task_instruction"]:
        labels = settings.raw["labels"]
        unknown = task_strings - set(labels["pa_to_stage"])
        if unknown:
            raise ValueError(f"Unmapped dataset PA task strings: {sorted(unknown)}")
        if settings.raw["stage"]["enabled"]:
            covered = {labels["pa_to_stage"][pa] for pa in task_strings}
            if covered != set(range(len(labels["stages"]))):
                raise ValueError(
                    "Dataset task table does not cover all configured stages"
                )
    return {
        "repo_id": config.data.repo_id,
        "fps": meta.fps,
        "info_sha256": hashlib.sha256(
            json.dumps(meta.info, sort_keys=True).encode()
        ).hexdigest(),
        "features": {k: meta.features[k] for k in required},
        "horizon_seconds": (config.model.action_horizon - 1) / meta.fps,
    }


def stats_contract(settings, config):
    # LeRobot may expose feature shapes as tuples. Canonicalize to the JSON form
    # used on disk so a matching dataset does not fail the reload check.
    value = {
        "data_signature": settings.data_signature,
        "dataset": dataset_info(settings, config),
        "normalization_mask_contract": "eligible_time_and_physical_dimensions_v1",
    }
    return json.loads(json.dumps(value, sort_keys=True))


def verify_stats(settings, config):
    path = settings.stats_dir / "data_contract.json"
    if not path.exists():
        raise FileNotFoundError(f"Run the norm command first: missing {path}")
    expected = stats_contract(settings, config)
    if json.loads(path.read_text())["contract"] != expected:
        raise ValueError(
            "Dataset schema/config changed since normalization; recompute stats in a new asset directory"
        )
    if not (settings.stats_dir / "norm_stats.json").exists():
        raise FileNotFoundError(
            "Normalization data_contract exists but norm_stats.json is missing"
        )


def inspect(settings, config):
    from openpi.training import data_loader

    info = dataset_info(settings, config)
    dc = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(
        dc, config.model.action_horizon, config.model
    )
    if len(raw) == 0:
        raise ValueError("Dataset is empty")
    from post_training.data import Inputs

    adapter = next(x for x in dc.data_transforms.inputs if isinstance(x, Inputs))
    sample = adapter(raw[0])
    if sample["actions"].shape != (
        config.model.action_horizon,
        settings.raw["data"]["action_dim"],
    ):
        raise ValueError(f"Unexpected action chunk shape: {sample['actions'].shape}")
    info.update(
        samples=len(raw),
        state_shape=list(sample["state"].shape),
        action_shape=list(sample["actions"].shape),
        instruction=sample["prompt"],
        active_cameras=[k for k, v in sample["image_mask"].items() if v],
    )
    if "state_history" in sample:
        info["proprio_history"] = {
            "shape": list(sample["state_history"].shape),
            "valid_slots": int(sample["state_history_mask"].sum()),
            "age_seconds": sample["state_history_age"].tolist(),
            "sampling": "recorded_past_state_v2_no_interpolation",
            "clock": adapter.history_time_key,
        }
    print(json.dumps(info, indent=2))


def normalize(settings, config, max_frames, force):
    import numpy as np
    from openpi.shared import normalize as norm

    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be positive")
    dest = settings.stats_dir
    if (dest / "norm_stats.json").exists() and not force:
        raise FileExistsError(
            f"Stats exist at {dest}; use a new assets_dir or explicitly pass --force"
        )
    helper = upstream_script("compute_norm_stats")
    dc = config.data.create(config.assets_dirs, config.model)
    contract = stats_contract(settings, config)
    loader, batches = helper.create_torch_dataloader(
        dc,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames,
    )
    if batches < 1:
        raise ValueError(
            "Dataset is smaller than one batch; lower batch_size for normalization"
        )
    from post_training.data import fit_normalization
    import tqdm

    statistics, counts = fit_normalization(
        tqdm.tqdm(loader, total=batches, desc="Normalization"),
        state_dim=settings.raw["data"]["state_dim"],
        action_dim=settings.raw["data"]["action_dim"],
    )
    dest.mkdir(parents=True, exist_ok=True)
    norm.save(dest, statistics)
    (dest / "data_contract.json").write_text(
        json.dumps({"contract": contract, **counts}, indent=2)
    )
    print(f"Saved normalization for {counts['sampled_anchors']} anchors: {dest}")


def validate_wandb_mode(config):
    if config.wandb_enabled and os.environ.get("WANDB_MODE", "online") not in (
        "",
        "online",
    ):
        raise ValueError(
            "W&B is enabled in config but WANDB_MODE is not online; fix the environment or disable it explicitly"
        )


@contextmanager
def run_lock(record):
    """Hold one local invocation's claim through preflight and trainer completion."""
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.with_suffix(".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError("Another invocation is already using this run name") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def admit_retry(record, out, resolved):
    # The caller holds run_lock throughout admission and the entire trainer call.
    with record.open("rb") as stream:
        original = stream.read()
        previous = json.loads(original)
        complete = record.with_name(record.stem + ".complete.json")
        if complete.exists() or any(
            p.is_dir() and re.match(r"\d+(?:\.|$)", p.name)
            for p in (out.iterdir() if out.exists() else ())
        ):
            raise FileExistsError(
                "Run has checkpoint progress; explicitly resume or choose a new run"
            )
        identity = ("config", "data_signature", "checkpoint_dir", "init_params")
        if any(previous.get(key) != resolved.get(key) for key in identity):
            raise ValueError("Retry config changed; choose a new run name")
        if previous.get("retry_lock_contract") != "local_flock_v1":
            raise ValueError("Legacy run ownership is unknown; choose a new run name")
        wandb_id = out / "wandb_id.txt"
        receipt = record.with_name(f"{record.stem}.retry-{time.time_ns()}.json")
        with receipt.open("x") as retry_stream:
            json.dump(
                {
                    "original_record_sha256": hashlib.sha256(original).hexdigest(),
                    "previous_wandb_id": wandb_id.read_text() if wandb_id.exists() else None,
                    "retry": resolved,
                },
                retry_stream,
                indent=2,
            )
        # The upstream manager admits an existing directory with resume=True,
        # then initializes fresh when it finds no checkpoint steps.
        return out.exists()


def prepare_run_record(record, out, resolved, *, resume):
    """Reserve a run, or admit an evidenced retry before any checkpoint progress."""
    if resume:
        if not record.exists() or not out.exists():
            raise FileNotFoundError(
                "Resume requires both the original run record and checkpoint directory"
            )
        committed = [
            int(p.name)
            for p in out.iterdir()
            if p.is_dir() and p.name.isdigit() and (p / "_CHECKPOINT_METADATA").exists()
        ]
        if not committed:
            raise ValueError("No committed checkpoint to resume; retry train without --resume")
        if resolved["steps"] <= max(committed) + 1:
            raise ValueError(
                "Run already reached this stop step; increase training.steps to continue"
            )
        previous = json.loads(record.read_text())
        old = json.loads(json.dumps(previous["config"]))
        new = json.loads(json.dumps(resolved["config"]))
        old["training"].pop("steps")
        new["training"].pop("steps")
        if old != new:
            raise ValueError("Resume config changed; only training.steps may be extended")
        return False
    if record.exists():
        return admit_retry(record, out, resolved)
    if out.exists():
        raise FileExistsError("Run directory has no original record; choose a new run name")
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("x") as stream:
        json.dump(resolved, stream, indent=2)
    return False


def train(settings, config):
    validate_wandb_mode(config)
    records = Path(config.checkpoint_base_dir) / "_run_records" / config.name
    with run_lock(records / (config.exp_name + ".json")):
        train_locked(settings, config)


def train_locked(settings, config):
    import jax
    from openpi.shared import normalize as norm

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "Training requires the NVIDIA GPU runtime; check installation and CUDA visibility"
        )
    if (
        config.batch_size % jax.device_count()
        or jax.device_count() % config.fsdp_devices
    ):
        raise ValueError(
            "batch_size must divide evenly across visible GPUs; fsdp_devices must divide GPU count"
        )
    if (
        not config.resume
        and not (settings.initialization_params() / "_METADATA").exists()
    ):
        raise FileNotFoundError(
            "init_params must be a complete AIRoA-0 release root or params directory"
        )
    out = Path(config.checkpoint_dir)
    source = settings.initialization_params()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError(
            "Training output must not overlap the initialization checkpoint"
        )
    verify_stats(settings, config)
    stats = norm.load(settings.stats_dir)
    for key, dim in [
        ("state", settings.raw["data"]["state_dim"]),
        ("actions", settings.raw["data"]["action_dim"]),
    ]:
        if stats[key].mean.shape != (dim,):
            raise ValueError(f"Normalization dimensions do not match {key}")
    records = Path(config.checkpoint_base_dir) / "_run_records" / config.name
    record = records / (config.exp_name + ".json")
    resolved = {
        "config": settings.raw,
        "config_path": str(settings.path),
        "data_signature": settings.data_signature,
        "checkpoint_dir": str(out),
        "init_params": str(source),
        "steps": config.num_train_steps,
        "devices": jax.device_count(),
        "jax_version": jax.__version__,
        "retry_lock_contract": "local_flock_v1",
        "invocation": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
        },
        "dataset_iterator_resume": "recreated; optimizer and update state restored, exact data cursor is not restored",
        "runtime_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (VENDOR / "src", VENDOR / "scripts")
            for path in sorted(folder.rglob("*.py"))
        },
        "wrapper_code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                *sorted((ROOT / "src/post_training").glob("*.py")),
            ]
        },
    }
    reuse_empty_output = prepare_run_record(record, out, resolved, resume=config.resume)
    if reuse_empty_output:
        import dataclasses

        config = dataclasses.replace(config, resume=True)
    trainer = upstream_script("train")
    try:
        trainer.main(config)
        import wandb

        wandb.finish(exit_code=0)
    except BaseException:
        import wandb

        wandb.finish(exit_code=1)
        raise
    (records / (config.exp_name + ".complete.json")).write_text(
        json.dumps(resolved, indent=2)
    )
    print(f"Finished: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "norm", "train", "resume"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", default="run01")
    parser.add_argument("--resume", action="store_true", help="Resume a train command")
    parser.add_argument(
        "--max-frames", type=int, help="Optional normalization anchor limit"
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace normalization explicitly"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("--run must contain only letters, numbers, underscores or hyphens")
    if args.resume and args.command != "train":
        parser.error(
            "--resume applies only to train; the resume command already enables it"
        )
    if args.force and args.command != "norm":
        parser.error("--force applies only to norm")
    if args.max_frames is not None and args.command != "norm":
        parser.error("--max-frames applies only to norm")
    settings = Settings.read(args.config)
    settings.require_data_contract()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    from post_training.configuration import build

    config = build(settings, args.run, resume=args.resume or args.command == "resume")
    if args.command == "inspect":
        inspect(settings, config)
    elif args.command == "norm":
        normalize(settings, config, args.max_frames, args.force)
    else:
        train(settings, config)


if __name__ == "__main__":
    main()
