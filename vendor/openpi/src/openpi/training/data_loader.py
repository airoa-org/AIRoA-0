# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar
import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            "Subclasses of IterableDataset should implement __iter__."
        )

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError(
            "Subclasses of DataLoader should implement data_config."
        )

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                batch_size = next((v.shape[0] for v in sample.values()))
                individual_samples = [
                    jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)
                ]
                transformed = [self._transform(s) for s in individual_samples]
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        (self._observation_spec, self._action_spec) = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            (rng, data_rng) = jax.random.split(rng)
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(
                    data_rng, shape=shape, minval=-1.0, maxval=1.0
                )
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        return {**observation.to_dict(), "actions": action}

    def __len__(self) -> int:
        return self._num_samples


def lerobot_video_backend() -> str:
    """LeRobot's default decoder, falling back to PyAV when torchcodec cannot load.

    LeRobot >= 0.4 picks torchcodec whenever the package is installed, but the
    wheel needs FFmpeg shared libraries at runtime. Probe an actual import so hosts
    without FFmpeg still read videos instead of failing in the first worker.
    """
    try:
        import torchcodec.decoders  # noqa: F401
    except Exception as error:  # noqa: BLE001 - any load failure means "unusable"
        logging.warning("torchcodec unavailable (%s); decoding videos with PyAV", error)
        return "pyav"
    return "torchcodec"


def lerobot_task_mapping(meta_or_tasks) -> dict[int, str]:
    """Return ``{task_index: task string}`` for a LeRobot task table.

    LeRobot >= 0.4 (dataset format v3.0) stores tasks as a pandas DataFrame indexed
    by the task string with a ``task_index`` column; older releases used a dict.
    """
    tasks = getattr(meta_or_tasks, "tasks", meta_or_tasks)
    if hasattr(tasks, "columns") and "task_index" in tasks.columns:
        mapping = {int(index): str(task) for task, index in tasks["task_index"].items()}
    else:
        mapping = {int(index): str(task) for index, task in dict(tasks).items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate task strings in the LeRobot task table")
    return mapping


class ConfiguredCameraDataset(lerobot_dataset.LeRobotDataset):
    """Decode only source cameras mapped to model logical views.

    Selected videos use the original decoder, timestamps and tolerance. Removing
    unused image columns prevents Hugging Face from decoding them in __getitem__.
    Logical camera mapping remains in the input adapter.
    """

    def __init__(
        self,
        *args,
        source_cameras,
        history_slots=0,
        history_state_key="observation.state",
        history_time_key="timestamp",
        history_valid_key="",
        history_rates_hz=(5.0, 8.0, 10.0, 12.0, 15.0),
        history_dropout=0.25,
        history_seed=42,
        **kwargs,
    ):
        # Set before LeRobot's __init__: its local-cache check below needs it.
        self.source_cameras = tuple(dict.fromkeys(source_cameras))
        super().__init__(*args, **kwargs)
        if not self.source_cameras or not set(self.source_cameras) <= set(
            self.meta.camera_keys
        ):
            raise ValueError("Configured source cameras must exist in dataset metadata")
        if self.image_transforms is not None:
            raise ValueError("Apply image transforms after configured-camera selection")
        unused_images = set(self.meta.image_keys) - set(self.source_cameras)
        columns = [key for key in self.hf_dataset.column_names if key in unused_images]
        if columns:
            self.hf_dataset = self.hf_dataset.remove_columns(columns)
        if type(history_slots) is not int or history_slots < 0:
            raise ValueError("History slots must be a nonnegative integer")
        self.history_slots = history_slots
        if history_slots:
            self._init_history(
                history_state_key,
                history_time_key,
                history_valid_key,
                history_rates_hz,
                history_dropout,
                history_seed,
            )

    def _check_cached_episodes_sufficient(self) -> bool:
        """LeRobot >= 0.4 local-cache check restricted to the decoded cameras.

        Upstream requires a video file for every camera listed in info.json and
        otherwise falls back to downloading from the Hub. Only configured source
        cameras are ever decoded here, so videos of unused cameras may be absent.
        """
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False
        available = {
            int(index) for index in self.hf_dataset.unique("episode_index")
        }
        requested = (
            set(range(self.meta.total_episodes))
            if self.episodes is None
            else set(self.episodes)
        )
        if not requested <= available:
            return False
        keys = [key for key in self.meta.video_keys if key in self.source_cameras]
        return all(
            (self.root / self.meta.get_video_file_path(episode, key)).exists()
            for episode in requested
            for key in keys
        )

    def _init_history(self, state_key, time_key, valid_key, rates, dropout, seed):
        # Numeric-only Arrow views never decode historical images. Only small
        # scalar clock/support arrays are cached; state rows stay Arrow-backed.
        if not len(self):
            raise ValueError("Proprio history requires a nonempty dataset")
        names = self.hf_dataset.column_names
        required = {state_key, time_key, "episode_index", "frame_index"}
        if valid_key:
            required.add(valid_key)
        missing = required - set(names)
        if missing:
            raise ValueError(f"Missing proprio history columns: {sorted(missing)}")
        validity_keys = list(
            dict.fromkeys(k for k in (valid_key, "state_valid") if k and k in names)
        )
        numeric = self.hf_dataset.select_columns(
            sorted(required | set(validity_keys))
        ).with_format("numpy")
        self._history_states = numeric.select_columns([state_key])
        self._history_state_key = state_key

        def scalar_column(key):
            values = np.asarray(numeric[key])
            if values.shape == (len(self), 1):
                values = values[:, 0]
            if values.shape != (len(self),):
                raise ValueError(f"History {key} must be a scalar per source row")
            return values

        times = scalar_column(time_key).astype(np.float64)
        episodes = scalar_column("episode_index")
        frames = scalar_column("frame_index")
        if (
            not np.isfinite(times).all()
            or not np.issubdtype(frames.dtype, np.integer)
            or not np.issubdtype(episodes.dtype, np.integer)
        ):
            raise ValueError(
                "History clocks must be finite and episode/frame identities integer"
            )
        valid = np.ones(len(self), bool)
        for key in validity_keys:
            values = scalar_column(key)
            if values.dtype != np.bool_:
                raise ValueError(f"History {key} must be boolean")
            valid &= values
        if (
            not rates
            or any(not np.isfinite(r) or r <= 0 for r in rates)
            or not 0 <= dropout < 1
        ):
            raise ValueError("Invalid proprio sampling rates/dropout")
        starts = np.zeros(len(self), np.int64)
        boundaries = np.r_[
            0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, len(self)
        ]
        if len(set(episodes[boundaries[:-1]].tolist())) != len(boundaries) - 1:
            raise ValueError("History episode rows must be contiguous")
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            differences = np.diff(times[begin:end])
            if np.any(differences <= 0) or np.any(np.diff(frames[begin:end]) <= 0):
                raise ValueError(
                    "History clock/frame must strictly increase within each episode"
                )
            # Clock-scale invariant; do not substitute the action query rate.
            gap_limit = (
                2 * np.median(differences) + 1e-7 if len(differences) else np.inf
            )
            start = int(begin)
            for row in range(int(begin), int(end)):
                if row > begin and (
                    frames[row] != frames[row - 1] + 1
                    or times[row] - times[row - 1] > gap_limit
                ):
                    start = row
                if not valid[row]:
                    start = row + 1
                starts[row] = start
        self._history_times, self._history_episodes = times, episodes
        self._history_frames, self._history_starts = frames, starts
        self._history_rates, self._history_dropout, self._history_seed = (
            tuple(rates),
            dropout,
            seed,
        )

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if not self.history_slots:
            return result
        from openpi.shared.proprio_frequency import (
            resampling_plan,
            sample_history_schedule,
        )

        row = int(index)
        rate, count = sample_history_schedule(
            slots=self.history_slots,
            rates_hz=self._history_rates,
            dropout=self._history_dropout,
            seed=self._history_seed,
            episode=int(self._history_episodes[row]),
            frame=int(self._history_frames[row]),
        )
        now = self._history_times[row]
        start = min(row, int(self._history_starts[row]))
        # Bound numeric search to the largest possible sampler support window.
        support = self._history_times[start:row]
        offset = max(
            0, int(np.searchsorted(support, now - (self.history_slots + 1) / rate)) - 1
        )
        start += offset
        plan = resampling_plan(
            self._history_times[start:row],
            now,
            interval=1 / rate,
            slots=self.history_slots,
            count=count,
            interpolate=False,
        )
        state = np.asarray(result[self._history_state_key], np.float32)
        if state.ndim != 1:
            raise ValueError("History source state must be a single measured vector")
        history = np.zeros((self.history_slots, len(state)), np.float32)
        mask = np.zeros(self.history_slots, bool)
        ages = np.zeros(self.history_slots, np.float32)
        selected = [
            (slot, start + entry[0], entry[3])
            for slot, entry in enumerate(plan)
            if entry is not None
        ]
        if selected:
            values = self._history_states.select([r for _, r, _ in selected])[
                self._history_state_key
            ]
            for (slot, _, age), value in zip(selected, values, strict=True):
                value = np.asarray(value, np.float32)
                if value.shape != state.shape or not np.isfinite(value).all():
                    raise ValueError(
                        "Eligible measured proprio history must be finite and match current state"
                    )
                history[slot], mask[slot], ages[slot] = value, True, age
        return {
            **result,
            "state_history": history,
            "state_history_mask": mask,
            "state_history_age": ages,
        }

    def _get_query_timestamps(self, current_ts, query_indices=None):
        timestamps = super()._get_query_timestamps(current_ts, query_indices)
        return {
            key: value
            for (key, value) in timestamps.items()
            if key in self.source_cameras
        }


def configured_source_cameras(data_config):
    for transform in data_config.data_transforms.inputs:
        cameras = getattr(transform, "cameras", None)
        if cameras:
            return tuple(dict.fromkeys((value for (_, value) in cameras)))
    raise ValueError(
        "Prepared training input requires an explicit logical-to-source camera mapping"
    )


def configured_proprio_history(data_config, model_config):
    for adapter in data_config.data_transforms.inputs:
        slots = getattr(adapter, "history_slots", 0)
        if slots:
            if not getattr(model_config, "use_temporal_proprio_memory", False):
                raise ValueError(
                    "Data proprio history requires model proprio memory to be enabled"
                )
            if len(model_config.temporal_proprio_history_lags) != slots:
                raise ValueError("Data/model proprio history slots differ")
            return dict(
                history_slots=slots,
                history_state_key=adapter.state_key,
                history_time_key=adapter.history_time_key,
                history_valid_key=adapter.history_valid_key,
                history_rates_hz=model_config.proprio_frequency_rates_hz,
                history_dropout=model_config.proprio_frequency_dropout,
                history_seed=model_config.proprio_frequency_seed,
            )
    if getattr(model_config, "use_temporal_proprio_memory", False):
        raise ValueError(
            "Model proprio memory requires an input adapter with positive history_slots"
        )
    return {}


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = ConfiguredCameraDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        source_cameras=configured_source_cameras(data_config),
        video_backend=lerobot_video_backend(),
        **configured_proprio_history(data_config, model_config),
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(lerobot_task_mapping(dataset_meta))]
        )
    return dataset


def transform_dataset(
    dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and (not skip_norm_stats):
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and (not skip_norm_stats):
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    if framework != "jax":
        raise ValueError("Only JAX post-training is supported")
    'Create a data loader for training.\n\n    Args:\n        config: The training configuration.\n        sharding: The sharding to use for the data loader (JAX only).\n        shuffle: Whether to shuffle the data.\n        num_batches: Determines the number of batches to return.\n        skip_norm_stats: Whether to skip data normalization.\n        framework: The framework to use ("jax" or "pytorch").\n    '
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )
    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        drop_last: bool = True,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            drop_last: Drop partial batches for training; normalization keeps them.
        """
        if jax.process_count() > 1:
            raise NotImplementedError(
                "Data loading with multiple processes is not supported."
            )
        if not len(dataset) or (drop_last and len(dataset) < local_batch_size):
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)})."
            )
        self._sharding = sharding
        if sharding is None and framework == "jax":
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=sampler is None and shuffle,
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(
                            self._sharding, x
                        ),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield (_model.Observation.from_dict(batch), batch["actions"])
