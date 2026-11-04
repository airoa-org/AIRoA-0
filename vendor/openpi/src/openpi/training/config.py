# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
import abc
from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any, Protocol, TypeAlias
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Locations of training normalization assets."""

    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    data_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    model_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ]
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        """Create a data config."""

    def create_base_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config
    )
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=_optimizer.AdamW
    )
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(
        default_factory=nnx.Nothing
    )
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (
            pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name
        ).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
