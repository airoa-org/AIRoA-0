# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
from torch.utils.data import Subset
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.RemoveStrings(),
        ],
    )
    if max_frames is not None:
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if max_frames < len(dataset):
            # Bound the sampled anchors exactly, including a partial last batch.
            indices = np.random.default_rng(0).choice(len(dataset), max_frames, replace=False)
            dataset = Subset(dataset, indices.tolist())
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        num_batches=num_batches,
        framework="pytorch",
        drop_last=False,
    )
    return (data_loader, num_batches)
