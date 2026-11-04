# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
import contextlib
import logging
import os

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def _activation_sharding_disabled() -> bool:
    value = os.environ.get("OPENPI_DISABLE_ACTIVATION_SHARDING", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _replicated_param_substrings() -> tuple[str, ...]:
    value = os.environ.get("OPENPI_REPLICATE_PARAM_SUBSTRINGS", "")
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _fsdp_min_size_mbytes(default: int = 4) -> int:
    value = os.environ.get("OPENPI_FSDP_MIN_SIZE_MB", "")
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"OPENPI_FSDP_MIN_SIZE_MB must be an integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(f"OPENPI_FSDP_MIN_SIZE_MB must be non-negative, got {parsed}")
    return parsed


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremely cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _activation_sharding_disabled():
        return pytree
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree,
        jax.sharding.NamedSharding(
            _MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS)
        ),
    )


def batch_activation_sharding_constraint(pytree):
    if _activation_sharding_disabled():
        return pytree
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree,
        jax.sharding.NamedSharding(
            _MeshState.active_mesh, jax.sharding.PartitionSpec(BATCH_AXIS)
        ),
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int | None = None,
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_mbytes = (
        _fsdp_min_size_mbytes() if min_size_mbytes is None else min_size_mbytes
    )
    min_size_bytes = min_size_mbytes * 2**20
    replicated_substrings = _replicated_param_substrings()

    if log:
        logging.info(
            "Applying FSDP sharding with fsdp_axis=%s min_size_mbytes=%d replicated_overrides=%s",
            mesh.shape[FSDP_AXIS],
            min_size_mbytes,
            replicated_substrings or "<none>",
        )

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        key = jax.tree_util.keystr(kp)
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if replicated_substrings and any(
            substr in key for substr in replicated_substrings
        ):
            if log:
                logging.info(
                    "Replicating %s due to OPENPI_REPLICATE_PARAM_SUBSTRINGS override",
                    key,
                )
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (
            arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize
        ) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {key} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec(*spec)
                )

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {key} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
