# Training data

Use LeRobot v3.0-format data with the reader pinned in `pyproject.toml`. For a local
dataset, set `HF_LEROBOT_HOME` to the cache root and `data.repo_id` to its ID, and set
`HF_HUB_OFFLINE=1`: LeRobot falls back to downloading from the Hub when it considers
the local copy incomplete.

The pinned LeRobot reads v3.0 only. Convert a LeRobot v2.1 dataset once with
LeRobot's converter; it works in place and keeps the original as `<repo_id>_old`:

```sh
HF_HUB_OFFLINE=1 python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id <repo_id> --root <cache> --push-to-hub=false
```

The local layout is:

```text
<cache>/<repo_id>/meta/info.json
<cache>/<repo_id>/data/...
<cache>/<repo_id>/videos/...
```

## Observations and actions

Configure the state/action feature names and dimensions using the units,
coordinate frames and action conventions expected by the pretrained model.
Training uses the supplied episodes and their timestamps.

Map `left_wrist_0_rgb` and `right_wrist_0_rgb` to the dataset's wrist-camera
image features; the default configuration reads only these two views, and the
unmapped `base_0_rgb` view is masked. Each configured feature must exist, and each
source camera maps to one model view. Cameras that are not mapped are never
decoded: they may be absent from the dataset, or listed in `meta/info.json`
without video files.
Images must be RGB, either uint8 in `[0, 255]` or floating point in `[0, 1]`.
CHW inputs are converted to HWC. Resizing preserves aspect ratio; padding masks
accompany the image tokens.

Run `inspect` to check the schema and an example observation before computing
normalization assets.

## Instructions and labels

Each frame's LeRobot task string is its PA label. Set the full action instruction
in `labels.task_instruction`, list canonical stage texts in `labels.stages`, and
map every dataset PA string to a zero-based stage ID in `labels.pa_to_stage`.
The dataset labels must cover every configured stage. Any number of stages,
two or more, is supported.

The action input uses the full task instruction. PA labels supervise the text
head; stage labels supervise an observation-based progress classifier.
Keep text within `auxiliary.max_tokens`, `stage.max_tokens` and
`model.max_token_len`; overlong labels raise an error.

## Validity and normalization

Optional action masks are `action_time_valid_mask` with shape `[H]` and
`action_time_dim_valid_mask` with shape `[H, action_dim]`. LeRobot endpoint
padding (`<action_key>_is_pad`, or `action_is_pad`) is masked from the loss.
The adapter carries validity through model padding and NUF packing. Action
values must be finite wherever their masks are true. Masked values are set to
zero, and padded model dimensions keep false validity masks.

Run `norm` after configuring the dataset. State statistics use anchors with
eligible targets; each action dimension uses its eligible values. History shares
the current state's statistics. Saved normalization assets include a data
contract covering the dataset metadata and mapped features. Generate new assets
when changing field mappings, action horizon, label vocabulary or representation.

## Proprio history

History selects strictly past measured states within the current episode,
ordered newest first. A deterministic schedule chooses the sampling frequency,
number of populated slots and whole-history dropout. Each slot carries its
actual elapsed age in seconds. Current observations supply the RGB inputs.

`data.history_time_key` selects the seconds clock (default `timestamp`).
`data.history_valid_key`, when set, names a boolean per-frame validity field.
The reader also honors `state_valid` when present; without either validity
field, the supplied episode frames are treated as valid.
Invalid rows, missing frame indices and clock gaps larger than twice the
episode's median clock step reset history support. Valid history uses the same
state normalization and model-width padding; masked slots remain zero.
