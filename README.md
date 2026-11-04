# AIRoA-0 post-training

<p>
  <a href="https://a-0.airoa.io"><img alt="Project page" src="https://img.shields.io/badge/Project-Page-2f6fde?logo=googlechrome&logoColor=white" height="26"></a>
  <a href="https://arxiv.org/abs/2611.03815"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2611.03815-b31b1b?logo=arxiv&logoColor=white" height="26"></a>
  <a href="https://huggingface.co/airoa-org/AIRoA-0"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-AIRoA--0-ffb000" height="26"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-3a3f44" height="26"></a>
</p>

Post-train AIRoA-0 (A-0) on a LeRobot dataset using action flow matching, PA text
supervision, stage classification and visual-budget regularization. The same
entry point supports anchor and NUF action training, with optional proprio history.

AIRoA-0 builds on [π0.5](https://www.pi.website/blog/pi05)
([paper](https://arxiv.org/abs/2504.16054)) by Physical Intelligence, and this
code derives from their [OpenPI](https://github.com/Physical-Intelligence/openpi)
implementation.

## Install

Use Python 3.11, Linux and an NVIDIA GPU with a compatible CUDA 12 driver.
Video datasets also require FFmpeg shared libraries available to TorchCodec;
see its [installation guide](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[train,test]'
cp configs/post_training.toml configs/local.toml
```

Provide the pretrained weights and your dataset. The tokenizer is downloaded on
first use. For online W&B logging, run `wandb login` before training; set
`training.wandb = false` to use local logging instead.

## Configure

Edit `configs/local.toml`. Relative paths resolve against the configuration file.

- Set `init_params` to the pretrained checkpoint directory or its `params/`
  subdirectory, and set writable `assets_dir` and `checkpoint_dir` paths.
- Set `data.repo_id`, the state/action fields and dimensions, and camera mappings.
  Use the units, coordinate frames and action conventions expected by the model
  (see [State and action layout](#state-and-action-layout)). The dataset must be
  in LeRobot v3.0 format; convert LeRobot v2.1 datasets first (see
  [the data guide](docs/data.md)).
- Set `labels.task_instruction` to the full task instruction. Each frame's
  dataset task string supplies its PA label; map each PA to a zero-based stage
  ID with `labels.pa_to_stage`. Stage supervision supports any number of stages
  (two or more).
- Set `training.fsdp_devices` and `training.batch_size` for your allocation.
  The global batch must divide evenly across visible GPUs, and the GPU count
  must divide evenly by `fsdp_devices`.

The initializer loads the pretrained core and visual-budget gate weights and
initializes the task's stage classifier. A new proprio-history branch is
initialized when the supplied checkpoint has none.

For local data, set `HF_LEROBOT_HOME` to the cache containing
`<repo_id>/meta/info.json` and the dataset's data/video files. See
[the data guide](docs/data.md) for feature, label and mask requirements.

## State and action layout

`data.state_dim` and `data.action_dim` are the widths of your dataset's state and
action features (1 to 32 each). Both are zero-padded to the model's 32 dimensions,
and padded action dimensions are masked from the loss. The values in
`configs/post_training.toml` are placeholders; set them to match your data.

The released post-trained checkpoints use an 11-dimensional state and a
20-dimensional action, and their `norm_stats.json` files only fit this layout.
Rotations use the 6D rotation representation.

| Feature | Indices | Content |
|---|---|---|
| state (11) | 0-2 | Position of the right gripper in the left gripper's frame |
| | 3-8 | Rotation of the right gripper in the left gripper's frame (6D) |
| | 9-10 | Left and right gripper opening |
| action (20) | 0-2, 9-11 | Left and right gripper translation, relative to its pose at the current frame |
| | 3-8, 12-17 | Left and right gripper rotation (6D), relative to its pose at the current frame |
| | 18-19 | Left and right gripper opening |

## Post-train

```sh
./run.sh inspect --config configs/local.toml
./run.sh norm --config configs/local.toml
./run.sh train --config configs/local.toml --run trial01
```

`inspect` checks the dataset schema and an example observation. `norm` computes
normalization assets. `train` starts the configured run.

The default recipe uses global batch 64, 20,000 updates, AdamW, a 500-step
warmup followed by constant learning rate `5e-6`, EMA `0.99`, and 16 loader
workers. Visual-budget regularization targets a mean gate value of `0.9`;
PA and stage loss weights are `0.01` and `0.1`. The example uses eight GPUs;
allocate at least 32 CPU cores for its input pipeline.

Anchor training uses action horizon 20. For C8/N2 NUF training, configure:

```toml
[model]
action_horizon = 16
max_token_len = 200
nuf_chunk_size = 8
nuf_buffer_size = 2
nuf_flow_times = [0.1, 1.0]
```

Other positive chunk lengths use `nuf_buffer_size = 2` and
`action_horizon = 2 * nuf_chunk_size`.

## Proprio history

The default samples up to 10 past measured states at 5/8/10/12/15 Hz per
observation, with a `0.25` probability of dropping the whole history. Each state
carries its elapsed age in seconds. Four learned prefix memory tokens summarize
this history, using the current state's normalization statistics and a prefix
scale of `0.1`. Set `proprio_history.slots = 0` to disable history.

Set `data.history_time_key` to the dataset's seconds clock. If available, use
`data.history_valid_key` to supply per-frame validity. Episode boundaries,
invalid frames and clock discontinuities limit the available history.

## Checkpoints and continuation

An action horizon counts dataset samples; its time span follows the dataset's
frame rate.

Outputs are saved under `<checkpoint_dir>/<name>/<run>/`. Checkpoints contain
working parameters and optimizer state. With EMA enabled, `params/` contains
EMA weights. To continue the same run, increase `training.steps` and run:

```sh
./run.sh train --config configs/local.toml --run trial01 --resume
```

Keep the other configuration settings unchanged when resuming. Model and
optimizer state are restored; data sampling restarts. Use a new run name and
matching normalization assets when changing the dataset or model configuration.

See [the objective definitions](docs/objectives.md) for formulas and
[verification](docs/verification.md) for test coverage.

## Tests

```sh
python -m pytest -q
```

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The code
derives from OpenPI and Big Vision, and the model builds on Gemma; see
[third-party notices](THIRD_PARTY.md). Model weights are distributed separately
and are subject to the Gemma Terms of Use.

## Contact

Questions about AIRoA-0: [ext-oss@airoa.org](mailto:ext-oss@airoa.org)

## Citation

```bibtex
@techreport{airoa0,
  title         = {{AIRoA-0}: Long-Horizon, Precise, Cross-Embodiment
                   Manipulation from Handheld Data},
  author        = {Yueh-Hua Wu and Kei Ota and Hanlong Li and
                   Hayata Yuasa and Jason Wang and Kengo Ikeuchi and
                   Kensuke Chiyoma and Makoto Sugiura and
                   Masafumi Takahashi and Masatoshi Tateno and
                   Naoaki Kanazawa and Nicola Scianca and
                   Petr Khrapchenkov and Satoshi Otsubo and
                   Takehiko Ohkawa and Takuya Okubo and
                   Tatsuya Matsushima and Yuki Noguchi and Yuma Shin and
                   Zixuan Zhong},
  institution   = {AIRoA},
  year          = {2026},
  month         = nov,
  eprint        = {2611.03815},
  archivePrefix = {arXiv}
}
```
