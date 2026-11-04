# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Task/stage labels and causal PA targets; no dataset-specific registry."""

import dataclasses
import functools
import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer


@functools.lru_cache(maxsize=8)
def tokenizer_for(length):
    return PaligemmaTokenizer(max_len=length)


def stage_tokens(tokenizer, texts, limit):
    rows = []
    for text in texts:
        ids = tokenizer._tokenizer.encode(" " + text + "\nAction: ")
        if not ids or len(ids) > limit:
            raise ValueError(
                "Stage text exceeds stage.max_tokens; truncation is prohibited"
            )
        rows.append(tuple(ids + [0] * (limit - len(ids))))
    return tuple(rows)


@dataclasses.dataclass(frozen=True)
class Labels:
    task: str
    pa_to_stage: tuple[tuple[str, int], ...]
    auxiliary: bool
    stages: bool
    max_tokens: int = 32

    def __call__(self, data):
        result = dict(data)
        prompt = result["prompt"]
        mapping = dict(self.pa_to_stage)
        if "actions" not in data:
            raise ValueError("Training records require action targets")
        if prompt not in mapping:
            raise ValueError(
                f"No reviewed PA label for training instruction: {prompt!r}"
            )
        result["prompt"] = self.task
        if self.stages:
            result["stage_label"] = np.asarray(mapping[prompt], np.int32)
        if self.auxiliary:
            sp = tokenizer_for(self.max_tokens)._tokenizer
            ids = [*sp.encode(prompt), sp.eos_id()]
            if len(ids) > self.max_tokens:
                raise ValueError(
                    "PA exceeds auxiliary.max_tokens; truncation is prohibited"
                )
            result["pa_input_tokens"] = np.asarray(
                [sp.bos_id(), *ids[:-1]] + [0] * (self.max_tokens - len(ids)), np.int32
            )
            result["pa_target_tokens"] = np.asarray(
                ids + [0] * (self.max_tokens - len(ids)), np.int32
            )
            result["pa_loss_mask"] = np.arange(self.max_tokens) < len(ids)
        return result


@dataclasses.dataclass(frozen=True)
class StagePrefix:
    tokenizer: object

    def __call__(self, data):
        result = dict(data)
        text = result.pop("prompt")
        state = np.digitize(result["state"], np.linspace(-1, 1, 257)[:-1]) - 1
        prefix = (
            f"Task: {text}\nState: " + " ".join(map(str, state)) + "\nCurrent subtask:"
        )
        ids = self.tokenizer._tokenizer.encode(prefix, add_bos=True)
        n = self.tokenizer._max_len
        if len(ids) > n:
            raise ValueError(
                "Stage query exceeds model.max_token_len; truncation is prohibited"
            )
        result.update(
            tokenized_prompt=np.asarray(ids + [0] * (n - len(ids)), np.int32),
            tokenized_prompt_mask=np.arange(n) < len(ids),
        )
        return result
