# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
#for svamp---
from datasets import Dataset
from core import CoT, PoT, PhP
#---
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    svamp = Dataset(Path("data/SVAMP.json"))
   
    CoT_correct = 0
    PoT_correct = 0
    PHP_correct = 0

    for i, problem in enumerate(svamp):
        cot_output = CoT(generator=generator, problem=problem)
        pot_output = PoT(generator=generator, problem=problem)
        #php_output = PhP(generator=generator, problem=problem)

        if cot_output == problem.answer:
            CoT_correct += 1
        if pot_output == problem.answer:
            PoT_correct += 1
        #if php_output == problem.answer:
        #    PHP_correct += 1

        #print(f"current corrects PoT: {PoT_correct}, CoT: {CoT_correct}, PHP: {PHP_correct}")
        print(f"{i}th current corrects PoT: {PoT_correct}, CoT: {CoT_correct}")

    # Save result json
    with open(Path(f"result_{MODEL_SIZE}.json"), 'w') as f:
        json.dump({
            "CoT": CoT_correct,
            "PoT": PoT_correct,
            #"PhP": PHP_correct,
        }, f)

if __name__ == "__main__":
    fire.Fire(main)
