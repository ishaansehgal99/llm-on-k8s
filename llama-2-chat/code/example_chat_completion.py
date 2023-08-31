# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import sys

from llama import Llama

def main(
    ckpt_dir: str = 'weights/',
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    role = None
    prompt = None
    if len(sys.argv) > 2:
        role = sys.argv[1]
        prompt = sys.argv[2]

    while True:
        if not role:
            role = input("Role: ")
        if not prompt:
            prompt = input("Prompt: ")
        if prompt in ['quit', 'q', 'exit']:
            sys.exit()

        results = generator.chat_completion(
            [[{"role": role, "content": prompt}]], # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        role = None
        prompt = None
        if len(results) == 0:
            print("No results")
            continue
        result = results[0]
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )


if __name__ == "__main__":
    fire.Fire(main)
