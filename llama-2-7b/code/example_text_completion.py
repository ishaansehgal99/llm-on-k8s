# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import sys

from llama import Llama

def main(
    ckpt_dir: str = 'llama-2-7b-weights/',
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompt = None
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    while True:
        if not prompt:
            prompt = input("Prompt: ")
        if prompt in ['quit', 'q', 'exit']:
            sys.exit()
        results = generator.text_completion(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        prompt = None
        if len(results) == 0:
            print("No results")
            continue
        print(f"> {results[0]['generation']}")


if __name__ == "__main__":
    fire.Fire(main)
