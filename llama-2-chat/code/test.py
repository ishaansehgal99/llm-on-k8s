from flask import Flask, request, jsonify
from llama import Llama
import torch
import os
import torch.distributed as dist
import sys
import logging
if __name__ == '__main__':
    # Example data and settings
    print("HELLO")
    ckpt_dir = "weights"
    tokenizer_path = "tokenizer.model"
    max_seq_len = 512
    max_batch_size = 2
    model_parallel_size = 2  # for 2 GPUs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print("LR", local_rank)

    llama = Llama.build(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, model_parallel_size)
    
    dialogs = [
        [{"role":"user","content":"Hello, how are you?"}]
    ]

        
    # predictions = llama.chat_completion(dialogs)
    sys.stdout = sys.__stdout__
    # print(predictions)

    # exit(0)
    if dist.get_rank() == 0:
        dist.broadcast_object_list(dialogs, src=0)
        predictions = llama.chat_completion(dialogs)
        print("1", predictions)
    else: 
        # while True:
        sys.stdout = sys.__stdout__
        print(f"Entered loop")
        config = [None] * 1  # Command and its associated data
        dist.broadcast_object_list(config, src=0)
        # print(f"broadcasted")
        # dialogs = config[0]
        # print("1", dialogs)
        predictions = llama.chat_completion(dialogs)
        print("2", predictions)
 