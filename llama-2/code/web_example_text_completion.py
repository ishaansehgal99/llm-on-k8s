from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel

from llama import Llama
import torch
import sys
import signal
import os
import torch.distributed as dist
import argparse

# Setup argparse
parser = argparse.ArgumentParser(description="Llama API server.")
parser.add_argument("--ckpt_dir", default="weights/", help="Checkpoint directory.")
parser.add_argument("--tokenizer_path", default="tokenizer.model", help="Path to the tokenizer model.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length.")
parser.add_argument("--max_batch_size", type=int, default=4, help="Maximum batch size.")
parser.add_argument("--model_parallel_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)), help="Model parallel size.")
args = parser.parse_args()

should_shutdown = False

app = FastAPI()

def build_generator(params):
    """Build Llama generator from provided parameters."""
    return Llama.build(**params)

def broadcast_for_shutdown():
    """Broadcasts shutdown command to worker processes."""
    dist.broadcast_object_list(["shutdown", None, None], src=0)

def broadcast_for_text_generation(prompts, max_gen_len, temperature, top_p):
    """Broadcasts generation parameters to worker processes."""
    dist.broadcast_object_list(["text_generate", prompts, {
        'max_gen_len': max_gen_len,
        'temperature': temperature,
        'top_p': top_p
    }], src=0)

def shutdown_server():
    """Shut down the server."""
    os.kill(os.getpid(), signal.SIGINT)

# Default values for the generator
gen_params = {
    'ckpt_dir': args.ckpt_dir,
    'tokenizer_path': args.tokenizer_path,
    'max_seq_len': args.max_seq_len,
    'max_batch_size': args.max_batch_size,
    'model_parallel_size': args.model_parallel_size,
}

generator = build_generator(gen_params)

@app.get('/')
async def home():
    return "Server is running", 200

@app.get("/healthz")
def health_check():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="No GPU available")
    if not generator:
        raise HTTPException(status_code=500, detail="Llama model not initialized")
    return {"status": "Healthy"}

# @app.teardown_request
# def check_shutdown(exception=None):
#     global should_shutdown
#     if should_shutdown:
#         print("Server shutting down...")
#         shutdown_server()

@app.post("/shutdown")
def shutdown():
    if dist.get_world_size() > 1:
        broadcast_for_shutdown()
    shutdown_server()
    return {}

class GenerationParameters(BaseModel):
    prompts: list
    parameters: dict = None

@app.post("/generate")
def generate_text(params: GenerationParameters):
    prompts = params.prompts
    # Check if the prompts are provided
    if not prompts or not isinstance(prompts, list):
        raise HTTPException(status_code=400, detail="Prompts are required and should be an array")

    parameters = params.parameters if params.parameters else {}
    max_gen_len = parameters.get('max_gen_len', None)
    temperature = parameters.get('temperature', 0.6)
    top_p = parameters.get('top_p', 0.9)

    if dist.get_world_size() > 1:
        broadcast_for_text_generation(prompts, max_gen_len, temperature, top_p)

    try: 
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Request Failed: " + str(e))

    if len(results) == 0:
        raise HTTPException(status_code=404, detail="No results")

    response_data = []
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
        entry = {
            "prompt": prompt,
            "response": result['generation']
        }
        response_data.append(entry)

    return {"results": response_data}

if __name__ == "__main__":
    if dist.get_rank() == 0:
        app.run(host='0.0.0.0', port=5000)
    else:
        while True:
            worker_num = dist.get_rank()
            print(f"Worker {worker_num} ready to receive next command")
            config = [None] * 3
            dist.broadcast_object_list(config, src=0)
            command = config[0]

            if command == "text_generate":
                try:
                    prompts = config[1]
                    parameters = config[2]
                    generator.text_completion(
                        prompts,
                        max_gen_len=parameters.get('max_gen_len', None),
                        temperature=parameters.get('temperature', 0.6),
                        top_p=parameters.get('top_p', 0.9)
                    )
                    print(f"Worker {worker_num} completed generation")              
                except Exception as e:
                    print(f"Error in generation: {str(e)}")
            elif command == "shutdown":
                print(f"Worker {worker_num} shutting down")
                sys.exit(0)
