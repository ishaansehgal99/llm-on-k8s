from flask import Flask, request, jsonify
from llama import Llama
import torch
import sys
import signal
import os
import torch.distributed as dist
import argparse
import time

# Setup argparse
parser = argparse.ArgumentParser(description="Llama API server.")
parser.add_argument("--ckpt_dir", default="weights/", help="Checkpoint directory.")
parser.add_argument("--tokenizer_path", default="tokenizer.model", help="Path to the tokenizer model.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length.")
parser.add_argument("--max_batch_size", type=int, default=4, help="Maximum batch size.")
parser.add_argument("--model_parallel_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)), help="Model parallel size.")
args = parser.parse_args()

should_shutdown = False

app = Flask(__name__)

def build_generator(params):
    """Build Llama generator from provided parameters."""
    return Llama.build(**params)

def broadcast_for_reconfig(new_params):
    """Broadcasts new configurations to worker processes."""
    dist.broadcast_object_list(["configure", new_params, None], src=0)

def broadcast_for_generation(input_string, max_gen_len, temperature, top_p):
    """Broadcasts generation parameters to worker processes."""
    dist.broadcast_object_list(["generate", input_string, {
        'max_gen_len': max_gen_len,
        'temperature': temperature,
        'top_p': top_p
    }], src=0)

def shutdown_server():
    """Shut down the server."""
    time.sleep(2)  # Delay for 2 seconds to ensure the response is sent
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

@app.route('/')
def home():
    return "Server is running", 200

@app.route('/healthz')
def health_check():
    # Check if a GPU is available
    if not torch.cuda.is_available():
        return "No GPU available", 500
    # Check Llama model initialization
    if not generator:
        return "Llama model not initialized", 500
    return "Healthy", 200

@app.teardown_request
def check_shutdown(exception=None):
    global should_shutdown
    if should_shutdown:
        shutdown_server()

@app.route('/configure', methods=['POST'])
def configure_generator():
    global generator
    global gen_params

    new_params = { key: request.json.get(key, value) for key, value in gen_params.items() }

    # Broadcasting the configuration changes
    broadcast_for_reconfig(new_params)

    # Reconfiguring the generator on master
    try:
        generator = build_generator(new_params)
        gen_params = new_params
    except Exception as e:
        print("Failed to reconfigure model: " + \
            str(e) + " without valid model, shutting down\n") 
        # Broadcast shutdown command to worker processes
        dist.broadcast_object_list(["shutdown", None, None], src=0)
        # Mark Flask server for shutdown
        global should_shutdown
        should_shutdown = True
        return jsonify(error="Failed to reconfigure model"), 400
    
    return jsonify(status="success"), 200

@app.route('/chat', methods=['POST'])
def chat_completion():
    data = request.json
    input_data = data.get('input_data')
    if not input_data:
        return jsonify(error="Input data is required"), 400
    
    input_string = input_data.get("input_string")
    if not input_string:
        return jsonify(error="Input string is required"), 400

    parameters = data.get("parameters", {})
    max_gen_len = parameters.get('max_gen_len', 64)
    temperature = parameters.get('temperature', 0.6)
    top_p = parameters.get('top_p', 0.9)

    # Broadcast generation params to worker processes
    broadcast_for_generation(input_string, max_gen_len, temperature, top_p)

    # Master's own generation
    try:
        results = generator.chat_completion(
            [input_string],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as e:
        return jsonify(error="Request Failed: " + str(e)), 400

    if len(results) == 0:
        return jsonify(error="No results"), 404

    result = results[0]
    return jsonify(
        role=result['generation']['role'].capitalize(),
        content=result['generation']['content']
    )

if __name__ == "__main__":
    if dist.get_rank() == 0:
        app.run(host='0.0.0.0', port=5000)
    else:
        # Note to enable logs to std out uncomment 
        sys.stdout = sys.__stdout__
        while True:
            worker_num = dist.get_rank()
            print(f"Worker {worker_num} ready to recieve next command")
            config = [None] * 3  # Command and its associated data
            dist.broadcast_object_list(config, src=0)
            command = config[0]

            if command == "generate":
                try:
                    input_string = config[1]
                    parameters = config[2]
                    generator.chat_completion(
                        [input_string],
                        max_gen_len=parameters.get('max_gen_len', 64),
                        temperature=parameters.get('temperature', 0.6),
                        top_p=parameters.get('top_p', 0.9)
                    )
                    print(f"Worker {worker_num} completed generation")              
                except Exception as e:
                    print(f"Error in generation: {str(e)}")
                    
            elif command == "configure":
                try:
                    new_params = config[1]
                    generator = build_generator(new_params)
                    print(f"Worker {worker_num} completed reconfigure")              
                except Exception as e:
                    print(f"Error in reconfiguring generator: {str(e)}")
            elif command == "shutdown":
                sys.exit(0)
