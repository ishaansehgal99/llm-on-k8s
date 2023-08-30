from flask import Flask, request, jsonify
from llama import Llama
import torch
import os
import gc

app = Flask(__name__)

class LlamaSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LlamaSingleton, cls).__new__(cls)
            cls._instance.initialize_generator(**kwargs)
        return cls._instance

    def initialize_generator(self, **kwargs):
        with torch.no_grad():
            self.generator = Llama.build(
                ckpt_dir=kwargs.get('ckpt_dir', 'weights/'),
                tokenizer_path=kwargs.get('tokenizer_path', 'tokenizer.model'),
                max_seq_len=kwargs.get('max_seq_len', 128),
                max_batch_size=kwargs.get('max_batch_size', 4),
                model_parallel_size=kwargs.get('model_parallel_size', 1)
            )

    def reconfigure(self, **kwargs):
        del self.generator
        torch.cuda.empty_cache()  # Clear CUDA cache
        gc.collect()  # Explicit garbage collection
        self.initialize_generator(**kwargs)


# Default values for the generator
gen_params = {
    'ckpt_dir': 'weights/',
    'tokenizer_path': 'tokenizer.model',
    'max_seq_len': 128,
    'max_batch_size': 4,
    'model_parallel_size': int(os.environ.get("WORLD_SIZE", 1)),
}

generator_instance = LlamaSingleton(**gen_params)
generator = generator_instance.generator

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

@app.route('/configure', methods=['POST'])
def configure_generator():
    global generator
    global gen_params

    new_params = {}
    for key, value in gen_params.items():
        new_params[key] = request.json.get(key, value)

    try:
        generator_instance.reconfigure(**new_params)
        generator = generator_instance.generator
        gen_params = new_params
    except Exception as e:
        return jsonify(error="Failed invalid parameters: " + str(e)), 400

    return jsonify(status="success"), 200

@app.route('/chat', methods=['POST'])
def chat_completion():
    data = request.json
    input_data = data.get('input_data')
    if not input_data:
        return jsonify(error="Input data is required"), 400

    input_string = input_data.get("input_string")
    if not input_data:
        return jsonify(error="Input string is required"), 400

    parameters = data.get("parameters", {})
    temperature = parameters.get('temperature', 0.6)
    top_p = parameters.get('top_p', 0.9)
    max_gen_len = parameters.get('max_gen_len', 64)

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
    app.run(host='0.0.0.0', port=5000)
