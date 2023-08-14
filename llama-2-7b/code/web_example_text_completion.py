from flask import Flask, request, jsonify
from llama import Llama
import os

app = Flask(__name__)

class SlidingWindow:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.token_history = []

    def append(self, max_gen_len, new_prompt_tokens):
        available_tokens = self.max_seq_len - len(new_prompt_tokens)
        if available_tokens < 0:
            print("User input exceeds the maximum token length")

        # Account for tokens required for model output
        available_tokens -= max_gen_len

        if available_tokens > len(self.token_history):
            self.token_history.extend(new_prompt_tokens)
            return self.token_history

        # If the total tokens exceed the allowed limit, remove the earliest tokens
        global generator
        self.token_history = self.token_history[-available_tokens:] + new_prompt_tokens
        assert len(self.token_history) == self.max_seq_len - max_gen_len
        return self.token_history

# Default values for the generator
gen_params = {
    'ckpt_dir': 'weights/',
    'tokenizer_path': 'tokenizer.model',
    'max_seq_len': 128,
    'max_batch_size': 4,
    'model_parallel_size': int(os.environ.get("WORLD_SIZE", 1)),
}

generator = Llama.build(
    ckpt_dir=gen_params['ckpt_dir'],
    tokenizer_path=gen_params['tokenizer_path'],
    max_seq_len=gen_params['max_seq_len'],
    max_batch_size=gen_params['max_batch_size'],
)

window = SlidingWindow(max_seq_len=gen_params['max_seq_len'])

@app.route('/configure', methods=['POST'])
def configure_generator():
    global generator

    for key, value in gen_params.items():
        gen_params[key] = request.json.get(key, value)

    generator = Llama.build(
        ckpt_dir=gen_params['ckpt_dir'],
        tokenizer_path=gen_params['tokenizer_path'],
        max_seq_len=gen_params['max_seq_len'],
        max_batch_size=gen_params['max_batch_size'],
        # Reinitializing generator requires this additional param
        model_parallel_size=gen_params['model_parallel_size']
    )

    window.max_seq_len = gen_params['max_seq_len']
    return jsonify(status="success"), 200

@app.route('/generate', methods=['GET'])
def generate_text():
    prompt = request.args.get('prompt')
    # Check if the prompt is provided
    if not prompt:
        return jsonify(error="Prompt is required"), 400

    temperature = float(request.args.get('temperature', 0.6))
    top_p = float(request.args.get('top_p', 0.9))
    max_gen_len = int(request.args.get('max_gen_len', 64))
    context = request.args.get('context')
    if context and context.strip().lower() == "true":
        # Tokenize the prompt using the tokenizer
        tokenizer = generator.tokenizer
        new_prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False)

        # Append new prompt to sliding window
        result_window = window.append(max_gen_len, new_prompt_tokens)

        # Decode resulting window
        prompt = tokenizer.decode(result_window)
        # print(prompt, "prompt with context")

    results = generator.text_completion(
        [prompt], # Note when we pass context its in the same prompt
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    if len(results) == 0:
        return jsonify(error="No results"), 404

    return jsonify(response=results[0]['generation'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
