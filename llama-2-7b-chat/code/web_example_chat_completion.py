from flask import Flask, request, jsonify
from llama import Llama
from typing import Optional

app = Flask(__name__)

# Default values for the generator
default_values = {
    'ckpt_dir': 'weights/',
    'tokenizer_path': 'tokenizer.model',
    'max_seq_len': 512,
    'max_batch_size': 4
}

generator = Llama.build(
    ckpt_dir=default_values['ckpt_dir'],
    tokenizer_path=default_values['tokenizer_path'],
    max_seq_len=default_values['max_seq_len'],
    max_batch_size=default_values['max_batch_size']
)

@app.route('/chat', methods=['GET'])
def chat_completion():
    role = request.args.get('role')
    prompt = request.args.get('prompt')

    # Check if the role and prompt are provided
    if not role or not prompt:
        return jsonify(error="Role and Prompt are required"), 400

    temperature = float(request.args.get('temperature', 0.6))
    top_p = float(request.args.get('top_p', 0.9))
    max_gen_len = request.args.get('max_gen_len')
    max_gen_len = int(max_gen_len) if max_gen_len else None

    results = generator.chat_completion(
        [[{"role": role, "content": prompt}]], # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    if len(results) == 0:
        return jsonify(error="No results"), 404

    result = results[0]
    return jsonify(
        role=result['generation']['role'].capitalize(),
        content=result['generation']['content']
    )

@app.route('/configure', methods=['POST'])
def configure_generator():
    global generator

    ckpt_dir = request.json.get('ckpt_dir', default_values['ckpt_dir'])
    tokenizer_path = request.json.get('tokenizer_path', default_values['tokenizer_path'])
    max_seq_len = int(request.json.get('max_seq_len', default_values['max_seq_len']))
    max_batch_size = int(request.json.get('max_batch_size', default_values['max_batch_size']))

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    return jsonify(status="success"), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
