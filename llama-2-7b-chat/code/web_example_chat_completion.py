from flask import Flask, request, jsonify
from llama import Llama
import os

app = Flask(__name__)

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

@app.route('/')
def health_check():
    return "Server is running", 200

@app.route('/configure', methods=['POST'])
def configure_generator():
    global generator
    global gen_params

    new_params = {}
    for key, value in gen_params.items():
        new_params[key] = request.json.get(key, value)

    try:
        generator = Llama.build(
            ckpt_dir=new_params['ckpt_dir'],
            tokenizer_path=new_params['tokenizer_path'],
            max_seq_len=new_params['max_seq_len'],
            max_batch_size=new_params['max_batch_size'],
            # Reinitializing generator requires this additional param
            model_parallel_size=new_params['model_parallel_size']
        )
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
