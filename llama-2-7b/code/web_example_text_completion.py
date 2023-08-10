from flask import Flask, request, jsonify, Response
from llama import Llama

app = Flask(__name__)

generator = Llama.build(
    ckpt_dir='weights/',
    tokenizer_path='tokenizer.model',
    max_seq_len=128,
    max_batch_size=4,
)

@app.route('/generate', methods=['GET'])
def generate_text():
    prompt = request.args.get('prompt')

    # Check if the prompt is 'quit'
    if prompt and prompt.lower() in ['quit', 'q', 'exit']:
        shutdown_server()
        return Response("Server shutting down..."), 200

    # Check if the prompt is provided
    if not prompt:
        return jsonify(error="Prompt is required"), 400

    temperature = float(request.args.get('temperature', 0.6))
    top_p = float(request.args.get('top_p', 0.9))
    max_gen_len = int(request.args.get('max_gen_len', 64))

    results = generator.text_completion(
        [prompt],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    if len(results) == 0:
        return jsonify(error="No results"), 404

    return jsonify(generation=results[0]['generation'])

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
