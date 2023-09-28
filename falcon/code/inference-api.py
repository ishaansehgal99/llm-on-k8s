# !pip install -U bitsandbytes
# !pip install -U git+https://github.com/huggingface/transformers.git
# !pip install -U git+https://github.com/huggingface/accelerate.git
# !pip install fastapi pydantic
# !pip install 'uvicorn[standard]'

# System
import os

# API
from fastapi import FastAPI
import uvicorn

# ML
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import torch.distributed as dist

app = FastAPI()
model_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # offload_folder="offload",
    # offload_state_dict = True
    # load_in_8bit=True,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# def worker_listen_tasks(): 
#     while True:
#         worker_num = dist.get_rank()
#         print(f"Worker {worker_num} ready to recieve next command")
#         config = [None] * 3  # Command and its associated data
#         dist.broadcast_object_list(config, src=0)
#         command = config[0]

#         if command == "generate":
#             try:
#                 input_string = config[1]
#                 parameters = config[2]
#                 generator.chat_completion(
#                     input_string,
#                     max_gen_len=parameters.get('max_gen_len', None),
#                     temperature=parameters.get('temperature', 0.6),
#                     top_p=parameters.get('top_p', 0.9)
#                 )
#                 print(f"Worker {worker_num} completed generation")              
#             except Exception as e:
#                 print(f"Error in generation: {str(e)}")
#         elif command == "shutdown":
#             print(f"Worker {worker_num} shutting down")
#             sys.exit(0)

@app.get("/generate/")
def generate_text(prompt: str):
    sequences = pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    result = ""
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        result += seq['generated_text']

    return {"Result": result}


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK"))
    if dist.get_rank() == 0:
        uvicorn.run(app=app, host='0.0.0.0', port=5000)
    # else: 
    #     worker_listen_tasks()
