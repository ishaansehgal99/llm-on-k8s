# !pip install -U bitsandbytes
# !pip install -U git+https://github.com/huggingface/transformers.git
# !pip install -U git+https://github.com/huggingface/accelerate.git
# !pip install fastapi pydantic
# !pip install 'uvicorn[standard]'

# System
import os

# ML
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import torch.distributed as dist

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


input_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"

sequences = pipeline(
    input_text,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)


for seq in sequences:
    print(f"Result: {seq['generated_text']}")
