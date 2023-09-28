from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch

# Initialize the accelerator
accelerator = Accelerator()
print(accelerator.device)

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Accelerator handles model placement on the device
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model, tokenizer = accelerator.prepare(model, tokenizer)

input_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"

# Tokenize the input_text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Move input_ids to GPU
input_ids = input_ids.to(accelerator.device)

# Generate sequences
sequences = model.generate(
    input_ids=input_ids,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode and print the generated text
for seq in sequences:
    print(f"Result: {tokenizer.decode(seq, skip_special_tokens=True)}")