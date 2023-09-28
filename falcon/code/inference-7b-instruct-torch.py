import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

def main():
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK")) 
    torch.cuda.set_device(local_rank)

    model_name = "tiiuae/falcon-40b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        #offload_to_cpu=True,
        # offload_folder="offload",
        # offload_state_dict = True
        # load_in_8bit=True,
    )

    # Wrap model for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Ensure everything is on the correct device
    device = torch.device("cuda", local_rank)
    model.to(device)


    print("hit")
    # input_text = "Girafatron is obsessed with giraffes..."
    
    # # Tokenize and move to the GPU
    # input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # # Generate sequences
    # sequences = model.module.generate(
    #     input_ids=input_ids,
    #     max_length=200,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    # )


    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # device_map="local_rank",
    )

    sequences = gen_pipeline(
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

if __name__ == '__main__':
    main()
