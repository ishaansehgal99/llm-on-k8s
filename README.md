# llm-on-k8s
Containerized LLM on Kubernetes

# Introduction
This guide provides insights on deploying two different APIs associated with Llama 2: one for text generation based on given prompts (llama-2-7b) and another tailored for chat-based applications (llama-2-7b-chat)

# Build and Deployment Process
1. Choose the Desired Model Directory: Navigate to either the llama-2-7b or llama-2-7b-chat directory, based on the desired model.
2. Build the Docker Image: ```docker build -t your-image-name:your-tag .```
3. Deploy the Image to a Container: ```docker run --name your-container-name your-image-name:your-tag```


# API Documentation

## Llama-2-7b Text Completion 
1. Server Health Check <br>
Endpoint: ```/``` <br>
Method: GET <br>
Purpose: Check if the server is running. <br>
Example: ```curl http://localhost:5000/```

2. Model Health Check <br>
Endpoint: ```/healthz``` <br>
Method: GET <br>
Purpose: Check if the model and GPU are properly initialized. <br>
Example: ```curl http://localhost:5000/healthz```

3. Configure Generator <br>
Endpoint: ```/configure``` <br>
Method: POST <br>
Purpose: Configure the generator's parameters. <br>
Example: ```curl -X POST -H "Content-Type: application/json" -d '{"ckpt_dir": "new_weights/", "max_seq_len": 150}' http://localhost:5000/configure```

4. Complete Text <br>
Endpoint: ```/generate``` <br>
Method: GET <br>
Purpose: Complete text based on a given prompt. <br>
Example: ```curl "http://localhost:5000/generate?prompt=Once%20upon%20a%20time&temperature=0.7&top_p=0.8&max_gen_len=100"```

## Llama-2-7b-chat Interaction
**Note:** Apart from the distinct chat interaction endpoint described below, all other endpoints (Server Health Check, Model Health Check, and Configure Generator) for Llama-2-7b-chat are identical to those in Llama-2-7b.

Chat Interaction <br>
Endpoint: ```/chat``` <br>
Method: POST <br>
Purpose: Facilitates chat-based text interactions. <br>
Example:
```curl -X POST -H "Content-Type: application/json" -d '{"input_data": {"input_string": [{"role":"user","content":"Hello, how are you?"}]}, "parameters": {"temperature": 0.7, "top_p"
: 0.8, "max_gen_len": 100}}' http://localhost:5000/chat```

# Conclusion
These APIs provide a streamlined approach to harness the capabilities of the Llama 2 model for both text generation and chat-oriented applications. Ensure the correct deployment and configuration for optimal utilization.



