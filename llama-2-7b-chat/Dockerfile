FROM nvcr.io/nvidia/pytorch:23.06-py3
WORKDIR /workspace

RUN git clone https://github.com/facebookresearch/llama

WORKDIR /workspace/llama

RUN pip install -e .

ADD llama-2-7b-chat /workspace/llama/llama-2-7b-chat
COPY tokenizer.model /workspace/llama/tokenizer.model
COPY example_chat_completion.py /workspace/llama/example_chat_completion.py