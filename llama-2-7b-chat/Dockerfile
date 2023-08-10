FROM nvcr.io/nvidia/pytorch:23.06-py3
WORKDIR /workspace

RUN git clone https://github.com/facebookresearch/llama

WORKDIR /workspace/llama

RUN pip install -e .

ADD code /workspace/llama/llama-2-7b-chat
COPY code/tokenizer.model /workspace/llama/tokenizer.model
COPY code/example_chat_completion.py /workspace/llama/example_chat_completion.py

ADD code/weights /workspace/llama/llama-2-7b-chat/weights
COPY code/weights/consolidated.00.pth /workspace/llama/llama-2-7b-chat/weights/consolidated.00.pth
COPY code/weights/params.json /workspace/llama/llama-2-7b-chat/weights/params.json
COPY code/weights/checklist.chk /workspace/llama/llama-2-7b-chat/weights/checklist.chk
