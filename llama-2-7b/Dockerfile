FROM nvcr.io/nvidia/pytorch:23.06-py3
WORKDIR /workspace

RUN git clone https://github.com/facebookresearch/llama

WORKDIR /workspace/llama

RUN pip install -e .
RUN pip install flask

ADD code /workspace/llama/llama-2-7b
COPY code/tokenizer.model /workspace/llama/tokenizer.model
COPY code/example_text_completion.py /workspace/llama/example_text_completion.py
COPY code/web_example_text_completion.py /workspace/llama/web_example_text_completion.py

# ADD code/weights /workspace/llama/llama-2-7b/weights
# COPY code/weights/consolidated.00.pth /workspace/llama/llama-2-7b/weights/consolidated.00.pth
# COPY code/weights/params.json /workspace/llama/llama-2-7b/weights/params.json
# COPY code/weights/checklist.chk /workspace/llama/llama-2-7b/weights/checklist.chk

CMD ["torchrun", "/workspace/llama/web_example_text_completion.py"]
