FROM nvcr.io/nvidia/pytorch:23.06-py3
WORKDIR /workspace

RUN git clone https://github.com/facebookresearch/llama

WORKDIR /workspace/llama

RUN pip install -e .
RUN pip install flask

COPY ../llama-readiness-check.sh /workspace/
RUN chmod +x /workspace/llama-readiness-check.sh

ADD code /workspace/llama/llama-2-chat
