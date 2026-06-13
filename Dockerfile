# syntax=docker/dockerfile:1.5
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	wget \
	ca-certificates \
	bzip2 \
	build-essential \
	python3-dev \
	&& rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
	&& bash /tmp/miniconda.sh -b -p $CONDA_DIR \
	&& rm /tmp/miniconda.sh \
	&& $CONDA_DIR/bin/conda install -y python=3.10 pip \
	&& $CONDA_DIR/bin/conda clean -ya

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /workspace

ENV PIP_CACHE_DIR=/root/.cache/pip

COPY requirements.txt /workspace/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
	pip install -r /workspace/requirements.txt

COPY . /workspace

ENV HF_HOME=/cache/hf \
	HF_DATASETS_CACHE=/cache/hf/datasets \
	TOKENIZERS_PARALLELISM=false \
	PYTHONUNBUFFERED=1

CMD ["bash"]