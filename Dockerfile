FROM nvcr.io/nvidia/pytorch:24.07-py3
# FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
COPY . /easy_ViTPose
WORKDIR /easy_ViTPose/
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0   \
        libsm6         \
        libxext6       \
        libxrender1    \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

# RUN pip uninstall -y $(pip list --format=freeze | grep opencv) && \
#     rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
RUN pip install -e . && pip install -r requirements.txt && pip install -r requirements_gpu.txt

# OpenCV dependency
RUN apt-get update && apt-get install -y libgl1
