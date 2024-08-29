FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && apt-get install python3-tk -y \
  && pip3 install --upgrade pip\
  && pip install jupyter \
  && pip install scipy \
  && pip install matplotlib \
  && pip install tqdm \
  && pip install -U scikit-learn \
  && pip install tensorflow_probability==0.8 \
  && pip install hdf5storage \
  && pip install h5py


RUN apt-get install -y git

WORKDIR /workspace

RUN git clone https://github.com/DaitTan/DNN-MFBO.git
# Default command
CMD ["/bin/bash"]
