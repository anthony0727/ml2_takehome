FROM ubuntu:18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        g++ \
        make \
        cmake \
        wget \
        unzip \
        vim \
        git \
        python3 \
        python3-pip

COPY app /home/app
WORKDIR /home/app
RUN pip3 install -r requirements.txt
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
RUN unzip libtorch-shared-with-deps-latest.zip -d /usr/local
RUN rm libtorch-shared-with-deps-latest.zip
RUN cd /home/app/lenet5_libtorch && cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch .. && make
