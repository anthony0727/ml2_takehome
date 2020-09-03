FROM ubuntu:18.04

# RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        g++ \
        make \
        cmake \
        wget \
        unzip \
        vim
        # python3 \
        # python3-pip

# Download and build libtorch with MKL support

ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN git clone --recurse-submodules -j8 https://github.com/pytorch/pytorch.git
RUN cd pytorch && mkdir build && cd build && BUILD_TEST=OFF USE_NCCL=OFF python3 ../tools/build_libtorch.py

# Prepare built package

RUN cd && mkdir -p libtorch/include && mkdir -p libtorch/share
RUN cp -r pytorch/build/lib libtorch
RUN cp -r pytorch/torch/share/cmake libtorch/share/cmake
RUN for dir in ATen c10 caffe2 torch; do cp -r pytorch/torch/include/$dir libtorch/include; done