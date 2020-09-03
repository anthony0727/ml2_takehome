if [ ! -d "/usr/local/libtorch" ]
then
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    sudo unzip libtorch-shared-with-deps-latest.zip -d /usr/local
    rm libtorch-shared-with-deps-latest.zip
fi
mkdir -p lenet5_libtorch
cd lenet5_libtorch
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
make
