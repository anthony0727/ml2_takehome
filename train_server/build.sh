mkdir -p app/lenet5_libtorch
cd app/lenet5_libtorch
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
make
