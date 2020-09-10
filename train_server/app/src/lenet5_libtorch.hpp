#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace torch;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Lenet5Impl : nn::Module
{
public:
  Lenet5Impl()
      : c1(nn::Conv2dOptions(1, 6, /*kernel_size=*/{5, 5}).padding({2, 2})),
        c3(nn::Conv2dOptions(6, 16, /*kernel_size=*/{5, 5})),
        c5(nn::Conv2dOptions(16, 120, /*kernel_size=*/{5, 5})),
        f6(120, 84),
        output(84, 10)
  {
    register_module("c1", c1);
    register_module("c3", c3);
    register_module("c5", c5);
    register_module("f6", f6);
    register_module("output", output);
  }

  Tensor forward(Tensor x)
  {
    x = max_pool2d(relu(c1(x)), {2, 2}, {2, 2});
    x = max_pool2d(relu(c3(x)), {2, 2}, {2, 2});
    x = relu(c5(x));
    x = x.view({x.size(0), -1});
    x = relu(f6(x));
    x = log_softmax(output(x), /*dim=*/1);

    return x;
  }

  nn::Conv2d c1, c3, c5;
  nn::Linear f6, output;
};
TORCH_MODULE(Lenet5);
