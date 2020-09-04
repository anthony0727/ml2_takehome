#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Net: torch::nn::Module {
public:
  Net()
      : c1(torch::nn::Conv2dOptions(1, 6, /*kernel_size=*/{5, 5}).padding({2, 2})),
        c3(torch::nn::Conv2dOptions(6, 16, /*kernel_size=*/{5, 5})),
        c5(torch::nn::Conv2dOptions(16, 120, /*kernel_size=*/{5, 5})),
        f6(120, 84),
        output(84, 10) {
    register_module("c1", c1);
    register_module("c3", c3);
    register_module("c5", c5);
    register_module("f6", f6);
    register_module("output", output);
  }

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv2d c1{nullptr};
  torch::nn::Conv2d c3{nullptr};
  torch::nn::Conv2d c5{nullptr};
  torch::nn::Linear f6{nullptr};
  torch::nn::Linear output{nullptr};
};
// TORCH_MODULE(Net);