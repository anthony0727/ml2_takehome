#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

struct Net : torch::nn::Module {
  Net()
      : c1(torch::nn::Conv2dOptions(1, 6, /*kernel_size=*/{5, 5})
                  .padding({2, 2})),
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

  torch::Tensor forward(torch::Tensor x) {
    x = c1->forward(x); // 6@28x28
    x = torch::max_pool2d(x, {2, 2}, {2, 2}); // 6@14x14
    x = c3->forward(x); // 16@10x10
    x = torch::max_pool2d(x, {2, 2}, {2, 2}); // 16@10x10
    x = c5->forward(x); // 120@1x1
    x = x.view({x.size(0), -1});
    x = f6->forward(x); // 120->84
    x = output->forward(x); // 84->10
    x = torch::log_softmax(x, /*dim=*/1);

    return x;
  }

  torch::nn::Conv2d c1;
  torch::nn::Conv2d c3;
  torch::nn::Conv2d c5;
  torch::nn::Linear f6;
  torch::nn::Linear output;
};
