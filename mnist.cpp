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
      : conv1(torch::nn::Conv2dOptions(1, 6, /*kernel_size=*/{5, 5})
                  .padding({2, 2})),
        conv2(torch::nn::Conv2dOptions(6, 16, /*kernel_size=*/{5, 5})),
        conv3(torch::nn::Conv2dOptions(16, 120, /*kernel_size=*/{5, 5})),
        fc1(120, 84),
        fc2(84, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv1->forward(x); // 6@28x28
    x = torch::max_pool2d(x, {2, 2}, {2, 2}); // 6@14x14
    x = conv2->forward(x); // 16@10x10
    x = torch::max_pool2d(x, {2, 2}, {2, 2}); // 16@10x10
    x = conv3->forward(x); // 120@1x1
    x = x.view({x.size(0), -1});
    x = fc1->forward(x); // 120->84
    x = fc2->forward(x); // 84->10
    x = torch::log_softmax(x, /*dim=*/1);

    return x;
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Conv2d conv3;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

template <typename DataLoader>
void train(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

void save_model(Net model, string model_path) {
  torch::serialize::OutputArchive output_archive;
  model.save(output_archive);
  output_archive.save_to(model_path);
}

auto main(int argc, char* argv[]) -> int {
  if (argc < 6) {
    cout
        << "pass arguments!\n"
        << "ex) data_root=./data train_batch_sz=64 test_batch_sz=1000 n_epoch=10 model_path=./model.pt"
        << endl;

    return 0;
  }
  // Where to find the MNIST dataset.
  const char* kDataRoot = argv[1];
  // The batch size for training.
  const int64_t kTrainBatchSize = stoi(argv[2]);
  // The batch size for testing.
  const int64_t kTestBatchSize = stoi(argv[3]);
  // The number of epochs to train.
  const int64_t kNumberOfEpochs = stoi(argv[4]);
  const string model_path = argv[5];

  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset =
      torch::data::datasets::MNIST(
          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

  save_model(model, model_path);
}
