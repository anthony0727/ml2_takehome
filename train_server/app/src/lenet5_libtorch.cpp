#include "lenet5_libtorch.hpp"

template <typename DataLoader>
void train(
    size_t epoch,
    Lenet5 &model,
    torch::Device device,
    DataLoader &data_loader,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size)
{
  model->train();
  size_t batch_idx = 0;
  for (auto &batch : data_loader)
  {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model->forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0)
    {
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
    Lenet5 &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size)
{
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto &batch : data_loader)
  {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model->forward(data);
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

void save_model(Lenet5 model, string model_path)
{
  torch::serialize::OutputArchive output_archive;
  model->save(output_archive);
  output_archive.save_to(model_path);
}

auto main(int argc, char *argv[]) -> int
{
  if (argc != 7)
  {
    cerr << "usage: ./main <data_path> <model_path> <train_batch_sz> <test_batch_sz> <n_epoch> <lr>\n"
         << "example: ./main ../data ../../models/libtorch/model.pt 64 1000 10 0.01" << endl;

    return -1;
  }
  // Where to find the MNIST dataset.
  const char *kDataRoot = argv[1];
  const string model_path = argv[2];
  // The batch size for training.
  const int64_t kTrainBatchSize = stoi(argv[3]);
  // The batch size for testing.
  const int64_t kTestBatchSize = stoi(argv[4]);
  // The number of epochs to train.
  const int64_t kNumberOfEpochs = stoi(argv[5]);
  const double lr = stod(argv[6]);

  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  }
  else
  {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Lenet5 model;
  model->to(device);

  auto train_dataset =
      torch::data::datasets::MNIST(kDataRoot)
          // .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset =
      torch::data::datasets::MNIST(
          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
          // .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::Adam optimizer(
      model->parameters(), torch::optim::AdamOptions(lr));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
  torch::save(model, model_path);

  // check if archived model is loadable
  Lenet5 module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::load(module, model_path);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // auto parameters = module->named_parameters();
  for (auto& p : module->named_parameters())
  {
    std::cout << p.key() << std::endl;
  }
}
