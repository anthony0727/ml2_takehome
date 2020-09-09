import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.c3 = nn.Conv2d(6, 16, 5, 1)
        self.c5 = nn.Conv2d(16, 120, 5, 1)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.c3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.c5(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.f6(x)
        x = F.relu(x)

        x = self.output(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_batch_sz', type=int)
    parser.add_argument('--test_batch_sz', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--lr', type=float)

    if len(sys.argv) != 13:
        print("usage: python3 lenet5_pytorch.py --data_path ./data --n_epochs 10 --train_batch_sz 64 --test_batch_sz 1024 --lr 0.01 --model_path ../models")
        exit()

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(args.data_path, train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, args.train_batch_sz)
    test_loader = torch.utils.data.DataLoader(dataset2, args.test_batch_sz)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, args.n_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
