"""
Task module for federated learning.
Implements a CNN model and data loading utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset

class Net(nn.Module):
    """ LeNet-style CNN for 32x32 RGB images."""
    def __init__(self, num_classes: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

fds = None

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.1, partition_by="label", min_partition_size=1)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset():
    """Load test set and return dataloader."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def train(net, trainloader, epochs: int, lr: float, device: torch.device):
    """Train a neural network on the given data."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    net.train()
    
    for _ in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def test(net, testloader, device: torch.device):
    """Test a neural network on the given data."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    
    net.eval()

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    loss = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    
    return loss, accuracy

def get_weights(net):
    """Extract model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)

def get_transforms():
    """Return the transforms used for the dataset."""
    return Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
