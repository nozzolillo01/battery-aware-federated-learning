"""
Task module for federated learning with Fashion-MNIST.
Implements a LeNet-style CNN model and data loading utilities.
"""

#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Global variable to store dataset
fds = None

# Create preprocessing transforms
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id: int, num_partitions: int) -> tuple:
    """
    Load a federated dataset partition and create train/test data loaders.
    
    Uses a global dataset object to avoid reloading for each client.
    Applies standard preprocessing transformations and creates data loaders.
    
    Args:
        partition_id: ID of the client's partition
        num_partitions: Total number of partitions (clients)
        
    Returns:
        tuple: (trainloader, testloader) DataLoaders for training and testing
    """
    global fds
    
    # Load dataset once and reuse for all clients
    if fds is None:
        #partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.3, partition_by="label")
        fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
    
    # Load client's specific partition
    partition = fds.load_partition(partition_id)
    
    # Split into train and test sets
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Apply transforms to the dataset
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    # Create data loaders
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    
    return trainloader, testloader

def train(net, trainloader, epochs: int, device: torch.device):
    """
    Train a neural network on the given data.
    
    Performs training for the specified number of epochs using Adam optimizer
    and cross entropy loss. Tracks and returns final loss and accuracy.
    
    Args:
        net: The neural network model
        trainloader: DataLoader with training data
        epochs: Number of training epochs
        device: Device to train on (CPU/GPU)
        
    Returns:
        tuple: (final_loss, final_accuracy)
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    
    net.train()
    
    for epoch in range(epochs):
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
    """
    Test a neural network on the given data.
    
    Evaluates model performance by computing loss and accuracy on the test set.
    Uses no_grad context to save memory during evaluation.
    
    Args:
        net: The neural network model
        testloader: DataLoader with testing data
        device: Device to test on (CPU/GPU)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()

    with torch.no_grad():
        for batch in testloader:

            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = loss / len(testloader)
    accuracy = correct / total
    
    return average_loss, accuracy


def get_weights(net):
    """
    Extract model weights as a list of NumPy arrays.
    
    Args:
        net: The neural network model
        
    Returns:
        list: Model weights as NumPy arrays
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """
    Set model weights from a list of NumPy arrays.
    
    Args:
        net: The neural network model
        parameters: List of NumPy arrays containing weights
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


def get_transforms():
    """
    Return the transforms used for Fashion-MNIST dataset.
    
    Returns:
        Compose: Composition of transforms for data preprocessing
    """
    return Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
