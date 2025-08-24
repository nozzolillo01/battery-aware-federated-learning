import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset

class Net(nn.Module):
    """
    LeNet-style CNN model for image classification.
    Designed for efficiency in federated learning scenarios with
    limited computational resources.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Global variable to store dataset (loaded only once)
fds = None

def load_data(partition_id: int, num_partitions: int):
    """
    Load a federated dataset partition and create train/test data loaders.
    
    Args:
        partition_id: ID of the client's partition
        num_partitions: Total number of partitions (clients)
        
    Returns:
        trainloader: DataLoader for training data
        testloader: DataLoader for testing data
    """
    global fds
    if fds is None:
        fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": num_partitions})
    
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def train(net, trainloader, epochs: int, device: torch.device):
    """
    Train a neural network on the given data.
    
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
            images, labels = batch["image"].to(device), batch["label"].to(device)
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
    
    Args:
        net: The neural network model
        testloader: DataLoader with testing data
        device: Device to test on (CPU/GPU)
        
    Returns:
        tuple: (loss, accuracy)
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader)
    accuracy = correct / total
    
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
    """Return the transforms used for Fashion-MNIST dataset."""
    return Compose([ToTensor(), Normalize((0.5,), (0.5,))])
