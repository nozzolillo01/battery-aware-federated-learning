"""
Client application for the battery-aware federated learning system.
Handles model training on client devices using the Flower framework.
"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from my_awesome_app.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    """
    Client implementation for federated learning.
    
    Handles local model training and evaluation, with proper
    device management and metrics tracking.
    """
    
    def __init__(self, net: Net, trainloader, valloader, local_epochs: int, context: Context):
        """
        Initialize the Flower client with model and data loaders.
        
        Args:
            net: Neural network model to train
            trainloader: DataLoader for training data
            valloader: DataLoader for validation data
            local_epochs: Number of local training epochs
            context: Flower client context
        """
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # Ensure metrics storage is initialized
        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration for training
            
        Returns:
            tuple: (updated_parameters, sample_size, metrics)
        """
        # Update model with server parameters
        set_weights(self.net, parameters)
        
        # Perform local training
        loss, accuracy = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        # Return updated model and metrics
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            },
        )

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters from the server
            config: Configuration for evaluation
            
        Returns:
            tuple: (loss, sample_size, metrics)
        """
        # Update model with server parameters
        set_weights(self.net, parameters)
        
        # Perform evaluation
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """
    Factory function to create a Flower client.
    
    Args:
        context: Client context from Flower
        
    Returns:
        Client: A configured Flower client
    """
    # Initialize model
    net = Net()
    
    # Get partition info from context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load data for this client
    trainloader, valloader = load_data(partition_id, num_partitions)
    
    # Get configuration
    local_epochs = context.run_config["local-epochs"]
    
    # Create and return client
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()


# Register client app
app = ClientApp(client_fn=client_fn)
