"""
Flower client application for federated learning.
"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from my_awesome_app.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):

    def __init__(self, net: Net, trainloader, valloader, local_epochs: int, lr: float, context: Context):
        """Initialize the Flower client with model and data loaders."""
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train the model on local data."""
        set_weights(self.net, parameters)
        
        loss, accuracy = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.lr,
            device=self.device,
        )

        return (get_weights(self.net), len(self.trainloader.dataset), {"train_loss": loss,"train_accuracy": accuracy})

    def evaluate(self, parameters, config):
        """Evaluate the model on local validation data."""

        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    """Factory function to create a Flower client."""
    net = Net()

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size=32)

    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config.get("lr", 0.01)

    return FlowerClient(net, trainloader, valloader, local_epochs, lr, context).to_client()

# Register client app
app = ClientApp(client_fn=client_fn)
