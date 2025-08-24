"""my-awesome-app: A Flower / PyTorch app."""

import logging
import os
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from my_awesome_app.task import Net, get_weights, load_data, set_weights, test, train

logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("ray").setLevel(logging.CRITICAL)
os.environ["RAY_DEDUP_LOGS"] = "0"


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()


app = ClientApp(client_fn=client_fn)
