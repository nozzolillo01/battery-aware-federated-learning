"""Simple battery-aware federated learning server."""

import logging
import os
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

from my_awesome_app.task import Net, get_weights, set_weights, test, get_transforms
from my_awesome_app.my_strategy import BatteryAwareFedAvg

logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("ray").setLevel(logging.CRITICAL)
os.environ["WANDB_SILENT"] = "true"
os.environ["RAY_DEDUP_LOGS"] = "0"


def get_evaluate_fn(testloader, device):

    def evaluate(server_round, parameters_ndarrays, config):
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def on_fit_config(server_round: int) -> Metrics:
    return {"lr": 0.01}


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    
    def apply_transforms(batch):
        transforms = get_transforms()
        batch["image"] = [transforms(img) for img in batch["image"]]
        return batch
    
    testset_transformed = testset.with_transform(apply_transforms)
    testloader = DataLoader(testset_transformed, batch_size=32)

    strategy = BatteryAwareFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
        min_battery_threshold=0.2,
    )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
