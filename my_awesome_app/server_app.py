"""
Server application for the battery-aware federated learning system.
Configures and runs the FL server with energy-aware client selection.
"""

import logging
import os
import sys
import tomli
import pathlib
from typing import List, Tuple, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

from my_awesome_app.task import Net, get_weights, set_weights, test, get_transforms
from my_awesome_app.battery_strategy import BatteryAwareFedAvg
from my_awesome_app.base_strategy import BaseStrategy

# Configure logging and environment
logging.getLogger("flwr").setLevel(logging.CRITICAL)
os.environ["WANDB_SILENT"] = "true"

def get_num_supernodes_from_config() -> int:
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--num-supernodes"):
            value: Optional[str] = None
            if "=" in arg:
                value = arg.split("=", 1)[1]
            elif i + 1 < len(sys.argv):
                value = sys.argv[i + 1]
            if value is not None:
                return int(value)


def get_evaluate_fn(testloader, device):
    """
    Create an evaluation function for the global model using a centralized test set.

    Args:
        testloader: DataLoader with test data
        device: Device to run evaluation on

    Returns:
        function: Evaluation function for the federated strategy
    """
    def evaluate(server_round, parameters_ndarrays, config):
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        # Emit server-side test metric with the final, explicit name
        return loss, {"test_accuracy_server": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client metrics using weighted average.
    
    Args:
        metrics: List of tuples (sample_size, metrics_dict)
        
    Returns:
        Metrics: Aggregated metrics
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    return {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0,
    }


def fit_metrics_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client fit metrics using weighted averages by sample size.

    Expects each client's metrics to potentially include:
      - train_loss, train_accuracy
      - val_loss, val_accuracy
    Missing keys are ignored gracefully.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    def wavg(key: str, invert: bool = False) -> Optional[float]:
        acc = 0.0
        denom = 0
        for num_examples, m in metrics:
            if key in m and m[key] is not None:
                acc += num_examples * float(m[key])
                denom += num_examples
        return (acc / denom) if denom > 0 else None

    return {
        "train_loss": wavg("train_loss"),
        "train_accuracy": wavg("train_accuracy"),
        "val_loss": wavg("val_loss"),
        "val_accuracy": wavg("val_accuracy"),
    }

def server_fn(context: Context) -> ServerAppComponents:
    """
    Create and configure the FL server components.
    
    Args:
        context: Server context from Flower
        
    Returns:
        ServerAppComponents: Configured server components
    """
    # Extract configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_battery_threshold = context.run_config["min-battery-threshold"]
    local_epochs = context.run_config.get("local-epochs", None)
    strategy = context.run_config.get("strategy", 0)
    alpha = context.run_config.get("alpha", 2.0)
    num_supernodes = get_num_supernodes_from_config()



    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load and prepare centralized test dataset for server-side evaluation
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    def apply_transforms(batch):
        transforms = get_transforms()
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch

    dataset = test_dataset.with_transform(apply_transforms)
    testloader = DataLoader(dataset, batch_size=128)

    if strategy == 1:
        # Create battery-aware strategy
        strategy = BatteryAwareFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_weighted_average,
            evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
            min_battery_threshold=min_battery_threshold,
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            num_supernodes=num_supernodes,
            alpha=alpha,
        )
    else:
        # Base strategy without battery awareness 
        strategy = BaseStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_weighted_average,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
        total_rounds=num_rounds,
        local_epochs=local_epochs,
        num_supernodes=num_supernodes,
    )

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Register server app
app = ServerApp(server_fn=server_fn)
