"""
Flower server application for federated learning.
"""

import logging
import os
import sys
from typing import List, Tuple, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .task import Net, get_weights, load_centralized_dataset, set_weights, test
from .strategies import BatteryAwareClientFedAvg, RandomClientFedAvg

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

def evaluate_global_model(server_round, parameters_ndarrays, config, testloader, device):
    """Evaluate the global model using a centralized test set."""
    model = Net()
    set_weights(model, parameters_ndarrays)
    model.to(device)
    loss, accuracy = test(model, testloader, device)

    return loss, {"test_accuracy_server": accuracy}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return { "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0 }

def fit_metrics_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client fit metrics using weighted averages by sample size."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    train_loss_sum = 0.0
    train_accuracy_sum = 0.0
    val_loss_sum = 0.0
    val_accuracy_sum = 0.0

    for num_examples, m in metrics:
        if "train_loss" in m and m["train_loss"] is not None:
            train_loss_sum += num_examples * float(m["train_loss"])
        if "train_accuracy" in m and m["train_accuracy"] is not None:
            train_accuracy_sum += num_examples * float(m["train_accuracy"])
        if "val_loss" in m and m["val_loss"] is not None:
            val_loss_sum += num_examples * float(m["val_loss"])
        if "val_accuracy" in m and m["val_accuracy"] is not None:
            val_accuracy_sum += num_examples * float(m["val_accuracy"])

    return {
        "train_loss": train_loss_sum / total_examples,
        "train_accuracy": train_accuracy_sum / total_examples,
        "val_loss": val_loss_sum / total_examples,
        "val_accuracy": val_accuracy_sum / total_examples,
    }

def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the FL server components."""

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    local_epochs = context.run_config.get("local-epochs", None)
    strategy_id = context.run_config.get("strategy", 0)
    alpha = context.run_config.get("alpha", 2.0)
    min_battery_threshold = context.run_config.get("min-battery-threshold", None)
    num_supernodes = get_num_supernodes_from_config()

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    testloader = load_centralized_dataset()

    if strategy_id == 1:
        # Create battery-aware strategy
        strategy_impl = BatteryAwareClientFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_weighted_average,
            evaluate_fn=lambda sr, pn, c: evaluate_global_model(sr, pn, c, testloader, "cpu"),
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            num_supernodes=num_supernodes,
            alpha=alpha,
            min_battery_threshold=min_battery_threshold,
        )
    else:
        # Base strategy without battery awareness 
        strategy_impl = RandomClientFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_weighted_average,
            evaluate_fn=lambda sr, pn, c: evaluate_global_model(sr, pn, c, testloader, "cpu"),
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            num_supernodes=num_supernodes,
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy_impl, config=config)

# Register server app
app = ServerApp(server_fn=server_fn)
