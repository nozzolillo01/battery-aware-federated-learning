import logging
import os
from typing import List, Tuple, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

from my_awesome_app.task import Net, get_weights, set_weights, test, get_transforms
from my_awesome_app.my_strategy import BatteryAwareFedAvg

logging.getLogger("flwr").setLevel(logging.CRITICAL)
os.environ["WANDB_SILENT"] = "true"



def get_evaluate_fn(testloader, device):

    def evaluate(server_round, parameters_ndarrays, config):
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    return {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0,
    }


def on_fit_config(server_round: int) -> Metrics:
    return {}


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_battery_threshold = context.run_config["min-battery-threshold"]

    # Read num-supernodes from pyproject.toml (server side) and pass it to the strategy
    def read_num_supernodes(default: Optional[int] = None) -> Optional[int]:
        try:
            with open("pyproject.toml", "r") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("options.num-supernodes"):
                        return int(s.split("=")[-1].strip())
        except Exception:
            pass
        return default

    num_supernodes = read_num_supernodes()

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
        min_battery_threshold=min_battery_threshold,
        total_rounds=num_rounds,
        local_epochs=context.run_config.get("local-epochs", None),
        num_supernodes=num_supernodes,
    )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
