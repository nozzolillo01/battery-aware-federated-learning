"""
Flower server application for federated learning with battery-aware client selection.

This universal server works with any registered selection strategy through
the SelectionRegistry. No need to modify this file when adding new strategies.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from flwr.common import Context, FitRes, Metrics, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .battery_simulator import FleetManager
from .selection import SelectionRegistry
from .task import Net, get_weights, load_centralized_dataset, set_weights, test

# Configure logging and environment
logging.getLogger("flwr").setLevel(logging.CRITICAL)
os.environ["WANDB_SILENT"] = "true"

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

def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the universal FL server with pluggable selection.
    
    This function creates a single universal strategy that works with any
    selection function registered in SelectionRegistry.
    """
    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    local_epochs = int(context.run_config.get("local-epochs", 5))
    
    # Selection strategy configuration
    selection_name = context.run_config.get("selection", "battery_aware")
    sample_fraction = float(context.run_config.get("sample-fraction", 0.5))
    alpha = float(context.run_config.get("alpha", 2.0))
    min_battery_threshold = float(context.run_config.get("min-battery-threshold", 0.2))
    
    num_supernodes = get_num_supernodes_from_config()

    # Initialize model and test data
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    testloader = load_centralized_dataset()

    # Get selection function from registry (raises error if not found)
    selection_fn = SelectionRegistry.get(selection_name)
    
    # Prepare parameters dict for selection function
    selection_params = {
        "sample_fraction": sample_fraction,
        "alpha": alpha,
        # Add any other custom params here - they'll be passed to selection function
    }

    # Print configuration header
    print(f"num-supernodes = {num_supernodes}")
    print(f"num-server-rounds = {num_rounds}")
    print(f"local-epochs = {local_epochs}")
    print(f"selection strategy = {selection_name}")
    print(f"sample-fraction = {sample_fraction}")
    if selection_name == "battery_aware":
        print(f"alpha = {alpha}")
        print(f"min-battery-threshold = {min_battery_threshold}")

    # Initialize fleet manager and W&B
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import wandb
    import numpy as np
    
    fleet_manager = FleetManager()
    
    # Initialize W&B
    tz = ZoneInfo("Europe/Rome")
    timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
    try:
        wandb.init(
            project="nuovo",
            name=f"{selection_name.upper()}-run-{timestamp}",
            config={
                "total_rounds": num_rounds,
                "local_epochs": local_epochs,
                "num_supernodes": num_supernodes,
                "selection": selection_name,
                "sample_fraction": sample_fraction,
                "alpha": alpha,
                "min_battery_threshold": min_battery_threshold,
            },
        )
        wandb.define_metric("round")
        wandb.define_metric("*", step="round")
    except Exception:
        pass

    # State tracking for logging
    class ServerState:
        """Tracks server state across rounds."""
        def __init__(self):
            self.last_selected_count = 0
            self.last_selected_battery_avg: Optional[float] = None
            self.last_selected_battery_min: Optional[float] = None
            self.last_deaths_count = 0
            self.current_eligible_count = 0
            self.prev_battery_levels: Dict[str, float] = {}
            self.client_id_order: List[str] = []
            self.rounds_since_selected: Dict[str, int] = {}
            self.selected_for_fit_last_round: List[str] = []
            self.last_train_accuracy: Optional[float] = None
            self.last_train_loss: Optional[float] = None
            self.last_server_test: Dict[str, Any] = {}
            self.acc_series = {"train": [], "val": [], "test": []}
            self.loss_series = {"train": [], "val": [], "test": []}

    state = ServerState()

    # Create universal strategy with battery awareness
    class UniversalBatteryAwareFedAvg(FedAvg):
        """Universal FedAvg strategy with pluggable client selection and battery tracking."""

        def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: Any,
        ) -> List[Tuple[ClientProxy, Dict[str, Any]]]:
            """Configure clients for training."""
            # Get base configuration from parent FedAvg
            original_config = super().configure_fit(server_round, parameters, client_manager)
            if not original_config:
                return []

            # Extract available clients and config
            available_clients = [client for client, _ in original_config]
            config = original_config[0][1] if original_config else {}

            # Call selection function
            selected_clients, prob_map = selection_fn(
                available_clients=available_clients,
                fleet_manager=fleet_manager,
                params=selection_params,
            )

            # Track eligible count for logging
            eligible_ids = fleet_manager.get_eligible_clients(
                [c.cid for c in available_clients],
                min_battery_threshold,
            )
            state.current_eligible_count = len(eligible_ids)

            # Ensure probability map covers all clients
            prob_map = {c.cid: prob_map.get(c.cid, 0.0) for c in available_clients}

            selected_client_ids = [c.cid for c in selected_clients]
            available_client_ids = [c.cid for c in available_clients]

            # Check for battery deaths during training
            deaths = fleet_manager.get_dead_clients(selected_client_ids, local_epochs)
            state.last_deaths_count = len(deaths)

            # Capture stats before battery update
            if selected_client_ids:
                levels = [fleet_manager.get_battery_level(cid) for cid in selected_client_ids]
                state.last_selected_battery_avg = sum(levels) / len(levels) if levels else None
                state.last_selected_battery_min = min(levels) if levels else None
            else:
                state.last_selected_battery_avg = None
                state.last_selected_battery_min = None
            state.last_selected_count = len(selected_client_ids)

            # Update battery levels (consume for selected, recharge for all)
            fleet_manager.update_round(selected_client_ids, available_client_ids, local_epochs)

            # Log to W&B
            _log_selection_to_wandb(
                server_round,
                available_clients,
                selected_client_ids,
                eligible_ids,
                prob_map,
                list(deaths),
                fleet_manager,
                state,
            )

            # Remove dead clients from selection
            selected_clients = [c for c in selected_clients if c.cid not in deaths]
            state.selected_for_fit_last_round = [c.cid for c in selected_clients]

            return [(client, config) for client in selected_clients]

        def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: Any,
        ) -> List[Tuple[ClientProxy, Dict[str, Any]]]:
            """Configure clients for evaluation (only those that trained)."""
            original = super().configure_evaluate(server_round, parameters, client_manager)
            if not original or not state.selected_for_fit_last_round:
                return original

            selected_set = set(state.selected_for_fit_last_round)
            filtered = [(c, cfg) for (c, cfg) in original if c.cid in selected_set]
            return filtered if filtered else original

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Tuple[ClientProxy, FitRes]],
        ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
            """Aggregate fit results and add battery metrics."""
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                server_round, results, failures
            )

            # Add battery metrics
            fleet_stats = fleet_manager.get_fleet_stats(min_battery_threshold)
            battery_metrics = {
                "fleet_avg_battery": fleet_stats.get("avg_battery", 0.0),
                "fleet_min_battery": fleet_stats.get("min_battery", 0.0),
                "fairness_jain": fleet_stats.get("fairness_jain", 0.0),
                "total_energy_consumed": fleet_stats.get("total_energy_consumed", 0.0),
            }

            if metrics_aggregated is None:
                metrics_aggregated = {}
            metrics_aggregated.update(battery_metrics)

            # Save train metrics for later logging
            state.last_train_accuracy = metrics_aggregated.get("train_accuracy")
            state.last_train_loss = metrics_aggregated.get("train_loss")

            return parameters_aggregated, metrics_aggregated

        def evaluate(
            self,
            server_round: int,
            parameters: Parameters,
        ) -> Optional[Tuple[float, Dict[str, Any]]]:
            """Evaluate global model on centralized test set."""
            result = super().evaluate(server_round, parameters)
            if result is not None:
                loss, metrics = result
                state.last_server_test = {"loss": loss, **metrics}
            return result

        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Tuple[ClientProxy, FitRes]],
        ) -> Tuple[Optional[float], Dict[str, Any]]:
            """Aggregate evaluation results and log complete metrics."""

            loss, metrics = super().aggregate_evaluate(server_round, results, failures)

            # Prepare complete metrics
            fleet_stats = fleet_manager.get_fleet_stats(min_battery_threshold)
            analysis_results = {
                "round": server_round,
                "battery_avg": round(fleet_stats.get("avg_battery", 0.0), 3),
                "battery_min": round(fleet_stats.get("min_battery", 0.0), 3),
                "fairness_jain": round(fleet_stats.get("fairness_jain", 0.0), 3),
                "total_energy": round(fleet_stats.get("total_energy_consumed", 0.0), 3),
                "eligible_clients": state.current_eligible_count,
                "selected_clients": state.last_selected_count,
                "selected_battery_avg": round(state.last_selected_battery_avg, 3)
                if state.last_selected_battery_avg is not None
                else None,
                "selected_battery_min": round(state.last_selected_battery_min, 3)
                if state.last_selected_battery_min is not None
                else None,
                "deaths_clients": state.last_deaths_count,
            }

            # Add client validation metrics
            analysis_results["val_accuracy_client"] = round(float(metrics.get("accuracy", 0.0)), 4)
            analysis_results["val_loss_client"] = round(loss, 4)

            # Add train metrics if available
            if state.last_train_accuracy is not None:
                analysis_results["train_accuracy_client"] = float(state.last_train_accuracy)
            if state.last_train_loss is not None:
                analysis_results["train_loss_client"] = float(state.last_train_loss)

            # Add server test metrics if available
            if state.last_server_test:
                if "test_accuracy_server" in state.last_server_test:
                    analysis_results["test_accuracy_server"] = round(
                        float(state.last_server_test["test_accuracy_server"]), 4
                    )
                if "loss" in state.last_server_test:
                    analysis_results["test_loss_server"] = round(
                        float(state.last_server_test["loss"]), 4
                    )
            state.last_server_test = {}

            # Log to W&B
            _log_results_to_wandb(server_round, analysis_results, state)

            # Print summary
            _print_evaluation_summary(server_round, analysis_results, num_supernodes)

            return loss, metrics

    # Helper functions for logging (defined outside class to keep it clean)
    def _log_selection_to_wandb(
        server_round: int,
        available_clients: List[ClientProxy],
        selected_client_ids: List[str],
        eligible_ids: List[str],
        prob_map: Dict[str, float],
        deaths_ids: List[str],
        fleet_mgr: FleetManager,
        state_obj: ServerState,
    ) -> None:
        """Log client selection details to W&B."""
        # Update client order
        newly_seen = [c.cid for c in available_clients if c.cid not in state_obj.client_id_order]
        if newly_seen:
            state_obj.client_id_order.extend(newly_seen)
            state_obj.client_id_order = sorted(state_obj.client_id_order)

        # Collect present client data
        present_data: Dict[str, Dict[str, Any]] = {}
        for c in available_clients:
            cid = c.cid
            battery = fleet_mgr.get_battery_level(cid)
            prev = state_obj.prev_battery_levels.get(cid)
            consumed = fleet_mgr.client_consumed_battery.get(cid, 0)
            recharged = fleet_mgr.client_recharged_battery.get(cid, 0)
            prob = prob_map.get(cid, 0.0)
            present_data[cid] = {
                "current_battery_level": round(battery, 4),
                "previous_battery_level": (round(prev, 4) if prev is not None else np.nan),
                "consumed_battery": round(consumed, 4),
                "recharged_battery": round(recharged, 4),
                "prob_selection": round(prob, 6),
                "selected": int(cid in selected_client_ids),
                "eligible": int(cid in eligible_ids),
            }

        # Update rounds since selected
        for cid in state_obj.client_id_order:
            if cid in selected_client_ids:
                state_obj.rounds_since_selected[cid] = 0
            else:
                if cid in state_obj.rounds_since_selected:
                    state_obj.rounds_since_selected[cid] += 1
                else:
                    state_obj.rounds_since_selected[cid] = 1 if cid not in selected_client_ids else 0

        # Create W&B table
        columns = [
            "round",
            "client_id",
            "device_class",
            "current_battery_level",
            "previous_battery_level",
            "consumed_battery",
            "recharged_battery",
            "prob_selection",
            "selected",
            "eligible",
            "is_dead_during_the_round",
            "rounds_since_selected",
        ]
        round_table = wandb.Table(columns=columns)

        for cid in state_obj.client_id_order:
            if cid in present_data:
                is_dead = int(cid in deaths_ids)
                row = [
                    server_round,
                    cid,
                    fleet_mgr.get_device_class(cid),
                    present_data[cid]["current_battery_level"],
                    present_data[cid]["previous_battery_level"],
                    present_data[cid]["consumed_battery"],
                    present_data[cid]["recharged_battery"],
                    present_data[cid]["prob_selection"],
                    present_data[cid]["selected"],
                    present_data[cid]["eligible"],
                    is_dead,
                    state_obj.rounds_since_selected.get(cid, np.nan),
                ]
            else:
                row = [
                    server_round,
                    cid,
                    fleet_mgr.get_device_class(cid),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0.0,
                    0,
                    0,
                    0,
                    state_obj.rounds_since_selected.get(cid, np.nan),
                ]
            round_table.add_data(*row)

        try:
            wandb.log(
                {
                    "round": server_round,
                    "selected_clients": state_obj.last_selected_count,
                    "deaths_clients": state_obj.last_deaths_count,
                    f"client_status_round_{server_round}": round_table,
                },
                step=server_round,
            )
        except Exception:
            pass

        # Update previous battery levels
        for c in available_clients:
            state_obj.prev_battery_levels[c.cid] = fleet_mgr.get_battery_level(c.cid)

    def _log_results_to_wandb(
        server_round: int,
        analysis_results: Dict[str, Any],
        state_obj: ServerState,
    ) -> None:
        """Log aggregated results and charts to W&B."""
        wandb_payload = {
            k: v for k, v in analysis_results.items() if k not in ["round"] and v is not None
        }
        try:
            wandb.log(wandb_payload, step=server_round)
        except Exception:
            pass

        # Update series and create charts
        try:

            def _append(target: List[Optional[float]], value: Optional[Any]) -> None:
                target.append(float(value) if value is not None else None)

            _append(state_obj.acc_series["train"], analysis_results.get("train_accuracy_client"))
            _append(state_obj.acc_series["val"], analysis_results.get("val_accuracy_client"))
            _append(state_obj.acc_series["test"], analysis_results.get("test_accuracy_server"))

            _append(state_obj.loss_series["train"], analysis_results.get("train_loss_client"))
            _append(state_obj.loss_series["val"], analysis_results.get("val_loss_client"))
            _append(state_obj.loss_series["test"], analysis_results.get("test_loss_server"))

            xs = list(range(1, len(state_obj.acc_series["train"]) + 1))
            acc_plot = wandb.plot.line_series(
                xs=xs,
                ys=[
                    state_obj.acc_series["train"],
                    state_obj.acc_series["val"],
                    state_obj.acc_series["test"],
                ],
                keys=[
                    "train_accuracy_client",
                    "val_accuracy_client",
                    "test_accuracy_server",
                ],
                title="Accuracy per round",
                xname="round",
            )
            loss_plot = wandb.plot.line_series(
                xs=xs,
                ys=[
                    state_obj.loss_series["train"],
                    state_obj.loss_series["val"],
                    state_obj.loss_series["test"],
                ],
                keys=[
                    "train_loss_client",
                    "val_loss_client",
                    "test_loss_server",
                ],
                title="Loss per round",
                xname="round",
            )
            wandb.log(
                {
                    "chart/accuracy_per_round": acc_plot,
                    "chart/loss_per_round": loss_plot,
                },
                step=server_round,
            )
        except Exception:
            pass

    def _print_evaluation_summary(
        server_round: int,
        analysis_results: Dict[str, Any],
        total_supernodes: Optional[int],
    ) -> None:
        """Print round evaluation summary to console."""
        try:
            val_acc = analysis_results.get("val_accuracy_client")
            val_loss = analysis_results.get("val_loss_client")
            test_acc = analysis_results.get("test_accuracy_server")
            test_loss = analysis_results.get("test_loss_server")
            
            test_acc_label = (
                f"test_accuracy_server={test_acc:.4f}" if test_acc is not None else "test_accuracy_server=NA"
            )
            test_loss_label = (
                f"test_loss_server={test_loss:.4f}" if test_loss is not None else "test_loss_server=NA"
            )
            val_acc_label = (
                f"val_accuracy_client={val_acc:.4f}" if val_acc is not None else "val_accuracy_client=NA"
            )
            val_loss_label = (
                f"val_loss_client={val_loss:.4f}" if val_loss is not None else "val_loss_client=NA"
            )
            
            eligible = analysis_results.get("eligible_clients", 0)
            total = total_supernodes if total_supernodes is not None else "?"
            selected = analysis_results.get("selected_clients", 0)
            deaths = analysis_results.get("deaths_clients", 0)

            print(
                f"[Round {server_round}] {test_acc_label} {test_loss_label} {val_acc_label} {val_loss_label} "
                f"eligible_clients={eligible} "
                f"selected_clients={selected}/{total} "
                f"deaths={deaths}/{selected}"
            )
        except Exception:
            pass

    # Create the universal strategy instance
    strategy = UniversalBatteryAwareFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_weighted_average,
        evaluate_fn=lambda sr, pn, c: evaluate_global_model(sr, pn, c, testloader, device="cpu"),
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Register server app
app = ServerApp(server_fn=server_fn)
