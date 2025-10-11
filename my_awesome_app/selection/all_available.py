from .base import SelectionRegistry


@SelectionRegistry.register("all_available")
def select_all_available(available_clients, fleet_manager, params):
    """Select all available clients."""
    probability_map = {client.cid: 1.0 for client in available_clients}
    return available_clients, probability_map