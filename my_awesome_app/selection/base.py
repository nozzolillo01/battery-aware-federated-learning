"""Registry for client selection strategies."""

from typing import Callable, Dict, List


class SelectionRegistry:
    """Simple registry for selection strategies.
    
    Usage:
        @SelectionRegistry.register("my_strategy")
        def select_my_way(available_clients, fleet_manager, params):
            return selected_clients, probability_map
    """

    _strategies: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Register a strategy with a decorator."""
        def decorator(func):
            cls._strategies[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        """Get a strategy by name."""
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")
        return cls._strategies[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered strategies."""
        return sorted(cls._strategies.keys())
