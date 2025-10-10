"""Core abstractions for client selection strategies.

This module provides a function-based selection system with automatic registration.
Selection functions are simple, pure functions decorated with @SelectionRegistry.register.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:  # pragma: no cover
    from ..battery_simulator import FleetManager


# Type alias for selection functions
SelectionFunction = Callable[
    [
        List[ClientProxy],  # available_clients
        Optional["FleetManager"],  # fleet_manager
        Dict[str, any],  # params from config
    ],
    Tuple[List[ClientProxy], Dict[str, float]],  # (selected, prob_map)
]


class SelectionRegistry:
    """Auto-discovery registry for client selection functions.
    
    Usage:
        @SelectionRegistry.register("my_custom")
        def select_my_way(eligible_clients, available_clients, fleet_manager, num_clients, params):
            # Your selection logic
            return selected_clients, probability_map
    
    Then use it from config:
        selection = "my_custom"
    """

    _selections: Dict[str, SelectionFunction] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[SelectionFunction], SelectionFunction]:
        """Decorator to register a selection function.
        
        Args:
            name: Name of the selection strategy (used in config).
        
        Returns:
            Decorator function.
        """

        def decorator(func: SelectionFunction) -> SelectionFunction:
            if name in cls._selections:
                raise ValueError(f"Selection strategy '{name}' already registered")
            cls._selections[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> SelectionFunction:
        """Get a selection function by name.
        
        Args:
            name: Name of the selection strategy.
        
        Returns:
            The selection function.
        
        Raises:
            ValueError: If strategy name not found.
        """
        if name not in cls._selections:
            available = ", ".join(sorted(cls._selections.keys()))
            raise ValueError(
                f"Unknown selection strategy: '{name}'. "
                f"Available strategies: [{available}]"
            )
        return cls._selections[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered selection strategies.
        
        Returns:
            Sorted list of strategy names.
        """
        return sorted(cls._selections.keys())
