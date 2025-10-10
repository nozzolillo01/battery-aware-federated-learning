"""Battery-aware federated learning application package.

This package provides a modular, function-based architecture for
federated learning with battery-aware client selection.

Key components:
- selection: Pluggable selection strategies (auto-discovered via SelectionRegistry)
- battery_simulator: Energy model and fleet management
- server_app, client_app: Flower applications
- task: ML model and dataset utilities
"""

from .selection import SelectionRegistry

__all__ = [
    "SelectionRegistry",
]
