"""Client selection strategy implementations with auto-discovery.

This module automatically discovers and registers all selection functions
in the package. Built-in strategies (random, battery_aware) are always available.

To add a custom selection strategy:
1. Create a new file in this directory (e.g., my_custom.py)
2. Import SelectionRegistry from .base
3. Decorate your function with @SelectionRegistry.register("my_custom")
4. Use it in config: selection = "my_custom"
"""

from .base import SelectionRegistry

# Import built-in strategies to trigger registration
from . import random_subset
from . import battery_weighted

# Auto-discover and import all user-defined selection strategies
import importlib
import pkgutil
from pathlib import Path

_selection_dir = Path(__file__).parent
for _module_info in pkgutil.iter_modules([str(_selection_dir)]):
    _module_name = _module_info.name
    # Skip base module, __init__, and built-ins (already imported above)
    if _module_name not in ["base", "__init__", "battery_weighted", "random_subset"]:
        importlib.import_module(f".{_module_name}", package=__name__)

__all__ = ["SelectionRegistry"]
