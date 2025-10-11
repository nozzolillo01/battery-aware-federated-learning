"""Client selection strategies with auto-discovery."""

import importlib
import pkgutil
from pathlib import Path

from .base import SelectionRegistry
from . import random_subset
from . import battery_weighted

_selection_dir = Path(__file__).parent
for _module_info in pkgutil.iter_modules([str(_selection_dir)]):
    _module_name = _module_info.name
    if _module_name not in ["base", "__init__", "battery_weighted", "random_subset"]:
        importlib.import_module(f".{_module_name}", package=__name__)

__all__ = ["SelectionRegistry"]
