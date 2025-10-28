"""Top-level package for cbcbeatx."""

from importlib.metadata import metadata

from .markerwisefield import Markerwise, rhs_with_markerwise_field
from .monodomainsolver import MonodomainSolver

meta = metadata("cbcbeatx")
__version__ = meta["Version"]
__author__ = meta.get("Author")
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["MonodomainSolver", "rhs_with_markerwise_field", "Markerwise"]
