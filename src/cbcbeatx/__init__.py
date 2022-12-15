"""Top-level package for cbcbeatx."""
from .monodomainsolver import MonodomainSolver
from .markerwisefield import rhs_with_markerwise_field, Markerwise
from importlib.metadata import metadata

meta = metadata("cbcbeatx")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["MonodomainSolver", "rhs_with_markerwise_field", "Markerwise"]
