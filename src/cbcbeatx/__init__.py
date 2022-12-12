"""Top-level package for cbcbeatx."""
from importlib.metadata import metadata

meta = metadata("cbcbeatx")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]
