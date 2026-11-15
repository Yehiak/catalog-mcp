"""MCP server for searching, comparing and deduplicating products in a catalog."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("catalog-mcp")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"
