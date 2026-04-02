"""MCP server composition root — thin router, delegates to tool modules."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("academic-helper")


def create_server() -> FastMCP:
    """Configure and return the MCP server with all tool groups registered."""
    from .tools.search import register as reg_search
    from .tools.gaps import register as reg_gaps
    from .tools.evaluate import register as reg_evaluate

    reg_search(mcp)
    reg_gaps(mcp)
    reg_evaluate(mcp)
    return mcp


def main() -> None:
    """Entry point for the MCP server."""
    create_server()
    mcp.run(transport="stdio")
