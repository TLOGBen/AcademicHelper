"""Tests for AcademicHelper server infrastructure."""

def test_package_imports():
    """Package should be importable and expose __version__."""
    import academic_helper

    assert hasattr(academic_helper, "__version__")
    assert isinstance(academic_helper.__version__, str)
    assert len(academic_helper.__version__) > 0


def test_server_module_imports():
    """Server module should be importable."""
    from academic_helper import server

    assert server is not None


def test_fastmcp_instance_exists():
    """Server should create a FastMCP instance."""
    from academic_helper.server import mcp

    assert mcp is not None
    assert mcp.name == "academic-helper"


def test_create_server_returns_mcp():
    """create_server() should return the configured FastMCP instance."""
    from academic_helper.server import create_server

    app = create_server()
    assert app is not None
    assert app.name == "academic-helper"


def test_server_has_main():
    """Server should expose a main() entry point."""
    from academic_helper.server import main

    assert callable(main)


def test_committee_tool_registered():
    """prepare_review_context is registered as an MCP tool."""
    from academic_helper.server import create_server

    server = create_server()
    tool_names = list(server._tool_manager._tools.keys())
    assert "prepare_review_context" in tool_names
