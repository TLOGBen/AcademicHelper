"""Exercise discovery and credential-free calls over an actual MCP stdio process."""

import os
import sys
import json
from datetime import timedelta

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from academic_helper.tools.zotero import TOOLS


@pytest.mark.asyncio
async def test_zotero_tools_over_stdio():
    environment = dict(os.environ)
    # Explicit empty value also blocks Windows User fallback: never use real credentials.
    environment["ZOTERO_API_KEY"] = ""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from academic_helper.server import main; main()"],
        env=environment,
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=20)) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "academic-helper"
            listing = await session.list_tools()
            expected = {tool.__name__ for tool in TOOLS}
            names = {tool.name for tool in listing.tools}
            assert expected <= names and len(names) == 15
            note = next(t for t in listing.tools if t.name == "zotero_save_note")
            assert note.inputSchema["properties"]["apply"]["default"] is False
            status = await session.call_tool("zotero_status", {})
            assert not status.isError
            payload = status.structuredContent or json.loads("\n".join(c.text for c in status.content if c.type == "text"))
            assert payload["status"] == "missing_config"
            expansion = await session.call_tool("expand_topics", {"seed_topic": "oral health", "limit": 1})
            assert not expansion.isError
