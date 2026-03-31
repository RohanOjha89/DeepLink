import asyncio
import json
import os
from typing import Any, Dict


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic tool caller that runs the mcp_server.main module as a subprocess
    and returns the parsed JSON result.

    This provides a simple host/server separation: the agentic system
    (host) calls this function instead of importing the server module
    directly.
    """
    # Build the JSON argument string for the CLI entrypoint.
    args_json = json.dumps(arguments or {})

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Do not pass large JSON on argv (embeddings/chunks exceed OS ARG_MAX → errno 7).
    process = await asyncio.create_subprocess_exec(
        "python",
        "-m",
        "mcp_server.main",
        tool_name,
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout, stderr = await process.communicate(input=args_json.encode("utf-8"))

    if process.returncode != 0:
        return {
            "success": False,
            "error": stderr.decode("utf-8") if stderr else "mcp_server failed",
        }

    output = stdout.decode("utf-8")
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        # Subprocess printed invalid/truncated JSON (e.g. stdout buffer); surface it.
        return {
            "success": False,
            "error": f"Invalid JSON from MCP ({e!s}). Output length: {len(output)}. First 200 chars: {output[:200]!r}",
        }

