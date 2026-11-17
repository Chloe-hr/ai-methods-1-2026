# MCP: let AI agents call your model

**Status: optional, Thursday. The MCP part is not yet tested in class.** Try it only if your app already works.

## What MCP is

- MCP means **Model Context Protocol**.
- It is a standard way for an AI agent to call a tool.
- A tool is a function with a name, a description and typed inputs.
- An MCP server lists its tools. The agent reads the list and chooses a tool.
- Your coding agent (Claude Code, Cursor) is an MCP client.

## Gradio as an MCP server (one flag)

Gradio can expose your functions as MCP tools. You need the `mcp` extra:

```
gradio[mcp]==6.32.0
```

In `app.py`, change the last line:

```python
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), mcp_server=True)
```

| Item | Value |
|---|---|
| MCP address | `https://<service-name>.onrender.com/gradio_api/mcp/` |
| Tool name | Taken from the function name |
| Tool description | Taken from the docstring and the type hints |

Write a clear docstring. The agent uses it to decide when to call your tool.

## Connect a coding agent

The exact steps depend on the client. Check the current docs of your tool. For Claude Code, the pattern is:

```bash
claude mcp add --transport http my-model https://<service-name>.onrender.com/gradio_api/mcp/
```

Then ask the agent: "Use the my-model tool to score this image." Look at the tool call in the agent output.

Notes:
- The Render free server sleeps. The first call can take 1 minute.
- Anyone with the URL can call the tool. Do not expose private data.

## Alternative: FastMCP (pure agent tool)

If you want **only** an agent tool, with no web page, use the `fastmcp` Python package.

```python
from fastmcp import FastMCP
mcp = FastMCP("my-model")

@mcp.tool
def score(image_b64: str) -> dict:
    """Score one image. Returns the anomaly score."""
    ...

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

Check the FastMCP docs for the current API before you use it.

## Which one?

| Need | Use |
|---|---|
| A page, an API and agent access | Gradio with `mcp_server=True` |
| Agent access only | FastMCP |
| Nothing from an agent | Skip this guide |
