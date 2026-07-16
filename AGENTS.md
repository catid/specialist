# Agent Instructions

## Web searches

- Use the Exa MCP for web searches.

## Claude CLI

- Claude CLI requests can take a long time. Inspect the session more carefully
  than relying on a short timeout such as 300 seconds.
- Retry a Claude CLI request if a recent attempt failed because it timed out.
