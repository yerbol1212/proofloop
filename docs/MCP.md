# Read-only agent interface

ProofLoop includes a dependency-free stdio MCP adapter for protocol **2025-06-18**. It connects to an already running local ProofLoop HTTP server. It does not start inference, submit images, activate tasks, follow repairs, or grant permission to act.

Start the workbench normally, then configure an MCP client to launch:

```bash
/absolute/path/to/proofloop/.venv/bin/python /absolute/path/to/proofloop/proofloop/mcp_server.py --port 8790
```

Using absolute executable/script paths avoids a client-specific working-directory requirement. From a repository shell, `python -m proofloop.mcp_server --port 8790` is equivalent. The process writes only newline-delimited JSON-RPC to stdout; CLI diagnostics go to stderr. Do not launch it as an ordinary interactive chat terminal expecting prose.

| Tool | Arguments | Result |
| --- | --- | --- |
| `proofloop_capabilities` | `{}` | Implemented predicates, evidence boundaries, policy version and worker availability |
| `proofloop_inspection` | `{"job_id":"<32 lowercase hex characters>"}` | One known job, integrity status and any verified historical completion result |
| `proofloop_events` | `{"job_id":"<id>","after_sequence":0}` | At most 200 events after the supplied cursor; cursor is optional |
| `proofloop_accounting` | `{}` | Persistent inspection request allowance, transport outcomes and reported token totals |

The client must initialize with the pinned protocol and send `notifications/initialized` before tool discovery/calls. Tool schemas and runtime argument validation reject extra fields. Both `structuredContent` and an equivalent JSON text content block are returned. Unknown tools/arguments produce protocol errors; backend failures produce fixed redacted tool errors.

Only four fixed HTTP GET routes on `127.0.0.1` are reachable. The port is the only connection option. Redirects, environment proxies, arbitrary URLs, private file reads, media access and HTTP mutations are excluded. Input and output messages are bounded to 1 MiB; sessions admit at most 10,000 request IDs. Tool annotations describe read-only behavior; actual dispatch restrictions implement it.

## What an agent can conclude

An agent can learn what evidence ProofLoop supports, read a known inspection, inspect its local journal, and identify required follow-up evidence. A completion record's `current_at_snapshot` does not mean the scene is current at a later tool call. `physical_action_authorized` remains false. When integrity validation fails, a stored success has no exposed verdict/result. Keep the authoritative workflow gate and human task review in the workbench.

This interface was tested with a real subprocess MCP session and an authored loopback HTTP backend: initialize → initialized notification → list tools → ping → all four tools. It was also exercised against the running production capability/accounting endpoints without inference. This is an implemented adapter, not an interoperability certification for every MCP client or a claim that Meta AI/glasses have connected to it.

Meta currently documents stdio and Streamable HTTP MCP in [Muse Code](https://dev.meta.ai/docs/muse-code/extending). Its [tool-calling API](https://dev.meta.ai/docs/tool-calling) has applications execute developer functions and enforce their own runtime limits. That makes a narrow verification tool surface a relevant integration direction. This is an engineering inference from public APIs, not inside knowledge of Meta's roadmap or judge preferences.

The implementation pins the official [MCP lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle), [stdio transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports) and [tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) specifications. See [durable inspection behavior and limitations](DURABLE-INSPECTIONS.md) before writing a client that submits paid work through the separate HTTP API.
