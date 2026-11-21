"""
LangGraph Event Streaming Utilities.

Provides helpers for consuming LangGraph's event streaming API
with typed projections for messages, state, subgraphs, and custom events.

Based on LangGraph Event Streaming documentation:
https://docs.langchain.com/oss/python/langgraph/event-streaming.md
"""

from typing import Any, AsyncGenerator, Dict, List, Optional


async def stream_messages(stream) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Extract message tokens from a LangGraph event stream.

    Yields dicts with:
        - type: "token" | "message_complete"
        - text: token text or full message text
        - node: node name that produced the message
        - usage: token usage metadata (on complete)
    """
    try:
        async for message in stream.messages:
            text = str(message.text) if hasattr(message, "text") else ""
            node = getattr(message, "node", "unknown")
            usage = None
            if hasattr(message, "output") and hasattr(message.output, "usage_metadata"):
                usage = message.output.usage_metadata

            yield {
                "type": "message",
                "text": text,
                "node": node,
                "usage": usage,
            }
    except Exception as e:
        yield {"type": "error", "error": str(e)}


async def stream_values(stream) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Extract state snapshots from a LangGraph event stream.

    Yields full state dicts after each graph step.
    """
    try:
        async for snapshot in stream.values:
            yield {"type": "state_snapshot", "state": snapshot}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


async def stream_subgraphs(stream) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Extract subgraph events from a LangGraph event stream.

    Yields dicts with subgraph name, path, and nested messages.
    """
    try:
        async for subgraph in stream.subgraphs:
            graph_name = getattr(subgraph, "graph_name", "unknown")
            path = getattr(subgraph, "path", [])
            yield {
                "type": "subgraph",
                "graph_name": graph_name,
                "path": path,
            }
    except Exception as e:
        yield {"type": "error", "error": str(e)}


async def stream_raw_events(stream) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream all raw protocol events from a LangGraph event stream.

    Each event is a ProtocolEvent envelope with:
        - seq: sequence number
        - method: channel name (messages, values, tools, lifecycle, etc.)
        - params: {namespace, timestamp, data}
    """
    try:
        async for event in stream:
            yield event
    except Exception as e:
        yield {"type": "error", "error": str(e)}


def format_sse_event(event_type: str, data: Any) -> str:
    """
    Format an event as a Server-Sent Events (SSE) string.

    Args:
        event_type: SSE event name
        data: JSON-serializable data payload

    Returns:
        Formatted SSE string ready to send over HTTP
    """
    import json
    json_data = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {json_data}\n\n"


class StreamEventAdapter:
    """
    Adapts LangGraph's event streaming to the existing SSE format
    used by the Orbit agent server.

    Bridges between:
    - LangGraph stream_events v3 API (typed projections)
    - Orbit SSE format (token, tool_start, tool_end, done, error)
    """

    def __init__(self, stream):
        self.stream = stream
        self._final_text = ""

    async def to_orbit_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Convert LangGraph event stream to Orbit SSE events.

        Yields the same event format as ReActAgent.astream_run():
            - {"type": "token", "data": {"token": "..."}}
            - {"type": "tool_start", "data": {...}}
            - {"type": "tool_end", "data": {...}}
            - {"type": "done", "data": {"content": "..."}}
        """
        try:
            for event in self.stream:
                method = event.get("method", "")
                params = event.get("params", {})
                data = params.get("data")

                if method == "messages":
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                evt = item.get("event", "")
                                if evt == "content-block-delta":
                                    delta = item.get("delta", {})
                                    delta_type = delta.get("type", "")
                                    if delta_type == "text-delta":
                                        token = delta.get("text", "")
                                        self._final_text += token
                                        yield {
                                            "type": "token",
                                            "data": {"token": token},
                                        }
                                    elif delta_type == "reasoning-delta":
                                        yield {
                                            "type": "token",
                                            "data": {
                                                "token": delta.get("reasoning", ""),
                                                "reasoning": True,
                                            },
                                        }

                elif method == "tools":
                    if isinstance(data, dict):
                        tool_event = data.get("event", "")
                        if tool_event == "tool-started":
                            yield {
                                "type": "tool_start",
                                "data": {
                                    "tool": data.get("tool_name", ""),
                                    "args": data.get("input", {}),
                                    "tool_call_id": data.get("tool_call_id", ""),
                                },
                            }
                        elif tool_event == "tool-finished":
                            yield {
                                "type": "tool_end",
                                "data": {
                                    "tool": data.get("tool_name", ""),
                                    "output": str(data.get("output", "")),
                                    "tool_call_id": data.get("tool_call_id", ""),
                                },
                            }
                        elif tool_event == "tool-error":
                            yield {
                                "type": "tool_end",
                                "data": {
                                    "tool": data.get("tool_name", ""),
                                    "output": f"Error: {data.get('error', '')}",
                                    "tool_call_id": data.get("tool_call_id", ""),
                                },
                            }

                elif method == "lifecycle":
                    if isinstance(data, dict):
                        lc_event = data.get("event", "")
                        if lc_event == "failed":
                            yield {
                                "type": "error",
                                "data": {"error": data.get("error", "Unknown error")},
                            }

            yield {
                "type": "done",
                "data": {"content": self._final_text.strip()},
            }

        except Exception as e:
            yield {"type": "error", "data": {"error": str(e)}}


# ── Interrupt handling helpers ──────────────────────────────────────────

def check_stream_interrupted(stream) -> bool:
    """Check if a stream was interrupted for human input."""
    return getattr(stream, "interrupted", False)


def get_stream_interrupts(stream) -> List[Any]:
    """Get interrupt payloads from a stream."""
    return list(getattr(stream, "interrupts", []))
