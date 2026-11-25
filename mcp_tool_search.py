"""Rank connected MCP tools for progressive, context-efficient discovery.

Uses the Python standard library: exact phrase/name matches, field-weighted
BM25-style lexical scoring, and curated synonym expansion. This is not embedding
or vector-based semantic search.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SYNONYMS = {
    "issue": {"ticket", "bug", "task"}, "ticket": {"issue", "bug"},
    "email": {"mail", "message"}, "mail": {"email", "message"},
    "repository": {"repo"}, "repo": {"repository"},
    "pull": {"pr", "merge"}, "request": {"pr"}, "pr": {"pull", "request"},
    "create": {"add", "open", "new"}, "add": {"create", "insert"},
    "remove": {"delete"}, "delete": {"remove"},
    "search": {"find", "query", "lookup"}, "find": {"search", "lookup"},
    "read": {"get", "fetch", "view"}, "get": {"read", "fetch"},
    "send": {"email", "message", "mail"},
    "calendar": {"event", "schedule"}, "event": {"calendar", "schedule"},
}

def _tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [token for key, item in value.items() for token in (_tokens(key) + _tokens(item))]
    if isinstance(value, (list, tuple, set)):
        return [token for item in value for token in _tokens(item)]
    return _TOKEN_RE.findall(str(value).lower())

def _schema_text(schema: Any) -> str:
    """Extract searchable parameter names/descriptions without changing schemas."""
    if not isinstance(schema, dict):
        return ""
    chunks: List[str] = []
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, spec in properties.items():
            chunks.append(str(name))
            if isinstance(spec, dict):
                chunks.append(str(spec.get("description", "")))
                chunks.append(_schema_text(spec))
    items = schema.get("items")
    if isinstance(items, dict):
        chunks.append(_schema_text(items))
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            chunks.extend(_schema_text(item) for item in variants)
    return " ".join(chunks)

def _field_score(query_tokens: List[str], field_tokens: List[str], idf: Dict[str, float], weight: float) -> float:
    if not query_tokens or not field_tokens:
        return 0.0
    counts = Counter(field_tokens)
    length_norm = 1.0 + 0.35 * (len(field_tokens) / 80.0)
    score = 0.0
    for token in query_tokens:
        tf = counts.get(token, 0)
        if tf:
            score += idf.get(token, 1.0) * (tf * 2.2 / (tf + 1.2)) / length_norm
    return score * weight

def rank_mcp_tools(query: str, tools: Iterable[Dict[str, Any]], service: Optional[str] = None, limit: int = 6) -> List[Dict[str, Any]]:
    """Return highest-ranked tools; optionally restrict to one connected service."""
    entries = list(tools)
    query_text = (query or "").strip().lower()
    query_tokens = _tokens(query_text)
    if not entries:
        return []
    service_text = (service or "").strip().lower()
    if service_text:
        entries = [entry for entry in entries if service_text in str(entry.get("connector_name", "")).lower()]
    if not entries:
        return []
    safe_limit = max(1, min(int(limit), 8))
    if not query_tokens:
        return entries[:safe_limit]
    expanded = set(query_tokens)
    for token in query_tokens:
        expanded.update(_SYNONYMS.get(token, set()))
    search_tokens = list(expanded)
    docs: List[Dict[str, List[str]]] = []
    df: Counter[str] = Counter()
    for entry in entries:
        fields = {
            "service": _tokens(entry.get("connector_name", "")),
            "name": _tokens(entry.get("raw_name", "")),
            "description": _tokens(entry.get("description", "")),
            "parameters": _tokens(_schema_text(entry.get("params") or entry.get("inputSchema") or {})),
        }
        docs.append(fields)
        df.update(set(token for values in fields.values() for token in values))
    total = max(1, len(docs))
    idf = {token: math.log(1.0 + (total - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5)) for token in search_tokens}
    ranked = []
    for entry, fields in zip(entries, docs):
        score = (
            _field_score(search_tokens, fields["name"], idf, 4.0)
            + _field_score(search_tokens, fields["service"], idf, 3.0)
            + _field_score(search_tokens, fields["description"], idf, 2.0)
            + _field_score(search_tokens, fields["parameters"], idf, 1.5)
        )
        name_text, service_name, desc_text = (" ".join(fields[k]) for k in ("name", "service", "description"))
        if query_text and query_text in name_text:
            score += 12.0
        if query_text and query_text in service_name:
            score += 8.0
        if query_text and query_text in desc_text:
            score += 3.0
        if score > 0:
            ranked.append((score, 1 if entry.get("perm") == "allow" else 0, str(entry.get("raw_name", "")).lower(), entry))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked[:safe_limit]]
