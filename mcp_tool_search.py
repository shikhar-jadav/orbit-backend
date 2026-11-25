"""Rank connected MCP tools for progressive, context-efficient discovery.

Implements hybrid retrieval (exact/prefix matches, field-weighted BM25 lexical
scoring, curated domain synonym expansion, and character trigram similarity),
plus thread-safe schema caching/invalidation and argument validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CAMEL_BOUNDARY_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")

_SCHEMA_CACHE_LOCK = Lock()
_SANITIZED_SCHEMA_CACHE: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]] = {}


def _split_camel(text: str) -> str:
    step1 = _CAMEL_BOUNDARY_1.sub(r"\1 \2", text)
    return _CAMEL_BOUNDARY_2.sub(r"\1 \2", step1)


def _tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [token for key, item in value.items() for token in (_tokens(key) + _tokens(item))]
    if isinstance(value, (list, tuple, set)):
        return [token for item in value for token in _tokens(item)]
    return _TOKEN_RE.findall(_split_camel(str(value)).lower())


def _trigrams(text: str) -> Set[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(cleaned) < 3:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}


def _trigram_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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


def get_cached_sanitized_schema(
    connector_id: str,
    tool_name: str,
    raw_schema: Any,
    sanitizer: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a cached sanitized JSON schema, invalidating automatically if raw_schema changes."""
    try:
        serialized = json.dumps(raw_schema or {}, sort_keys=True, default=str)
    except Exception:
        serialized = str(raw_schema)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    cache_key = (str(connector_id or ""), str(tool_name or ""))

    with _SCHEMA_CACHE_LOCK:
        cached = _SANITIZED_SCHEMA_CACHE.get(cache_key)
        if cached and cached[0] == digest:
            return cached[1]

    sanitized = sanitizer(raw_schema)
    with _SCHEMA_CACHE_LOCK:
        _SANITIZED_SCHEMA_CACHE[cache_key] = (digest, sanitized)
    return sanitized


def invalidate_mcp_tool_cache(connector_id: Optional[str] = None) -> int:
    """Invalidate cached tool schemas for one connector or all connectors."""
    with _SCHEMA_CACHE_LOCK:
        if connector_id is None:
            count = len(_SANITIZED_SCHEMA_CACHE)
            _SANITIZED_SCHEMA_CACHE.clear()
            return count
        target = str(connector_id)
        keys = [k for k in _SANITIZED_SCHEMA_CACHE if k[0] == target]
        for k in keys:
            _SANITIZED_SCHEMA_CACHE.pop(k, None)
        return len(keys)


def validate_mcp_tool_args(tool_args: Any, schema: Optional[Dict[str, Any]]) -> Optional[str]:
    """Validate MCP tool arguments against JSON schema required fields and object structure."""
    if tool_args is None:
        args_dict: Dict[str, Any] = {}
    elif isinstance(tool_args, dict):
        args_dict = tool_args
    else:
        return "Invalid tool arguments: expected a JSON object."

    if not isinstance(schema, dict):
        return None

    required = schema.get("required")
    if isinstance(required, list):
        missing = [str(field) for field in required if field not in args_dict or args_dict[field] is None]
        if missing:
            return f"Missing required parameter(s): {', '.join(missing)}"
    return None


def rank_mcp_tools(
    query: str,
    tools: Iterable[Dict[str, Any]],
    service: Optional[str] = None,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Return highest-ranked tools using hybrid lexical + synonym + trigram scoring."""
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

    search_tokens = list(dict.fromkeys(query_tokens))

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
    idf = {
        token: math.log(1.0 + (total - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
        for token in search_tokens
    }

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

        # Lightweight fuzzy/trigram fallback for partial stems & typos
        fuzzy_name = _trigram_similarity(query_text, name_text)
        fuzzy_desc = _trigram_similarity(query_text, desc_text)
        if fuzzy_name >= 0.2:
            score += fuzzy_name * 4.0
        if fuzzy_desc >= 0.2:
            score += fuzzy_desc * 1.5

        if score > 0:
            ranked.append((score, 1 if entry.get("perm") == "allow" else 0, str(entry.get("raw_name", "")).lower(), entry))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked[:safe_limit]]
