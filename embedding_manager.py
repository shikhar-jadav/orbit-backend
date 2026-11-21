"""
Multi-Provider Rotating Embedding Manager with Rate-Limit Cooldown & Automatic Failover.

Mirrors the architecture of RoundRobinProviderManager (config_manager.py):
- Scans numbered environment variables:
  EMBEDDING_PROVIDER_NAME{i}, EMBEDDING_API_KEY{i}, EMBEDDING_BASE_URL{i},
  EMBEDDING_MODEL{i}, EMBEDDING_DIMS{i}
- Supports thread-safe round-robin rotation across healthy providers
- Automatic rate-limit cooldown tracking (skips 429-cooling providers)
- High-level LangChain `RotatingEmbeddings` conforming to `langchain_core.embeddings.Embeddings`
  with zero-downtime failover across providers on 429 or network errors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("orbit.embedding_manager")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect HTTP 429 / rate-limit errors across OpenAI-compatible embedding endpoints."""
    if getattr(exc, "status_code", None) == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    err_str = str(exc).lower()
    return any(tok in err_str for tok in ("429", "rate limit", "rate_limit", "too many requests", "quota"))


def _extract_retry_after_seconds(exc: Exception, default: float = 3.0) -> float:
    """Parse Retry-After header or string from error response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is not None:
            try:
                val = float(str(raw).strip())
                if val > 0:
                    return max(1.0, min(val, 60.0))
            except ValueError:
                pass
    return default


@dataclass
class EmbeddingProviderConfig:
    """Configuration for an individual embedding provider endpoint."""

    index: int
    name: str
    api_key: str
    base_url: Optional[str] = None
    default_model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    call_count: int = 0
    last_request_at: float = 0.0
    cooldown_until: float = 0.0

    def masked_key(self) -> str:
        """Returns masked API key for safe logging and status reporting."""
        if not self.api_key:
            return "(empty)"
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def get_model(self) -> str:
        return self.default_model or "text-embedding-3-small"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "masked_key": self.masked_key(),
            "base_url": self.base_url or "https://api.openai.com/v1 (default OpenAI)",
            "active_model": self.get_model(),
            "dimensions": self.dimensions,
            "call_count": self.call_count,
            "is_cooling_down": time.monotonic() < self.cooldown_until,
            "cooldown_remaining_sec": max(0.0, round(self.cooldown_until - time.monotonic(), 1)),
        }


class RoundRobinEmbeddingManager:
    """
    Manages a pool of embedding API configurations loaded from environment variables.
    Provides thread-safe round-robin rotation across multiple embedding API endpoints,
    each with its own base_url, API key, model name, and optional dimension limit.
    """

    def __init__(
        self,
        env_path: Optional[str] = None,
        env_dict: Optional[Dict[str, str]] = None,
        allow_fallback_to_llm: bool = False,
    ):
        self.env_source: Dict[str, str] = {}
        if env_dict is not None:
            self.env_source = dict(env_dict)
        else:
            if env_path:
                load_dotenv(dotenv_path=env_path, override=True)
            else:
                load_dotenv(override=True)
            self.env_source = dict(os.environ)

        self.allow_fallback_to_llm = allow_fallback_to_llm
        self.providers: List[EmbeddingProviderConfig] = self._load_providers()

        self._lock = threading.Lock()
        self._current_index = 0
        self._total_requests = 0

    @property
    def is_configured(self) -> bool:
        """True if at least one embedding provider is available."""
        return len(self.providers) > 0

    def _get_val(self, key: str) -> Optional[str]:
        return self.env_source.get(key)

    def _load_providers(self) -> List[EmbeddingProviderConfig]:
        """
        Dynamically finds all EMBEDDING_API_KEY{i} or EMBEDDING_API_KEY_{i} entries.
        Also parses single EMBEDDING_API_KEY if no numbered keys exist.
        """
        indices = set()
        pattern = re.compile(r"^EMBEDDING_API_KEY_?(\d+)$", re.IGNORECASE)

        for key in self.env_source:
            match = pattern.match(key.strip())
            if match:
                indices.add(int(match.group(1)))

        sorted_indices = sorted(list(indices))
        providers: List[EmbeddingProviderConfig] = []

        for idx in sorted_indices:
            # 1. Fetch API Key
            api_key = self._get_val(f"EMBEDDING_API_KEY{idx}") or self._get_val(f"EMBEDDING_API_KEY_{idx}")
            if not api_key or not api_key.strip():
                continue
            api_key = api_key.strip()

            # 2. Fetch Base URL (optional)
            base_url = (
                self._get_val(f"EMBEDDING_BASE_URL{idx}")
                or self._get_val(f"EMBEDDING_BASE_URL_{idx}")
                or self._get_val(f"EMBEDDING_OPENAI_BASE_URL{idx}")
                or self._get_val(f"EMBEDDING_OPENAI_BASE_URL_{idx}")
            )
            if base_url:
                base_url = base_url.strip()

            # 3. Fetch Model
            model = (
                self._get_val(f"EMBEDDING_MODEL{idx}")
                or self._get_val(f"EMBEDDING_MODEL_{idx}")
                or "text-embedding-3-small"
            ).strip()

            # 4. Fetch Dimensions (optional)
            raw_dims = (
                self._get_val(f"EMBEDDING_DIMS{idx}")
                or self._get_val(f"EMBEDDING_DIMS_{idx}")
                or self._get_val(f"EMBEDDING_DIMENSIONS{idx}")
            )
            dimensions: Optional[int] = None
            if raw_dims:
                try:
                    dimensions = int(raw_dims.strip())
                except ValueError:
                    dimensions = None

            # 5. Fetch Provider Name
            name = (
                self._get_val(f"EMBEDDING_PROVIDER_NAME{idx}")
                or self._get_val(f"EMBEDDING_PROVIDER_NAME_{idx}")
            )
            if not name:
                if base_url:
                    if "together" in base_url.lower():
                        name = f"Together AI (Embedding #{idx})"
                    elif "voyage" in base_url.lower():
                        name = f"Voyage AI (Embedding #{idx})"
                    elif "mistral" in base_url.lower():
                        name = f"Mistral Embed (Embedding #{idx})"
                    elif "jina" in base_url.lower():
                        name = f"Jina AI (Embedding #{idx})"
                    elif "fireworks" in base_url.lower():
                        name = f"Fireworks (Embedding #{idx})"
                    elif "openrouter" in base_url.lower():
                        name = f"OpenRouter (Embedding #{idx})"
                    elif "localhost" in base_url.lower() or "127.0.0.1" in base_url:
                        name = f"Local/Ollama (Embedding #{idx})"
                    else:
                        name = f"Custom Embedding Provider #{idx}"
                else:
                    name = f"OpenAI (Embedding #{idx})"

            provider = EmbeddingProviderConfig(
                index=idx,
                name=name,
                api_key=api_key,
                base_url=base_url if base_url else None,
                default_model=model,
                dimensions=dimensions,
            )
            providers.append(provider)

        # Check for unnumbered EMBEDDING_API_KEY if no numbered keys were found
        if not providers:
            single_key = self._get_val("EMBEDDING_API_KEY")
            if single_key and single_key.strip():
                base_url = self._get_val("EMBEDDING_BASE_URL")
                model = self._get_val("EMBEDDING_MODEL") or "text-embedding-3-small"
                raw_dims = self._get_val("EMBEDDING_DIMS") or self._get_val("EMBEDDING_DIMENSIONS")
                dims = int(raw_dims.strip()) if raw_dims and raw_dims.strip().isdigit() else None
                name = self._get_val("EMBEDDING_PROVIDER_NAME") or "Default Embedding Provider"
                providers.append(
                    EmbeddingProviderConfig(
                        index=1,
                        name=name,
                        api_key=single_key.strip(),
                        base_url=base_url.strip() if base_url else None,
                        default_model=model.strip(),
                        dimensions=dims,
                    )
                )

        # Optional fallback to LLM provider 1 if allowed
        if not providers and self.allow_fallback_to_llm:
            llm_key = self._get_val("API_KEY1") or self._get_val("OPENAI_API_KEY")
            if llm_key and llm_key.strip():
                base_url = self._get_val("BASE_URL1")
                providers.append(
                    EmbeddingProviderConfig(
                        index=1,
                        name="LLM-Fallback Embedding Provider",
                        api_key=llm_key.strip(),
                        base_url=base_url.strip() if base_url else None,
                        default_model="text-embedding-3-small",
                    )
                )

        return providers

    def get_provider_count(self) -> int:
        return len(self.providers)

    def mark_request_started(self, provider: EmbeddingProviderConfig) -> None:
        """Record when an embedding request was dispatched."""
        with self._lock:
            provider.last_request_at = time.monotonic()

    def mark_rate_limited(self, provider: EmbeddingProviderConfig, retry_after: float = 3.0) -> None:
        """Temporarily cool down an embedding provider after a 429 response."""
        cooldown = max(1.0, min(float(retry_after), 60.0))
        with self._lock:
            now = time.monotonic()
            provider.cooldown_until = max(provider.cooldown_until, now + cooldown)
            logger.warning(
                f"[EmbeddingPool] Provider #{provider.index} ('{provider.name}') rate-limited; "
                f"cooling down for {cooldown:.1f}s."
            )

    def get_next_provider(self) -> EmbeddingProviderConfig:
        """
        Returns the next healthy EmbeddingProviderConfig in round-robin sequence (1 -> 2 -> ... -> N -> 1),
        skipping providers currently in cooldown when another provider is ready. Thread-safe.
        """
        if not self.providers:
            raise ValueError(
                "No embedding providers configured. "
                "Set EMBEDDING_API_KEY1 (and optionally EMBEDDING_BASE_URL1, EMBEDDING_MODEL1) in .env."
            )

        with self._lock:
            now = time.monotonic()
            count = len(self.providers)
            picked_idx: Optional[int] = None

            for offset in range(count):
                candidate_idx = (self._current_index + offset) % count
                if self.providers[candidate_idx].cooldown_until <= now:
                    picked_idx = candidate_idx
                    break

            if picked_idx is None:
                # All providers in cooldown; choose the one recovering soonest
                picked_idx = min(range(count), key=lambda idx: self.providers[idx].cooldown_until)

            provider = self.providers[picked_idx]
            self._total_requests += 1
            provider.call_count += 1
            self._current_index = (picked_idx + 1) % count
            return provider

    def create_embeddings_client(
        self,
        provider: EmbeddingProviderConfig,
        **kwargs: Any,
    ) -> OpenAIEmbeddings:
        """
        Creates a LangChain OpenAIEmbeddings client for the given provider configuration.
        Disables internal SDK retries to allow the manager's round-robin failover to handle 429s.
        """
        init_kwargs: Dict[str, Any] = {
            "model": provider.get_model(),
            "api_key": provider.api_key,
            "max_retries": 0,
            "request_timeout": kwargs.pop("request_timeout", 60.0),
            **kwargs,
        }

        if provider.dimensions is not None:
            init_kwargs["dimensions"] = provider.dimensions

        if provider.base_url:
            init_kwargs["base_url"] = provider.base_url
            if "ngrok" in provider.base_url.lower():
                headers = dict(init_kwargs.get("default_headers") or {})
                headers.setdefault("ngrok-skip-browser-warning", "true")
                init_kwargs["default_headers"] = headers

        return OpenAIEmbeddings(**init_kwargs)

    def get_next_embeddings(
        self,
        **kwargs: Any,
    ) -> Tuple[OpenAIEmbeddings, EmbeddingProviderConfig, int]:
        """
        Fetches the next healthy provider in round-robin sequence and returns:
        (OpenAIEmbeddings instance, EmbeddingProviderConfig, request_number)
        """
        provider = self.get_next_provider()
        req_num = self._total_requests
        client = self.create_embeddings_client(provider, **kwargs)
        return client, provider, req_num

    def get_status(self) -> Dict[str, Any]:
        """Returns the current pool metrics and provider health for FastAPI endpoints."""
        with self._lock:
            next_idx = self.providers[self._current_index].index if self.providers else 0
            return {
                "total_providers": len(self.providers),
                "next_provider_index": next_idx,
                "total_requests_served": self._total_requests,
                "providers": [p.to_dict() for p in self.providers],
            }

    def print_summary(self) -> None:
        """Prints a summary table of all loaded embedding providers."""
        print("\n" + "=" * 70)
        print(f"Loaded {len(self.providers)} Embedding Provider(s) from .env (Round-Robin Pool)")
        print("=" * 70)
        for i, p in enumerate(self.providers):
            print(f" [{i+1}] Name        : {p.name}")
            print(f"     Key         : {p.masked_key()}")
            print(f"     Base URL    : {p.base_url or 'https://api.openai.com/v1 (default OpenAI)'}")
            print(f"     Model       : {p.get_model()}")
            if p.dimensions:
                print(f"     Dimensions  : {p.dimensions}")
            print("-" * 70)
        print("=" * 70 + "\n")


class RotatingEmbeddings(Embeddings):
    """
    Production-grade LangChain Embeddings implementation that transparently rotates
    and fails over across a pool of OpenAI-compatible embedding providers.

    Conforms to the standard `langchain_core.embeddings.Embeddings` contract:
    - embed_documents(texts: List[str]) -> List[List[float]]
    - embed_query(text: str) -> List[float]
    - aembed_documents(texts: List[str]) -> List[List[float]]
    - aembed_query(text: str) -> List[float]

    On HTTP 429 or network errors, automatically cools down the failing provider
    and immediately retries with the next healthy provider in the pool.
    """

    def __init__(
        self,
        manager: Optional[RoundRobinEmbeddingManager] = None,
        *,
        env_dict: Optional[Dict[str, str]] = None,
        max_retries_per_provider: int = 1,
    ):
        if manager is not None:
            self.manager = manager
        else:
            self.manager = RoundRobinEmbeddingManager(env_dict=env_dict)
        self.max_retries_per_provider = max_retries_per_provider

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with automatic round-robin rotation and failover."""
        if not texts:
            return []

        total_providers = self.manager.get_provider_count()
        if total_providers == 0:
            raise ValueError("No embedding providers configured in RoundRobinEmbeddingManager.")

        last_error: Optional[Exception] = None

        for attempt in range(total_providers):
            client, provider, req_num = self.manager.get_next_embeddings()
            self.manager.mark_request_started(provider)
            try:
                return client.embed_documents(texts)
            except Exception as e:
                last_error = e
                is_rate_limit = _is_rate_limit_error(e)
                retry_after = _extract_retry_after_seconds(e, default=3.0)

                if is_rate_limit:
                    self.manager.mark_rate_limited(provider, retry_after=retry_after)

                logger.warning(
                    f"[RotatingEmbeddings] Provider #{provider.index} ('{provider.name}') failed on "
                    f"embed_documents: {e}. Failing over to next provider ({attempt + 1}/{total_providers})..."
                )

        raise RuntimeError(
            f"All {total_providers} embedding provider(s) failed. Last error: {last_error}"
        ) from last_error

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with automatic round-robin rotation and failover."""
        if not text:
            return []

        total_providers = self.manager.get_provider_count()
        if total_providers == 0:
            raise ValueError("No embedding providers configured in RoundRobinEmbeddingManager.")

        last_error: Optional[Exception] = None

        for attempt in range(total_providers):
            client, provider, req_num = self.manager.get_next_embeddings()
            self.manager.mark_request_started(provider)
            try:
                return client.embed_query(text)
            except Exception as e:
                last_error = e
                is_rate_limit = _is_rate_limit_error(e)
                retry_after = _extract_retry_after_seconds(e, default=3.0)

                if is_rate_limit:
                    self.manager.mark_rate_limited(provider, retry_after=retry_after)

                logger.warning(
                    f"[RotatingEmbeddings] Provider #{provider.index} ('{provider.name}') failed on "
                    f"embed_query: {e}. Failing over to next provider ({attempt + 1}/{total_providers})..."
                )

        raise RuntimeError(
            f"All {total_providers} embedding provider(s) failed. Last error: {last_error}"
        ) from last_error

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed documents with failover."""
        # Run blocking embed_documents in a thread pool to avoid blocking the event loop
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a query with failover."""
        return await asyncio.to_thread(self.embed_query, text)
