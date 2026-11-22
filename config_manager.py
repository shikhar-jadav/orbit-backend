import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from embedding_manager import (
        EmbeddingProviderConfig,
        RoundRobinEmbeddingManager,
        RotatingEmbeddings,
    )
except ImportError:
    pass


@dataclass
class ProviderConfig:
    index: int
    name: str
    api_key: str
    base_url: Optional[str] = None
    default_model: str = "gpt-4o-mini"
    model_list: List[str] = field(default_factory=list)
    call_count: int = 0
    last_request_at: float = 0.0
    cooldown_until: float = 0.0

    def masked_key(self) -> str:
        """Returns masked API key for secure logging and API responses."""
        if not self.api_key:
            return "(empty)"
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def get_model(self) -> str:
        """Returns the primary model to use for this provider."""
        if self.default_model:
            return self.default_model
        if self.model_list:
            return self.model_list[0]
        return "gpt-4o-mini"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "masked_key": self.masked_key(),
            "base_url": self.base_url or "https://api.openai.com/v1 (default OpenAI)",
            "active_model": self.get_model(),
            "model_list": self.model_list,
            "call_count": self.call_count,
        }


class RoundRobinProviderManager:
    """
    Manages a pool of LLM API configurations loaded from environment variables.
    Provides thread-safe round-robin rotation across multiple API keys,
    each with its own base_url, default model, and supported model list.
    """

    def __init__(self, env_path: Optional[str] = None, env_dict: Optional[Dict[str, str]] = None):
        self.env_source: Dict[str, str] = {}
        if env_dict is not None:
            self.env_source = dict(env_dict)
        else:
            if env_path:
                load_dotenv(dotenv_path=env_path, override=True)
            else:
                load_dotenv(override=True)
            self.env_source = dict(os.environ)

        self.providers: List[ProviderConfig] = self._load_providers()
        if not self.providers:
            raise ValueError(
                "No API keys found in environment! "
                "Please configure API_KEY1, API_KEY2, etc. in your .env file."
            )

        self._lock = threading.Lock()
        self._current_index = 0
        self._total_requests = 0

    def _get_val(self, key: str) -> Optional[str]:
        return self.env_source.get(key)

    def _load_providers(self) -> List[ProviderConfig]:
        """
        Dynamically finds all API_KEY{i} or API_KEY_{i} entries in environment variables.
        Pairs them with corresponding BASE_URL{i}, MODEL{i}, MODEL_LIST{i}, etc.
        """
        indices = set()
        pattern = re.compile(r"^API_KEY_?(\d+)$", re.IGNORECASE)

        for key in self.env_source:
            match = pattern.match(key.strip())
            if match:
                indices.add(int(match.group(1)))

        sorted_indices = sorted(list(indices))
        providers: List[ProviderConfig] = []

        for idx in sorted_indices:
            # 1. Fetch API Key (e.g. API_KEY1 or API_KEY_1)
            api_key = self._get_val(f"API_KEY{idx}") or self._get_val(f"API_KEY_{idx}")
            if not api_key or not api_key.strip():
                continue
            api_key = api_key.strip()

            # 2. Fetch Base URL (optional)
            base_url = (
                self._get_val(f"BASE_URL{idx}")
                or self._get_val(f"BASE_URL_{idx}")
                or self._get_val(f"OPENAI_BASE_URL{idx}")
                or self._get_val(f"OPENAI_BASE_URL_{idx}")
            )
            if base_url:
                base_url = base_url.strip()

            # 3. Fetch Model and Model List
            raw_model_list = (
                self._get_val(f"MODEL_LIST{idx}")
                or self._get_val(f"MODEL_LIST_{idx}")
                or self._get_val(f"MODELS{idx}")
                or self._get_val(f"MODELS_{idx}")
            )
            model_list: List[str] = []
            if raw_model_list:
                model_list = [m.strip() for m in raw_model_list.split(",") if m.strip()]

            default_model = (
                self._get_val(f"MODEL{idx}")
                or self._get_val(f"MODEL_{idx}")
                or self._get_val(f"OPENAI_MODEL{idx}")
                or self._get_val(f"OPENAI_MODEL_{idx}")
            )
            if default_model:
                default_model = default_model.strip()
            elif model_list:
                default_model = model_list[0]
            else:
                default_model = "gpt-4o-mini"

            if default_model and default_model not in model_list:
                model_list.insert(0, default_model)

            # 4. Fetch Provider Name / Label (optional, for clean logging)
            name = (
                self._get_val(f"PROVIDER_NAME{idx}")
                or self._get_val(f"PROVIDER_NAME_{idx}")
                or self._get_val(f"NAME{idx}")
                or self._get_val(f"NAME_{idx}")
            )
            if not name:
                if base_url:
                    if "groq.com" in base_url.lower():
                        name = f"Groq (API #{idx})"
                    elif "openrouter.ai" in base_url.lower():
                        name = f"OpenRouter (API #{idx})"
                    elif "localhost" in base_url.lower() or "127.0.0.1" in base_url:
                        name = f"Local/Ollama (API #{idx})"
                    elif "deepseek.com" in base_url.lower():
                        name = f"DeepSeek (API #{idx})"
                    else:
                        name = f"Custom Provider (API #{idx})"
                else:
                    name = f"OpenAI (API #{idx})"

            provider = ProviderConfig(
                index=idx,
                name=name,
                api_key=api_key,
                base_url=base_url if base_url else None,
                default_model=default_model,
                model_list=model_list,
            )
            providers.append(provider)

        return providers

    def get_provider_count(self) -> int:
        return len(self.providers)

    def mark_request_started(self, provider: ProviderConfig) -> None:
        """Record when a request was dispatched to a provider."""
        with self._lock:
            provider.last_request_at = time.monotonic()

    def mark_rate_limited(self, provider: ProviderConfig, retry_after: float = 3.0) -> None:
        """Temporarily cool down a provider after a 429 rate-limit response."""
        cooldown = max(1.0, min(float(retry_after), 30.0))
        with self._lock:
            now = time.monotonic()
            provider.cooldown_until = max(provider.cooldown_until, now + cooldown)

    def get_next_provider(self) -> ProviderConfig:
        """
        Returns the next healthy ProviderConfig in round-robin sequence (1 -> 2 -> ... -> N -> 1),
        skipping providers currently in a rate-limit cooldown when another provider is ready.
        Thread-safe.
        """
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
                # All providers are cooling down; choose the one that recovers soonest.
                picked_idx = min(range(count), key=lambda idx: self.providers[idx].cooldown_until)

            provider = self.providers[picked_idx]
            self._total_requests += 1
            provider.call_count += 1
            self._current_index = (picked_idx + 1) % count
            return provider

    def create_chat_model(
        self,
        provider: ProviderConfig,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = True,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """
        Creates a ChatOpenAI instance configured with the provider's API key,
        base URL, streaming mode, and selected model.
        """
        chosen_model = model_name or provider.get_model()

        init_kwargs = {
            "model": chosen_model,
            "api_key": provider.api_key,
            "temperature": temperature,
            "streaming": streaming,
            "max_retries": 0,
            "request_timeout": 90.0,
            **kwargs,
        }

        if provider.base_url:
            init_kwargs["base_url"] = provider.base_url
            if "ngrok" in provider.base_url.lower():
                headers = dict(init_kwargs.get("default_headers") or {})
                headers.setdefault("ngrok-skip-browser-warning", "true")
                init_kwargs["default_headers"] = headers

        # Ollama-specific options must never be sent to other OpenAI-compatible
        # providers (for example Mistral), which reject the non-standard field.
        provider_hint = f"{provider.name} {provider.base_url or ''}".lower()
        is_ollama = (
            "ollama" in provider_hint
            or (provider.base_url is not None and ":11434" in provider.base_url.lower())
        )
        if is_ollama:
            extra_body = dict(init_kwargs.get("extra_body") or {})
            extra_body.setdefault("options", {"num_ctx": 8192})
            init_kwargs["extra_body"] = extra_body

        return ChatOpenAI(**init_kwargs)

    def get_next_llm(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = True,
        **kwargs: Any,
    ) -> tuple[ChatOpenAI, ProviderConfig, int]:
        """
        Fetches the next provider in round-robin order and returns:
        (ChatOpenAI instance, ProviderConfig used, request_number)
        """
        provider = self.get_next_provider()
        request_num = self._total_requests
        llm = self.create_chat_model(
            provider,
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
            **kwargs,
        )
        return llm, provider, request_num

    def get_provider_by_index(
        self,
        index: int,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        streaming: bool = True,
        **kwargs: Any,
    ) -> tuple[ChatOpenAI, ProviderConfig, int]:
        """
        Fetches a specific provider by its index (e.g. from Open WebUI model selection)
        and increments request counters.
        """
        for p in self.providers:
            if p.index == index:
                with self._lock:
                    self._total_requests += 1
                    p.call_count += 1
                    req_num = self._total_requests
                llm = self.create_chat_model(
                    p,
                    model_name=model_name,
                    temperature=temperature,
                    streaming=streaming,
                    **kwargs,
                )
                return llm, p, req_num
        # Fallback to round robin
        return self.get_next_llm(model_name=model_name, temperature=temperature, streaming=streaming, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Returns the current pool status and metrics for the FastAPI server."""
        with self._lock:
            return {
                "total_providers": len(self.providers),
                "next_provider_index": self.providers[self._current_index].index,
                "total_requests_served": self._total_requests,
                "providers": [p.to_dict() for p in self.providers],
            }

    def print_summary(self):
        """Prints a summary table of all loaded providers."""
        print("\n" + "=" * 70)
        print(f"Loaded {len(self.providers)} API Provider(s) from .env (Round-Robin Pool)")
        print("=" * 70)
        for i, p in enumerate(self.providers):
            print(f" [{i+1}] Name        : {p.name}")
            print(f"     Key         : {p.masked_key()}")
            print(f"     Base URL    : {p.base_url or 'https://api.openai.com/v1 (default OpenAI)'}")
            print(f"     Active Model: {p.get_model()}")
            if len(p.model_list) > 1:
                print(f"     Model List  : {', '.join(p.model_list)}")
            print("-" * 70)
        print("=" * 70 + "\n")
