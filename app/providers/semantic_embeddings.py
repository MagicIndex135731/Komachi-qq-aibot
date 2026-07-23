from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import math
from pathlib import Path
import threading
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import httpx


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Immutable metadata used to version derived semantic indexes."""

    provider: str
    model: str
    version: str
    dimensions: int


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...

    @property
    def available(self) -> bool: ...

    def embed_query(self, query: str) -> list[float] | None: ...

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]] | None: ...


class DisabledEmbeddingProvider:
    """A deliberate vector-channel off switch; callers fall back to FTS."""

    def __init__(self, *, dimensions: int = 512, version: str = "") -> None:
        self._identity = EmbeddingIdentity(
            provider="disabled",
            model="",
            version=version,
            dimensions=max(1, int(dimensions)),
        )

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    @property
    def available(self) -> bool:
        return False

    def embed_query(self, query: str) -> list[float] | None:
        del query
        return None

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]] | None:
        del documents
        return None


class _ValidatedEmbeddingProvider:
    def __init__(self, *, identity: EmbeddingIdentity) -> None:
        self._identity = identity
        self._available = True

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    @property
    def available(self) -> bool:
        return self._available

    def _disable(self, *, error: BaseException) -> None:
        self._available = False
        logger.warning(
            "semantic_embedding_disabled provider=%s model=%s error=%s",
            self.identity.provider,
            self.identity.model,
            type(error).__name__,
        )

    def _validate_vectors(self, vectors: Sequence[object], *, expected_count: int) -> list[list[float]] | None:
        if len(vectors) != expected_count:
            self._disable(error=ValueError("unexpected embedding count"))
            return None

        validated: list[list[float]] = []
        try:
            for vector in vectors:
                values = [float(value) for value in vector]  # type: ignore[union-attr]
                if len(values) != self.identity.dimensions or not all(math.isfinite(value) for value in values):
                    raise ValueError("invalid embedding vector")
                validated.append(values)
        except (TypeError, ValueError, OverflowError) as exc:
            self._disable(error=exc)
            return None
        return validated


class LocalFastEmbedProvider(_ValidatedEmbeddingProvider):
    """Lazy FastEmbed adapter that leaves the vector channel disabled on failure."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        cache_dir: Path | str,
        device: str = "cpu",
        local_files_only: bool = False,
        version: str = "",
        embedding_class: Callable[..., Any] | None = None,
        available_providers: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        super().__init__(
            identity=EmbeddingIdentity(
                provider="local",
                model=model,
                version=version,
                dimensions=max(1, int(dimensions)),
            )
        )
        self.cache_dir = Path(cache_dir)
        self.local_files_only = local_files_only
        self.device = device.strip().lower()
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"unsupported embedding device: {device}")
        self._embedding_class = embedding_class
        self._available_providers = available_providers
        self._embedder: Any | None = None
        self._active_accelerator = "cpu"
        self._inference_lock = threading.RLock()

    @property
    def active_accelerator(self) -> str:
        return self._active_accelerator

    def _runtime_providers(self) -> list[str]:
        if self.device == "cpu":
            return ["CPUExecutionProvider"]
        loader = self._available_providers
        if loader is None:
            runtime = importlib.import_module("onnxruntime")
            loader = runtime.get_available_providers
        available = set(loader())
        if "CUDAExecutionProvider" in available:
            self._active_accelerator = "cuda"
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "cuda":
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        self._active_accelerator = "cpu"
        return ["CPUExecutionProvider"]

    def _construct_embedder(self, *, providers: Sequence[str]) -> Any:
        embedding_class = self._embedding_class
        if embedding_class is None:
            embedding_class = importlib.import_module("fastembed").TextEmbedding
        return embedding_class(
            model_name=self.identity.model,
            cache_dir=str(self.cache_dir),
            providers=list(providers),
            local_files_only=self.local_files_only,
        )

    def _replace_with_cpu_embedder(self, *, error: BaseException) -> Any | None:
        if self.device != "auto" or self._active_accelerator != "cuda":
            return None
        logger.warning(
            "semantic_embedding_cuda_fallback provider=%s model=%s error=%s",
            self.identity.provider,
            self.identity.model,
            type(error).__name__,
        )
        try:
            self._active_accelerator = "cpu"
            self._embedder = self._construct_embedder(providers=["CPUExecutionProvider"])
        except Exception as fallback_error:
            self._disable(error=fallback_error)
            return None
        return self._embedder

    def _ensure_embedder(self) -> Any | None:
        with self._inference_lock:
            if not self.available:
                return None
            if self._embedder is not None:
                return self._embedder
            try:
                self._embedder = self._construct_embedder(providers=self._runtime_providers())
            except Exception as exc:
                fallback = self._replace_with_cpu_embedder(error=exc)
                if fallback is not None:
                    return fallback
                self._disable(error=exc)
                return None
            logger.info(
                "semantic_embedding_runtime provider=%s model=%s accelerator=%s",
                self.identity.provider,
                self.identity.model,
                self._active_accelerator,
            )
            return self._embedder

    def _embed(self, *, method_names: Sequence[str], values: Sequence[str]) -> list[object] | None:
        embedder = self._ensure_embedder()
        if embedder is None:
            return None
        method = next((getattr(embedder, name, None) for name in method_names if getattr(embedder, name, None)), None)
        if method is None:
            self._disable(error=AttributeError("embedding method is unavailable"))
            return None
        try:
            return list(method(list(values)))
        except Exception as exc:
            fallback = self._replace_with_cpu_embedder(error=exc)
            if fallback is None:
                self._disable(error=exc)
                return None
            fallback_method = next(
                (getattr(fallback, name, None) for name in method_names if getattr(fallback, name, None)),
                None,
            )
            if fallback_method is None:
                self._disable(error=AttributeError("embedding method is unavailable"))
                return None
            try:
                return list(fallback_method(list(values)))
            except Exception as fallback_error:
                self._disable(error=fallback_error)
                return None

    def embed_query(self, query: str) -> list[float] | None:
        with self._inference_lock:
            vectors = self._embed(method_names=("query_embed", "embed"), values=[str(query)])
            if vectors is None:
                return None
            validated = self._validate_vectors(vectors, expected_count=1)
            return validated[0] if validated else None

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]] | None:
        values = [str(document) for document in documents]
        if not values:
            return []
        with self._inference_lock:
            vectors = self._embed(method_names=("passage_embed", "embed"), values=values)
            if vectors is None:
                return None
            return self._validate_vectors(vectors, expected_count=len(values))


class OpenAICompatibleEmbeddingProvider(_ValidatedEmbeddingProvider):
    """OpenAI embeddings endpoint adapter with strict response ordering."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        version: str = "",
        timeout_seconds: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            identity=EmbeddingIdentity(
                provider="openai_compatible",
                model=model,
                version=version,
                dimensions=max(1, int(dimensions)),
            )
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.http_client = http_client or httpx.Client(timeout=self.timeout_seconds, trust_env=False)
        if not self.base_url or not self.identity.model:
            self._disable(error=ValueError("embedding endpoint configuration is incomplete"))

    def embed_query(self, query: str) -> list[float] | None:
        vectors = self.embed_documents([query])
        return vectors[0] if vectors else None

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]] | None:
        values = [str(document) for document in documents]
        if not values:
            return []
        if not self.available:
            return None
        try:
            response = self.http_client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={"model": self.identity.model, "input": values},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload["data"] if isinstance(payload, dict) else None
            if not isinstance(data, list) or len(data) != len(values):
                raise ValueError("invalid embedding response shape")
            vectors: list[object] = []
            for expected_index, item in enumerate(data):
                if not isinstance(item, dict) or item.get("index") != expected_index:
                    raise ValueError("embedding response indexes are not sequential")
                vectors.append(item.get("embedding"))
        except (httpx.HTTPError, TypeError, ValueError, KeyError) as exc:
            self._disable(error=exc)
            return None
        return self._validate_vectors(vectors, expected_count=len(values))


def build_embedding_provider(
    *,
    provider: str,
    device: str = "cpu",
    model: str,
    dimensions: int,
    cache_dir: Path | str,
    local_files_only: bool = False,
    version: str = "",
    base_url: str = "",
    api_key: str = "",
    timeout_seconds: float = 10.0,
) -> EmbeddingProvider:
    """Build an optional provider without ever falling back to hashed vectors."""

    normalized = provider.strip().lower()
    if normalized == "local":
        return LocalFastEmbedProvider(
            model=model,
            dimensions=dimensions,
            cache_dir=cache_dir,
            device=device,
            local_files_only=local_files_only,
            version=version,
        )
    if normalized == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            version=version,
            timeout_seconds=timeout_seconds,
        )
    return DisabledEmbeddingProvider(dimensions=dimensions, version=version)
