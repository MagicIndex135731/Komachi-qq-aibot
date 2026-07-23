from __future__ import annotations

import json
import math
import threading
import time

import httpx

from app.providers.semantic_embeddings import (
    DisabledEmbeddingProvider,
    LocalFastEmbedProvider,
    OpenAICompatibleEmbeddingProvider,
)


def test_disabled_provider_has_stable_identity_and_never_embeds() -> None:
    provider = DisabledEmbeddingProvider(dimensions=512, version="v2")

    assert provider.identity.provider == "disabled"
    assert provider.identity.dimensions == 512
    assert provider.embed_query("历史问题") is None
    assert provider.embed_documents(["一", "二"]) is None


def test_local_fastembed_is_lazy_uses_cache_dir_and_supports_query_and_documents(tmp_path) -> None:
    constructed: list[dict[str, object]] = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)

        def embed(self, texts: list[str]):
            return iter([[float(index), 0.5] for index, _text in enumerate(texts)])

    provider = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path / "models",
        version="model-v1",
        embedding_class=FakeTextEmbedding,
    )

    assert constructed == []
    assert provider.identity.model == "test-model"
    assert provider.identity.version == "model-v1"
    assert provider.embed_query("问题") == [0.0, 0.5]
    assert provider.embed_documents(["文档一", "文档二"]) == [[0.0, 0.5], [1.0, 0.5]]
    assert constructed == [
        {
            "model_name": "test-model",
            "cache_dir": str(tmp_path / "models"),
            "providers": ["CPUExecutionProvider"],
            "local_files_only": False,
        }
    ]


def test_local_fastembed_can_require_an_existing_offline_cache(tmp_path) -> None:
    constructed: list[dict[str, object]] = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)

        def embed(self, texts: list[str]):
            return iter([[1.0, 0.0] for _ in texts])

    provider = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        local_files_only=True,
        embedding_class=FakeTextEmbedding,
    )

    assert provider.embed_query("query") == [1.0, 0.0]
    assert constructed[0]["local_files_only"] is True


def test_local_fastembed_auto_prefers_cuda_and_falls_back_to_cpu(tmp_path) -> None:
    constructed: list[list[str]] = []

    class ProbeEmbedding:
        def __init__(self, **kwargs) -> None:
            providers = list(kwargs["providers"])
            constructed.append(providers)
            self.providers = providers

        def embed(self, texts: list[str]):
            if self.providers[0] == "CUDAExecutionProvider":
                raise RuntimeError("cuda inference failed")
            return iter([[1.0, 0.0] for _ in texts])

    provider = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        device="auto",
        embedding_class=ProbeEmbedding,
        available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert provider.embed_query("query") == [1.0, 0.0]
    assert provider.available is True
    assert constructed == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_local_fastembed_explicit_cuda_fails_closed_when_provider_is_unavailable(tmp_path) -> None:
    provider = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        device="cuda",
        embedding_class=lambda **_kwargs: object(),
        available_providers=lambda: ["CPUExecutionProvider"],
    )

    assert provider.embed_query("query") is None
    assert provider.available is False


def test_local_fastembed_disables_vector_channel_after_initialization_or_vector_validation_failure(tmp_path) -> None:
    class BrokenTextEmbedding:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("model download failed")

    broken = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        embedding_class=BrokenTextEmbedding,
    )
    assert broken.embed_query("问题") is None
    assert broken.available is False

    class InvalidVectorTextEmbedding:
        def __init__(self, **_kwargs) -> None:
            pass

        def embed(self, _texts: list[str]):
            return iter([[1.0, math.nan]])

    invalid = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        embedding_class=InvalidVectorTextEmbedding,
    )
    assert invalid.embed_documents(["文档"]) is None
    assert invalid.available is False


def test_local_fastembed_uses_asymmetric_query_and_passage_methods_when_available(tmp_path) -> None:
    calls: list[tuple[str, list[str]]] = []

    class FakeTextEmbedding:
        def __init__(self, **_kwargs) -> None:
            pass

        def query_embed(self, texts: list[str]):
            calls.append(("query", texts))
            return iter([[1.0, 0.0]])

        def passage_embed(self, texts: list[str]):
            calls.append(("passage", texts))
            return iter([[0.0, 1.0] for _ in texts])

    provider = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        embedding_class=FakeTextEmbedding,
    )

    assert provider.embed_query("问题") == [1.0, 0.0]
    assert provider.embed_documents(["证据一", "证据二"]) == [[0.0, 1.0], [0.0, 1.0]]
    assert calls == [("query", ["问题"]), ("passage", ["证据一", "证据二"])]


def test_local_fastembed_serializes_concurrent_first_load_and_inference(tmp_path) -> None:
    constructed = 0
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    class ConcurrentProbeEmbedding:
        def __init__(self, **_kwargs) -> None:
            nonlocal constructed
            with state_lock:
                constructed += 1

        def embed(self, texts: list[str]):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return iter([[1.0, 0.0] for _ in texts])

    provider = LocalFastEmbedProvider(
        model="test-model",
        dimensions=2,
        cache_dir=tmp_path,
        embedding_class=ConcurrentProbeEmbedding,
    )
    barrier = threading.Barrier(3)
    results: list[object] = []

    def invoke_query() -> None:
        barrier.wait()
        results.append(provider.embed_query("query"))

    def invoke_documents() -> None:
        barrier.wait()
        results.append(provider.embed_documents(["document"]))

    query_thread = threading.Thread(target=invoke_query)
    document_thread = threading.Thread(target=invoke_documents)
    query_thread.start()
    document_thread.start()
    barrier.wait()
    query_thread.join()
    document_thread.join()

    assert constructed == 1
    assert max_active == 1
    assert len(results) == 2
    assert [1.0, 0.0] in results
    assert [[1.0, 0.0]] in results


def test_openai_compatible_provider_uses_finite_timeout_and_validates_response_indexes_and_dimensions() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://embedding.example.test/v1",
        api_key="test-secret-key",
        model="embedding-model",
        dimensions=2,
        version="remote-v1",
        timeout_seconds=3.0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.embed_documents(["一", "二"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert captured == {
        "url": "https://embedding.example.test/v1/embeddings",
        "authorization": "Bearer test-secret-key",
        "payload": {"model": "embedding-model", "input": ["一", "二"]},
    }
    assert provider.identity.provider == "openai_compatible"

    invalid_response = OpenAICompatibleEmbeddingProvider(
        base_url="https://embedding.example.test/v1",
        api_key="test-secret-key",
        model="embedding-model",
        dimensions=2,
        timeout_seconds=2.0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.1, 0.2]}]})
            )
        ),
    )
    assert invalid_response.embed_query("问题") is None
    assert invalid_response.available is False
