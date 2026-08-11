"""Benchmark bge-small-zh-v1.5 embedding on CPU vs GPU (runs inside the image)."""

from __future__ import annotations

import time
from pathlib import Path

from fastembed import TextEmbedding


TEXTS = [
    "阿渣喜欢什么动画",
    "昨天我在群里说了什么",
    "帮我总结一下今天的群聊",
    "大蛇丸是哪个球队的粉丝",
    "你记得我上次约了什么时间吗",
    "这张图里画的是什么",
    "台风白海豚什么时候登陆",
    "晚上吃什么比较好",
] * 8


def _find_cache_dir() -> str:
    for base in ("/workspace/data", "/root/.cache"):
        root = Path(base)
        if root.exists():
            hits = sorted(root.glob("**/bge-small-zh-v1.5"))
            if hits:
                return str(hits[0].parent.parent)
    return "/workspace/data/embed_models"


def bench(providers: list[str], label: str) -> None:
    cache_dir = _find_cache_dir()
    started = time.perf_counter()
    model = TextEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_dir=cache_dir,
        providers=providers,
    )
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    list(model.embed([TEXTS[0]]))
    single_seconds = time.perf_counter() - started

    started = time.perf_counter()
    list(model.embed(TEXTS))
    batch_seconds = time.perf_counter() - started

    print(
        f"{label}: load={load_seconds:.2f}s single={single_seconds*1000:.1f}ms "
        f"batch64={batch_seconds:.2f}s ({len(TEXTS)/batch_seconds:.0f} texts/s)",
        flush=True,
    )


if __name__ == "__main__":
    bench(["CPUExecutionProvider"], "cpu")
    bench(["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda")
