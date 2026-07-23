# FastEmbed API verification

Verified against Qdrant's official FastEmbed documentation on 2026-07-23.

## Confirmed

- Install package: `fastembed`.
- Dense text entry point: `from fastembed import TextEmbedding`.
- Construction accepts a supported `model_name`; `.embed(list[str])` returns a
  generator of NumPy arrays.
- `BAAI/bge-small-zh-v1.5` is in the official supported-model list with
  dimension 512 and an approximately 0.09 GB model payload.
- FastEmbed uses ONNX Runtime and quantized model files.
- Current FastEmbed supports a `local_files_only` option passed through to the
  underlying Hugging Face download layer; after an initial cache population it
  can guarantee no network attempt.

## Project integration consequence

- The provider adapter must turn the generator into concrete `list[float]`
  values and validate the returned dimension before touching sqlite-vec.
- The configured cache directory must be passed to the FastEmbed constructor
  and must live under `/workspace/data/models`.
- Normal startup should attempt to load the cached model. A cache miss or model
  initialization failure must disable the vector channel and keep FTS active.
- Image build installs the Python/ONNX dependency, but model weights remain in
  the persistent runtime volume. No `pip install` occurs at startup.

## Sources

- https://qdrant.github.io/fastembed/
- https://qdrant.github.io/fastembed/examples/Supported_Models/
- https://github.com/qdrant/fastembed
