from __future__ import annotations

import json
from pathlib import Path


def load_export(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("history export must be a list of message objects")
    return data
