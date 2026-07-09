from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from PIL import Image

from .schemas import TaskPacket


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_hf_image(record: Dict[str, Any], output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_value = record.get("image")
    decoded = record.get("decoded_image")
    pid = str(record.get("pid") or record.get("sample_id") or "sample")

    if isinstance(image_value, str) and Path(image_value).exists():
        return str(Path(image_value).resolve())

    if isinstance(decoded, Image.Image):
        out = output_dir / f"{pid}.png"
        decoded.save(out)
        return str(out.resolve())

    if isinstance(image_value, dict) and image_value.get("bytes"):
        out = output_dir / f"{pid}.png"
        out.write_bytes(image_value["bytes"])
        return str(out.resolve())

    raise FileNotFoundError(
        "Could not materialize image from record. Provide a local image path or a decoded_image."
    )


def task_from_mathvista_record(record: Dict[str, Any], image_root: str | Path | None = None) -> TaskPacket:
    record = dict(record)
    if image_root and record.get("image"):
        candidate = Path(image_root) / str(record["image"])
        if candidate.exists():
            record["image_path"] = str(candidate.resolve())
    elif record.get("image") and Path(str(record["image"])).exists():
        record["image_path"] = str(Path(str(record["image"])).resolve())
    elif record.get("image_path"):
        record["image_path"] = str(Path(str(record["image_path"])).resolve())
    return TaskPacket.from_dict(record)
