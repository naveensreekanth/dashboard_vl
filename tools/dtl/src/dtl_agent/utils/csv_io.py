"""CSV utilities supporting full load, header inspection, and chunked iteration."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"CSV has no header row: {path}")
    return list(header)


def iter_csv_dicts(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, str]]:
    """Yield row dicts. If ``columns`` is set, project to those keys only."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        for row in reader:
            if columns is None:
                yield dict(row)
            else:
                yield {c: row.get(c, "") for c in columns}


def load_csv_dicts(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    return list(iter_csv_dicts(path, columns=columns))


def iter_csv_chunks(
    path: Path,
    *,
    chunk_size: int = 50_000,
    columns: Sequence[str] | None = None,
) -> Iterator[list[dict[str, str]]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    chunk: list[dict[str, str]] = []
    for row in iter_csv_dicts(path, columns=columns):
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        return sum(1 for _ in reader)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
