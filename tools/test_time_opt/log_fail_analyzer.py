"""
Parse ATE / Tessent-style test logs and count PASS/FAIL per pattern.

Primary format (Advantest V93000 scan logs in this project):

  P12 | CH3 EXPECTED_OUTPUT:...
           ACTUAL_OUTPUT:...
           STATUS:F

Log pattern labels are 1-based (P1..P1000). Verilumen / STIL ids are
0-based (0..999), so Pn maps to pattern_id n-1.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Advantest pattern header: "P12 | CH3 ..."
_ATE_PAT_LINE = re.compile(r"^P(\d+)\s*\|", re.IGNORECASE)
_ATE_STATUS = re.compile(r"STATUS\s*:\s*([FP])\b", re.IGNORECASE)

# Legacy / alternate text formats
_PAT_STATUS = re.compile(
    r"(?:"
    r"pattern[_\s]*(?:numer|number|id|num)?\s*[:=#]?\s*"
    r"|pat(?:tern)?\s*[#:]?\s*"
    r"|p\s*"
    r")"
    r"(\d{1,5})"
    r"(?!\d)"
    r".{0,80}?"
    r"\b(fail(?:ed|ure|s)?|pass(?:ed)?)\b",
    re.IGNORECASE,
)
_STATUS_PAT = re.compile(
    r"\b(fail(?:ed|ure|s)?|pass(?:ed)?)\b"
    r".{0,80}?"
    r"(?:"
    r"pattern[_\s]*(?:numer|number|id|num)?\s*[:=#]?\s*"
    r"|pat(?:tern)?\s*[#:]?\s*"
    r"|p\s*"
    r")"
    r"(\d{1,5})"
    r"(?!\d)",
    re.IGNORECASE,
)
_CSV_ROW = re.compile(
    r"^\s*(?:P|pat(?:tern)?[_\s]*)?(\d{1,5})\s*[,;\t]\s*(fail(?:ed)?|pass(?:ed)?|[FP])\b",
    re.IGNORECASE,
)
_FAIL_LIST = re.compile(
    r"(?:failed?\s+patterns?|failing\s+patterns?|fail\s+list)\s*[:=]\s*([0-9P,\s]+)",
    re.IGNORECASE,
)

_TEXT_EXTS = {
    ".log",
    ".txt",
    ".csv",
    ".tsv",
    ".out",
    ".rpt",
    ".report",
    ".sum",
    ".summary",
    ".dat",
    "",
}


def log_label_to_pattern_id(label: int) -> int:
    """Map log Pn (1-based) → STIL/Verilumen pattern_id (0-based)."""
    if label <= 0:
        return label
    return label - 1


def _is_fail_word(word: str) -> bool:
    w = word.lower()
    return w.startswith("fail") or w == "f"


def _add(
    counts: dict[int, dict[str, int]],
    pid: int,
    fail: bool,
    n: int = 1,
) -> None:
    bucket = counts.setdefault(pid, {"fails": 0, "passes": 0})
    if fail:
        bucket["fails"] += n
    else:
        bucket["passes"] += n


def parse_ate_v93000(text: str) -> dict[int, dict[str, int]] | None:
    """
    Parse Advantest-style channel STATUS lines.

    Per pattern in one die log:
      - fail if any channel STATUS:F
      - pass if at least one STATUS and no F
    Returns None if this does not look like the Advantest format.
    """
    cur_label: int | None = None
    # label -> has_fail / has_status
    has_fail: dict[int, bool] = {}
    has_status: dict[int, bool] = {}
    status_hits = 0

    for line in text.splitlines():
        m = _ATE_PAT_LINE.match(line.strip())
        if m:
            cur_label = int(m.group(1))
            continue
        sm = _ATE_STATUS.search(line)
        if not sm or cur_label is None:
            continue
        status_hits += 1
        has_status[cur_label] = True
        if sm.group(1).upper() == "F":
            has_fail[cur_label] = True

    if status_hits == 0:
        return None

    counts: dict[int, dict[str, int]] = {}
    for label in has_status:
        pid = log_label_to_pattern_id(label)
        _add(counts, pid, bool(has_fail.get(label)))
    return counts


def parse_legacy_text(text: str) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {}
    seen: set[tuple[int, bool, int]] = set()

    def record(raw_id: int, fail: bool, line_no: int, one_based: bool = False) -> None:
        pid = log_label_to_pattern_id(raw_id) if one_based else raw_id
        # If ids look 1-based (no zero, max ~1000), still map when prefixed with P
        key = (pid, fail, line_no)
        if key in seen:
            return
        seen.add(key)
        _add(counts, pid, fail)

    for line_no, line in enumerate(text.splitlines()):
        for m in _PAT_STATUS.finditer(line):
            # "P12 FAIL" style → treat as 1-based label when line has P-prefix near id
            one_based = bool(re.search(rf"\bP\s*{m.group(1)}\b", line, re.I))
            record(int(m.group(1)), _is_fail_word(m.group(2)), line_no, one_based)
        for m in _STATUS_PAT.finditer(line):
            one_based = bool(re.search(rf"\bP\s*{m.group(2)}\b", line, re.I))
            record(int(m.group(2)), _is_fail_word(m.group(1)), line_no, one_based)
        for m in _CSV_ROW.finditer(line):
            record(int(m.group(1)), _is_fail_word(m.group(2)), line_no, True)
        for m in _FAIL_LIST.finditer(line):
            for pid_m in re.finditer(r"\d+", m.group(1)):
                record(int(pid_m.group(0)), True, line_no, True)

    return counts


def parse_log_text(text: str) -> dict[int, dict[str, int]]:
    ate = parse_ate_v93000(text)
    if ate is not None:
        return ate
    return parse_legacy_text(text)


def parse_log_file(path: Path) -> dict[int, dict[str, int]]:
    try:
        # Stream large logs; avoid loading issues on huge files
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    return parse_log_text(text)


def folder_key_from_rel(rel: str) -> str:
    parts = Path(rel.replace("\\", "/")).parts
    if len(parts) >= 2:
        return parts[0]
    return "(root)"


def analyze_uploaded_files(
    files: Iterable[tuple[str, Path]],
    selected_ids: list[int],
    discarded_ids: list[int],
) -> dict:
    selected_set = set(int(x) for x in selected_ids)
    discarded_set = set(int(x) for x in discarded_ids)
    known = selected_set | discarded_set

    per_pattern: dict[int, dict] = defaultdict(
        lambda: {
            "fails": 0,
            "passes": 0,
            "by_folder": defaultdict(lambda: {"fails": 0, "passes": 0}),
            "files": [],
        }
    )
    folders_seen: set[str] = set()
    files_parsed = 0
    files_skipped = 0
    files_with_status = 0

    for rel, abs_path in files:
        folders_seen.add(folder_key_from_rel(rel))
        if not abs_path.is_file():
            files_skipped += 1
            continue
        # skip manifest
        if abs_path.name.startswith("_"):
            files_skipped += 1
            continue

        file_counts = parse_log_file(abs_path)
        files_parsed += 1
        if not file_counts:
            continue
        files_with_status += 1
        folder = folder_key_from_rel(rel)
        for pid, cp in file_counts.items():
            row = per_pattern[pid]
            row["fails"] += cp["fails"]
            row["passes"] += cp["passes"]
            row["by_folder"][folder]["fails"] += cp["fails"]
            row["by_folder"][folder]["passes"] += cp["passes"]
            if cp["fails"] or cp["passes"]:
                row["files"].append(
                    {
                        "path": rel.replace("\\", "/"),
                        "fails": cp["fails"],
                        "passes": cp["passes"],
                    }
                )

    def serialize_rows(ids: list[int]) -> list[dict]:
        out = []
        for pid in ids:
            row = per_pattern.get(pid)
            fails = int(row["fails"]) if row else 0
            passes = int(row["passes"]) if row else 0
            out.append(
                {
                    "pattern_id": pid,
                    "fails": fails,
                    "passes": passes,
                    "total": fails + passes,
                    "fail_rate": round(fails / max(fails + passes, 1), 4),
                    "by_folder": {
                        k: dict(v) for k, v in (row["by_folder"].items() if row else [])
                    },
                    "files": (row["files"][:20] if row else []),
                }
            )
        out.sort(key=lambda r: (-r["fails"], r["pattern_id"]))
        return out

    kept_rows = serialize_rows(list(selected_ids))
    discarded_rows = serialize_rows(list(discarded_ids))

    unmatched = []
    for pid, row in sorted(per_pattern.items(), key=lambda kv: -kv[1]["fails"]):
        if pid in known:
            continue
        if row["fails"] == 0 and row["passes"] == 0:
            continue
        unmatched.append(
            {
                "pattern_id": pid,
                "fails": row["fails"],
                "passes": row["passes"],
            }
        )

    kept_fails = sum(r["fails"] for r in kept_rows)
    disc_fails = sum(r["fails"] for r in discarded_rows)

    return {
        "folders": sorted(folders_seen),
        "n_folders": len(folders_seen),
        "files_parsed": files_parsed,
        "files_with_status": files_with_status,
        "files_skipped": files_skipped,
        "id_mapping": "log_Pn -> pattern_id n-1 (1-based logs, 0-based STIL)",
        "summary": {
            "kept_n": len(selected_ids),
            "discarded_n": len(discarded_ids),
            "kept_with_fails": sum(1 for r in kept_rows if r["fails"] > 0),
            "discarded_with_fails": sum(1 for r in discarded_rows if r["fails"] > 0),
            "kept_total_fails": kept_fails,
            "discarded_total_fails": disc_fails,
            "all_total_fails": kept_fails + disc_fails,
            "unmatched_patterns": len(unmatched),
            "unmatched_fails": sum(u["fails"] for u in unmatched),
        },
        "kept": kept_rows,
        "discarded": discarded_rows,
        "kept_failing": [r for r in kept_rows if r["fails"] > 0],
        "discarded_failing": [r for r in discarded_rows if r["fails"] > 0],
        "unmatched": unmatched[:100],
    }


def analyze_dirs(
    dirs: list[Path],
    selected_ids: list[int],
    discarded_ids: list[int],
) -> dict:
    files: list[tuple[str, Path]] = []
    for d in dirs:
        d = d.resolve()
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                rel = f"{d.name}/{p.relative_to(d).as_posix()}"
                files.append((rel, p))
    return analyze_uploaded_files(files, selected_ids, discarded_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze pattern fails in log folders")
    ap.add_argument("dirs", nargs="+", type=Path, help="Log folders")
    ap.add_argument("--selected", type=str, default="[]")
    ap.add_argument("--discarded", type=str, default="[]")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    selected = json.loads(args.selected)
    discarded = json.loads(args.discarded)
    result = analyze_dirs(args.dirs, selected, discarded)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
