"""The shared half of an append-only JSONL log.

``state.py`` and ``activity.py`` both keep one: append a line per event, never
rewrite, so a process killed mid-write costs one event rather than the file.
Both eventually have to prune, and what they prune *by* differs — sessions carry
an old event forward to preserve a topic, activity simply drops anything outside
the window — so the selection stays in each module.

What does not differ is the rewrite, and that is the part worth having in one
place: **write beside the target, then ``os.replace``**. The replace is atomic,
so a concurrent reader sees either the whole old log or the whole new one and
never a half-written file. Rewriting an append-only log in place is exactly how
append-only logs get corrupted, and two copies of that reasoning is one copy too
many — a fix to one would not reach the other.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: Compact past this size, not before. Reading and rewriting the whole log is
#: not free, and a log that has not grown has nothing to gain from it.
COMPACT_OVER_BYTES = 2_000_000


def is_large(target: Path, max_bytes: int = COMPACT_OVER_BYTES) -> bool:
    """Has this log actually got big enough to be worth rewriting?

    Never raises: a stat that fails means we cannot tell, and "do nothing" is
    the right answer to that on a housekeeping path.
    """
    try:
        return target.is_file() and target.stat().st_size > max_bytes
    except OSError:
        return False


def rewrite(target: Path, records: list) -> None:
    """Replace ``target`` with these records, atomically.

    ``records`` are anything with a ``to_json()``. The temporary file sits
    beside the target rather than in the system temp directory, because
    ``os.replace`` is only atomic within a filesystem.
    """
    temp = target.with_suffix(target.suffix + ".compacting")
    with open(temp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    os.replace(temp, target)
