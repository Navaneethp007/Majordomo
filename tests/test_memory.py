"""Tests for the memory layer — pure filesystem, no model, no network."""
from __future__ import annotations

from unittest import mock

import pytest

from majordomo import memory


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_slugify_lowercases_and_hyphenates():
    assert memory.slugify("Prefers Windows-native!") == "prefers-windows-native"


def test_slugify_never_truncates_mid_word():
    """A slug is the filename and the index label; 'later-co' reads like a bug."""
    slug = memory.slugify("builds for windows first and treats cross platform as a later concern")
    assert len(slug) <= 60
    assert not slug.endswith("-")
    # every segment is a whole word from the input
    assert all(part in "builds for windows first and treats cross platform as a later concern"
               for part in slug.split("-"))


def test_auto_name_uses_first_content_words():
    name = memory.auto_name("Builds for Windows first; treats cross-platform as a later concern.")
    assert name.startswith("builds")
    assert len(name.split("-")) <= 6


def test_slugify_falls_back_when_nothing_survives():
    assert memory.slugify("!!!") == "memory"


def test_score_counts_meaningful_overlap_only():
    m = memory.Memory(name="windows-first", description="Builds for Windows first", body="")
    # "for" and "the" are stopwords and must not inflate the score
    assert memory.score(m, "windows") == 1
    assert memory.score(m, "for the a an") == 0


def test_score_ignores_the_body():
    """Scoring on hidden text makes `relevant` unpredictable from the index."""
    m = memory.Memory(name="a", description="unrelated", body="kubernetes everywhere")
    assert memory.score(m, "kubernetes") == 0


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


def test_render_parse_round_trip():
    original = memory.Memory(
        name="x", description="A description", body="A body.", type="preference",
        created="2026-09-01",
    )
    parsed = memory.parse(memory.render(original))
    assert parsed == original


def test_a_dash_rule_in_a_field_does_not_split_the_file():
    """Splitting on the bare substring '---' silently mangles the memory.

    The description is cut in half and the remaining metadata spills into the
    body, so `type` and `created` vanish too — data loss with no error.
    """
    original = memory.Memory(
        name="x",
        description="use --- as a separator",
        body="the real body",
        type="reference",
        created="2026-09-01",
    )
    parsed = memory.parse(memory.render(original))

    assert parsed == original
    assert parsed.description == "use --- as a separator"
    assert parsed.body == "the real body"
    assert parsed.type == "reference"


def test_a_fence_line_inside_the_body_survives():
    original = memory.Memory(
        name="x", description="d", body="before\n---\nafter", created="2026-09-01"
    )
    assert memory.parse(memory.render(original)).body == "before\n---\nafter"


def test_a_description_that_is_only_a_fence_survives():
    original = memory.Memory(name="x", description="---", body="b", created="2026-09-01")
    assert memory.parse(memory.render(original)) == original


def test_colon_in_description_survives_the_round_trip():
    """Hand-written `description: {value}` breaks the moment there is a colon."""
    original = memory.Memory(
        name="x", description="Rule: always test on Windows: no exceptions",
        body="b", created="2026-09-01",
    )
    parsed = memory.parse(memory.render(original))
    assert parsed is not None
    assert parsed.description == "Rule: always test on Windows: no exceptions"


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter at all",
        "---\nname: [unclosed\n---\nbody",       # broken YAML
        "---\njust: a mapping\n---\nbody",       # no name/description
        "---\nname: x\n---\nbody",               # no description
        "---\n- a list\n---\nbody",              # frontmatter is not a mapping
        "---\nname: x\ndescription: '  '\n---\n",  # blank description
    ],
)
def test_parse_returns_none_rather_than_raising(text):
    """Reads never throw. A bad file costs one memory, not the briefing."""
    assert memory.parse(text) is None


def test_parse_strips_a_bom():
    text = "﻿" + memory.render(
        memory.Memory(name="x", description="d", body="b")
    )
    assert memory.parse(text) is not None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_then_read(tmp_path):
    written = memory.write_memory(
        memory.MemoryCandidate(description="Likes short answers", type="preference"),
        root=tmp_path,
    )
    assert written.type == "preference"
    assert memory.read_memory(written.name, root=tmp_path) == written


def test_body_defaults_to_the_description(tmp_path):
    written = memory.write_memory(
        memory.MemoryCandidate(description="One-liner"), root=tmp_path
    )
    assert written.body == "One-liner"


def test_unknown_type_falls_back_to_user(tmp_path):
    written = memory.write_memory(
        memory.MemoryCandidate(description="d", type="nonsense"), root=tmp_path
    )
    assert written.type == "user"


def test_name_collision_gets_a_suffix(tmp_path):
    a = memory.write_memory(memory.MemoryCandidate(description="Same words here"), root=tmp_path)
    b = memory.write_memory(
        memory.MemoryCandidate(description="Same words here"), root=tmp_path
    )
    assert a.name != b.name
    assert b.name.endswith("-2")


def test_overwrite_replaces_in_place(tmp_path):
    first = memory.write_memory(memory.MemoryCandidate(description="Old fact"), root=tmp_path)
    memory.write_memory(
        memory.MemoryCandidate(description="New fact"), root=tmp_path, overwrite=first.name
    )
    assert len(memory.read_all(tmp_path)) == 1
    assert memory.read_memory(first.name, root=tmp_path).description == "New fact"


def test_empty_description_is_refused(tmp_path):
    with pytest.raises(memory.MemoryError_):
        memory.write_memory(memory.MemoryCandidate(description="   "), root=tmp_path)


def test_overlong_description_is_refused(tmp_path):
    with pytest.raises(memory.MemoryError_, match="scannable"):
        memory.write_memory(
            memory.MemoryCandidate(description="x" * 300), root=tmp_path
        )


def test_overlong_body_is_refused(tmp_path):
    with pytest.raises(memory.MemoryError_, match="document"):
        memory.write_memory(
            memory.MemoryCandidate(description="d", body="x" * 5000), root=tmp_path
        )


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnop1234567890",
        "ghp_abcdefghijklmnopqrstuvwxyz1234",
        "xoxb-1234567890-abcdefghij",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_credentials_are_refused(tmp_path, secret):
    """A memory is replayed into every future conversation that matches it."""
    with pytest.raises(memory.MemoryError_, match="credential"):
        memory.write_memory(
            memory.MemoryCandidate(description="my key", body=f"use {secret}"),
            root=tmp_path,
        )


def test_ordinary_text_is_not_mistaken_for_a_secret(tmp_path):
    memory.write_memory(
        memory.MemoryCandidate(
            description="Uses OPENROUTER_API_KEY from the env, never inline"
        ),
        root=tmp_path,
    )
    assert len(memory.read_all(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Reading a directory
# ---------------------------------------------------------------------------


def test_corrupt_file_is_skipped_and_counted(tmp_path):
    memory.write_memory(memory.MemoryCandidate(description="Good one"), root=tmp_path)
    (tmp_path / "broken.md").write_text("---\nname: [unclosed\n---\nx", encoding="utf-8")

    result = memory.read_all_detailed(tmp_path)
    assert len(result.memories) == 1
    assert result.skipped == 1


def test_missing_directory_reads_empty(tmp_path):
    result = memory.read_all_detailed(tmp_path / "nope")
    assert result.memories == [] and result.skipped == 0


def test_index_is_excluded_from_the_memories(tmp_path):
    memory.write_memory(memory.MemoryCandidate(description="A fact"), root=tmp_path)
    assert (tmp_path / "INDEX.md").is_file()
    assert len(memory.read_all(tmp_path)) == 1


def test_index_stays_consistent_after_delete(tmp_path):
    a = memory.write_memory(memory.MemoryCandidate(description="First fact"), root=tmp_path)
    memory.write_memory(memory.MemoryCandidate(description="Second fact"), root=tmp_path)

    assert memory.delete_memory(a.name, root=tmp_path) is True

    index = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert a.name not in index
    assert "Second fact" in index
    assert a.name not in memory.read_index(tmp_path)


def test_delete_reports_a_miss(tmp_path):
    assert memory.delete_memory("never-existed", root=tmp_path) is False


def test_read_index_derives_from_files_not_the_stale_file(tmp_path):
    """INDEX.md is for a human to read; the prompt uses what is actually there."""
    memory.write_memory(memory.MemoryCandidate(description="Real fact"), root=tmp_path)
    (tmp_path / "INDEX.md").write_text("# lies\n\n- [ghost] not real\n", encoding="utf-8")

    index = memory.read_index(tmp_path)
    assert "ghost" not in index
    assert "Real fact" in index


def test_empty_index_says_so(tmp_path):
    assert "nothing remembered" in memory.read_index(tmp_path)


# ---------------------------------------------------------------------------
# Relevance and duplicates
# ---------------------------------------------------------------------------


def test_relevant_ranks_and_limits(tmp_path):
    memory.write_memory(
        memory.MemoryCandidate(description="Windows first, cross-platform later"),
        root=tmp_path,
    )
    memory.write_memory(
        memory.MemoryCandidate(description="Prefers concise answers"), root=tmp_path
    )

    hits = memory.relevant("windows", root=tmp_path)
    assert len(hits) == 1
    assert "Windows" in hits[0].description


def test_relevant_returns_nothing_on_no_overlap(tmp_path):
    memory.write_memory(memory.MemoryCandidate(description="Windows first"), root=tmp_path)
    assert memory.relevant("kubernetes helm charts", root=tmp_path) == []


def test_relevant_respects_the_limit(tmp_path):
    for i in range(5):
        memory.write_memory(
            memory.MemoryCandidate(description=f"Windows fact number {i}"), root=tmp_path
        )
    assert len(memory.relevant("windows", limit=2, root=tmp_path)) == 2


def test_find_similar_catches_a_near_duplicate(tmp_path):
    memory.write_memory(
        memory.MemoryCandidate(description="Builds for Windows first, cross-platform later"),
        root=tmp_path,
    )
    hit = memory.find_similar("He builds for Windows first; cross-platform later", root=tmp_path)
    assert hit is not None


def test_find_similar_ignores_an_unrelated_memory(tmp_path):
    memory.write_memory(
        memory.MemoryCandidate(description="Builds for Windows first"), root=tmp_path
    )
    assert memory.find_similar("Enjoys dark roast coffee", root=tmp_path) is None


def test_find_similar_on_an_empty_store(tmp_path):
    assert memory.find_similar("anything at all", root=tmp_path) is None


# ---------------------------------------------------------------------------
# An unwritable directory
# ---------------------------------------------------------------------------


def test_a_failed_write_raises_memory_error_not_oserror(tmp_path):
    """Callers catch MemoryError_, so a raw OSError escaped all of them — and
    it surfaces on the chat REPL's exit path and out of `mj remember`, the two
    places least able to afford a traceback."""
    with mock.patch.object(
        memory.Path, "write_text", side_effect=OSError("read-only file system")
    ):
        with pytest.raises(memory.MemoryError_) as exc_info:
            memory.write_memory(
                memory.MemoryCandidate(description="something"), root=tmp_path
            )

    assert "read-only file system" in str(exc_info.value)


def test_a_directory_that_cannot_be_created_raises_memory_error(tmp_path):
    with mock.patch.object(memory.Path, "mkdir", side_effect=OSError("denied")):
        with pytest.raises(memory.MemoryError_):
            memory.write_memory(
                memory.MemoryCandidate(description="something"),
                root=tmp_path / "nested",
            )


def test_a_failed_delete_raises_memory_error(tmp_path):
    memory.write_memory(memory.MemoryCandidate(description="a fact"), root=tmp_path)
    name = memory.read_all(tmp_path)[0].name

    with mock.patch.object(memory.Path, "unlink", side_effect=OSError("in use")):
        with pytest.raises(memory.MemoryError_):
            memory.delete_memory(name, root=tmp_path)


def test_deleting_something_absent_is_still_just_false(tmp_path):
    """Absent is not an error — only a failure to remove one that is there."""
    assert memory.delete_memory("never-existed", root=tmp_path) is False
