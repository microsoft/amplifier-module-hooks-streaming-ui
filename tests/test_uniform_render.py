"""TDD tests for uniform Amplifier:+Markdown rendering for all parent text.

These tests assert the behaviour introduced in feat/uniform-assistant-render
AND updated in fix/256-hook-skip-final:

  1. _text_renderable: body is full-width Markdown (no ▍ rail glyph).
  2. _paint_interleaved_text parent (agent_name=None): output has "Amplifier:"
     + markdown text, NO ▍.  Sub-agent path still uses [agent_name] label (unchanged).
  3. handle_content_block_end: FINAL text block (is_last_block=True, parent,
     not in _overlay_painted_text) is NOT painted by the hook (#256 fix).
     app-cli's render_message is the sole owner. Interleaved asides are still painted.
  4. Overlay-painted interleaved aside (index in _overlay_painted_text) is
     still skipped by handle_content_block_end (de-dup guard unchanged).

Section 3 (TestFinalBlockNotPaintedByHook, formerly TestFinalBlockPaintedByHook)
was inverted in fix/256-hook-skip-final to encode the new single-owner contract.
"""

from __future__ import annotations

import io

import amplifier_module_hooks_streaming_ui as _mod
import pytest
from amplifier_core import HookResult
from amplifier_module_hooks_streaming_ui import StreamingUIHooks
from rich.console import Console


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hooks(**overrides) -> StreamingUIHooks:
    defaults = {"show_thinking": True, "show_tool_lines": 5, "show_token_usage": True}
    defaults.update(overrides)
    return StreamingUIHooks(**defaults)


def _render_text_renderable(content: str, width: int = 80) -> str:
    """Render _text_renderable to a string via a non-terminal Console."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False).print(
        _mod._text_renderable(content)
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. _text_renderable — body is full-width Markdown (no ▍)
# ---------------------------------------------------------------------------


class TestTextRenderableUniform:
    """Body must be Markdown, not a rail renderable."""

    def test_body_has_no_rail_glyph(self):
        """_text_renderable must NOT contain ▍ — body is now full-width Markdown."""
        out = _render_text_renderable("hello world")
        assert "▍" not in out, (
            f"_text_renderable must not contain ▍ (Markdown body); got: {out!r}"
        )

    def test_body_contains_content_text(self):
        """Content text must be present in the rendered output."""
        out = _render_text_renderable("some important text")
        assert "some important text" in out

    def test_label_amplifier_still_present(self):
        """The bold-green 'Amplifier:' label must still appear."""
        out = _render_text_renderable("hello world")
        assert "Amplifier:" in out

    def test_label_precedes_content(self):
        """'Amplifier:' label must appear before the content text."""
        out = _render_text_renderable("the body text")
        assert out.index("Amplifier:") < out.index("the body text")

    def test_no_whisper_glyph(self):
        """▸ whisper glyph must never appear."""
        out = _render_text_renderable("any text")
        assert "▸" not in out


# ---------------------------------------------------------------------------
# 2. _paint_interleaved_text — parent uses uniform renderable
# ---------------------------------------------------------------------------


class TestPaintInterleavedUniform:
    """Parent (agent_name=None) must use _text_renderable; sub-agent uses rail."""

    def test_parent_has_amplifier_label(self, capsys):
        """Parent path must output 'Amplifier:' label."""
        hooks = _hooks()
        hooks._paint_interleaved_text("Let me check that.", None)
        out = capsys.readouterr().out
        assert "Amplifier:" in out, (
            f"Parent _paint_interleaved_text must contain 'Amplifier:'; got: {out!r}"
        )

    def test_parent_contains_text_content(self, capsys):
        """Parent path must output the content text."""
        hooks = _hooks()
        hooks._paint_interleaved_text("analysis content", None)
        out = capsys.readouterr().out
        assert "analysis content" in out

    def test_parent_no_rail_glyph(self, capsys):
        """Parent path must NOT contain ▍ — uses full-width Markdown."""
        hooks = _hooks()
        hooks._paint_interleaved_text("some text here", None)
        out = capsys.readouterr().out
        assert "▍" not in out, (
            f"Parent _paint_interleaved_text must not use ▍ rail glyph; got: {out!r}"
        )

    def test_parent_no_whisper_glyph(self, capsys):
        """Parent path must never contain ▸."""
        hooks = _hooks()
        hooks._paint_interleaved_text("Short.", None)
        out = capsys.readouterr().out
        assert "▸" not in out

    def test_sub_agent_uses_agent_label_not_rail(self, capsys):
        """Sub-agent path (agent_name set) must use [agent_name] label, NOT ▍ rail.

        After Change A (feat/subagent-fullwidth-and-spinner): sub-agent final result
        renders as full-width dimmed Markdown with a dim-cyan [agent_name] label —
        no rail, no 4-space indent, no 52-col wrap.
        """
        hooks = _hooks()
        hooks._paint_interleaved_text("Checking structure.", "foundation:explorer")
        out = capsys.readouterr().out
        # Must have [agent_name] label
        assert "[foundation:explorer]" in out, (
            f"Sub-agent must have [foundation:explorer] label; got: {out!r}"
        )
        # Must NOT use rail glyph
        assert "▍" not in out, (
            f"Sub-agent must NOT use ▍ rail glyph after Change A; got: {out!r}"
        )
        # Must contain the content
        assert "Checking structure." in out

    def test_sub_agent_no_amplifier_label(self, capsys):
        """Sub-agent path must NOT emit 'Amplifier:' — uses [agent_name] label."""
        hooks = _hooks()
        hooks._paint_interleaved_text("some detail", "foundation:explorer")
        out = capsys.readouterr().out
        assert "Amplifier:" not in out, (
            f"Sub-agent output should not contain 'Amplifier:'; got: {out!r}"
        )

    def test_parent_has_trailing_blank_line(self, capsys):
        """Parent output must end with at least one blank line (separator)."""
        hooks = _hooks()
        hooks._paint_interleaved_text("content", None)
        out = capsys.readouterr().out
        # Must end with a newline (the print() call after _text_renderable)
        assert out.endswith("\n"), f"Parent output must end with newline; got: {out!r}"

    def test_parent_leading_blank_line_from_text_renderable(self, capsys):
        """_text_renderable's leading blank is part of the output (no extra blank)."""
        hooks = _hooks()
        hooks._paint_interleaved_text("some text", None)
        out = capsys.readouterr().out
        # Output should start with a blank line (from Text("") inside _text_renderable)
        assert out.startswith("\n"), (
            f"Parent output should start with blank line from _text_renderable; got: {out!r}"
        )


# ---------------------------------------------------------------------------
# 3. handle_content_block_end — final block NOT painted by hook (#256)
# ---------------------------------------------------------------------------
#
# Changed in fix/256-hook-skip-final: the hook no longer paints the parent final
# response. app-cli's render_message is the single owner (single-owner pattern;
# avoids the #256 double-render).  The old "TestFinalBlockPaintedByHook" class
# encoded the now-reversed old-#22 behavior.
# ---------------------------------------------------------------------------


class TestFinalBlockNotPaintedByHook:
    """handle_content_block_end must NOT paint the FINAL parent text block (#256).

    app-cli's render_message is the sole owner. The hook paints interleaved
    ASIDES only; it skips is_last_block=True for parent (agent_name=None) sessions.
    Both overlay_active=False and overlay_active=True paths must skip.
    (Previously named TestFinalBlockPaintedByHook -- behavior inverted by #256 fix.)
    """

    @pytest.mark.asyncio
    async def test_final_text_block_is_NOT_painted(self, capsys):
        """block_index=0, total_blocks=1 (final, only block) must NOT be painted.

        Changed in fix/256-hook-skip-final: hook skips is_last_block for parent.
        Old test asserted the opposite (test_final_text_block_is_painted).
        """
        hooks = _hooks()
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "Final answer here."},
        }
        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        out = capsys.readouterr().out
        assert "Final answer here." not in out, (
            f"Hook must NOT paint the final text block (#256); got: {out!r}"
        )

    @pytest.mark.asyncio
    async def test_final_text_block_not_painted_overlay_active(self, capsys):
        """Same skip when overlay_active=True (both overlay paths must skip final)."""
        hooks = _hooks(overlay_active=True)
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "Final overlay answer."},
        }
        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        out = capsys.readouterr().out
        assert "Final overlay answer." not in out, (
            f"Hook must skip parent final when overlay_active=True; got: {out!r}"
        )

    @pytest.mark.asyncio
    async def test_intermediate_block_is_painted(self, capsys):
        """Intermediate text block (not in overlay_painted_text) must still be painted."""
        hooks = _hooks()
        # block_index=0, total_blocks=2 → intermediate (not last)
        data = {
            "block_index": 0,
            "total_blocks": 2,
            "block": {"type": "text", "text": "Intermediate aside."},
        }
        await hooks.handle_content_block_end("content_block:end", data)
        out = capsys.readouterr().out
        assert "Intermediate aside." in out
        assert "Amplifier:" in out
        assert "▍" not in out

    @pytest.mark.asyncio
    async def test_overlay_painted_aside_still_skipped(self, capsys):
        """A block already in _overlay_painted_text must still be skipped (de-dup guard)."""
        hooks = _hooks()
        sid = "test-session"
        # Pre-populate the overlay-painted set for block 0
        hooks._overlay_painted_text[sid] = {0}

        data = {
            "block_index": 0,
            "total_blocks": 2,
            "session_id": sid,
            "block": {"type": "text", "text": "Should be skipped."},
        }
        await hooks.handle_content_block_end("content_block:end", data)
        out = capsys.readouterr().out
        assert "Should be skipped." not in out, (
            f"Overlay-painted block must be skipped; got: {out!r}"
        )
        # Index should be consumed (discarded from set)
        assert 0 not in hooks._overlay_painted_text.get(sid, set()), (
            "Block index must be discarded after skip"
        )
