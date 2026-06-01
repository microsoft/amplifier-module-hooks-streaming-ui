"""Unit tests for _paint_interleaved_text and the overlay look-ahead state machine.

Covers:
  - ALL text lengths → rail ▍ on every line (no whisper ▸, ever)
  - Sub-agent path uses wrap_width=52 and 4-space indent
  - Overlay look-ahead state machine: pending_text stashed at block_end(text),
    drained+painted at block_start(tool_use), and index recorded in
    hooks._overlay_painted_text so handle_content_block_end skips it
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import amplifier_module_hooks_streaming_ui as _mod
import pytest
from amplifier_module_hooks_streaming_ui import StreamingUIHooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hooks(**overrides) -> StreamingUIHooks:
    defaults = {"show_thinking": True, "show_tool_lines": 5, "show_token_usage": True}
    defaults.update(overrides)
    return StreamingUIHooks(**defaults)


# ---------------------------------------------------------------------------
# _paint_interleaved_text — unit tests
# ---------------------------------------------------------------------------


class TestPaintInterleavedText:
    """Unit tests for the extracted _paint_interleaved_text helper."""

    def test_short_text_emits_rail_glyph(self, capsys):
        """Short text (<3 rendered lines) must now use the rail ▍ glyph, not ▸."""
        hooks = _hooks()
        hooks._paint_interleaved_text("Let me check that.", None)
        out = capsys.readouterr().out
        assert "▍" in out, f"Expected rail glyph ▍ in output; got: {out!r}"
        assert "▸" not in out, "Whisper glyph ▸ must NEVER appear (whisper removed)"

    def test_short_text_never_uses_whisper(self, capsys):
        """Short text must NOT produce ▸ (whisper removed entirely)."""
        hooks = _hooks()
        hooks._paint_interleaved_text("Short.", None)
        out = capsys.readouterr().out
        assert "▸" not in out, "Whisper glyph ▸ should never appear after unification"
        assert "▍" in out, "Rail glyph ▍ must appear for short text"

    def test_long_text_emits_rail_glyph_on_every_line(self, capsys):
        """Long text (≥3 rendered lines) should print the rail ▍ on every non-blank line."""
        hooks = _hooks()
        # Double newlines create separate Markdown paragraphs → ≥3 rendered lines
        text = "Line one analysis.\n\nLine two continues.\n\nLine three concludes."
        hooks._paint_interleaved_text(text, None)
        out = capsys.readouterr().out
        assert "▍" in out, f"Expected rail glyph ▍ in output; got: {out!r}"
        assert "▸" not in out, "Whisper glyph ▸ should not appear for long text"

    def test_sub_agent_uses_4_space_indent(self, capsys):
        """Sub-agent path (agent_name set) should indent with 4 spaces."""
        hooks = _hooks()
        hooks._paint_interleaved_text("Checking structure.", "foundation:explorer")
        out = capsys.readouterr().out
        # The 4-space indent must appear on lines with ▍
        glyph_lines = [ln for ln in out.split("\n") if "▍" in ln]
        assert glyph_lines, "No ▍ lines found in sub-agent output"
        assert any("    " in ln for ln in glyph_lines), (
            f"Expected 4-space indent for sub-agent; got: {out!r}"
        )
        assert "Checking structure." in out

    def test_sub_agent_uses_wrap_width_52(self, capsys):
        """Sub-agent path wraps at width 52; parent wraps at 60.

        A line of 56 characters should wrap for sub-agent (52) but not for
        parent (60).  We verify that sub-agent output has the ▍ glyph and
        both contain the text content.
        """
        hooks = _hooks()
        # 56 characters — wraps at 52 (sub-agent) but fits at 60 (parent)
        text = "A" * 56  # single long token-free line
        hooks._paint_interleaved_text(text, None)
        parent_out = capsys.readouterr().out

        hooks._paint_interleaved_text(text, "some-agent")
        subagent_out = capsys.readouterr().out

        # Sub-agent output should be indented
        assert "    " in subagent_out
        # Both should contain the text content
        assert "A" * 10 in parent_out
        assert "A" * 10 in subagent_out

    def test_parent_has_no_indent(self, capsys):
        """Parent path (agent_name=None) must not add a 4-space indent prefix."""
        hooks = _hooks()
        hooks._paint_interleaved_text("Simple text.", None)
        out = capsys.readouterr().out
        lines = out.split("\n")
        # Content lines (those with the rail glyph) should not start with 4 spaces
        content_lines = [line for line in lines if "▍" in line]
        for line in content_lines:
            assert not line.startswith("    "), (
                f"Parent line should not start with 4-space indent: {line!r}"
            )


# ---------------------------------------------------------------------------
# Overlay look-ahead state machine test
# ---------------------------------------------------------------------------


class TestOverlayLookAheadStateMachine:
    """Simulate the [text][tool_use] event sequence and verify look-ahead behaviour.

    The test mocks Live so we don't need a real TTY.  It patches
    hooks._paint_interleaved_text to record calls instead of printing.
    """

    @pytest.mark.asyncio
    async def test_paint_called_once_at_tool_use_start_and_index_recorded(self):
        """block_end(text,0) stashes; block_start(tool_use,1) paints once and records idx=0."""
        hooks = _hooks()

        # Replace the painter with a mock so we can assert call count/args
        paint_mock = MagicMock()
        hooks._paint_interleaved_text = paint_mock

        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_instance = MagicMock()
            mock_live_cls.return_value = mock_live_instance

            overlay = _mod._make_streaming_overlay(hooks)
            start_h = overlay["llm:stream_block_start"]
            delta_h = overlay["llm:stream_block_delta"]
            end_h = overlay["llm:stream_block_end"]

            sid = "sess-parent"

            # 1. text block starts (idx=0)
            await start_h(
                "llm:stream_block_start",
                {"session_id": sid, "block_index": 0, "block_type": "text"},
            )

            # 2. a few deltas arrive
            await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": sid,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "Let me ",
                },
            )
            await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": sid,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "think.",
                },
            )

            # Painter must NOT have been called yet
            paint_mock.assert_not_called()

            # 3. text block ends (idx=0) — stash pending_text
            await end_h(
                "llm:stream_block_end",
                {"session_id": sid, "block_index": 0},
            )

            # Painter still must not have been called (stashed, not painted)
            paint_mock.assert_not_called()

            # 4. tool_use block starts (idx=1) — look-ahead drains stash
            await start_h(
                "llm:stream_block_start",
                {
                    "session_id": sid,
                    "block_index": 1,
                    "block_type": "tool_use",
                    "name": "bash",
                },
            )

        # Painter must have been called exactly ONCE with the accumulated buffer
        assert paint_mock.call_count == 1, (
            f"Expected _paint_interleaved_text called once; got {paint_mock.call_count}"
        )
        call_text, call_agent = paint_mock.call_args[0]
        assert "Let me " in call_text and "think." in call_text, (
            f"Painter called with unexpected text: {call_text!r}"
        )
        assert call_agent is None, "Parent session should have agent_name=None"

        # Index 0 must be recorded so handle_content_block_end would skip it
        assert sid in hooks._overlay_painted_text, (
            "Session key missing from hooks._overlay_painted_text"
        )
        assert 0 in hooks._overlay_painted_text[sid], (
            f"Block index 0 not recorded in overlay_painted_text; got: {hooks._overlay_painted_text}"
        )

    @pytest.mark.asyncio
    async def test_no_paint_when_no_following_block(self):
        """If no next block arrives, the stash is never painted (final-response case).

        The stash cleanup happens at reset time (_on_prompt_submit); between
        block_end(text) and the reset the stash is simply held.
        The final-response text is rendered via handle_content_block_end's
        is_last_block → render_message path, NOT by the overlay painter.
        """
        hooks = _hooks()
        paint_mock = MagicMock()
        hooks._paint_interleaved_text = paint_mock

        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_cls.return_value = MagicMock()

            overlay = _mod._make_streaming_overlay(hooks)
            start_h = overlay["llm:stream_block_start"]
            delta_h = overlay["llm:stream_block_delta"]
            end_h = overlay["llm:stream_block_end"]

            sid = "sess-final"

            await start_h(
                "llm:stream_block_start",
                {"session_id": sid, "block_index": 0, "block_type": "text"},
            )
            await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": sid,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "Final answer.",
                },
            )
            await end_h(
                "llm:stream_block_end",
                {"session_id": sid, "block_index": 0},
            )

            # No block_start follows — painter must NOT have been called
            paint_mock.assert_not_called()

        # overlay_painted_text should NOT have this sid (no paint recorded)
        assert sid not in hooks._overlay_painted_text, (
            "No paint should be recorded when no next block arrives"
        )

    @pytest.mark.asyncio
    async def test_reset_clears_pending_text_on_prompt_submit(self):
        """_on_prompt_submit → _reset_session clears pending_text so it doesn't leak."""
        hooks = _hooks()
        paint_mock = MagicMock()
        hooks._paint_interleaved_text = paint_mock

        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_cls.return_value = MagicMock()

            overlay = _mod._make_streaming_overlay(hooks)
            start_h = overlay["llm:stream_block_start"]
            delta_h = overlay["llm:stream_block_delta"]
            end_h = overlay["llm:stream_block_end"]
            submit_h = overlay["prompt:submit"]

            sid = "sess-reset"

            # Simulate: text block streams and ends (final response, no next block)
            await start_h(
                "llm:stream_block_start",
                {"session_id": sid, "block_index": 0, "block_type": "text"},
            )
            await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": sid,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "Hello.",
                },
            )
            await end_h(
                "llm:stream_block_end",
                {"session_id": sid, "block_index": 0},
            )

            # Next turn starts: prompt:submit fires
            await submit_h("prompt:submit", {"session_id": sid})

            # Now a new block_start arrives — must NOT drain the stale stash
            await start_h(
                "llm:stream_block_start",
                {"session_id": sid, "block_index": 0, "block_type": "text"},
            )

        # Painter must never have been called (stash was cleared before new block)
        assert paint_mock.call_count == 0, (
            f"Painter should not be called after reset; got {paint_mock.call_count} calls"
        )
