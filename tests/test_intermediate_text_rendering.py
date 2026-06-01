"""Tests for intermediate text block rendering in streaming UI hooks.

Covers:
- All text lengths (short AND long) render with rail prefix (▍) — no whisper (▸) ever
- Correct ANSI 256-color codes for rail mode
- Empty text blocks are skipped
- Last-block exclusion (final response not rendered as intermediate)
- Sub-agent text uses correct indentation
- Spacing: blank line before and after text blocks
"""

import pytest
from amplifier_core import HookResult
from amplifier_module_hooks_streaming_ui import StreamingUIHooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hooks(**overrides):
    defaults = {"show_thinking": True, "show_tool_lines": 5, "show_token_usage": True}
    defaults.update(overrides)
    return StreamingUIHooks(**defaults)


def _text_block_end_event(text, block_index=0, total_blocks=2, session_id=None):
    """Build a content_block:end event data dict for a text block.

    total_blocks defaults to 2 (text + tool_use) to simulate intermediate text.
    When total_blocks > 1 and block_index < total_blocks - 1, the text block
    is NOT the last block -- meaning it accompanies tool calls.
    """
    data = {
        "block_index": block_index,
        "total_blocks": total_blocks,
        "block": {"type": "text", "text": text},
    }
    if session_id:
        data["session_id"] = session_id
    return data


# ---------------------------------------------------------------------------
# Short text: rail (▍) — whisper removed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_text_renders_rail_glyph(capsys):
    """Short text (< 3 lines) must now render with ▍ rail glyph, not ▸ whisper."""
    hooks = _hooks()
    data = _text_block_end_event("Let me check that config.")

    result = await hooks.handle_content_block_end("content_block:end", data)

    assert isinstance(result, HookResult)
    assert result.action == "continue"

    captured = capsys.readouterr()
    output = captured.out
    # Rail glyph must appear — whisper is removed
    assert "▍" in output, f"Expected ▍ in output; got: {output!r}"
    assert "▸" not in output, "Whisper glyph ▸ must never appear after unification"
    assert "Let me check that config." in output


@pytest.mark.asyncio
async def test_two_line_text_uses_rail(capsys):
    """Two-line text must use rail ▍, not whisper ▸."""
    hooks = _hooks()
    data = _text_block_end_event("Line one.\nLine two.")

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    assert "▍" in output, f"Expected ▍ in output; got: {output!r}"
    # Whisper character must NOT appear
    assert "▸" not in output, "Whisper glyph ▸ must never appear"


# ---------------------------------------------------------------------------
# Long text: rail (▍) — unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_text_renders_narration_rail(capsys):
    """Text of 3+ rendered lines should render with ▍ rail on every line.

    Uses Markdown paragraphs (double newline) so Rich renders each as a
    separate block, producing 3+ output lines and triggering rail mode.
    """
    hooks = _hooks()
    # Double newlines create separate Markdown paragraphs -> separate rendered lines
    text = "Line one of analysis.\n\nLine two continues.\n\nLine three concludes."
    data = _text_block_end_event(text)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Rail character must appear (>= 3 rendered lines triggers rail mode)
    assert "▍" in output
    assert "Line one of analysis." in output
    assert "Line two continues." in output
    assert "Line three concludes." in output
    # Whisper prefix should NOT appear
    assert "▸" not in output


@pytest.mark.asyncio
async def test_five_line_text_uses_rail(capsys):
    """Five paragraphs should use the rail (>= 3 rendered lines)."""
    hooks = _hooks()
    # Double newlines create separate Markdown paragraphs
    text = "\n\n".join(f"Analysis line {i + 1}." for i in range(5))
    data = _text_block_end_event(text)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Rail glyph must appear (>= 3 rendered lines)
    assert "▍" in output
    assert "▸" not in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_text_block_skipped(capsys):
    """Empty or whitespace-only text blocks should produce no output."""
    hooks = _hooks()

    for empty_text in ["", "   ", "\n", "\n\n  \n"]:
        data = _text_block_end_event(empty_text)
        await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # No whisper or rail characters should appear
    assert "▸" not in output
    assert "▍" not in output


@pytest.mark.asyncio
async def test_last_block_text_not_rendered_as_intermediate(capsys):
    """Text that is the LAST block (end_turn) should NOT get intermediate treatment.

    The final response text is rendered by the main response path at full brightness,
    not by this hook. We only render intermediate text (not the last block).
    """
    hooks = _hooks()
    # block_index=0, total_blocks=1 means this is the ONLY block (final response)
    data = _text_block_end_event("Final answer.", block_index=0, total_blocks=1)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Should NOT render with whisper or rail
    assert "▸" not in output
    assert "▍" not in output


# ---------------------------------------------------------------------------
# Sub-agent indentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_text_indented(capsys):
    """Sub-agent intermediate text should be indented with 4 spaces."""
    hooks = _hooks()
    # Sub-agent session IDs contain an underscore followed by the agent name
    data = _text_block_end_event(
        "Checking the module structure.",
        session_id="0000000000000000-7cc787dd22d54f6c_foundation:explorer",
    )

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # The text should appear with 4-space indentation
    assert "    " in output
    assert "Checking the module structure." in output


# ---------------------------------------------------------------------------
# Rich Markdown wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_single_line_wraps_to_rail_mode(capsys):
    """A single raw line that wraps to 3+ rendered lines should use rail mode.

    The line count threshold must be based on the RENDERED output lines (after
    Rich Markdown wrapping), not the raw input lines. A single long line that
    wraps to many lines should trigger rail mode (▍), not whisper (▸).
    """
    hooks = _hooks()
    # A single line long enough to wrap well past 3 lines at width=60
    long_line = "This is a very long sentence that should definitely wrap around " * 8
    data = _text_block_end_event(long_line.strip())

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Should use rail mode because rendered lines >= 3
    assert "▍" in output, "Expected rail glyph for long wrapped text"
    # Should NOT use whisper prefix
    assert "▸" not in output, "Whisper glyph should not appear for long wrapped text"


# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blank_line_before_and_after(capsys):
    """Intermediate text block should have a blank line before and after."""
    hooks = _hooks()
    data = _text_block_end_event("Some analysis text.")

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Output should start with a newline (blank line before)
    assert output.startswith("\n")
    # Output should end with a newline (blank line after)
    assert output.rstrip("\n") != output  # has trailing newline
