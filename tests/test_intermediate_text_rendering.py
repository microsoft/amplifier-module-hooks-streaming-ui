"""Tests for intermediate text block rendering in streaming UI hooks.

Covers:
- All text lengths (short AND long) render via _text_renderable (Amplifier: + Markdown,
  no ▍ rail glyph) — the uniform parent-text renderable
- Empty text blocks are skipped
- Final-block inclusion: hook now paints the FINAL response via _text_renderable
- Sub-agent text still uses ▍ rail indentation (unchanged)
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
# Short text: uniform Amplifier: + Markdown (no ▍)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_text_renders_amplifier_label(capsys):
    """Short intermediate text must render with 'Amplifier:' label, no ▍ glyph."""
    hooks = _hooks()
    data = _text_block_end_event("Let me check that config.")

    result = await hooks.handle_content_block_end("content_block:end", data)

    assert isinstance(result, HookResult)
    assert result.action == "continue"

    captured = capsys.readouterr()
    output = captured.out
    # Uniform renderable: Amplifier: label + Markdown body, no rail glyph
    assert "Amplifier:" in output, f"Expected 'Amplifier:' in output; got: {output!r}"
    assert "▍" not in output, "Rail glyph ▍ must NOT appear — parent uses Markdown body"
    assert "▸" not in output, "Whisper glyph ▸ must never appear"
    assert "Let me check that config." in output


@pytest.mark.asyncio
async def test_two_line_text_renders_amplifier_label(capsys):
    """Two-line intermediate text must use Amplifier: + Markdown, not ▍."""
    hooks = _hooks()
    data = _text_block_end_event("Line one.\nLine two.")

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    assert "Amplifier:" in output, f"Expected 'Amplifier:' in output; got: {output!r}"
    assert "▍" not in output, "Rail glyph ▍ must NOT appear in parent output"
    assert "▸" not in output, "Whisper glyph ▸ must never appear"


# ---------------------------------------------------------------------------
# Long text: uniform Amplifier: + Markdown (no ▍)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_text_renders_amplifier_label(capsys):
    """Multi-paragraph intermediate text renders via Amplifier: + Markdown."""
    hooks = _hooks()
    text = "Line one of analysis.\n\nLine two continues.\n\nLine three concludes."
    data = _text_block_end_event(text)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    assert "Amplifier:" in output
    assert "▍" not in output, "No ▍ glyph for parent — uses full-width Markdown"
    assert "Line one of analysis." in output
    assert "Line two continues." in output
    assert "Line three concludes." in output
    assert "▸" not in output


@pytest.mark.asyncio
async def test_five_line_text_renders_amplifier_label(capsys):
    """Five paragraphs should render via Amplifier: + Markdown (not rail)."""
    hooks = _hooks()
    text = "\n\n".join(f"Analysis line {i + 1}." for i in range(5))
    data = _text_block_end_event(text)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    assert "Amplifier:" in output
    assert "▍" not in output
    assert "▸" not in output


# ---------------------------------------------------------------------------
# Final block: hook now PAINTS the final response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_block_IS_painted_by_hook(capsys):
    """Text that is the LAST block (end_turn) IS now painted by the hook.

    Changed in feat/uniform-assistant-render: handle_content_block_end no longer
    skips is_last_block text.  The hook owns the final-response paint via
    _paint_interleaved_text → _text_renderable (Amplifier: + Markdown).
    """
    hooks = _hooks()
    # block_index=0, total_blocks=1 means this is the ONLY block (final response)
    data = _text_block_end_event("Final answer.", block_index=0, total_blocks=1)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Hook must paint the final block with the uniform renderable
    assert "Final answer." in output, (
        f"Final text block must be painted by hook; got: {output!r}"
    )
    assert "Amplifier:" in output, "Final block must use uniform Amplifier: label"
    assert "▍" not in output, "Final block must use Markdown body (no ▍)"
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
    # No output at all for empty blocks
    assert "▸" not in output
    assert "▍" not in output
    assert "Amplifier:" not in output


# ---------------------------------------------------------------------------
# Sub-agent indentation (unchanged — still uses ▍ rail)
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
    # Output should start with a newline (blank line before, from Text("") in _text_renderable)
    assert output.startswith("\n")
    # Output should end with a newline (blank line after)
    assert output.rstrip("\n") != output  # has trailing newline


@pytest.mark.asyncio
async def test_long_single_line_renders_amplifier_label(capsys):
    """A single raw line that wraps renders via Amplifier: + Markdown, no rail glyph.

    The render uses full terminal width Markdown, not the narrow 60-char rail wrapper.
    """
    hooks = _hooks()
    long_line = "This is a very long sentence that should definitely wrap around " * 8
    data = _text_block_end_event(long_line.strip())

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Should use uniform Amplifier: + Markdown — no rail glyph
    assert "Amplifier:" in output, "Expected 'Amplifier:' for long wrapped text"
    assert "▍" not in output, "Rail glyph ▍ must not appear — parent uses Markdown"
    assert "▸" not in output, "Whisper glyph should not appear"
