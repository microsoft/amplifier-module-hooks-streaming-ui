"""Tests for intermediate text block rendering in streaming UI hooks.

Covers:
- All text lengths (short AND long) render via _text_renderable (Amplifier: + Markdown,
  no ▍ rail glyph) — the uniform parent-text renderable
- Empty text blocks are skipped
- Final-block exclusion: hook does NOT paint the FINAL parent response (#256 fix) —
  app-cli's render_message is the sole owner (single-owner contract)
- Both overlay_active=True and overlay_active=False skip the parent final
- Interleaved ASIDES (not final) are still painted and dimmed
- Sub-agent final still rendered attributed; sub-agent intermediate suppressed
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
# Final block: hook does NOT paint the final response (#256 fix)
# ---------------------------------------------------------------------------
#
# Changed in fix/256-hook-skip-final: handle_content_block_end SKIPS
# is_last_block=True for parent (agent_name=None) sessions.
# app-cli's render_message is the sole owner (single-owner pattern; avoids
# the #256 double-render that two-sided coordination caused).
# Old test test_final_block_IS_painted_by_hook encoded the reversed old-#22
# behavior and has been replaced with test_final_block_NOT_painted_by_hook.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_block_NOT_painted_by_hook(capsys):
    """Text that is the LAST block (end_turn) is NOT painted by the hook.

    Changed in fix/256-hook-skip-final: handle_content_block_end now skips
    is_last_block=True for parent sessions. app-cli's render_message is the
    sole owner of the final-response paint (single owner; fixes #256 double-render).
    (Previously named test_final_block_IS_painted_by_hook — behavior reversed.)
    """
    hooks = _hooks()
    # block_index=0, total_blocks=1 means this is the ONLY block (final response)
    data = _text_block_end_event("Final answer.", block_index=0, total_blocks=1)

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Hook must NOT paint the final block — app-cli's render_message owns it
    assert "Final answer." not in output, (
        f"Final text block must NOT be painted by hook (#256); got: {output!r}"
    )


@pytest.mark.asyncio
async def test_parent_final_not_painted_overlay_inactive(capsys):
    """Parent final (is_last_block=True, agent_name=None) skipped when overlay_active=False."""
    hooks = _hooks(overlay_active=False)
    data = _text_block_end_event(
        "Response text overlay-off.", block_index=0, total_blocks=1
    )

    await hooks.handle_content_block_end("content_block:end", data)

    out = capsys.readouterr().out
    assert "Response text overlay-off." not in out, (
        f"Hook must skip parent final block (overlay_active=False); got: {out!r}"
    )


@pytest.mark.asyncio
async def test_parent_final_not_painted_overlay_active(capsys):
    """Parent final (is_last_block=True, agent_name=None) skipped when overlay_active=True."""
    hooks = _hooks(overlay_active=True)
    data = _text_block_end_event(
        "Response text overlay-on.", block_index=0, total_blocks=1
    )

    await hooks.handle_content_block_end("content_block:end", data)

    out = capsys.readouterr().out
    assert "Response text overlay-on." not in out, (
        f"Hook must skip parent final block (overlay_active=True); got: {out!r}"
    )


@pytest.mark.asyncio
async def test_parent_aside_still_painted(capsys):
    """Parent interleaved aside (is_last_block=False) is still painted and dimmed."""
    hooks = _hooks()
    # block_index=0, total_blocks=2 → is_last_block=False (aside)
    data = _text_block_end_event(
        "Aside text before tool call.", block_index=0, total_blocks=2
    )

    await hooks.handle_content_block_end("content_block:end", data)

    out = capsys.readouterr().out
    assert "Aside text before tool call." in out, (
        f"Parent aside must still be painted; got: {out!r}"
    )
    assert "Amplifier:" in out


@pytest.mark.asyncio
async def test_sub_agent_final_still_painted_attributed(capsys):
    """Sub-agent final (agent_name set, is_last_block=True) is still painted attributed.

    Sub-agent handling is UNCHANGED by #256 fix. Only parent final is skipped.
    """
    hooks = _hooks()
    # block_index=0, total_blocks=1 → is_last_block=True; sub-agent session_id
    data = _text_block_end_event(
        "Sub-agent result text.",
        block_index=0,
        total_blocks=1,
        session_id="0000000000000000-7cc787dd22d54f6c_foundation:explorer",
    )

    await hooks.handle_content_block_end("content_block:end", data)

    out = capsys.readouterr().out
    assert "Sub-agent result text." in out, (
        f"Sub-agent final block must still be painted; got: {out!r}"
    )
    assert "[foundation:explorer]" in out, (
        f"Sub-agent attribution must still appear; got: {out!r}"
    )


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
# Sub-agent rendering (Change A: full-width dimmed Markdown with [agent] label)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_agent_intermediate_text_painted(capsys):
    """Sub-agent intermediate text (not last block) is NOW PAINTED attributed.

    Previously (curation commit 35fddbc / feat/subagent-curated change 4) this
    test asserted suppression.  The regression has been restored: intermediate
    asides render dim + attributed, same as the final block.  Streaming
    (token-level) output remains suppressed via the overlay _on_delta pass —
    this only fires at content_block:end once the block is fully settled.
    """
    hooks = _hooks()
    # Sub-agent session IDs contain an underscore followed by the agent name.
    # total_blocks=2, block_index=0 → NOT last block (is_last_block=False).
    data = _text_block_end_event(
        "Checking the module structure.",
        session_id="0000000000000000-7cc787dd22d54f6c_foundation:explorer",
    )

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Intermediate aside must now render with attribution.
    assert "[foundation:explorer]" in output, (
        f"Sub-agent intermediate aside must include attribution; got: {output!r}"
    )
    assert "Checking the module structure." in output, (
        f"Sub-agent intermediate text must appear in output; got: {output!r}"
    )
    assert captured.err == "", (
        f"No stderr expected for sub-agent intermediate text; got: {captured.err!r}"
    )


@pytest.mark.asyncio
async def test_sub_agent_final_text_attributed(capsys):
    """Sub-agent FINAL text (last block) is rendered with [agent_name] attribution.

    feat/subagent-fullwidth-and-spinner (Change A): when is_last_block=True,
    the sub-agent result is painted with a dim-cyan [agent_name] label above
    full-width dimmed Markdown — no ▍ rail, no 4-space indent.
    """
    hooks = _hooks()
    # block_index=0, total_blocks=1 → is_last_block=True
    data = _text_block_end_event(
        "The answer is 42.",
        block_index=0,
        total_blocks=1,
        session_id="0000000000000000-7cc787dd22d54f6c_foundation:explorer",
    )

    await hooks.handle_content_block_end("content_block:end", data)

    captured = capsys.readouterr()
    output = captured.out
    # Attribution header must appear
    assert "[foundation:explorer]" in output, (
        f"Attribution header missing from sub-agent final result; got: {output!r}"
    )
    # Result text must appear
    assert "The answer is 42." in output, (
        f"Result text missing from sub-agent final result; got: {output!r}"
    )


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
