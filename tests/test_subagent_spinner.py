"""Tests for Change A (_text_renderable generalization) and Change B (sub-agent spinner).

Change A:
  - _text_renderable with label/label_style args: contains [x], dim, full-width Markdown, NO ▍
  - Default call (parent): byte-identical to before (existing parent tests pass)

Change B:
  - Counter: handle_tool_pre delegate → count 1; second → 2; handle_tool_post → 1 → 0
  - Non-spawn tools don't change counter
  - Spinner does NOT start when stdout is not a TTY
  - Spinner is reset/stopped on handle_render_end
  - No exception or corruption when not a TTY (counter still works headlessly)
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from amplifier_module_hooks_streaming_ui import StreamingUIHooks

import amplifier_module_hooks_streaming_ui as _mod
from rich.console import Console


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hooks(**overrides) -> StreamingUIHooks:
    defaults = {"show_thinking": True, "show_tool_lines": 5, "show_token_usage": True}
    defaults.update(overrides)
    return StreamingUIHooks(**defaults)


def _render_text_renderable(
    content: str,
    *,
    dim: bool = False,
    label: str = "Amplifier:",
    label_style: str | None = None,
    width: int = 80,
) -> str:
    """Render _text_renderable to a string via a non-terminal Console."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False).print(
        _mod._text_renderable(content, dim=dim, label=label, label_style=label_style)
    )
    return buf.getvalue()


# ===========================================================================
# Change A: _text_renderable generalization
# ===========================================================================


class TestTextRenderableGeneralized:
    """_text_renderable with custom label/label_style (Change A)."""

    def test_custom_label_appears_in_output(self):
        """A custom label like '[x]' must appear in the output."""
        out = _render_text_renderable("some text", label="[x]")
        assert "[x]" in out, f"Expected '[x]' in output; got: {out!r}"

    def test_agent_label_dim_cyan_no_rail_glyph(self):
        """label='[agent]', label_style='dim cyan', dim=True: contains [agent], no ▍."""
        out = _render_text_renderable(
            "result text",
            dim=True,
            label="[foundation:explorer]",
            label_style="dim cyan",
        )
        assert "[foundation:explorer]" in out, (
            f"Expected '[foundation:explorer]' in output; got: {out!r}"
        )
        assert "result text" in out, f"Expected content in output; got: {out!r}"
        assert "▍" not in out, f"Rail glyph ▍ must never appear; got: {out!r}"

    def test_custom_label_precedes_content(self):
        """Custom label must appear before the content text in the output."""
        out = _render_text_renderable(
            "body content", label="[my-agent]", label_style="dim cyan"
        )
        assert out.index("[my-agent]") < out.index("body content"), (
            f"Label must precede content; got: {out!r}"
        )

    def test_default_call_uses_amplifier_label(self):
        """Default call (no label kwarg) still uses 'Amplifier:' — parent unchanged."""
        out = _render_text_renderable("hello")
        assert "Amplifier:" in out, f"Default label must be 'Amplifier:'; got: {out!r}"
        assert "▍" not in out

    def test_default_call_bright_uses_bold_green(self):
        """Default call with dim=False → label_style 'bold green' (parent path)."""
        buf = io.StringIO()
        Console(file=buf, width=80, force_terminal=True, color_system="standard").print(
            _mod._text_renderable("test")
        )
        out = buf.getvalue()
        # bold green → ANSI \x1b[1m ... \x1b[32m or combined
        # We just verify "Amplifier:" is present and no rail glyph
        assert "Amplifier:" in out
        assert "▍" not in out

    def test_default_call_dim_uses_dim_green(self):
        """Default call with dim=True → label_style 'dim green' (parent aside path)."""
        out = _render_text_renderable("dimmed", dim=True)
        assert "Amplifier:" in out
        assert "▍" not in out

    def test_leading_blank_line_present(self):
        """_text_renderable's leading Text('') produces a blank line at the start."""
        out = _render_text_renderable("text here", label="[x]")
        assert out.startswith("\n"), (
            f"Output must start with blank line from Text(''); got: {out!r}"
        )


# ===========================================================================
# Change A: sub-agent final via _paint_interleaved_text
# ===========================================================================


class TestSubAgentPaintInterleavedChangeA:
    """After Change A, sub-agent final renders as full-width dim Markdown."""

    def test_subagent_contains_agent_label(self, capsys):
        """Sub-agent final output must contain [agent_name] label."""
        hooks = _hooks()
        hooks._paint_interleaved_text("The final answer.", "foundation:explorer")
        out = capsys.readouterr().out
        assert "[foundation:explorer]" in out, (
            f"Expected '[foundation:explorer]' label; got: {out!r}"
        )

    def test_subagent_contains_text(self, capsys):
        """Sub-agent final output must contain the actual text."""
        hooks = _hooks()
        hooks._paint_interleaved_text("The final answer.", "foundation:explorer")
        out = capsys.readouterr().out
        assert "The final answer." in out

    def test_subagent_no_rail_glyph(self, capsys):
        """Sub-agent final must NOT use ▍ rail glyph — uses full-width Markdown."""
        hooks = _hooks()
        hooks._paint_interleaved_text("sub result", "my-agent")
        out = capsys.readouterr().out
        assert "▍" not in out, (
            f"Sub-agent final must not use ▍ after Change A; got: {out!r}"
        )

    def test_subagent_no_4_space_indent_on_label(self, capsys):
        """Sub-agent label line must NOT start with 4 spaces (no indent anymore)."""
        hooks = _hooks()
        hooks._paint_interleaved_text("result", "my-agent")
        out = capsys.readouterr().out
        label_lines = [ln for ln in out.split("\n") if "[my-agent]" in ln]
        assert label_lines, f"No label line found; got: {out!r}"
        for ln in label_lines:
            assert not ln.startswith("    "), (
                f"Label must not be 4-space indented; got: {ln!r}"
            )

    def test_subagent_no_amplifier_label(self, capsys):
        """Sub-agent output must NOT contain 'Amplifier:' (parent label)."""
        hooks = _hooks()
        hooks._paint_interleaved_text("some text", "some-agent")
        out = capsys.readouterr().out
        assert "Amplifier:" not in out

    def test_subagent_has_trailing_blank(self, capsys):
        """Sub-agent output must end with at least one newline (trailing blank)."""
        hooks = _hooks()
        hooks._paint_interleaved_text("text", "agent")
        out = capsys.readouterr().out
        assert out.endswith("\n"), f"Output must end with newline; got: {out!r}"


# ===========================================================================
# Change B: spinner counter logic
# ===========================================================================


class TestSpinnerCounter:
    """Spinner counter increments/decrements correctly for delegate/task tools."""

    @pytest.mark.asyncio
    async def test_delegate_pre_increments_counter(self):
        """handle_tool_pre with tool_name='delegate' increments _subagents_running."""
        hooks = _hooks()
        assert hooks._subagents_running == 0

        await hooks.handle_tool_pre(
            "tool:pre",
            {"tool_name": "delegate", "tool_input": {}, "session_id": None},
        )
        assert hooks._subagents_running == 1, (
            f"Expected count=1 after delegate:pre; got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_task_pre_increments_counter(self):
        """handle_tool_pre with tool_name='task' also increments _subagents_running."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {"tool_name": "task", "tool_input": {}, "session_id": None},
        )
        assert hooks._subagents_running == 1

    @pytest.mark.asyncio
    async def test_two_spawns_counter_reaches_two(self):
        """Two delegate:pre calls push the counter to 2 (parallel spawns)."""
        hooks = _hooks()
        for _ in range(2):
            await hooks.handle_tool_pre(
                "tool:pre",
                {"tool_name": "delegate", "tool_input": {}, "session_id": None},
            )
        assert hooks._subagents_running == 2, (
            f"Expected count=2 after two delegate:pre; got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_delegate_post_decrements_counter(self):
        """handle_tool_post with tool_name='delegate' decrements _subagents_running."""
        hooks = _hooks()
        # Pre: count → 1
        await hooks.handle_tool_pre(
            "tool:pre",
            {"tool_name": "delegate", "tool_input": {}, "session_id": None},
        )
        assert hooks._subagents_running == 1
        # Post (suppress path): count → 0
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_response": {"output": {"response": "done"}},
                "session_id": None,
            },
        )
        assert hooks._subagents_running == 0, (
            f"Expected count=0 after delegate:post; got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_two_pre_one_post_leaves_count_one(self):
        """Two pre, one post → count=1 (not yet zero)."""
        hooks = _hooks()
        for _ in range(2):
            await hooks.handle_tool_pre(
                "tool:pre",
                {"tool_name": "delegate", "tool_input": {}, "session_id": None},
            )
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_response": {"output": {"response": "done"}},
                "session_id": None,
            },
        )
        assert hooks._subagents_running == 1, (
            f"Expected count=1; got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_non_spawn_tool_pre_does_not_change_counter(self):
        """tool:pre for non-spawn tools (bash, read_file, etc.) must not change counter."""
        hooks = _hooks()
        for tool in ("bash", "read_file", "write_file", "grep"):
            await hooks.handle_tool_pre(
                "tool:pre",
                {"tool_name": tool, "tool_input": {}, "session_id": None},
            )
        assert hooks._subagents_running == 0, (
            f"Non-spawn tools must not increment counter; got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_non_spawn_tool_post_does_not_change_counter(self):
        """tool:post for non-spawn tools must not change counter."""
        hooks = _hooks()
        hooks._subagents_running = 1  # set manually
        for tool in ("bash", "read_file"):
            await hooks.handle_tool_post(
                "tool:post",
                {"tool_name": tool, "tool_response": {"output": "x"}},
            )
        assert hooks._subagents_running == 1, (
            f"Non-spawn tool post must not decrement counter; got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_counter_never_goes_negative(self):
        """Counter must never go below 0 even if post arrives without a matching pre."""
        hooks = _hooks()
        assert hooks._subagents_running == 0
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_response": {"output": {"response": "x"}},
            },
        )
        assert hooks._subagents_running == 0, (
            f"Counter must not go negative; got {hooks._subagents_running}"
        )


# ===========================================================================
# Change B: TTY gate — spinner not started when not a TTY
# ===========================================================================


class TestSpinnerTTYGate:
    """Spinner Live must NOT be started when stdout is not a TTY."""

    @pytest.mark.asyncio
    async def test_spinner_not_started_when_not_tty(self):
        """_subagents_running increments correctly but spinner stays off in non-TTY."""
        hooks = _hooks()
        # _is_tty is set in __init__; during tests stdout is not a TTY
        assert not hooks._is_tty, "Test harness must have non-TTY stdout"

        await hooks.handle_tool_pre(
            "tool:pre",
            {"tool_name": "delegate", "tool_input": {}, "session_id": None},
        )

        assert hooks._subagents_running == 1
        # Spinner Live must NOT be created/started when not a TTY
        assert hooks._spinner_live is None or not hooks._spinner_live.is_started, (
            "Spinner must not start when stdout is not a TTY"
        )

    @pytest.mark.asyncio
    async def test_counter_works_without_spinner(self, capsys):
        """Counter increments/decrements correctly even without spinner (non-TTY path)."""
        hooks = _hooks()
        assert not hooks._is_tty

        # Full cycle: pre → post (suppressed) → count=0
        await hooks.handle_tool_pre(
            "tool:pre",
            {"tool_name": "delegate", "tool_input": {}, "session_id": None},
        )
        assert hooks._subagents_running == 1

        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_response": {"output": {"response": "done"}},
            },
        )
        assert hooks._subagents_running == 0

        # No exception, no corruption
        _ = capsys.readouterr()  # consume any output

    @pytest.mark.asyncio
    async def test_no_exception_or_corruption_in_non_tty(self, capsys):
        """Running the full sub-agent cycle in non-TTY mode raises no exceptions."""
        hooks = _hooks()

        # Simulate: parent delegate pre
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_input": {"prompt": "do stuff"},
                "session_id": None,
            },
        )
        # Simulate: sub-agent tool call (with sub-session ID)
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "bash",
                "tool_input": {"command": "ls"},
                "session_id": "parent-child_my-agent",
            },
        )
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "bash",
                "tool_response": {"returncode": 0, "stdout": "file.py", "stderr": ""},
                "session_id": "parent-child_my-agent",
            },
        )
        # Simulate: sub-agent final result
        from amplifier_core import HookResult

        result = await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {"type": "text", "text": "Done."},
                "session_id": "parent-child_my-agent",
            },
        )
        assert isinstance(result, HookResult)
        assert result.action == "continue"

        # Parent delegate post (suppressed)
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_response": {"output": {"response": "sub-agent done"}},
                "session_id": None,
            },
        )
        assert hooks._subagents_running == 0
        out = capsys.readouterr().out
        # Sub-agent final result must appear in output
        assert "Done." in out
        assert "[my-agent]" in out


# ===========================================================================
# Change B: handle_render_end resets spinner
# ===========================================================================


class TestSpinnerResetOnRenderEnd:
    """handle_render_end resets _subagents_running to 0 and stops the spinner."""

    @pytest.mark.asyncio
    async def test_render_end_resets_counter(self):
        """handle_render_end resets _subagents_running to 0."""
        hooks = _hooks()
        hooks._subagents_running = 2  # simulate orphaned sub-agents

        await hooks.handle_render_end("cleanup:render_end", {})
        assert hooks._subagents_running == 0, (
            f"handle_render_end must reset _subagents_running to 0; "
            f"got {hooks._subagents_running}"
        )

    @pytest.mark.asyncio
    async def test_render_end_stops_spinner(self):
        """handle_render_end stops the spinner if running."""
        hooks = _hooks()
        mock_live = MagicMock()
        mock_live.is_started = True
        hooks._spinner_live = mock_live
        hooks._subagents_running = 1

        await hooks.handle_render_end("cleanup:render_end", {})

        mock_live.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_render_end_no_error_when_no_spinner(self):
        """handle_render_end is safe even when no spinner has been created."""
        hooks = _hooks()
        assert hooks._spinner_live is None
        hooks._subagents_running = 0
        # Must not raise
        await hooks.handle_render_end("cleanup:render_end", {})
        assert hooks._subagents_running == 0
