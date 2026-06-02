"""Tests for sub-agent rendering — static per-spawn marker (replaces Live panel).

Change history:
  feat/per-agent-spinner-list: replaced integer counter with _active_spawns dict +
    named Live panel (TestSpawnToolsConfig, TestActiveSpawns, TestAgentLabelFallback,
    TestSpawnsRenderable, TestSpinnerTTYGate, TestSpinnerResetOnRenderEnd).

  Current (restore full sub-agent rendering): removed animated Live panel entirely.
    - _active_spawns, _spawn_counter, _spinner_live, _spinner_console, _is_tty
      are all DELETED from StreamingUIHooks.
    - _make_spawns_renderable, _start_or_update_spinner, _stop_spinner,
      _with_spinner_paused are all DELETED.
    - handle_tool_pre for spawn tools now prints a static dim ⏳ marker instead
      of starting the Live panel.

Removed test classes (obsolete — code they tested is deleted):
  - TestSpawnToolsConfig       (_active_spawns tracking gated by spawn_tools config)
  - TestActiveSpawns           (_active_spawns insertion-ordered dict lifecycle)
  - TestAgentLabelFallback     (_active_spawns label fallback logic)
  - TestSpawnsRenderable       (_make_spawns_renderable renderable content)
  - TestSpinnerTTYGate         (_spinner_live / _is_tty TTY-gate)
  - TestSpinnerResetOnRenderEnd(_active_spawns.clear() + _stop_spinner in render_end)

Kept:
  - TestTextRenderableGeneralized     (_text_renderable helper — still in use)
  - TestSubAgentPaintInterleavedChangeA (_paint_interleaved_text — still in use)

Added:
  - TestStaticSpawnMarker  (new static ⏳ marker behavior + absence of old attrs)
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

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
# Change A: _text_renderable generalization  (unchanged from prior test suite)
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
# Change A: sub-agent final via _paint_interleaved_text (unchanged)
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
# Static per-spawn marker: new behavior replacing the Live panel
# ===========================================================================


class TestStaticSpawnMarker:
    """handle_tool_pre for spawn tools prints a static ⏳ dim marker; no Live panel."""

    # --- Absence of old attributes ---

    def test_no_active_spawns_attribute(self):
        """StreamingUIHooks must NOT have _active_spawns (old Live-panel tracking removed)."""
        hooks = _hooks()
        assert not hasattr(hooks, "_active_spawns"), (
            "_active_spawns must be removed (replaced by static marker)"
        )

    def test_no_spinner_live_attribute(self):
        """StreamingUIHooks must NOT have _spinner_live."""
        hooks = _hooks()
        assert not hasattr(hooks, "_spinner_live"), "_spinner_live must be removed"

    def test_no_spawn_counter_attribute(self):
        """StreamingUIHooks must NOT have _spawn_counter."""
        hooks = _hooks()
        assert not hasattr(hooks, "_spawn_counter"), "_spawn_counter must be removed"

    def test_no_is_tty_attribute(self):
        """StreamingUIHooks must NOT have _is_tty (TTY-gate for spinner removed)."""
        hooks = _hooks()
        assert not hasattr(hooks, "_is_tty"), "_is_tty must be removed"

    def test_no_subagents_running_attribute(self):
        """StreamingUIHooks must NOT have _subagents_running (old integer counter)."""
        hooks = _hooks()
        assert not hasattr(hooks, "_subagents_running"), (
            "_subagents_running int counter must be absent"
        )

    def test_spawn_tools_still_present(self):
        """_spawn_tools config attribute must still exist (gates the static marker)."""
        hooks = _hooks()
        assert hasattr(hooks, "_spawn_tools"), "_spawn_tools must still exist"
        assert "delegate" in hooks._spawn_tools
        assert "task" in hooks._spawn_tools

    # --- Static marker content ---

    @pytest.mark.asyncio
    async def test_spawn_tool_pre_prints_hourglass_marker(self, capsys):
        """tool:pre for a spawn tool prints a ⏳ marker with the agent label."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_input": {"agent": "foundation:explorer"},
                "session_id": None,
            },
        )
        out = capsys.readouterr().out
        assert "⏳" in out, f"Expected ⏳ marker in output; got: {out!r}"
        assert "foundation:explorer" in out, (
            f"Expected agent label in marker; got: {out!r}"
        )
        assert "working" in out.lower(), f"Expected 'working' in marker; got: {out!r}"

    @pytest.mark.asyncio
    async def test_spawn_tool_pre_marker_uses_agent_label(self, capsys):
        """Static marker uses tool_input.agent as the label."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t1",
                "tool_input": {"agent": "zen-architect"},
                "session_id": None,
            },
        )
        out = capsys.readouterr().out
        assert "zen-architect" in out, (
            f"Agent label must appear in marker; got: {out!r}"
        )

    @pytest.mark.asyncio
    async def test_non_spawn_tool_pre_no_hourglass_marker(self, capsys):
        """Non-spawn tool:pre must NOT print a ⏳ marker."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "bash",
                "tool_call_id": "t1",
                "tool_input": {"command": "ls"},
                "session_id": None,
            },
        )
        out = capsys.readouterr().out
        assert "⏳" not in out, f"Non-spawn tool must not emit ⏳ marker; got: {out!r}"

    @pytest.mark.asyncio
    async def test_parallel_spawns_print_multiple_markers(self, capsys):
        """N parallel spawns print N static markers — one per spawn, in order."""
        hooks = _hooks()
        for agent in ("alpha", "beta"):
            await hooks.handle_tool_pre(
                "tool:pre",
                {
                    "tool_name": "delegate",
                    "tool_call_id": f"t-{agent}",
                    "tool_input": {"agent": agent},
                    "session_id": None,
                },
            )
        out = capsys.readouterr().out
        assert "alpha" in out, f"Expected 'alpha' in output; got: {out!r}"
        assert "beta" in out, f"Expected 'beta' in output; got: {out!r}"
        assert out.count("⏳") == 2, f"Expected 2 ⏳ markers for 2 spawns; got: {out!r}"

    @pytest.mark.asyncio
    async def test_no_live_object_created_on_spawn(self, capsys):
        """No Live instance is created when a spawn tool is invoked."""
        hooks = _hooks()
        with patch.object(_mod, "Live") as mock_live_cls:
            await hooks.handle_tool_pre(
                "tool:pre",
                {
                    "tool_name": "delegate",
                    "tool_call_id": "t1",
                    "tool_input": {"agent": "my-agent"},
                    "session_id": None,
                },
            )
        # Live class must NOT have been instantiated (no Live panel)
        mock_live_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_label_fallback_to_tool_name(self, capsys):
        """No 'agent' key in tool_input → label falls back to tool_name."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t1",
                "tool_input": {"prompt": "do something"},
                "session_id": None,
            },
        )
        out = capsys.readouterr().out
        # tool_name should appear in the static marker
        assert "task" in out, f"Marker must fall back to tool_name; got: {out!r}"

    @pytest.mark.asyncio
    async def test_custom_spawn_tools_config(self, capsys):
        """Custom spawn_tools config is honoured for the static marker."""
        hooks = _hooks(spawn_tools=("mytool",))
        # 'mytool' → marker; 'delegate' → no marker
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "mytool",
                "tool_call_id": "t1",
                "tool_input": {"agent": "special-agent"},
                "session_id": None,
            },
        )
        out_mytool = capsys.readouterr().out
        assert "⏳" in out_mytool, (
            f"Custom spawn tool must emit ⏳ marker; got: {out_mytool!r}"
        )

        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t2",
                "tool_input": {"agent": "other"},
                "session_id": None,
            },
        )
        out_delegate = capsys.readouterr().out
        assert "⏳" not in out_delegate, (
            "Non-configured tool must not emit ⏳ marker when not in spawn_tools"
        )

    # --- Regression: sub-agent live streaming still suppressed ---

    @pytest.mark.asyncio
    async def test_sub_agent_delta_no_stderr_regression(self, capsys):
        """Regression: overlay _on_delta for sub-agents still writes nothing to stderr."""
        hooks = _hooks()
        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_cls.return_value = MagicMock()
            overlay = _mod._make_streaming_overlay(hooks)
            _sub_sid = "0000000000000000-abc_my-agent"
            await overlay["llm:stream_block_start"](
                "llm:stream_block_start",
                {"session_id": _sub_sid, "block_index": 0, "block_type": "text"},
            )
            await overlay["llm:stream_block_delta"](
                "llm:stream_block_delta",
                {
                    "session_id": _sub_sid,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "live streaming token",
                },
            )
        captured = capsys.readouterr()
        assert captured.err == "", (
            f"Sub-agent live streaming must still be suppressed; got: {captured.err!r}"
        )

    # --- Regression: render_end still works ---

    @pytest.mark.asyncio
    async def test_render_end_no_error_without_spinner(self, capsys):
        """handle_render_end is safe without any spinner machinery."""
        hooks = _hooks()
        # Must not raise AttributeError for _active_spawns or _spinner_live
        from amplifier_core import HookResult

        result = await hooks.handle_render_end("cleanup:render_end", {})
        assert isinstance(result, HookResult)
        assert result.action == "continue"
