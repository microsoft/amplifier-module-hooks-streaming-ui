"""Tests for per-agent named-list panel (feat/per-agent-spinner-list).

Change A (unchanged):
  _text_renderable with label/label_style args: contains [x], dim, full-width
  Markdown, NO ▍ — existing tests in TestTextRenderableGeneralized still pass.

Replaced: the old _subagents_running integer counter tests (Change B) are
replaced by the new _active_spawns dict-model tests below.  Specifically:

Old tests removed / migrated
-----------------------------
TestSpinnerCounter
  test_delegate_pre_increments_counter  → TestActiveSpawns.test_delegate_pre_adds_entry
  test_task_pre_increments_counter      → TestActiveSpawns.test_task_pre_adds_entry
  test_two_spawns_counter_reaches_two   → TestActiveSpawns.test_two_spawns_two_entries_in_order
  test_delegate_post_decrements_counter → TestActiveSpawns.test_post_removes_entry
  test_two_pre_one_post_leaves_count_one→ TestActiveSpawns.test_two_pre_one_post_leaves_one_entry
  test_non_spawn_tool_pre_…             → TestActiveSpawns.test_non_spawn_tool_pre_no_entry
  test_non_spawn_tool_post_…            → TestActiveSpawns.test_non_spawn_tool_post_no_change
  test_counter_never_goes_negative      → TestActiveSpawns.test_post_without_pre_does_not_break

TestSpinnerTTYGate
  Updated: _subagents_running checks → _active_spawns checks

TestSpinnerResetOnRenderEnd
  Updated: _subagents_running checks → _active_spawns checks
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


def _make_spawns_text(hooks: StreamingUIHooks, width: int = 80) -> str:
    """Render _make_spawns_renderable to a plain string (no ANSI)."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False, no_color=True).print(
        hooks._make_spawns_renderable()
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
# spawn_tools config: default and custom
# ===========================================================================


class TestSpawnToolsConfig:
    """spawn_tools is config-driven; hardcoded 'task'/'delegate' removed."""

    @pytest.mark.asyncio
    async def test_default_recognises_delegate(self):
        """Default config: 'delegate' tool:pre adds an entry to _active_spawns."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_input": {"agent": "my-agent"},
                "session_id": None,
            },
        )
        assert "t1" in hooks._active_spawns, (
            f"'delegate' must be in default spawn_tools; _active_spawns={hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_default_recognises_task(self):
        """Default config: 'task' tool:pre adds an entry to _active_spawns."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t2",
                "tool_input": {"agent": "my-agent"},
                "session_id": None,
            },
        )
        assert "t2" in hooks._active_spawns, (
            f"'task' must be in default spawn_tools; _active_spawns={hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_custom_delegate_only_task_not_added(self):
        """spawn_tools=['delegate']: 'task' tool:pre does NOT add an entry."""
        hooks = _hooks(spawn_tools=("delegate",))
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t3",
                "tool_input": {"agent": "my-agent"},
                "session_id": None,
            },
        )
        assert "t3" not in hooks._active_spawns, (
            f"'task' must NOT be tracked when spawn_tools=['delegate']; "
            f"_active_spawns={hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_custom_mytool_adds_entry(self):
        """spawn_tools=['mytool']: 'mytool' tool:pre adds an entry."""
        hooks = _hooks(spawn_tools=("mytool",))
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "mytool",
                "tool_call_id": "t4",
                "tool_input": {"agent": "special-agent"},
                "session_id": None,
            },
        )
        assert "t4" in hooks._active_spawns, (
            f"'mytool' must be tracked when spawn_tools=['mytool']; "
            f"_active_spawns={hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_custom_config_post_removes_entry(self):
        """spawn_tools config is honoured at the remove (tool:post) site."""
        hooks = _hooks(spawn_tools=("mytool",))
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "mytool",
                "tool_call_id": "t5",
                "tool_input": {"agent": "alpha"},
                "session_id": None,
            },
        )
        assert "t5" in hooks._active_spawns
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "mytool",
                "tool_call_id": "t5",
                "tool_response": {"output": {"response": "done"}},
                "session_id": None,
            },
        )
        assert "t5" not in hooks._active_spawns, (
            f"Post must remove entry for configured spawn tool; "
            f"_active_spawns={hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_custom_config_dedup_only_for_configured_tool(self):
        """Dedup (early return) fires only for tools in spawn_tools, not for others."""
        hooks = _hooks(spawn_tools=("delegate",))
        # 'task' is NOT in spawn_tools → result body must NOT be suppressed
        # (we can verify indirectly: _active_spawns is empty, so no tracking happened)
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t6",
                "tool_input": {"agent": "agt"},
                "session_id": None,
            },
        )
        assert not hooks._active_spawns, (
            "Non-configured spawn tool must not add to _active_spawns"
        )
        # Calling post for 'task' (not in spawn_tools) should also not touch _active_spawns
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "task",
                "tool_call_id": "t6",
                "tool_response": {"output": "plain text result"},
                "session_id": None,
            },
        )
        assert not hooks._active_spawns


# ===========================================================================
# Per-agent dict tracking (_active_spawns)
# ===========================================================================


class TestActiveSpawns:
    """_active_spawns dict tracks tool_call_id -> agent_label in spawn order."""

    @pytest.mark.asyncio
    async def test_delegate_pre_adds_entry(self):
        """tool:pre for 'delegate' with tool_call_id adds an entry to _active_spawns."""
        hooks = _hooks()
        assert hooks._active_spawns == {}

        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_input": {"agent": "foundation:explorer"},
                "session_id": None,
            },
        )
        assert hooks._active_spawns == {"t1": "foundation:explorer"}, (
            f"Expected {{'t1': 'foundation:explorer'}}; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_task_pre_adds_entry(self):
        """tool:pre for 'task' also adds an entry to _active_spawns."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t2",
                "tool_input": {"agent": "zen-architect"},
                "session_id": None,
            },
        )
        assert "t2" in hooks._active_spawns
        assert hooks._active_spawns["t2"] == "zen-architect"

    @pytest.mark.asyncio
    async def test_two_spawns_two_entries_in_order(self):
        """Two spawns produce two entries; insertion order is preserved."""
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
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t2",
                "tool_input": {"agent": "zen-architect"},
                "session_id": None,
            },
        )
        assert list(hooks._active_spawns.keys()) == ["t1", "t2"], (
            f"Expected ['t1', 't2'] in order; got {list(hooks._active_spawns.keys())}"
        )
        assert list(hooks._active_spawns.values()) == [
            "foundation:explorer",
            "zen-architect",
        ]

    @pytest.mark.asyncio
    async def test_post_removes_entry(self):
        """tool:post with tool_call_id removes that entry; other entries remain."""
        hooks = _hooks()
        # Add t1 and t2
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_input": {"agent": "foundation:explorer"},
                "session_id": None,
            },
        )
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t2",
                "tool_input": {"agent": "zen-architect"},
                "session_id": None,
            },
        )
        assert len(hooks._active_spawns) == 2

        # Remove t1
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_response": {"output": {"response": "done"}},
                "session_id": None,
            },
        )
        assert "t1" not in hooks._active_spawns, (
            f"t1 must be removed after post; got {hooks._active_spawns}"
        )
        assert hooks._active_spawns == {"t2": "zen-architect"}, (
            f"t2 must remain; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_two_pre_one_post_leaves_one_entry(self):
        """Two pre, one post → one entry remains."""
        hooks = _hooks()
        for i, ag in enumerate(["alpha", "beta"], start=1):
            await hooks.handle_tool_pre(
                "tool:pre",
                {
                    "tool_name": "delegate",
                    "tool_call_id": f"t{i}",
                    "tool_input": {"agent": ag},
                    "session_id": None,
                },
            )
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_response": {"output": {"response": "done"}},
                "session_id": None,
            },
        )
        assert len(hooks._active_spawns) == 1, (
            f"Expected 1 entry; got {hooks._active_spawns}"
        )
        assert "t2" in hooks._active_spawns

    @pytest.mark.asyncio
    async def test_post_with_all_removed_empty_dict(self):
        """After all posts, _active_spawns is empty."""
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
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_response": {"output": {"response": "done"}},
                "session_id": None,
            },
        )
        assert hooks._active_spawns == {}, (
            f"Expected empty dict after final post; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_non_spawn_tool_pre_no_entry(self):
        """tool:pre for non-spawn tools must NOT add to _active_spawns."""
        hooks = _hooks()
        for tool in ("bash", "read_file", "write_file", "grep"):
            await hooks.handle_tool_pre(
                "tool:pre",
                {
                    "tool_name": tool,
                    "tool_call_id": f"x_{tool}",
                    "tool_input": {},
                    "session_id": None,
                },
            )
        assert hooks._active_spawns == {}, (
            f"Non-spawn tools must not touch _active_spawns; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_non_spawn_tool_post_no_change(self):
        """tool:post for non-spawn tools must not change _active_spawns."""
        hooks = _hooks()
        # Manually insert an entry so we can verify it survives
        hooks._active_spawns["t99"] = "live-agent"
        for tool in ("bash", "read_file"):
            await hooks.handle_tool_post(
                "tool:post",
                {
                    "tool_name": tool,
                    "tool_call_id": f"x_{tool}",
                    "tool_response": {"output": "x"},
                },
            )
        assert hooks._active_spawns == {"t99": "live-agent"}, (
            f"Non-spawn tool post must not modify _active_spawns; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_post_without_pre_does_not_break(self):
        """tool:post for a tool_call_id not in _active_spawns is a no-op (no KeyError)."""
        hooks = _hooks()
        assert hooks._active_spawns == {}
        # Must not raise
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_call_id": "nonexistent",
                "tool_response": {"output": {"response": "x"}},
            },
        )
        assert hooks._active_spawns == {}


# ===========================================================================
# Agent label fallback
# ===========================================================================


class TestAgentLabelFallback:
    """tool_input.agent is preferred; falls back to tool_name or 'sub-agent'."""

    @pytest.mark.asyncio
    async def test_agent_key_in_input_uses_agent(self):
        """tool_input with 'agent' key → label is that agent name."""
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
        assert hooks._active_spawns.get("t1") == "foundation:explorer"

    @pytest.mark.asyncio
    async def test_no_agent_key_falls_back_to_tool_name(self):
        """tool_input without 'agent' → label falls back to tool_name."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "task",
                "tool_call_id": "t2",
                "tool_input": {"prompt": "do something"},
                "session_id": None,
            },
        )
        assert hooks._active_spawns.get("t2") == "task", (
            f"Expected label='task'; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_empty_agent_key_falls_back_to_tool_name(self):
        """tool_input with empty 'agent' key (falsy) → label falls back to tool_name."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t3",
                "tool_input": {"agent": ""},
                "session_id": None,
            },
        )
        assert hooks._active_spawns.get("t3") == "delegate", (
            f"Empty agent key must fall back to tool_name; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_no_tool_name_falls_back_to_sub_agent(self):
        """If both agent and tool_name are absent/falsy → label is 'sub-agent'."""
        hooks = _hooks(spawn_tools=("",))
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "",
                "tool_call_id": "t4",
                "tool_input": {},
                "session_id": None,
            },
        )
        assert hooks._active_spawns.get("t4") == "sub-agent", (
            f"No agent + no tool_name must yield 'sub-agent'; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_missing_tool_call_id_uses_fallback_key(self):
        """When tool_call_id is absent, a synthetic _fallback_N key is used."""
        hooks = _hooks()
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                # no tool_call_id
                "tool_input": {"agent": "some-agent"},
                "session_id": None,
            },
        )
        assert len(hooks._active_spawns) == 1
        key = next(iter(hooks._active_spawns))
        assert key.startswith("_fallback_"), (
            f"Synthetic key must start with '_fallback_'; got {key!r}"
        )
        assert hooks._active_spawns[key] == "some-agent"


# ===========================================================================
# Rendered panel content
# ===========================================================================


class TestSpawnsRenderable:
    """_make_spawns_renderable contains each active label, count, and glyph."""

    def test_empty_spawns_header_zero(self):
        """With no active spawns, header shows '0 agents working'."""
        hooks = _hooks()
        out = _make_spawns_text(hooks)
        assert "0 agents" in out, f"Expected '0 agents' in header; got: {out!r}"

    def test_single_spawn_header_singular(self):
        """With 1 active spawn, header uses singular 'agent'."""
        hooks = _hooks()
        hooks._active_spawns["t1"] = "foundation:explorer"
        out = _make_spawns_text(hooks)
        assert "1 agent " in out or "1 agent\n" in out or "1 agent w" in out, (
            f"Expected '1 agent working' (singular); got: {out!r}"
        )

    def test_two_spawns_header_plural(self):
        """With 2 active spawns, header uses plural 'agents'."""
        hooks = _hooks()
        hooks._active_spawns["t1"] = "alpha"
        hooks._active_spawns["t2"] = "beta"
        out = _make_spawns_text(hooks)
        assert "2 agents" in out, f"Expected '2 agents'; got: {out!r}"

    def test_each_label_appears_in_output(self):
        """Every active agent label must appear in the renderable output."""
        hooks = _hooks()
        hooks._active_spawns["t1"] = "foundation:explorer"
        hooks._active_spawns["t2"] = "zen-architect"
        out = _make_spawns_text(hooks)
        assert "foundation:explorer" in out, (
            f"Expected 'foundation:explorer' in renderable; got: {out!r}"
        )
        assert "zen-architect" in out, (
            f"Expected 'zen-architect' in renderable; got: {out!r}"
        )

    def test_row_glyph_present(self):
        """Each row must contain the ▍ glyph."""
        hooks = _hooks()
        hooks._active_spawns["t1"] = "my-agent"
        out = _make_spawns_text(hooks)
        assert "▍" in out, f"Expected ▍ glyph in row output; got: {out!r}"

    def test_no_subagents_running_attribute(self):
        """There must be no _subagents_running attribute on StreamingUIHooks."""
        hooks = _hooks()
        assert not hasattr(hooks, "_subagents_running"), (
            "_subagents_running int counter must be removed; use _active_spawns"
        )


# ===========================================================================
# TTY gate — live panel not started when not a TTY
# ===========================================================================


class TestSpinnerTTYGate:
    """Live panel must NOT be started when stdout is not a TTY."""

    @pytest.mark.asyncio
    async def test_panel_not_started_when_not_tty(self):
        """_active_spawns is populated correctly but panel stays off in non-TTY."""
        hooks = _hooks()
        # _is_tty is set in __init__; during tests stdout is not a TTY
        assert not hooks._is_tty, "Test harness must have non-TTY stdout"

        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_input": {"agent": "my-agent"},
                "session_id": None,
            },
        )

        assert "t1" in hooks._active_spawns, (
            f"_active_spawns must be populated even in non-TTY; got {hooks._active_spawns}"
        )
        # Live panel must NOT be created/started when not a TTY
        assert hooks._spinner_live is None or not hooks._spinner_live.is_started, (
            "Live panel must not start when stdout is not a TTY"
        )

    @pytest.mark.asyncio
    async def test_active_spawns_works_without_panel(self, capsys):
        """_active_spawns increments/decrements correctly without live panel (non-TTY)."""
        hooks = _hooks()
        assert not hooks._is_tty

        # Full cycle: pre → post (suppressed) → dict empty
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_input": {"agent": "my-agent"},
                "session_id": None,
            },
        )
        assert "t1" in hooks._active_spawns

        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_call_id": "t1",
                "tool_response": {"output": {"response": "done"}},
            },
        )
        assert hooks._active_spawns == {}

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
                "tool_call_id": "outer-1",
                "tool_input": {"agent": "inner-agent", "prompt": "do stuff"},
                "session_id": None,
            },
        )
        # Simulate: sub-agent tool call (with sub-session ID)
        await hooks.handle_tool_pre(
            "tool:pre",
            {
                "tool_name": "bash",
                "tool_call_id": "bash-1",
                "tool_input": {"command": "ls"},
                "session_id": "parent-child_my-agent",
            },
        )
        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "bash",
                "tool_call_id": "bash-1",
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
                "tool_call_id": "outer-1",
                "tool_response": {"output": {"response": "sub-agent done"}},
                "session_id": None,
            },
        )
        assert hooks._active_spawns == {}
        out = capsys.readouterr().out
        # Sub-agent final result must appear in output
        assert "Done." in out
        assert "[my-agent]" in out


# ===========================================================================
# handle_render_end: clears _active_spawns and stops panel
# ===========================================================================


class TestSpinnerResetOnRenderEnd:
    """handle_render_end clears _active_spawns and stops the live panel."""

    @pytest.mark.asyncio
    async def test_render_end_clears_active_spawns(self):
        """handle_render_end empties _active_spawns."""
        hooks = _hooks()
        hooks._active_spawns["t1"] = "orphaned-agent"
        hooks._active_spawns["t2"] = "another-agent"

        await hooks.handle_render_end("cleanup:render_end", {})
        assert hooks._active_spawns == {}, (
            f"handle_render_end must clear _active_spawns; got {hooks._active_spawns}"
        )

    @pytest.mark.asyncio
    async def test_render_end_stops_spinner(self):
        """handle_render_end stops the live panel if running."""
        hooks = _hooks()
        mock_live = MagicMock()
        mock_live.is_started = True
        hooks._spinner_live = mock_live
        hooks._active_spawns["t1"] = "orphaned-agent"

        await hooks.handle_render_end("cleanup:render_end", {})

        mock_live.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_render_end_no_error_when_no_spinner(self):
        """handle_render_end is safe even when no live panel has been created."""
        hooks = _hooks()
        assert hooks._spinner_live is None
        assert hooks._active_spawns == {}
        # Must not raise
        await hooks.handle_render_end("cleanup:render_end", {})
        assert hooks._active_spawns == {}

    @pytest.mark.asyncio
    async def test_render_end_clears_even_with_orphaned_spawns(self):
        """Any leftover spawns are cleared by handle_render_end (safety net)."""
        hooks = _hooks()
        # Simulate a scenario where tool:post was never called
        hooks._active_spawns["lost-1"] = "agent-a"
        hooks._active_spawns["lost-2"] = "agent-b"

        await hooks.handle_render_end("cleanup:render_end", {})
        assert hooks._active_spawns == {}
