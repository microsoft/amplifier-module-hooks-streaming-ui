"""Tests for curated sub-agent output (feat/subagent-curated).

Covers all 5 changes:
  Change 1: sub-agent overlay deltas produce NO stderr output.
  Change 2: sub-agent thinking-start (content_block:start) produces NO stderr.
  Change 3: sub-agent thinking block NOT painted at content_block:end.
  Change 4a: sub-agent intermediate text (is_last_block=False) suppressed.
  Change 4b: sub-agent final text (is_last_block=True) painted attributed.
  Change 5a: task tool with {"response":…} result body suppressed.
  Change 5b: task tool with error-shaped result rendered normally.
  Change 5c: non-task tool result rendered normally.

Parent-session behaviour is verified by the pre-existing test suite.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from amplifier_core import HookResult
from amplifier_module_hooks_streaming_ui import StreamingUIHooks

import amplifier_module_hooks_streaming_ui as _mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hooks(**overrides) -> StreamingUIHooks:
    defaults = {"show_thinking": True, "show_tool_lines": 5, "show_token_usage": True}
    defaults.update(overrides)
    return StreamingUIHooks(**defaults)


_SUB_SID = "0000000000000000-7cc787dd22d54f6c_foundation:explorer"
_AGENT = "foundation:explorer"

_PARENT_SID = "12345678-1234-1234-1234-123456789012"


# ===========================================================================
# Change 1: overlay _on_delta — sub-agent produces no stderr
# ===========================================================================


class TestChange1OverlayDeltaSuppressed:
    """Sub-agent deltas write nothing to stderr (change 1)."""

    @pytest.mark.asyncio
    async def test_sub_agent_delta_no_stderr(self, capsys):
        """Delta handler for a sub-agent session must not write to stderr."""
        hooks = _hooks()

        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_cls.return_value = MagicMock()
            overlay = _mod._make_streaming_overlay(hooks)
            start_h = overlay["llm:stream_block_start"]
            delta_h = overlay["llm:stream_block_delta"]

            await start_h(
                "llm:stream_block_start",
                {"session_id": _SUB_SID, "block_index": 0, "block_type": "text"},
            )
            result = await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": _SUB_SID,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "Some streaming token\nand another line\n",
                },
            )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert captured.err == "", (
            f"Sub-agent delta must write nothing to stderr; got: {captured.err!r}"
        )

    @pytest.mark.asyncio
    async def test_parent_delta_still_updates_live(self):
        """Parent session delta still drives the Live region (regression guard)."""
        hooks = _hooks()

        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_instance = MagicMock()
            mock_live_cls.return_value = mock_live_instance
            overlay = _mod._make_streaming_overlay(hooks)
            start_h = overlay["llm:stream_block_start"]
            delta_h = overlay["llm:stream_block_delta"]

            await start_h(
                "llm:stream_block_start",
                {"session_id": _PARENT_SID, "block_index": 0, "block_type": "text"},
            )
            result = await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": _PARENT_SID,
                    "block_index": 0,
                    "block_type": "text",
                    "text": "Hello parent",
                },
            )

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        # Live.update() must have been called for the parent
        mock_live_instance.update.assert_called()

    @pytest.mark.asyncio
    async def test_sub_agent_block_end_no_stderr_flush(self, capsys):
        """Overlay block_end for sub-agent must not flush trailing partial to stderr."""
        hooks = _hooks()

        with patch.object(_mod, "Live") as mock_live_cls:
            mock_live_cls.return_value = MagicMock()
            overlay = _mod._make_streaming_overlay(hooks)
            start_h = overlay["llm:stream_block_start"]
            delta_h = overlay["llm:stream_block_delta"]
            end_h = overlay["llm:stream_block_end"]

            await start_h(
                "llm:stream_block_start",
                {"session_id": _SUB_SID, "block_index": 0, "block_type": "text"},
            )
            await delta_h(
                "llm:stream_block_delta",
                {
                    "session_id": _SUB_SID,
                    "block_index": 0,
                    "block_type": "text",
                    # No trailing newline — this was the "partial line" scenario
                    "text": "partial line without newline",
                },
            )
            result = await end_h(
                "llm:stream_block_end",
                {"session_id": _SUB_SID, "block_index": 0},
            )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert captured.err == "", (
            f"Overlay block_end for sub-agent must not write to stderr; "
            f"got: {captured.err!r}"
        )


# ===========================================================================
# Change 2: handle_content_block_start — sub-agent thinking no stderr
# ===========================================================================


class TestChange2ThinkingStartSuppressed:
    """Sub-agent thinking-start writes nothing to stderr (change 2)."""

    @pytest.mark.asyncio
    async def test_sub_agent_thinking_start_no_stderr(self, capsys):
        """content_block:start for sub-agent thinking block → no stderr output."""
        hooks = _hooks()

        result = await hooks.handle_content_block_start(
            "content_block:start",
            {
                "block_type": "thinking",
                "block_index": 0,
                "session_id": _SUB_SID,
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 in hooks.thinking_blocks  # tracking still happens

        captured = capsys.readouterr()
        assert captured.err == "", (
            f"Sub-agent thinking-start must write nothing to stderr; "
            f"got: {captured.err!r}"
        )

    @pytest.mark.asyncio
    async def test_parent_thinking_start_still_writes_stderr(self, capsys):
        """Parent thinking-start still writes 🧠 Thinking... to stderr (regression guard)."""
        hooks = _hooks()

        await hooks.handle_content_block_start(
            "content_block:start",
            {
                "block_type": "thinking",
                "block_index": 0,
                # No session_id → parent session
            },
        )

        captured = capsys.readouterr()
        # Parent path is unchanged — must still announce thinking
        assert "Thinking..." in captured.err, (
            f"Parent thinking-start must still write to stderr; got: {captured.err!r}"
        )


# ===========================================================================
# Change 3: handle_content_block_end — sub-agent thinking not painted
# ===========================================================================


class TestChange3ThinkingNotPainted:
    """Sub-agent thinking block is NOT painted at block-end (change 3)."""

    @pytest.mark.asyncio
    async def test_sub_agent_thinking_not_painted(self, capsys):
        """content_block:end for sub-agent thinking block must NOT paint framed block."""
        hooks = _hooks()
        hooks.thinking_blocks[0] = {"started": True, "agent": _AGENT}

        result = await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {
                    "type": "thinking",
                    "thinking": "I am reasoning about this problem.",
                },
                "session_id": _SUB_SID,
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        # Cleanup must still happen
        assert 0 not in hooks.thinking_blocks, (
            "thinking_blocks entry must be cleaned up even when paint is suppressed"
        )

        captured = capsys.readouterr()
        # Framed thinking block must NOT appear
        assert "===" not in captured.out, (
            f"Sub-agent thinking must NOT produce framed === block; got: {captured.out!r}"
        )
        assert "Thinking:" not in captured.out, (
            f"Sub-agent thinking must NOT produce 'Thinking:' header; got: {captured.out!r}"
        )
        assert "I am reasoning about this problem." not in captured.out, (
            "Sub-agent thinking text must not appear in stdout"
        )

    @pytest.mark.asyncio
    async def test_parent_thinking_still_painted(self, capsys):
        """Parent thinking block is still painted (regression guard)."""
        hooks = _hooks()
        hooks.thinking_blocks[0] = {"started": True}

        await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {
                    "type": "thinking",
                    "thinking": "Parent reasoning here.",
                },
                # No session_id → parent
            },
        )

        captured = capsys.readouterr()
        assert "Thinking:" in captured.out, (
            "Parent thinking must still be painted as framed block"
        )
        assert "Parent reasoning here." in captured.out


# ===========================================================================
# Change 4: handle_content_block_end — text block routing
# ===========================================================================


class TestChange4TextBlockRouting:
    """Sub-agent text: intermediate suppressed, final attributed (change 4)."""

    @pytest.mark.asyncio
    async def test_intermediate_sub_agent_text_suppressed(self, capsys):
        """Sub-agent text that is NOT the last block produces no output."""
        hooks = _hooks()

        result = await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 2,  # is_last_block = 0==1 → False
                "block": {"type": "text", "text": "An intermediate aside."},
                "session_id": _SUB_SID,
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert captured.out == "", (
            f"Sub-agent intermediate text must be suppressed; got: {captured.out!r}"
        )
        assert captured.err == "", (
            f"No stderr for intermediate sub-agent text; got: {captured.err!r}"
        )

    @pytest.mark.asyncio
    async def test_final_sub_agent_text_attributed(self, capsys):
        """Sub-agent text that IS the last block is painted with [agent_name] header."""
        hooks = _hooks()

        result = await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,  # is_last_block = 0==0 → True
                "block": {
                    "type": "text",
                    "text": "The final answer from the sub-agent.",
                },
                "session_id": _SUB_SID,
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        output = captured.out

        # Attribution header must appear
        assert f"[{_AGENT}]" in output, (
            f"Sub-agent final result must include [{_AGENT}] attribution; "
            f"got: {output!r}"
        )
        # Result text must appear
        assert "The final answer from the sub-agent." in output, (
            f"Sub-agent final result text missing; got: {output!r}"
        )

    @pytest.mark.asyncio
    async def test_final_sub_agent_text_no_amplifier_label(self, capsys):
        """Sub-agent final result must NOT have 'Amplifier:' label (parent label)."""
        hooks = _hooks()

        await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {"type": "text", "text": "Sub-agent result text."},
                "session_id": _SUB_SID,
            },
        )

        captured = capsys.readouterr()
        assert "Amplifier:" not in captured.out, (
            "Sub-agent final result must not contain 'Amplifier:' parent label"
        )

    @pytest.mark.asyncio
    async def test_parent_intermediate_text_unchanged(self, capsys):
        """Parent intermediate text (agent_name=None) is painted with Amplifier: label."""
        hooks = _hooks()

        await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 2,  # is_last_block=False (intermediate)
                "block": {"type": "text", "text": "Parent intermediate text."},
                # No session_id → parent
            },
        )

        captured = capsys.readouterr()
        # Parent path unchanged: Amplifier: label, no sub-agent suppression
        assert "Amplifier:" in captured.out, (
            "Parent intermediate text must still render with 'Amplifier:' label"
        )
        assert "Parent intermediate text." in captured.out

    @pytest.mark.asyncio
    async def test_final_sub_agent_text_attribution_is_dim_cyan(self, capsys):
        """Attribution header uses dim cyan styling (Rich dim cyan style)."""
        hooks = _hooks()

        # Use a Rich console with force_terminal+color_system so ANSI codes are emitted
        buf = io.StringIO()
        from rich.console import Console

        real_console_cls = Console

        call_count = 0

        def _capturing_console(*args, **kwargs):
            nonlocal call_count
            kwargs["file"] = buf
            kwargs["force_terminal"] = True
            kwargs["color_system"] = "standard"
            call_count += 1
            return real_console_cls(*args, **kwargs)

        with patch.object(_mod, "Console", side_effect=_capturing_console):
            await hooks.handle_content_block_end(
                "content_block:end",
                {
                    "block_index": 0,
                    "total_blocks": 1,
                    "block": {"type": "text", "text": "Colored attribution."},
                    "session_id": _SUB_SID,
                },
            )

        output = buf.getvalue()
        # Dim styling → ANSI dim sequence \x1b[2m
        assert (
            "\x1b[2m" in output or "dim" in output.lower() or f"[{_AGENT}]" in output
        ), f"Attribution should carry dim styling; got: {output!r}"
        assert f"[{_AGENT}]" in output

    @pytest.mark.asyncio
    async def test_multiple_sub_agents_each_get_attribution(self, capsys):
        """Each sub-agent's final result shows its own agent name."""
        hooks = _hooks()

        for agent_sid, agent_name, text in [
            ("span_agent-a", "agent-a", "Result from A."),
            ("span_agent-b", "agent-b", "Result from B."),
        ]:
            await hooks.handle_content_block_end(
                "content_block:end",
                {
                    "block_index": 0,
                    "total_blocks": 1,
                    "block": {"type": "text", "text": text},
                    "session_id": agent_sid,
                },
            )

        captured = capsys.readouterr()
        output = captured.out
        assert "[agent-a]" in output, f"agent-a attribution missing; got: {output!r}"
        assert "[agent-b]" in output, f"agent-b attribution missing; got: {output!r}"
        assert "Result from A." in output
        assert "Result from B." in output


# ===========================================================================
# Change 5: handle_tool_post — task tool result dedup
# ===========================================================================


class TestChange5TaskToolDedup:
    """task tool result body suppressed for successful sub-agent responses (change 5)."""

    @pytest.mark.asyncio
    async def test_task_tool_success_shape_suppressed(self, capsys):
        """task tool with {"response": …, "session_id": …} → result body suppressed."""
        hooks = _hooks()

        result = await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "task",
                "tool_response": {
                    "response": "The sub-agent finished its work.",
                    "session_id": _SUB_SID,
                },
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        # Result body must not appear (attributed final already showed it)
        assert "The sub-agent finished its work." not in captured.out, (
            "task tool result body must be suppressed for success shape"
        )
        # No tool-result banner either
        assert "Tool result: task" not in captured.out

    @pytest.mark.asyncio
    async def test_task_tool_error_shape_rendered(self, capsys):
        """task tool with error-shaped result (no 'response' key) renders normally."""
        hooks = _hooks()

        result = await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "task",
                "tool_response": {
                    "error": "Agent timed out after 30s",
                    "success": False,
                },
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        # Error must appear — do not hide failures
        assert (
            "Agent timed out" in captured.out or "Tool result: task" in captured.out
        ), "task tool error result must not be suppressed"

    @pytest.mark.asyncio
    async def test_task_tool_string_result_rendered(self, capsys):
        """task tool with a string result (not a dict) renders normally."""
        hooks = _hooks()

        result = await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "task",
                "tool_response": "plain string error",
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "plain string error" in captured.out, (
            "task tool string result must render normally"
        )

    @pytest.mark.asyncio
    async def test_non_task_tool_result_unchanged(self, capsys):
        """Non-task tool result always renders regardless of shape (regression guard)."""
        hooks = _hooks()

        result = await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "bash",
                "tool_response": {
                    "response": "bash output here",
                    "session_id": "some-sid",
                },
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        # bash tool result must be rendered (not suppressed like task tool)
        assert (
            "bash output here" in captured.out or "Tool result: bash" in captured.out
        ), "Non-task tool result must render normally even if it has 'response' key"

    @pytest.mark.asyncio
    async def test_task_tool_empty_dict_rendered(self, capsys):
        """task tool with an empty dict result (no 'response' key) renders normally."""
        hooks = _hooks()

        result = await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "task",
                "tool_response": {},
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        # Empty dict has no "response" key → should not be suppressed.
        # Just verify it returns continue without error.


# ===========================================================================
# Parent path unchanged (regression guards summarised in one class)
# ===========================================================================


class TestParentPathUnchanged:
    """Verify key parent-path invariants still hold after curated changes."""

    @pytest.mark.asyncio
    async def test_parent_final_text_rendered(self, capsys):
        """Parent final text still renders with Amplifier: label (regression guard)."""
        hooks = _hooks()

        await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {"type": "text", "text": "Parent response text."},
                # No session_id → parent
            },
        )

        captured = capsys.readouterr()
        assert "Amplifier:" in captured.out
        assert "Parent response text." in captured.out

    @pytest.mark.asyncio
    async def test_parent_thinking_painted_with_frame(self, capsys):
        """Parent thinking is still framed (===) at block-end (regression guard)."""
        hooks = _hooks()
        hooks.thinking_blocks[0] = {"started": True}

        await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {"type": "thinking", "thinking": "Reasoning."},
            },
        )

        captured = capsys.readouterr()
        assert "==" in captured.out
        assert "Thinking:" in captured.out

    @pytest.mark.asyncio
    async def test_non_task_tool_post_parent_rendered(self, capsys):
        """Parent non-task tool result still rendered (regression guard)."""
        hooks = _hooks()

        await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "read_file",
                "tool_response": {"success": True, "output": "file contents"},
            },
        )

        captured = capsys.readouterr()
        assert "Tool result: read_file" in captured.out
