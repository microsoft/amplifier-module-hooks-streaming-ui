"""Tests for sub-agent rendering behavior.

Restored behavior (reverted curation, except Change 1 live-token-stream suppression):
  Change 1 (KEPT):  sub-agent overlay deltas produce NO stderr output (streaming OFF).
  Change 2 (reverted): sub-agent thinking-start emits 🤔 [agent] Thinking... to stderr.
  Change 3 (reverted): sub-agent thinking block IS painted at content_block:end.
  Change 4a (unchanged): sub-agent intermediate text painted attributed.
  Change 4b (unchanged): sub-agent final text painted attributed.
  Change 5 (reverted): spawn-tool result envelope IS rendered (dedup removed).

Parent-session behaviour is verified by the pre-existing test suite.
"""

from __future__ import annotations

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
# Change 1 (KEPT): overlay _on_delta — sub-agent produces no stderr
# ===========================================================================


class TestChange1OverlayDeltaSuppressed:
    """Sub-agent deltas write nothing to stderr (change 1 — KEPT)."""

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
# Change 2 (REVERTED): handle_content_block_start — sub-agent thinking emits marker
# ===========================================================================


class TestSubAgentThinkingStart:
    """Sub-agent thinking-start now emits 🤔 [agent] Thinking... to stderr (reverted)."""

    @pytest.mark.asyncio
    async def test_sub_agent_thinking_start_emits_marker(self, capsys):
        """content_block:start for sub-agent thinking block → emits attributed marker."""
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
        assert f"[{_AGENT}] Thinking..." in captured.err, (
            f"Sub-agent thinking-start must emit attributed marker to stderr; "
            f"got: {captured.err!r}"
        )

    @pytest.mark.asyncio
    async def test_sub_agent_thinking_marker_contains_agent_name(self, capsys):
        """The thinking marker must include the agent name in brackets."""
        hooks = _hooks()
        await hooks.handle_content_block_start(
            "content_block:start",
            {
                "block_type": "thinking",
                "block_index": 0,
                "session_id": "span_my-agent",
            },
        )
        captured = capsys.readouterr()
        assert "[my-agent] Thinking..." in captured.err, (
            f"Marker must include agent name; got: {captured.err!r}"
        )

    @pytest.mark.asyncio
    async def test_parent_thinking_start_still_writes_stderr(self, capsys):
        """Parent thinking-start still writes 🤔 Thinking... to stderr (regression guard)."""
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
# Change 3 (REVERTED): handle_content_block_end — sub-agent thinking IS painted
# ===========================================================================


class TestSubAgentThinkingPainted:
    """Sub-agent thinking block IS painted as a framed block at block-end (reverted)."""

    @pytest.mark.asyncio
    async def test_sub_agent_thinking_is_painted(self, capsys):
        """content_block:end for sub-agent thinking block MUST paint the framed block."""
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
            "thinking_blocks entry must be cleaned up after paint"
        )

        captured = capsys.readouterr()
        # Framed thinking block MUST appear
        assert "===" in captured.out, (
            f"Sub-agent thinking MUST produce framed === block; got: {captured.out!r}"
        )
        assert "Thinking:" in captured.out, (
            f"Sub-agent thinking MUST produce 'Thinking:' header; got: {captured.out!r}"
        )
        assert "I am reasoning about this problem." in captured.out, (
            "Sub-agent thinking text must appear in stdout"
        )

    @pytest.mark.asyncio
    async def test_sub_agent_thinking_attributed(self, capsys):
        """Sub-agent thinking block must include [agent_name] in the header."""
        hooks = _hooks()
        hooks.thinking_blocks[0] = {"started": True, "agent": _AGENT}

        await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
                "block": {
                    "type": "thinking",
                    "thinking": "Attribution check.",
                },
                "session_id": _SUB_SID,
            },
        )

        captured = capsys.readouterr()
        # The header should show "[foundation:explorer] Thinking:" or similar
        assert _AGENT in captured.out, (
            f"Sub-agent thinking block must include agent name; got: {captured.out!r}"
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
# Change 4: handle_content_block_end — text block routing (unchanged)
# ===========================================================================


class TestChange4TextBlockRouting:
    """Sub-agent text: ALL settled blocks painted attributed (intermediate + final)."""

    @pytest.mark.asyncio
    async def test_intermediate_sub_agent_text_painted(self, capsys):
        """Sub-agent intermediate text is painted attributed (not suppressed)."""
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
        output = captured.out

        assert f"[{_AGENT}]" in output, (
            f"Sub-agent intermediate aside must include [{_AGENT}] attribution; "
            f"got: {output!r}"
        )
        assert "An intermediate aside." in output, (
            f"Sub-agent intermediate text must appear in output; got: {output!r}"
        )

    @pytest.mark.asyncio
    async def test_final_sub_agent_text_attributed(self, capsys):
        """Sub-agent text that IS the last block is painted with [agent_name] header."""
        hooks = _hooks()

        result = await hooks.handle_content_block_end(
            "content_block:end",
            {
                "block_index": 0,
                "total_blocks": 1,
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

        assert f"[{_AGENT}]" in output, (
            f"Sub-agent final result must include [{_AGENT}] attribution; "
            f"got: {output!r}"
        )
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
        assert "Amplifier:" in captured.out, (
            "Parent intermediate text must still render with 'Amplifier:' label"
        )
        assert "Parent intermediate text." in captured.out

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
# Change 5 (REVERTED): handle_tool_post — spawn-tool result IS rendered
# ===========================================================================


class TestSpawnToolResultRendered:
    """Spawn-tool result envelope is rendered (dedup removed in revert of change 5)."""

    @pytest.mark.asyncio
    async def test_task_tool_success_shape_rendered(self, capsys):
        """task tool with {\"response\": …, \"session_id\": …} → result IS rendered."""
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
        # Result envelope MUST appear (dedup removed)
        assert "Tool result: task" in captured.out, (
            f"task tool result envelope must render (dedup removed); got: {captured.out!r}"
        )

    @pytest.mark.asyncio
    async def test_delegate_tool_success_shape_rendered(self, capsys):
        """delegate tool with nested output.response → result IS rendered."""
        hooks = _hooks()

        result = await hooks.handle_tool_post(
            "tool:post",
            {
                "tool_name": "delegate",
                "tool_response": {
                    "error": None,
                    "success": True,
                    "output": {
                        "agent": "foundation:explorer",
                        "response": "The delegated agent finished its work.",
                        "session_id": _SUB_SID,
                        "metadata": {},
                        "provider_routing": {"model_role": "fast"},
                        "status": "success",
                        "turn_count": 7,
                    },
                },
            },
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        # Result envelope MUST appear (dedup removed)
        assert "Tool result: delegate" in captured.out, (
            f"delegate tool result envelope must render (dedup removed); got: {captured.out!r}"
        )

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
        # Empty dict has no "response" key — should not be suppressed.
        # Just verify it returns continue without error.


# ===========================================================================
# Parent path unchanged (regression guards summarised in one class)
# ===========================================================================


class TestParentPathUnchanged:
    """Verify key parent-path invariants still hold after reverted changes."""

    @pytest.mark.asyncio
    async def test_parent_final_text_NOT_rendered_by_hook(self, capsys):
        """Parent final text is NOT painted by the hook (#256 fix; regression guard).

        handle_content_block_end skips is_last_block=True for parent (agent_name=None).
        app-cli's render_message is the sole owner.
        """
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
        # Hook must NOT paint the final parent text (#256 double-render fix)
        assert "Parent response text." not in captured.out, (
            f"Hook must skip parent final text; got: {captured.out!r}"
        )

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
