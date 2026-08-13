"""Tests for streaming UI hooks module."""

import io
import re
from unittest.mock import MagicMock, patch

import pytest
from amplifier_core import HookResult
from amplifier_module_hooks_streaming_ui import StreamingUIHooks
from amplifier_module_hooks_streaming_ui import mount


class TestStreamingUIHooksInit:
    """Test StreamingUIHooks initialization."""

    def test_last_llm_info_initialized_to_none(self):
        """Test that last_llm_info is initialized to None in __init__."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        assert hasattr(hooks, "last_llm_info"), "last_llm_info attribute should exist"
        assert hooks.last_llm_info is None, (
            "last_llm_info should be initialized to None"
        )


class TestHandleLLMResponse:
    """Test handle_llm_response handler method."""

    @pytest.mark.asyncio
    async def test_handle_llm_response_exists(self):
        """Test that handle_llm_response method exists on StreamingUIHooks."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        assert hasattr(hooks, "handle_llm_response"), (
            "handle_llm_response method should exist"
        )

    @pytest.mark.asyncio
    async def test_handle_llm_response_captures_llm_info(self):
        """Test that handle_llm_response captures provider, model, and duration_ms."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        data = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "duration_ms": 1234,
        }

        await hooks.handle_llm_response("llm:response", data)

        # Verify last_llm_info is populated correctly
        assert hooks.last_llm_info is not None, "last_llm_info should be set"
        assert hooks.last_llm_info["provider"] == "anthropic"
        assert hooks.last_llm_info["model"] == "claude-3-sonnet"
        assert hooks.last_llm_info["duration_ms"] == 1234

    @pytest.mark.asyncio
    async def test_handle_llm_response_returns_hook_result_continue(self):
        """Test that handle_llm_response returns HookResult with action='continue'."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        data = {
            "provider": "openai",
            "model": "gpt-4",
            "duration_ms": 500,
        }

        result = await hooks.handle_llm_response("llm:response", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

    @pytest.mark.asyncio
    async def test_handle_llm_response_handles_missing_fields(self):
        """Test that handle_llm_response handles missing fields gracefully."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Empty data dict
        data = {}

        result = await hooks.handle_llm_response("llm:response", data)

        # Should still work with None values
        assert hooks.last_llm_info is not None
        assert hooks.last_llm_info["provider"] is None
        assert hooks.last_llm_info["model"] is None
        assert hooks.last_llm_info["duration_ms"] is None
        assert isinstance(result, HookResult)
        assert result.action == "continue"


@pytest.mark.asyncio
async def test_mount_registers_hooks():
    """Test that mount registers all required hooks."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.register = MagicMock()

    config = {
        "ui": {
            "show_thinking_stream": True,
            "show_tool_lines": 5,
            "show_token_usage": True,
        }
    }

    await mount(coordinator, config)

    # Verify all hooks are registered
    expected_events = [
        "content_block:start",
        "content_block:end",
        "tool:pre",
        "tool:post",
        "llm:response",
    ]

    for event in expected_events:
        # Find if this event was registered
        registered = any(
            call[0][0] == event for call in coordinator.hooks.register.call_args_list
        )
        assert registered, f"Event {event} was not registered"


@pytest.mark.asyncio
async def test_mount_with_defaults():
    """Test mount works with default config."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.register = MagicMock()

    # Empty config should use defaults
    config = {}

    await mount(coordinator, config)

    # Should register 8 hooks:
    #   content_block:start, content_block:end, tool:pre, tool:post,
    #   llm:response, provider:resolve, orchestrator:complete, prompt:submit
    #   (the last two are the cost-summary and cost-seed handlers, both
    #   gated on show_token_usage which defaults to True)
    assert coordinator.hooks.register.call_count == 8


@pytest.mark.asyncio
async def test_mount_with_token_usage_disabled_skips_cost_handler():
    """When show_token_usage=False, orchestrator:complete is not registered.

    Users who disable token usage display should not see the cost footer either.
    The per-call cost line and the 💰 turn/session footer are gated together.
    """
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.register = MagicMock()

    config = {"ui": {"show_token_usage": False}}

    await mount(coordinator, config)

    # Only 6 hooks: content_block:start, content_block:end, tool:pre, tool:post,
    # llm:response, provider:resolve.
    # orchestrator:complete and prompt:submit (cost handlers) are NOT
    # registered when show_token_usage=False.
    assert coordinator.hooks.register.call_count == 6

    registered_events = [
        call[0][0] for call in coordinator.hooks.register.call_args_list
    ]
    assert "orchestrator:complete" not in registered_events


class TestStreamingUIHooks:
    """Test the StreamingUIHooks class."""

    @pytest.mark.asyncio
    async def test_thinking_block_start(self, capsys):
        """Test thinking block start detection."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        data = {"block_type": "thinking", "block_index": 0}

        result = await hooks.handle_content_block_start("content_block:start", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 in hooks.thinking_blocks

        captured = capsys.readouterr()
        assert "🧠 Thinking..." in (captured.err or "")

    @pytest.mark.asyncio
    async def test_thinking_block_disabled(self, capsys):
        """Test thinking blocks are not shown when disabled."""
        hooks = StreamingUIHooks(
            show_thinking=False, show_tool_lines=5, show_token_usage=True
        )

        data = {"block_type": "thinking", "block_index": 0}

        result = await hooks.handle_content_block_start("content_block:start", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 not in hooks.thinking_blocks

        captured = capsys.readouterr()
        assert "Thinking" not in captured.out

    @pytest.mark.asyncio
    async def test_thinking_block_end(self, capsys):
        """Test thinking block display on end."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Track the block first
        hooks.thinking_blocks[0] = {"started": True}

        data = {
            "block_index": 0,
            "block": {
                "type": "thinking",
                "thinking": "This is a test thought process.",
            },
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 0 not in hooks.thinking_blocks  # Should be cleaned up

        captured = capsys.readouterr()
        assert "=" * 60 in captured.out
        assert "Thinking:" in captured.out
        assert "This is a test thought process." in captured.out

    @pytest.mark.asyncio
    async def test_reasoning_block_end(self, capsys):
        """Reasoning blocks should be treated like thinking blocks."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        hooks.thinking_blocks[1] = {"started": True}
        data = {
            "block_index": 1,
            "block": {
                "type": "reasoning",
                "summary": [{"text": "Summary insight"}],
                "content": [{"text": "Detailed chain of thought"}],
            },
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert 1 not in hooks.thinking_blocks

        captured = capsys.readouterr()
        assert "Thinking:" in captured.out
        assert "Summary insight" in captured.out
        assert "Detailed chain of thought" in captured.out

    @pytest.mark.asyncio
    async def test_tool_pre(self, capsys):
        """Test tool invocation display."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=3, show_token_usage=True
        )

        data = {
            "tool_name": "filesystem_read",
            "tool_input": {"path": "/some/long/path/to/file.txt", "encoding": "utf-8"},
        }

        result = await hooks.handle_tool_pre("tool:pre", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "🔧 Using tool: filesystem_read" in captured.out
        assert "path:" in captured.out  # YAML-style key display

    @pytest.mark.asyncio
    async def test_tool_post_success(self, capsys):
        """Test successful tool result display."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=3, show_token_usage=True
        )

        data = {
            "tool_name": "filesystem_read",
            "tool_response": {"success": True, "output": "File contents here"},
        }

        result = await hooks.handle_tool_post("tool:post", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "✅ Tool result: filesystem_read" in captured.out
        assert "File contents here" in captured.out

    @pytest.mark.asyncio
    async def test_tool_post_failure(self, capsys):
        """Test failed tool result display."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=3, show_token_usage=True
        )

        data = {
            "tool_name": "filesystem_read",
            "tool_response": {"success": False, "output": "Error: File not found"},
        }

        result = await hooks.handle_tool_post("tool:post", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "❌ Tool result: filesystem_read" in captured.out
        assert "Error: File not found" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_display_with_thinking(self, capsys):
        """Test token usage displayed after last block when included in event data."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Track the thinking block first
        hooks.thinking_blocks[0] = {"started": True}

        # content_block:end now includes usage from parent response and total_blocks
        # This is the last (and only) block
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "thinking", "thinking": "Test thinking"},
            "usage": {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801},
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "📊 Token Usage" in captured.out
        assert "Input: 1,234" in captured.out
        assert "Output: 567" in captured.out
        assert "Total: 1,801" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_not_displayed_for_non_last_block(self, capsys):
        """Test token usage NOT displayed for blocks that aren't last."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        hooks.thinking_blocks[0] = {"started": True}

        # This is block 0 of 2 (not last)
        data = {
            "block_index": 0,
            "total_blocks": 2,
            "block": {"type": "thinking", "thinking": "Test"},
            "usage": {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801},
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "Token Usage" not in captured.out  # Should NOT display

    @pytest.mark.asyncio
    async def test_token_usage_disabled(self, capsys):
        """Test token usage is not shown when disabled."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=False
        )

        hooks.thinking_blocks[0] = {"started": True}

        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "thinking", "thinking": "Test"},
            "usage": {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801},
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "Token Usage" not in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_missing_from_event(self, capsys):
        """Test token usage handles missing usage data gracefully."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        hooks.thinking_blocks[0] = {"started": True}

        # No usage field in event data (but is last block)
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "thinking", "thinking": "Test"},
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert "Token Usage" not in captured.out

    def test_truncate_lines(self):
        """Test line truncation logic."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=3, show_token_usage=True
        )

        # Test short text (no truncation)
        text = "line1\nline2\nline3"
        result = hooks._truncate_lines(text, 3)
        assert result == text

        # Test long text (truncation)
        text = "line1\nline2\nline3\nline4\nline5"
        result = hooks._truncate_lines(text, 3)
        assert result == "line1\nline2\nline3\n... (2 more lines)"

        # Test empty text
        result = hooks._truncate_lines("", 3)
        assert result == "(empty)"

        # Test single line
        text = "single line"
        result = hooks._truncate_lines(text, 3)
        assert result == text

    @pytest.mark.asyncio
    async def test_llm_response_captures_model_info(self):
        """Test that handle_llm_response captures provider/model/duration."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Verify initial state
        assert hooks.last_llm_info is None

        data = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250514",
            "duration_ms": 2345,
        }

        result = await hooks.handle_llm_response("llm:response", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert hooks.last_llm_info is not None
        assert hooks.last_llm_info["provider"] == "anthropic"
        assert hooks.last_llm_info["model"] == "claude-sonnet-4-5-20250514"
        assert hooks.last_llm_info["duration_ms"] == 2345

    @pytest.mark.asyncio
    async def test_token_usage_displays_model_info(self, capsys):
        """Test token usage header includes provider/model/duration when available."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Simulate llm:response having been received
        hooks.last_llm_info = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250514",
            "duration_ms": 2345,
        }

        # Last block with token usage
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "Hello"},
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        assert (
            "📊 Token Usage (anthropic/claude-sonnet-4-5-20250514) [2.3s]"
            in captured.out
        )
        assert "Input: 1,000" in captured.out
        assert "Output: 500" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_fallback_without_model_info(self, capsys):
        """Test token usage renders gracefully when last_llm_info is None."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Explicitly ensure no llm info
        assert hooks.last_llm_info is None

        # Last block with token usage
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "Hello"},
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }

        result = await hooks.handle_content_block_end("content_block:end", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"

        captured = capsys.readouterr()
        # Should show basic header without model info
        assert "📊 Token Usage" in captured.out
        # Should NOT have parentheses with provider/model
        assert "📊 Token Usage (" not in captured.out
        assert "Input: 1,000" in captured.out

    @pytest.mark.asyncio
    async def test_last_llm_info_cleared_after_render(self, capsys):
        """Test that last_llm_info is cleared after token usage is rendered."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Simulate llm:response having been received
        hooks.last_llm_info = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250514",
            "duration_ms": 2345,
        }

        # Last block with token usage
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "Hello"},
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }

        await hooks.handle_content_block_end("content_block:end", data)

        # State should be cleared after rendering
        assert hooks.last_llm_info is None


@pytest.mark.asyncio
async def test_non_thinking_blocks_ignored():
    """Test that non-thinking blocks are ignored."""
    hooks = StreamingUIHooks(
        show_thinking=True, show_tool_lines=5, show_token_usage=True
    )

    # Test text block (should be ignored)
    data = {"block_type": "text", "block_index": 0}

    result = await hooks.handle_content_block_start("content_block:start", data)
    assert isinstance(result, HookResult)
    assert result.action == "continue"
    assert 0 not in hooks.thinking_blocks


@pytest.mark.asyncio
async def test_tool_with_string_result(capsys):
    """Test tool result when result is a plain string."""
    hooks = StreamingUIHooks(
        show_thinking=True, show_tool_lines=5, show_token_usage=True
    )

    data = {"tool_name": "some_tool", "tool_response": "Simple string result"}

    result = await hooks.handle_tool_post("tool:post", data)

    assert isinstance(result, HookResult)
    assert result.action == "continue"

    captured = capsys.readouterr()
    assert "✅ Tool result: some_tool" in captured.out
    assert "Simple string result" in captured.out


class TestTokenUsageHeaderWithModelInfo:
    """Test token usage header displays model info when available."""

    @pytest.mark.asyncio
    async def test_token_usage_header_includes_model_info(self, capsys):
        """Test token usage header shows provider/model and duration when last_llm_info is set."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Simulate llm:response event setting last_llm_info
        hooks.last_llm_info = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "duration_ms": 1500,
        }

        # Last block with usage data
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": ""},
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        await hooks.handle_content_block_end("content_block:end", data)

        captured = capsys.readouterr()
        # Should include provider/model and duration in header
        assert "📊 Token Usage (anthropic/claude-3-sonnet) [1.5s]" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_header_without_duration(self, capsys):
        """Test token usage header shows provider/model without duration when duration_ms is None."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # No duration_ms
        hooks.last_llm_info = {
            "provider": "openai",
            "model": "gpt-4",
            "duration_ms": None,
        }

        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": ""},
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        await hooks.handle_content_block_end("content_block:end", data)

        captured = capsys.readouterr()
        # Should include provider/model but no duration
        assert "📊 Token Usage (openai/gpt-4)" in captured.out
        # Should NOT have duration like [1.5s] - check for pattern [digits.digits s]
        header_line = captured.out.split("Token Usage")[1].split("\n")[0]
        assert not re.search(r"\[\d+\.\d+s\]", header_line)  # No duration bracket

    @pytest.mark.asyncio
    async def test_token_usage_header_without_llm_info(self, capsys):
        """Test token usage header shows basic format when last_llm_info is None."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # No llm info captured
        hooks.last_llm_info = None

        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": ""},
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        await hooks.handle_content_block_end("content_block:end", data)

        captured = capsys.readouterr()
        # Should show basic header without model info
        assert "📊 Token Usage" in captured.out
        # Should NOT have parentheses (no model info)
        header_line = [
            line for line in captured.out.split("\n") if "Token Usage" in line
        ][0]
        assert "(" not in header_line

    @pytest.mark.asyncio
    async def test_last_llm_info_cleared_after_token_usage_rendered(self, capsys):
        """Test that last_llm_info is cleared to None after token usage is displayed.

        This prevents stale model info from bleeding into subsequent requests.
        """
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )

        # Simulate llm:response event setting last_llm_info
        hooks.last_llm_info = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "duration_ms": 1500,
        }

        # Last block with usage data triggers token usage display
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": ""},
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        await hooks.handle_content_block_end("content_block:end", data)

        # After rendering token usage, last_llm_info should be cleared
        assert hooks.last_llm_info is None, (
            "last_llm_info should be cleared after token usage is rendered "
            "to avoid stale data in subsequent requests"
        )


# ─── Per-call cost display on Token Usage line ───────────────────────────────


class TestTokenUsageCostDisplay:
    """Test that per-call cost_usd from M2 providers appears on the Token Usage line."""

    @pytest.mark.asyncio
    async def test_token_usage_shows_cost_when_present(self, capsys):
        """cost_usd on the usage dict produces '| Cost: $X.XX' on the Token Usage line."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "hi"},
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                # Real but under half a cent: renders as the sub-cent marker,
                # never "$0.00" (which would read as free).
                "cost_usd": "0.0043",
            },
        }

        await hooks.handle_content_block_end("content_block:end", data)

        captured = capsys.readouterr()
        assert "Cost:" in captured.out, "Cost field missing from Token Usage line"
        assert "<$0.01" in captured.out, (
            f"Expected <$0.01 in output, got: {captured.out}"
        )
        assert "$0.00" not in captured.out, "a real cost must never read as free"
        # Full expected format check
        assert "Input: 1,000" in captured.out
        assert "Output: 500" in captured.out
        assert "Total: 1,500" in captured.out

    @pytest.mark.asyncio
    async def test_token_usage_omits_cost_when_absent(self, capsys):
        """When cost_usd is absent, the Token Usage line does not show 'Cost:'."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "hi"},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                # no cost_usd — self-hosted / unknown model
            },
        }

        await hooks.handle_content_block_end("content_block:end", data)

        captured = capsys.readouterr()
        assert "Cost:" not in captured.out, (
            f"Cost: should not appear when cost_usd is absent, got: {captured.out}"
        )

    @pytest.mark.asyncio
    async def test_token_usage_omits_cost_when_none(self, capsys):
        """When cost_usd is explicitly None, the Token Usage line does not show 'Cost:'."""
        hooks = StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        data = {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "hi"},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost_usd": None,  # self-hosted provider returns None
            },
        }

        await hooks.handle_content_block_end("content_block:end", data)

        captured = capsys.readouterr()
        assert "Cost:" not in captured.out, (
            f"Cost: should not appear when cost_usd is None, got: {captured.out}"
        )


# ---------------------------------------------------------------------------
# Tests for the streaming overlay label behaviour (feat/label-in-stream)
# ---------------------------------------------------------------------------


class TestTextRenderable:
    """_text_renderable leads streaming parent text with a transient bold-green
    'Amplifier:' label above the rail aside body (Option C)."""

    def test_includes_label_and_content(self):
        import amplifier_module_hooks_streaming_ui as _mod
        from rich.console import Console

        b = io.StringIO()
        Console(file=b, width=80, force_terminal=False).print(
            _mod._text_renderable("hello world")
        )
        out = b.getvalue()
        assert "Amplifier:" in out
        assert "hello world" in out

    def test_body_uses_full_width_markdown(self):
        """_text_renderable body must use full-width Markdown (no ▍ rail glyph).

        The body is Markdown(content) so the streaming preview and the settled
        paint (_paint_interleaved_text → _text_renderable) are byte-identical —
        the whole point of the uniform-render change.
        """
        import amplifier_module_hooks_streaming_ui as _mod
        from rich.console import Console

        b = io.StringIO()
        Console(file=b, width=80, force_terminal=False).print(
            _mod._text_renderable("hello world")
        )
        out = b.getvalue()
        assert "▍" not in out, (
            f"_text_renderable body must NOT use ▍ (now full-width Markdown); got: {out!r}"
        )
        assert "hello world" in out, "Content text must be present in rendered output"
        assert "▸" not in out, "Whisper glyph ▸ must never appear in _text_renderable"

    def test_empty_content_still_shows_label(self):
        import amplifier_module_hooks_streaming_ui as _mod
        from rich.console import Console

        b = io.StringIO()
        Console(file=b, width=80, force_terminal=False).print(_mod._text_renderable(""))
        assert "Amplifier:" in b.getvalue()

    def test_label_precedes_body(self):
        import amplifier_module_hooks_streaming_ui as _mod
        from rich.console import Console

        b = io.StringIO()
        Console(file=b, width=80, force_terminal=False).print(
            _mod._text_renderable("the body text")
        )
        out = b.getvalue()
        assert out.index("Amplifier:") < out.index("the body text")


class TestStreamingOverlayLabel:
    """The overlay never prints 'Amplifier:' PERMANENTLY (no direct
    parent_console.print). For text blocks the label now leads the text
    *inside* the transient Live region via _text_renderable, so it clears
    with the stream: interleaved asides settle to the rail note (no label),
    and the final response's persistent label is owned by render_message."""

    @pytest.mark.asyncio
    async def test_label_not_printed_for_parent_text_block(self):
        """Overlay must NOT print 'Amplifier:' as a permanent line when a parent
        text block starts (the label lives inside the transient Live instead;
        see TestTextRenderable). With Live mocked, no direct print should occur."""
        import amplifier_module_hooks_streaming_ui as _mod

        buf = io.StringIO()
        hooks_stub = _mod.StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        with patch.object(_mod, "Live") as mock_live_cls, patch("sys.stdout", buf):
            mock_live_cls.return_value = MagicMock()
            overlay = _mod._make_streaming_overlay(hooks_stub)
            handler = overlay["llm:stream_block_start"]

            result = await handler(
                "llm:stream_block_start",
                {
                    "session_id": "sess-parent",  # no underscore -> agent is None
                    "block_index": 0,
                    "block_type": "text",
                },
            )

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert "Amplifier:" not in buf.getvalue(), (
            f"'Amplifier:' must NOT appear in overlay output for text block; got: {buf.getvalue()!r}"
        )

    @pytest.mark.asyncio
    async def test_label_not_printed_for_thinking_block(self):
        """Overlay does NOT print 'Amplifier:' for thinking blocks (they have their own header)."""
        import amplifier_module_hooks_streaming_ui as _mod

        buf = io.StringIO()
        hooks_stub = _mod.StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        with patch.object(_mod, "Live") as mock_live_cls, patch("sys.stdout", buf):
            mock_live_cls.return_value = MagicMock()
            overlay = _mod._make_streaming_overlay(hooks_stub)
            handler = overlay["llm:stream_block_start"]

            result = await handler(
                "llm:stream_block_start",
                {
                    "session_id": "sess-parent",
                    "block_index": 0,
                    "block_type": "thinking",
                },
            )

        assert result.action == "continue"
        assert "Amplifier:" not in buf.getvalue(), (
            f"'Amplifier:' should not appear for thinking block; got: {buf.getvalue()!r}"
        )

    @pytest.mark.asyncio
    async def test_label_not_printed_for_sub_agent_text_block(self):
        """Overlay does NOT print 'Amplifier:' for sub-agent sessions (they use stderr)."""
        import amplifier_module_hooks_streaming_ui as _mod

        buf = io.StringIO()
        hooks_stub = _mod.StreamingUIHooks(
            show_thinking=True, show_tool_lines=5, show_token_usage=True
        )
        with patch.object(_mod, "Live") as mock_live_cls, patch("sys.stdout", buf):
            mock_live_cls.return_value = MagicMock()
            overlay = _mod._make_streaming_overlay(hooks_stub)
            handler = overlay["llm:stream_block_start"]

            # Sub-agent session IDs contain an underscore after the span portion.
            result = await handler(
                "llm:stream_block_start",
                {
                    "session_id": "parent-span_explorer",  # underscore -> agent='explorer'
                    "block_index": 0,
                    "block_type": "text",
                },
            )

        assert result.action == "continue"
        assert "Amplifier:" not in buf.getvalue(), (
            f"'Amplifier:' should not appear for sub-agent block; got: {buf.getvalue()!r}"
        )


# ---------------------------------------------------------------------------
# Settled-aside dimming (dim=True) vs bright final/streaming (dim=False)
# ---------------------------------------------------------------------------


def _render_text_renderable(dim: bool) -> str:
    import io as _io

    import amplifier_module_hooks_streaming_ui as _mod
    from rich.console import Console

    buf = _io.StringIO()
    Console(file=buf, width=80, force_terminal=True, color_system="standard").print(
        _mod._text_renderable("hello world", dim=dim)
    )
    return buf.getvalue()


def test_dim_aside_dims_label_and_body():
    """dim=True emits a dim ANSI sequence (label + body); content/label intact."""
    out = _render_text_renderable(dim=True)
    assert "Amplifier:" in out and "hello world" in out
    assert "\x1b[2m" in out  # dim SGR present


def test_bright_final_has_no_dim():
    """dim=False (final response / streaming preview) emits NO dim SGR."""
    out = _render_text_renderable(dim=False)
    assert "Amplifier:" in out and "hello world" in out
    assert "\x1b[2m" not in out
