"""Tests for the pinned-provider marker in the token-usage footer.

Covers observation of the `provider:resolve` event (basis="pinned" vs
"priority", scope filtering) and the resulting footer rendering, including
the honest-absence requirement: no marker is ever shown unless a
scope="conversation" resolution was actually observed.
"""

from __future__ import annotations

from typing import Any

import pytest
from amplifier_core.models import HookResult
from amplifier_module_hooks_streaming_ui import StreamingUIHooks


def _make_hooks() -> StreamingUIHooks:
    return StreamingUIHooks(
        show_thinking=False, show_tool_lines=5, show_token_usage=True
    )


class TestHandleProviderResolve:
    """Unit tests for the handle_provider_resolve hook handler itself."""

    def test_initial_state_is_none(self):
        hooks = _make_hooks()
        assert hooks._last_conversation_resolve is None

    @pytest.mark.asyncio
    async def test_conversation_scope_is_recorded(self):
        hooks = _make_hooks()
        data = {
            "provider": "anthropic-haiku",
            "model": "claude-haiku-4-5-20251001",
            "basis": "pinned",
            "scope": "conversation",
        }

        result = await hooks.handle_provider_resolve("provider:resolve", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert hooks._last_conversation_resolve == data

    @pytest.mark.asyncio
    async def test_non_conversation_scope_is_ignored(self):
        """A goal_utility (or any non-conversation) scope must never be recorded.

        These describe internal utility calls, not the user's visible turn,
        and must not influence the footer even if their basis is "pinned".
        """
        hooks = _make_hooks()
        data = {
            "provider": "anthropic-haiku",
            "model": "claude-haiku-4-5-20251001",
            "basis": "pinned",
            "scope": "goal_utility",
        }

        result = await hooks.handle_provider_resolve("provider:resolve", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert hooks._last_conversation_resolve is None

    @pytest.mark.asyncio
    async def test_non_conversation_scope_does_not_overwrite_prior_conversation_state(
        self,
    ):
        """A later non-conversation event must not clobber the last known
        conversation-scope resolution."""
        hooks = _make_hooks()
        conversation_event = {
            "provider": "anthropic-sonnet",
            "model": "claude-sonnet-5",
            "basis": "pinned",
            "scope": "conversation",
        }
        await hooks.handle_provider_resolve("provider:resolve", conversation_event)

        utility_event = {
            "provider": "anthropic-haiku",
            "model": "claude-haiku-4-5-20251001",
            "basis": "priority",
            "scope": "goal_utility",
        }
        await hooks.handle_provider_resolve("provider:resolve", utility_event)

        assert hooks._last_conversation_resolve == conversation_event

    @pytest.mark.asyncio
    async def test_missing_scope_key_is_treated_as_non_conversation(self):
        hooks = _make_hooks()
        data = {"provider": "anthropic", "model": "claude-x", "basis": "pinned"}

        result = await hooks.handle_provider_resolve("provider:resolve", data)

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert hooks._last_conversation_resolve is None

    @pytest.mark.asyncio
    async def test_malformed_payload_never_raises(self):
        """A malformed event payload must never break the hook (or, by
        extension, footer rendering later)."""
        hooks = _make_hooks()

        # `None` has no .get(); handler must swallow the AttributeError.
        result = await hooks.handle_provider_resolve("provider:resolve", None)  # type: ignore[arg-type]

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        assert hooks._last_conversation_resolve is None

    @pytest.mark.asyncio
    async def test_later_conversation_event_overwrites_earlier_one(self):
        """The footer must reflect the MOST RECENT conversation-scope
        resolution (e.g. the user unpinned mid-session)."""
        hooks = _make_hooks()
        pinned_event = {
            "provider": "anthropic-haiku",
            "model": "claude-haiku-4-5-20251001",
            "basis": "pinned",
            "scope": "conversation",
        }
        priority_event = {
            "provider": "anthropic-sonnet",
            "model": "claude-sonnet-5",
            "basis": "priority",
            "scope": "conversation",
        }

        await hooks.handle_provider_resolve("provider:resolve", pinned_event)
        await hooks.handle_provider_resolve("provider:resolve", priority_event)

        assert hooks._last_conversation_resolve == priority_event


class TestTokenUsageFooterPinnedMarker:
    """End-to-end: provider:resolve -> llm:response -> content_block:end footer."""

    def _usage_data(self) -> dict[str, Any]:
        return {
            "block_index": 0,
            "total_blocks": 1,
            "block": {"type": "text", "text": "Hello"},
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }

    @pytest.mark.asyncio
    async def test_pinned_marker_shown_in_footer(self, capsys):
        hooks = _make_hooks()

        await hooks.handle_provider_resolve(
            "provider:resolve",
            {
                "provider": "anthropic-haiku",
                "model": "claude-haiku-4-5-20251001",
                "basis": "pinned",
                "scope": "conversation",
            },
        )
        await hooks.handle_llm_response(
            "llm:response",
            {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "duration_ms": 1700,
            },
        )

        result = await hooks.handle_content_block_end(
            "content_block:end", self._usage_data()
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        captured = capsys.readouterr()
        assert (
            "📊 Token Usage (anthropic/claude-haiku-4-5-20251001 · 📌 pinned) [1.7s]"
            in captured.out
        )

    @pytest.mark.asyncio
    async def test_priority_basis_shows_no_marker(self, capsys):
        hooks = _make_hooks()

        await hooks.handle_provider_resolve(
            "provider:resolve",
            {
                "provider": "anthropic-sonnet",
                "model": "claude-sonnet-5",
                "basis": "priority",
                "scope": "conversation",
            },
        )
        await hooks.handle_llm_response(
            "llm:response",
            {
                "provider": "anthropic-sonnet",
                "model": "claude-sonnet-5",
                "duration_ms": 1700,
            },
        )

        await hooks.handle_content_block_end("content_block:end", self._usage_data())

        captured = capsys.readouterr()
        assert (
            "📊 Token Usage (anthropic-sonnet/claude-sonnet-5) [1.7s]" in captured.out
        )
        assert "📌" not in captured.out
        assert "pinned" not in captured.out

    @pytest.mark.asyncio
    async def test_no_event_seen_matches_todays_exact_output(self, capsys):
        """Honest absence: no provider:resolve ever observed -> footer renders
        EXACTLY as it did before this feature existed. No marker, no
        placeholder, no 'auto' label."""
        hooks = _make_hooks()
        assert hooks._last_conversation_resolve is None

        await hooks.handle_llm_response(
            "llm:response",
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250514",
                "duration_ms": 1700,
            },
        )

        await hooks.handle_content_block_end("content_block:end", self._usage_data())

        captured = capsys.readouterr()
        assert (
            "📊 Token Usage (anthropic/claude-sonnet-4-5-20250514) [1.7s]"
            in captured.out
        )
        assert "📌" not in captured.out
        assert "pinned" not in captured.out

    @pytest.mark.asyncio
    async def test_non_conversation_scope_never_produces_marker(self, capsys):
        """A goal_utility resolution with basis="pinned" must never leak a
        marker into the conversation-turn footer."""
        hooks = _make_hooks()

        await hooks.handle_provider_resolve(
            "provider:resolve",
            {
                "provider": "anthropic-haiku",
                "model": "claude-haiku-4-5-20251001",
                "basis": "pinned",
                "scope": "goal_utility",
            },
        )
        await hooks.handle_llm_response(
            "llm:response",
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250514",
                "duration_ms": 1700,
            },
        )

        await hooks.handle_content_block_end("content_block:end", self._usage_data())

        captured = capsys.readouterr()
        assert (
            "📊 Token Usage (anthropic/claude-sonnet-4-5-20250514) [1.7s]"
            in captured.out
        )
        assert "📌" not in captured.out
        assert "pinned" not in captured.out

    @pytest.mark.asyncio
    async def test_malformed_provider_resolve_event_never_breaks_footer(self, capsys):
        """Constraint: a malformed/unexpected provider:resolve payload must
        never break footer rendering -- the footer must always render."""
        hooks = _make_hooks()

        # Malformed payload observed first; must not raise and must not set state.
        await hooks.handle_provider_resolve("provider:resolve", None)  # type: ignore[arg-type]

        await hooks.handle_llm_response(
            "llm:response",
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250514",
                "duration_ms": 1700,
            },
        )

        result = await hooks.handle_content_block_end(
            "content_block:end", self._usage_data()
        )

        assert isinstance(result, HookResult)
        assert result.action == "continue"
        captured = capsys.readouterr()
        assert (
            "📊 Token Usage (anthropic/claude-sonnet-4-5-20250514) [1.7s]"
            in captured.out
        )
