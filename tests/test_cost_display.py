"""Tests for cost display formatting in hooks-streaming-ui."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.models import HookResult

from amplifier_module_hooks_streaming_ui import _sum_cost_usd, format_cost_usd


class TestFormatCostUsd:
    def test_none_returns_question_mark(self):
        assert format_cost_usd(None) == "?"

    def test_zero_returns_zero_dollars(self):
        assert format_cost_usd(Decimal("0")) == "$0.00"

    def test_above_one_cent_uses_two_decimal(self):
        assert format_cost_usd(Decimal("0.09")) == "$0.09"
        assert format_cost_usd(Decimal("1.23")) == "$1.23"

    def test_sub_cent_uses_two_significant_figures(self):
        assert format_cost_usd(Decimal("0.0043")) == "$0.0043"
        assert format_cost_usd(Decimal("0.0001")) == "$0.0001"

    def test_never_returns_float(self):
        result = format_cost_usd(Decimal("0.05"))
        assert isinstance(result, str)


class TestSumCostUsd:
    def test_sums_contributions(self):
        contributions = [{"cost_usd": Decimal("0.03")}, {"cost_usd": Decimal("0.05")}]
        assert _sum_cost_usd(contributions) == Decimal("0.08")

    def test_empty_returns_none(self):
        assert _sum_cost_usd([]) is None

    def test_all_none_returns_none(self):
        assert _sum_cost_usd([{"cost_usd": None}]) is None

    def test_mixed_none_and_values(self):
        contributions = [{"cost_usd": Decimal("0.03")}, {"cost_usd": None}]
        assert _sum_cost_usd(contributions) == Decimal("0.03")


@pytest.mark.asyncio
async def test_orchestrator_complete_prints_cost_line(capsys):
    """On orchestrator:complete, hook prints the 💰 line with turn and session cost."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.09")}]
    )

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, _ = _make_cost_handler(coordinator)

    result = await handler("orchestrator:complete", {})

    captured = capsys.readouterr()
    assert "💰" in captured.out
    assert "$0.09" in captured.out  # both turn and session are $0.09 on first turn
    assert isinstance(result, HookResult)
    assert result.action == "continue"


@pytest.mark.asyncio
async def test_orchestrator_complete_computes_turn_delta(capsys):
    """Turn cost is the delta since the last orchestrator:complete."""
    coordinator = MagicMock()
    # First turn: $0.09 session total
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.09")}]
    )

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, _ = _make_cost_handler(coordinator)

    await handler("orchestrator:complete", {})
    capsys.readouterr()  # discard first turn output

    # Second turn: $0.18 session total → turn cost = $0.09
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.18")}]
    )
    await handler("orchestrator:complete", {})
    captured = capsys.readouterr()

    assert "Turn: $0.09" in captured.out
    assert "Session: $0.18" in captured.out


@pytest.mark.asyncio
async def test_orchestrator_complete_shows_question_mark_for_unknown_model(capsys):
    """When no cost data, displays ? for both turn and session."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(return_value=[])

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, _ = _make_cost_handler(coordinator)

    await handler("orchestrator:complete", {})
    captured = capsys.readouterr()

    assert "Turn: ?" in captured.out
    assert "Session: ?" in captured.out


@pytest.mark.asyncio
async def test_mount_registers_orchestrator_complete_handler():
    """mount() registers a handler for orchestrator:complete."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(return_value=[])
    registered_hooks = {}

    def capture_hook(event, handler, **kwargs):
        registered_hooks[event] = handler

    coordinator.hooks.register = capture_hook
    coordinator.mount = AsyncMock()

    from amplifier_module_hooks_streaming_ui import mount

    await mount(coordinator, config={})

    assert "orchestrator:complete" in registered_hooks


# ---------------------------------------------------------------------------
# Tests for the session cost accumulator — the unconditional llm:response
# hook that owns session.cost contributions (previously lived in providers).
# ---------------------------------------------------------------------------


class TestSessionCostAccumulator:
    """mount() registers an unconditional llm:response accumulator on session.cost."""

    def _make_coordinator(self):
        """Return a mock coordinator that captures hooks and contributors."""
        coordinator = MagicMock()
        coordinator.collect_contributions = AsyncMock(return_value=[])

        registered_hooks: dict = {}
        registered_contributors: dict = {}

        def capture_hook(event, handler, **kwargs):
            registered_hooks.setdefault(event, []).append(handler)

        def capture_contributor(channel, name, callback):
            registered_contributors[(channel, name)] = callback

        coordinator.hooks.register = capture_hook
        coordinator.register_contributor = capture_contributor

        return coordinator, registered_hooks, registered_contributors

    @pytest.mark.asyncio
    async def test_accumulator_registered_unconditionally(self):
        """session.cost contributor is registered regardless of show_token_usage."""
        from amplifier_module_hooks_streaming_ui import mount

        for show in (True, False):
            coordinator, _, contributors = self._make_coordinator()
            await mount(coordinator, config={"ui": {"show_token_usage": show}})
            assert ("session.cost", "streaming-ui-cost-accumulator") in contributors, (
                f"session.cost contributor missing when show_token_usage={show}"
            )

    @pytest.mark.asyncio
    async def test_accumulator_returns_none_before_any_events(self):
        """Contributor returns None (not $0) before any llm:response events."""
        from amplifier_module_hooks_streaming_ui import mount

        coordinator, _, contributors = self._make_coordinator()
        await mount(coordinator, config={})

        callback = contributors[("session.cost", "streaming-ui-cost-accumulator")]
        assert callback() is None

    @pytest.mark.asyncio
    async def test_accumulator_sums_cost_across_events(self):
        """Accumulator adds cost_usd from successive llm:response events."""
        from amplifier_module_hooks_streaming_ui import mount

        coordinator, hooks, contributors = self._make_coordinator()
        await mount(coordinator, config={})

        # Drive all registered llm:response handlers (display + accumulator)
        accumulators = [
            h
            for h in hooks.get("llm:response", [])
            if h.__name__ == "_accumulate_session_cost"
        ]
        assert len(accumulators) == 1, (
            "Expected exactly one _accumulate_session_cost handler"
        )
        accumulate = accumulators[0]

        await accumulate("llm:response", {"usage": {"cost_usd": Decimal("0.05")}})
        await accumulate("llm:response", {"usage": {"cost_usd": Decimal("0.03")}})

        callback = contributors[("session.cost", "streaming-ui-cost-accumulator")]
        result = callback()
        assert result == {"cost_usd": Decimal("0.08")}
        assert isinstance(result["cost_usd"], Decimal)

    @pytest.mark.asyncio
    async def test_accumulator_ignores_none_cost(self):
        """Events with cost_usd=None leave the contributor returning None."""
        from amplifier_module_hooks_streaming_ui import mount

        coordinator, hooks, contributors = self._make_coordinator()
        await mount(coordinator, config={})

        accumulators = [
            h
            for h in hooks.get("llm:response", [])
            if h.__name__ == "_accumulate_session_cost"
        ]
        accumulate = accumulators[0]

        await accumulate("llm:response", {"usage": {"cost_usd": None}})

        callback = contributors[("session.cost", "streaming-ui-cost-accumulator")]
        assert callback() is None
