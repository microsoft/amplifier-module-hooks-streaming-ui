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

    def test_above_one_cent_uses_four_significant_figures(self):
        assert format_cost_usd(Decimal("0.09")) == "$0.09"
        assert format_cost_usd(Decimal("1.23")) == "$1.23"

    def test_sub_cent_uses_four_significant_figures(self):
        assert format_cost_usd(Decimal("0.0043")) == "$0.0043"
        assert format_cost_usd(Decimal("0.0001")) == "$0.0001"

    def test_boundary_precision_near_one_cent(self):
        """Regression for #231: no precision cliff at the $0.01 boundary.

        The old two-regime split rounded $0.014 to "$0.01" and showed "$0.0099"
        for an almost identical amount — same money, radically different precision.
        """
        # Values just below $0.01 — must use 4 sig figs, not be rounded to $0.00xx
        assert format_cost_usd(Decimal("0.0099")) == "$0.0099"
        assert format_cost_usd(Decimal("0.0047")) == "$0.0047"
        # Values just above $0.01 — old code rounded to 2dp, losing significant digits
        assert format_cost_usd(Decimal("0.0101")) == "$0.0101"  # was "$0.01" before fix
        assert format_cost_usd(Decimal("0.014")) == "$0.014"    # was "$0.01" before fix
        assert format_cost_usd(Decimal("0.0933")) == "$0.0933"  # was "$0.09" before fix

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


@pytest.mark.asyncio
async def test_orchestrator_complete_degrades_on_error(capsys):
    """Handler degrades to '?' when collect_contributions raises — never crashes the orchestrator."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(side_effect=RuntimeError("contributor exploded"))

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, _ = _make_cost_handler(coordinator)

    result = await handler("orchestrator:complete", {})

    captured = capsys.readouterr()
    assert "💰" in captured.out
    assert "?" in captured.out
    assert isinstance(result, HookResult)
    assert result.action == "continue"


# ---------------------------------------------------------------------------
# Bug #230 regression: sub-session events must not fire the cost summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_complete_skips_subsession_none_session_id(capsys):
    """Regression for #230: session_id=None in payload must NOT fire the cost summary.

    The old guard `if session_id and "_" in session_id` short-circuits to False
    when session_id is None, letting sub-session events without an explicit ID
    slip through and print a spurious mid-turn line.
    """
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.04")}]
    )

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, state = _make_cost_handler(coordinator)

    result = await handler("orchestrator:complete", {"session_id": None})

    captured = capsys.readouterr()
    assert captured.out == ""  # no 💰 line printed
    assert state["prev_total"] is None  # state must not be mutated
    assert isinstance(result, HookResult)
    assert result.action == "continue"


@pytest.mark.asyncio
async def test_orchestrator_complete_skips_subsession_explicit_id(capsys):
    """Sub-session with explicit underscore session ID must NOT fire the cost summary."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.04")}]
    )

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, state = _make_cost_handler(coordinator)

    result = await handler("orchestrator:complete", {"session_id": "abc-def_agent-name"})

    captured = capsys.readouterr()
    assert captured.out == ""  # no 💰 line printed
    assert state["prev_total"] is None  # state must not be mutated
    assert isinstance(result, HookResult)
    assert result.action == "continue"


@pytest.mark.asyncio
async def test_orchestrator_complete_state_not_corrupted_by_subsession(capsys):
    """Regression for #230: a sub-session event must not corrupt prev_total.

    Old bug sequence:
      1. Sub-session completes mid-turn (session_id=None) → handler fires
      2. state["prev_total"] is set to the in-progress session total ($0.04)
      3. Root turn completes → turn_cost = $0.09 - $0.04 = $0.05 (WRONG)
         should be $0.09 (first root turn, prev_total should still be None)
    """
    coordinator = MagicMock()

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, state = _make_cost_handler(coordinator)

    # Sub-session fires first (mid-turn delegated agent completes)
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.04")}]
    )
    await handler("orchestrator:complete", {"session_id": None})
    capsys.readouterr()  # discard (should be empty; assertion covered by other test)

    # Root session fires (the real turn completion)
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.09")}]
    )
    await handler("orchestrator:complete", {})
    captured = capsys.readouterr()

    # Turn cost must be the full $0.09 (first root turn), not the $0.05 delta
    # that a corrupted prev_total of $0.04 would have produced.
    assert "Turn: $0.09" in captured.out
    assert "Session: $0.09" in captured.out


@pytest.mark.asyncio
async def test_orchestrator_complete_fires_for_root_with_uuid_session_id(capsys):
    """Root session with an explicit UUID (no underscore) still fires the cost summary."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.09")}]
    )

    from amplifier_module_hooks_streaming_ui import _make_cost_handler

    handler, _ = _make_cost_handler(coordinator)

    result = await handler(
        "orchestrator:complete",
        {"session_id": "12345678-abcd-1234-abcd-123456789012"},
    )

    captured = capsys.readouterr()
    assert "💰" in captured.out
    assert "$0.09" in captured.out
    assert isinstance(result, HookResult)
    assert result.action == "continue"
