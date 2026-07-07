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

    def test_above_one_cent_uses_two_decimal_places(self):
        assert format_cost_usd(Decimal("0.09")) == "$0.09"
        assert format_cost_usd(Decimal("1.23")) == "$1.23"

    def test_sub_cent_uses_two_significant_figures(self):
        assert format_cost_usd(Decimal("0.0043")) == "$0.0043"
        assert format_cost_usd(Decimal("0.0001")) == "$0.0001"

    def test_sub_cent_truncates_raw_decimal_arithmetic(self):
        """Regression for #231: raw Decimal values with many decimal places are
        formatted to 2 significant figures, not shown as-is.

        Brian saw "$0.00639785" in the display — the correct output is "$0.0064".
        The formatter must round to 2 sig figs, never leak raw Decimal precision.
        """
        assert format_cost_usd(Decimal("0.00639785")) == "$0.0064"
        assert format_cost_usd(Decimal("0.0099")) == "$0.0099"
        assert format_cost_usd(Decimal("0.0047")) == "$0.0047"

    def test_string_input_is_coerced(self):
        """format_cost_usd must accept str — cost_usd travels as str through event dicts.

        If a caller passes the raw event value directly (without first wrapping in
        Decimal), the function should format it correctly rather than silently
        returning '?' via a TypeError on the Decimal comparison.
        """
        assert format_cost_usd("0.09") == "$0.09"
        assert format_cost_usd("0.0064") == "$0.0064"
        assert format_cost_usd("0") == "$0.00"
        assert format_cost_usd("not-a-number") == "?"

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
    coordinator.collect_contributions = AsyncMock(
        side_effect=RuntimeError("contributor exploded")
    )

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

    result = await handler(
        "orchestrator:complete", {"session_id": "abc-def_agent-name"}
    )

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


# ---------------------------------------------------------------------------
# Resume cost baseline: the prompt:submit seed handler
#
# On a resumed session app-cli restores prior spend by registering a
# history:<session_id> contributor on the session.cost channel *before* the
# first turn runs.  Without a pre-turn baseline the first orchestrator:complete
# reports the entire cumulative total as this turn's cost (Turn == Session on
# the first line after resume).  The prompt:submit seed handler captures the
# pre-turn snapshot into state["prev_total"] so the first turn's delta is right.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_seed_produces_correct_first_turn_delta(capsys):
    """Resumed session: first turn shows the delta, not the whole restored total.

    Sequence:
      1. prompt:submit fires -> channel has the restored history contributor
         ($5.00). Seed captures prev_total = $5.00.
      2. First turn runs; provider adds $0.10 -> channel total $5.10.
      3. orchestrator:complete -> Turn = $5.10 - $5.00 = $0.10, Session = $5.10.
    """
    coordinator = MagicMock()

    from amplifier_module_hooks_streaming_ui import (
        _make_cost_handler,
        _make_cost_seed_handler,
    )

    complete_handler, state = _make_cost_handler(coordinator)
    seed_handler = _make_cost_seed_handler(coordinator, state)

    # 1. Pre-turn snapshot on resume: only the restored history contributor.
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("5.00")}]
    )
    await seed_handler("prompt:submit", {})
    assert state["prev_total"] == Decimal("5.00")

    # 2 + 3. First turn completes with the restored total plus this turn's cost.
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("5.10")}]
    )
    await complete_handler("orchestrator:complete", {})
    captured = capsys.readouterr()

    assert "Turn: $0.10" in captured.out
    assert "Session: $5.10" in captured.out


@pytest.mark.asyncio
async def test_fresh_session_seed_is_noop(capsys):
    """Fresh session: nothing on the channel at prompt:submit -> baseline stays None.

    The first turn must still report Turn == Session (the whole first-turn cost),
    exactly as before this fix.
    """
    coordinator = MagicMock()

    from amplifier_module_hooks_streaming_ui import (
        _make_cost_handler,
        _make_cost_seed_handler,
    )

    complete_handler, state = _make_cost_handler(coordinator)
    seed_handler = _make_cost_seed_handler(coordinator, state)

    # Fresh session: no contributors yet at prompt:submit.
    coordinator.collect_contributions = AsyncMock(return_value=[])
    await seed_handler("prompt:submit", {})
    assert state["prev_total"] is None

    # First turn completes at $0.09.
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.09")}]
    )
    await complete_handler("orchestrator:complete", {})
    captured = capsys.readouterr()

    assert "Turn: $0.09" in captured.out
    assert "Session: $0.09" in captured.out


@pytest.mark.asyncio
async def test_seed_only_fires_once(capsys):
    """The seed handler acts only on the first turn; turn 2+ baselines are owned
    by the complete handler and must not be re-seeded."""
    coordinator = MagicMock()

    from amplifier_module_hooks_streaming_ui import (
        _make_cost_handler,
        _make_cost_seed_handler,
    )

    complete_handler, state = _make_cost_handler(coordinator)
    seed_handler = _make_cost_seed_handler(coordinator, state)

    # Turn 1: seed from restored history, complete at $5.10.
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("5.00")}]
    )
    await seed_handler("prompt:submit", {})
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("5.10")}]
    )
    await complete_handler("orchestrator:complete", {})
    capsys.readouterr()

    # Turn 2: seed must be a no-op even though the channel total changed.
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("5.10")}]
    )
    await seed_handler("prompt:submit", {})
    assert state["prev_total"] == Decimal("5.10")  # unchanged by the seed

    # Turn 2 completes at $5.25 -> Turn = $0.15.
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("5.25")}]
    )
    await complete_handler("orchestrator:complete", {})
    captured = capsys.readouterr()
    assert "Turn: $0.15" in captured.out
    assert "Session: $5.25" in captured.out


@pytest.mark.asyncio
async def test_seed_skips_subsession_none_session_id():
    """A sub-session prompt:submit (session_id=None) must not seed the root baseline."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.04")}]
    )

    from amplifier_module_hooks_streaming_ui import (
        _make_cost_handler,
        _make_cost_seed_handler,
    )

    _, state = _make_cost_handler(coordinator)
    seed_handler = _make_cost_seed_handler(coordinator, state)

    await seed_handler("prompt:submit", {"session_id": None})
    assert state["prev_total"] is None


@pytest.mark.asyncio
async def test_seed_skips_subsession_explicit_id():
    """A sub-session prompt:submit (underscore session_id) must not seed the baseline."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        return_value=[{"cost_usd": Decimal("0.04")}]
    )

    from amplifier_module_hooks_streaming_ui import (
        _make_cost_handler,
        _make_cost_seed_handler,
    )

    _, state = _make_cost_handler(coordinator)
    seed_handler = _make_cost_seed_handler(coordinator, state)

    await seed_handler("prompt:submit", {"session_id": "abc-def_agent-name"})
    assert state["prev_total"] is None


@pytest.mark.asyncio
async def test_seed_degrades_on_error():
    """Seeding is best-effort: if collect_contributions raises, the turn proceeds
    with prev_total left as None (falls back to first-turn behavior)."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(
        side_effect=RuntimeError("contributor exploded")
    )

    from amplifier_module_hooks_streaming_ui import (
        _make_cost_handler,
        _make_cost_seed_handler,
    )

    _, state = _make_cost_handler(coordinator)
    seed_handler = _make_cost_seed_handler(coordinator, state)

    result = await seed_handler("prompt:submit", {})
    assert state["prev_total"] is None
    assert isinstance(result, HookResult)
    assert result.action == "continue"


@pytest.mark.asyncio
async def test_mount_registers_prompt_submit_cost_seed():
    """mount() registers a prompt:submit handler to seed the cost baseline."""
    coordinator = MagicMock()
    coordinator.collect_contributions = AsyncMock(return_value=[])
    registered_hooks = {}

    def capture_hook(event, handler, **kwargs):
        registered_hooks.setdefault(event, []).append(kwargs.get("name"))

    coordinator.hooks.register = capture_hook
    coordinator.mount = AsyncMock()

    from amplifier_module_hooks_streaming_ui import mount

    await mount(coordinator, config={})

    assert "prompt:submit" in registered_hooks
    assert "streaming-ui-cost-seed" in registered_hooks["prompt:submit"]
