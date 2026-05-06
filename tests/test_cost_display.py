"""Tests for cost display formatting in hooks-streaming-ui."""

from decimal import Decimal

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
