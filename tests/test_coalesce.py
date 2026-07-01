"""FAILING-TEST-FIRST: THROTTLE/COALESCE spike for the streaming Live repaint.

Covers the pure decision helper ``should_refresh`` that governs whether the
transient streaming ``Live`` region should physically repaint on a given
delta, per docs/designs (mid-turn steering compose-state coalescing):

  - not composing                          -> always True  (today's feel,
                                               refresh on every delta)
  - composing, elapsed < COALESCE_S         -> False (skip; buffer keeps
                                               accumulating, no repaint)
  - composing, elapsed >= COALESCE_S        -> True  (repaint, reset the
                                               last-refresh clock)
  - force=True (block-end flush path)       -> always True regardless of
                                               composing / elapsed

These tests are written BEFORE ``should_refresh`` exists in
``amplifier_module_hooks_streaming_ui`` and MUST fail on an unmodified
checkout (ImportError / AttributeError), then pass once the helper is
implemented.
"""

from __future__ import annotations

import amplifier_module_hooks_streaming_ui as _mod

# COALESCE_S is expected to be a named module constant (default 0.25s).
COALESCE_S = getattr(_mod, "COALESCE_S", 0.25)


def _should_refresh(*args, **kwargs):
    """Import indirection so a missing symbol produces a clear failure."""
    return _mod.should_refresh(*args, **kwargs)


class TestNotComposingAlwaysRefreshes:
    """Not composing => always refresh, regardless of elapsed time."""

    def test_zero_elapsed(self):
        assert (
            _should_refresh(False, now=100.0, last_refresh=100.0, coalesce_s=COALESCE_S)
            is True
        )

    def test_tiny_elapsed(self):
        assert (
            _should_refresh(
                False, now=100.001, last_refresh=100.0, coalesce_s=COALESCE_S
            )
            is True
        )

    def test_large_elapsed(self):
        assert (
            _should_refresh(False, now=200.0, last_refresh=100.0, coalesce_s=COALESCE_S)
            is True
        )


class TestComposingWithinWindowSkips:
    """Composing + elapsed < coalesce_s => skip the repaint (return False)."""

    def test_just_under_window(self):
        assert (
            _should_refresh(True, now=100.24, last_refresh=100.0, coalesce_s=0.25)
            is False
        )

    def test_no_elapsed_time(self):
        assert (
            _should_refresh(True, now=100.0, last_refresh=100.0, coalesce_s=0.25)
            is False
        )

    def test_small_elapsed(self):
        assert (
            _should_refresh(True, now=100.1, last_refresh=100.0, coalesce_s=0.25)
            is False
        )


class TestComposingAfterWindowRefreshes:
    """Composing + elapsed >= coalesce_s => refresh (return True)."""

    def test_exactly_at_window(self):
        assert (
            _should_refresh(True, now=100.25, last_refresh=100.0, coalesce_s=0.25)
            is True
        )

    def test_past_window(self):
        assert (
            _should_refresh(True, now=101.0, last_refresh=100.0, coalesce_s=0.25)
            is True
        )


class TestForceFlushAlwaysRefreshes:
    """The block-end force-flush path always refreshes regardless of state."""

    def test_force_while_composing_within_window(self):
        assert (
            _should_refresh(
                True, now=100.01, last_refresh=100.0, coalesce_s=0.25, force=True
            )
            is True
        )

    def test_force_while_not_composing(self):
        assert (
            _should_refresh(
                False, now=100.0, last_refresh=100.0, coalesce_s=0.25, force=True
            )
            is True
        )
