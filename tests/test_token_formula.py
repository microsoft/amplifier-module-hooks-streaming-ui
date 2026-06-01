"""Tests for token formula correctness."""

from amplifier_module_hooks_streaming_ui import StreamingUIHooks


def test_total_input_does_not_add_cache_read():
    """input_tokens is gross total — cache_read is already inside it, never add again."""
    hooks = StreamingUIHooks(
        show_thinking=False, show_tool_lines=5, show_token_usage=True
    )
    usage = {
        "input_tokens": 110_000,  # gross: 10k fresh + 100k cache_read
        "cache_read_tokens": 100_000,
        "cache_write_tokens": 0,
        "output_tokens": 200,
        "total_tokens": 110_200,
    }
    total_input = hooks._compute_total_input(usage)
    assert total_input == 110_000  # NOT 210_000


def test_cache_write_tokens_added_to_total():
    """cache_write_tokens (cache creation cost) is NOT inside gross — add it."""
    hooks = StreamingUIHooks(
        show_thinking=False, show_tool_lines=5, show_token_usage=True
    )
    usage = {
        "input_tokens": 10_000,  # gross fresh input
        "cache_read_tokens": 0,
        "cache_write_tokens": 3_000,  # new cache being created — billed separately
        "output_tokens": 200,
    }
    total_input = hooks._compute_total_input(usage)
    assert total_input == 13_000  # input + cache_create


def test_hybrid_event_raw_and_canonical_fields_both_present():
    """The content_block:end event carries both raw Anthropic SDK field names
    (cache_read_input_tokens) and canonical amplifier-core field names
    (cache_read_tokens) simultaneously. Since input_tokens is already the
    gross total, cache_read_input_tokens must be ignored — not added again."""
    hooks = StreamingUIHooks(
        show_thinking=False, show_tool_lines=5, show_token_usage=True
    )
    # Real shape emitted by the streaming orchestrator for Anthropic responses
    usage = {
        "input_tokens": 51_095,  # GROSS (10 fresh + 51085 cached)
        "cache_read_tokens": 51_085,  # canonical
        "cache_read_input_tokens": 51_085,  # raw Anthropic — must be ignored
        "cache_write_tokens": 3_135,  # canonical
        "cache_creation_input_tokens": 3_135,  # raw Anthropic — used as fallback only
        "output_tokens": 75,
    }
    total_input = hooks._compute_total_input(usage)
    assert total_input == 54_230  # 51095 + 3135, NOT 105315 (double-count)
