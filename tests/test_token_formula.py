"""Tests for token formula correctness."""

from amplifier_module_hooks_streaming_ui import StreamingUIHooks


def test_total_input_does_not_add_cache_read():
    """After Anthropic gross-total fix, cache_read is already in input_tokens."""
    hooks = StreamingUIHooks(show_thinking=False, show_tool_lines=5, show_token_usage=True)
    usage = {
        "input_tokens": 110_000,
        "cache_read_tokens": 100_000,
        "cache_write_tokens": 0,
        "output_tokens": 200,
        "total_tokens": 110_200,
    }
    total_input = hooks._compute_total_input(usage)
    assert total_input == 110_000  # NOT 210_000
