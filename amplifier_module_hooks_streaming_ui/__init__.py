"""Streaming UI Hooks Module

Display streaming LLM output (thinking blocks, tool calls, and token usage) to console.
"""

# Amplifier module metadata
__amplifier_module_type__ = "hook"

import logging
import math
import sys
from decimal import Decimal
from typing import Any

from amplifier_core.models import HookResult
from rich.console import Console
from rich.markdown import Markdown

logger = logging.getLogger(__name__)


async def mount(coordinator: Any, config: dict[str, Any]) -> None:
    """Mount streaming UI hooks module.

    Args:
        coordinator: The amplifier coordinator instance
        config: Configuration from profile
    """
    # Extract config from ui section
    ui_config = config.get("ui", {})
    show_thinking = ui_config.get("show_thinking_stream", True)
    show_tool_lines = ui_config.get("show_tool_lines", 5)
    show_token_usage = ui_config.get("show_token_usage", True)

    # Create hook handlers
    hooks = StreamingUIHooks(show_thinking, show_tool_lines, show_token_usage)

    # Register hooks on the coordinator
    coordinator.hooks.register(
        "content_block:start",
        hooks.handle_content_block_start,
        name="streaming-ui-content-block-start",
    )
    coordinator.hooks.register(
        "content_block:end",
        hooks.handle_content_block_end,
        name="streaming-ui-content-block-end",
    )
    coordinator.hooks.register(
        "tool:pre", hooks.handle_tool_pre, name="streaming-ui-tool-pre"
    )
    coordinator.hooks.register(
        "tool:post", hooks.handle_tool_post, name="streaming-ui-tool-post"
    )
    coordinator.hooks.register(
        "llm:response", hooks.handle_llm_response, name="streaming-ui-llm-response"
    )
    if show_token_usage:
        _cost_handler, _ = _make_cost_handler(coordinator)
        coordinator.hooks.register(
            "orchestrator:complete", _cost_handler, name="streaming-ui-cost-summary"
        )
    # Log successful mount
    logger.info("Mounted hooks-streaming-ui")

    return


class StreamingUIHooks:
    """Hooks for displaying streaming UI output."""

    def __init__(
        self, show_thinking: bool, show_tool_lines: int, show_token_usage: bool
    ):
        """Initialize streaming UI hooks.

        Args:
            show_thinking: Whether to display thinking blocks
            show_tool_lines: Number of lines to show for tool I/O
            show_token_usage: Whether to display token usage
        """
        self.show_thinking = show_thinking
        self.show_tool_lines = show_tool_lines
        self.show_token_usage = show_token_usage
        self.thinking_blocks: dict[int, dict[str, Any]] = {}
        self.last_llm_info: dict | None = None

    # ── Formula helper ─────────────────────────────────────────────────────

    def _compute_total_input(self, usage: dict) -> int:
        """Compute gross total input tokens.

        All providers report input_tokens as the gross total —
        fresh tokens plus any tokens read from the prompt cache.
        cache_read is already counted inside input_tokens, so adding
        it again would double-count. cache_write_tokens is the
        exception: cache creation cost is billed on top of gross
        and is NOT included in input_tokens.

        Args:
            usage: Usage dict from the event

        Returns:
            Gross total input token count
        """
        input_tokens = usage.get("input_tokens") or 0
        cache_create = (
            usage.get("cache_write_tokens")
            or usage.get("cache_creation_input_tokens")
            or 0
        )
        return input_tokens + cache_create

    # ── Hook handlers ──────────────────────────────────────────────────────

    async def handle_llm_response(
        self, _event: str, data: dict[str, Any]
    ) -> HookResult:
        """Capture model/provider info for display with token usage.

        Args:
            _event: Event name (llm:response) - unused
            data: Event data containing provider, model, duration_ms

        Returns:
            HookResult with action="continue"
        """
        self.last_llm_info = {
            "provider": data.get("provider"),
            "model": data.get("model"),
            "duration_ms": data.get("duration_ms"),
        }
        return HookResult(action="continue")

    def _parse_agent_from_session_id(self, session_id: str | None) -> str | None:
        """Extract agent name from hierarchical session ID.

        Session ID format follows W3C Trace Context principles:
        {parent-span}-{child-span}_{agent-name}

        Examples:
        - Sub-session: 0000000000000000-7cc787dd22d54f6c_developer-expertise-zen-architect
        - Parent session: 12345678-1234-1234-1234-123456789012 (no underscore, no agent)

        Args:
            session_id: Session ID with optional agent name after underscore

        Returns:
            Agent name if child session (contains underscore), None if parent session
        """
        if not session_id:
            return None

        # W3C Trace Context format: {parent-span}-{child-span}_{agent-name}
        # Underscore separator marks the boundary before agent name
        if "_" in session_id:
            parts = session_id.split("_", 1)  # Split on first underscore only
            if len(parts) == 2:
                # Everything after underscore is agent name
                # Handles namespaced agents like "developer-expertise-zen-architect"
                return parts[1]

        # No underscore = parent session (no agent name)
        return None

    async def handle_content_block_start(
        self, _event: str, data: dict[str, Any]
    ) -> HookResult:
        """Detect thinking blocks and prepare for display.

        Args:
            _event: Event name (content_block:start) - unused
            data: Event data containing block information

        Returns:
            HookResult with action="continue"
        """
        block_type = data.get("block_type")
        block_index = data.get("block_index")

        # Detect sub-agent context for visual distinction
        session_id = data.get("session_id")
        agent_name = self._parse_agent_from_session_id(session_id)

        # Only track thinking blocks if configured to show them
        if (
            block_type in {"thinking", "reasoning"}
            and self.show_thinking
            and block_index is not None
        ):
            self.thinking_blocks[block_index] = {"started": True, "agent": agent_name}
            if agent_name:
                # Sub-agent thinking: status line cyan, 4-space indent
                sys.stderr.write(
                    f"\n    \033[36m🤔 [{agent_name}] Thinking...\033[0m\n"
                )
                sys.stderr.flush()
            else:
                # Parent thinking: status line cyan
                sys.stderr.write("\n\033[36m🧠 Thinking...\033[0m\n")
                sys.stderr.flush()

        return HookResult(action="continue")

    async def handle_content_block_end(
        self, _event: str, data: dict[str, Any]
    ) -> HookResult:
        """Display complete thinking block and token usage.

        Args:
            _event: Event name (content_block:end) - unused
            data: Event data containing complete block, usage, and total count

        Returns:
            HookResult with action="continue"
        """
        block_index = data.get("block_index")
        total_blocks = data.get("total_blocks")
        block = data.get("block", {})
        block_type = block.get("type")
        usage = data.get("usage")  # Usage from parent response
        is_last_block = block_index == total_blocks - 1 if total_blocks else False

        # Parse agent name from session_id for consistent indentation
        # (used for both thinking blocks and token usage display)
        session_id = data.get("session_id")
        agent_name = self._parse_agent_from_session_id(session_id)

        # Override with tracked thinking block agent if available (for consistency)
        if block_index in self.thinking_blocks:
            tracked_agent = self.thinking_blocks[block_index].get("agent")
            if tracked_agent:
                agent_name = tracked_agent

        # Display thinking block if we were tracking it
        if (
            block_type in {"thinking", "reasoning"}
            and block_index is not None
            and block_index in self.thinking_blocks
        ):
            # Extract thinking text from block
            thinking_text = (
                block.get("thinking", "")
                or block.get("text", "")
                or _flatten_reasoning_block(block)
            )

            if thinking_text:
                # Display formatted thinking block with agent context
                if agent_name:
                    # Sub-agent thinking: dark gray, 4-space indent, markdown wrapped in dim ANSI codes
                    print(f"\n    \033[90m{'=' * 56}\033[0m")
                    print(f"    \033[90m[{agent_name}] Thinking:\033[0m")
                    print(f"    \033[90m{'-' * 56}\033[0m")
                    # Render markdown and wrap each line in dim ANSI code with indent
                    from io import StringIO

                    buffer = StringIO()
                    temp_console = Console(file=buffer, highlight=False, width=52)
                    temp_console.print(Markdown(thinking_text))
                    rendered = buffer.getvalue()
                    for line in rendered.rstrip().split("\n"):
                        # Wrap each line in dim ANSI code (same approach as tool results)
                        print(f"    \033[2m{line}\033[0m")
                    print(f"    \033[90m{'=' * 56}\033[0m\n")
                else:
                    # Parent thinking: markdown rendered and wrapped in dim ANSI codes
                    from io import StringIO

                    buffer = StringIO()
                    temp_console = Console(file=buffer, highlight=False, width=60)
                    temp_console.print(Markdown(thinking_text))
                    rendered = buffer.getvalue()

                    print(f"\n\033[90m{'=' * 60}\033[0m")
                    print("\033[90mThinking:\033[0m")
                    print(f"\033[90m{'-' * 60}\033[0m")
                    # Wrap markdown in dim ANSI code (same approach as tool results)
                    print(f"\033[2m{rendered.rstrip()}\033[0m")
                    print(f"\033[90m{'=' * 60}\033[0m\n")

            # Clean up tracking
            del self.thinking_blocks[block_index]

        # Display intermediate text blocks (P2 fix)
        # Only render text that accompanies tool calls (not the final response).
        # The final response (last block when stop_reason=end_turn) is rendered
        # by the main response path at full brightness.
        if block_type == "text" and not is_last_block and block.get("text", "").strip():
            text = block["text"]
            indent = "    " if agent_name else ""

            # Render through Rich Console + Markdown for proper line wrapping
            # (matches the pattern used by thinking blocks above)
            from io import StringIO

            wrap_width = 52 if agent_name else 60
            buffer = StringIO()
            temp_console = Console(file=buffer, highlight=False, width=wrap_width)
            temp_console.print(Markdown(text))
            rendered = buffer.getvalue()
            lines = rendered.rstrip().split("\n")
            line_count = len(lines)

            # ANSI 256-color escape sequences
            RESET = "\033[0m"

            if line_count < 3:
                # Whisper mode: ▸ prefix on first line, 2-space indent on continuation
                GLYPH_COLOR = "\033[38;5;110m"  # Soft blue for ▸
                TEXT_COLOR = "\033[38;5;188m"  # Muted warm white for text
                print(
                    f"\n{indent}{GLYPH_COLOR}\u25b8{RESET} {TEXT_COLOR}{lines[0]}{RESET}"
                )
                for line in lines[1:]:
                    print(f"{indent}  {TEXT_COLOR}{line}{RESET}")
                print()  # Blank line after
            else:
                # Rail mode: ▍ on every line
                RAIL_COLOR = "\033[38;5;103m"  # Muted lavender for ▍
                TEXT_COLOR = "\033[38;5;145m"  # Warm gray for text
                print()  # Blank line before
                for line in lines:
                    print(
                        f"{indent}{RAIL_COLOR}\u258d{RESET} {TEXT_COLOR}{line}{RESET}"
                    )
                print()  # Blank line after

        # Display token usage after last block (if present and configured)
        if is_last_block and self.show_token_usage and usage:
            indent = "    " if agent_name else ""

            # Get raw token counts (guard against None values from model_dump())
            output_tokens = usage.get("output_tokens") or 0

            # Cache metrics (Anthropic splits input into cached/uncached buckets)
            # Support both Anthropic-SDK field names and amplifier-core Usage model names
            cache_read = (
                usage.get("cache_read_input_tokens")
                or usage.get("cache_read_tokens")
                or 0
            )

            # Compute actual total input using helper (fixes double-count bug)
            total_input = self._compute_total_input(usage)
            total_tokens = total_input + output_tokens

            # Format numbers with thousands separators
            input_str = f"{total_input:,}"
            output_str = f"{output_tokens:,}"
            total_str = f"{total_tokens:,}"

            # Build cache info string if caching is active
            cache_info = ""
            if cache_read > 0:
                cache_pct = (
                    int((cache_read / total_input) * 100) if total_input > 0 else 0
                )
                cache_info = f" ({cache_pct}% cached)"
            else:
                cache_create = (
                    usage.get("cache_creation_input_tokens")
                    or usage.get("cache_write_tokens")
                    or 0
                )
                if cache_create > 0:
                    # First request - cache being created
                    cache_info = " (caching...)"

            # Build the header with model info if available
            if self.last_llm_info:
                provider = self.last_llm_info.get("provider") or ""
                model = self.last_llm_info.get("model") or ""
                duration_ms = self.last_llm_info.get("duration_ms")

                # Format duration as seconds with 1 decimal
                duration_str = f" [{duration_ms / 1000:.1f}s]" if duration_ms else ""

                header = f"📊 Token Usage ({provider}/{model}){duration_str}"
            else:
                header = "📊 Token Usage"

            # cost_usd may arrive as Decimal (from Pydantic model fields) or str
            # (from providers that serialize before emitting). Decimal(str(cost_raw))
            # handles both safely.
            cost_raw = usage.get("cost_usd")
            cost_part = ""
            if cost_raw is not None:
                try:
                    cost_part = f" | Cost: {format_cost_usd(Decimal(str(cost_raw)))}"
                except Exception:
                    cost_part = " | Cost: ?"

            print(f"{indent}\033[2m│  {header}\033[0m")
            print(
                f"{indent}\033[2m└─ Input: {input_str}{cache_info} | Output: {output_str} | Total: {total_str}{cost_part}\033[0m"
            )
            # Clear for next request to avoid stale data
            self.last_llm_info = None

        return HookResult(action="continue")

    async def handle_tool_pre(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display tool invocation with truncated input.

        Shows sub-agent tool calls with indentation and agent name for clarity.

        Args:
            _event: Event name (tool:pre) - unused
            data: Event data containing tool and arguments (includes session_id from defaults)

        Returns:
            HookResult with action="continue"
        """
        tool_name = data.get("tool_name", "unknown")
        tool_input = data.get("tool_input", {})
        session_id = data.get("session_id")

        # Detect if this is a sub-agent's tool call
        agent_name = self._parse_agent_from_session_id(session_id)

        # Format tool input for display with proper formatting
        input_str = self._format_for_display(tool_input)
        truncated = self._truncate_lines(input_str, self.show_tool_lines)

        if agent_name:
            # Sub-agent tool call: status line cyan, 4-space indent, box drawing
            print(f"\n    \033[36m┌─ 🔧 [{agent_name}] Using tool: {tool_name}\033[0m")
            # Indent each line of arguments
            for line in truncated.split("\n"):
                print(f"    \033[36m│\033[0m  \033[2m{line}\033[0m")
        else:
            # Parent tool call: status line cyan
            print(f"\n\033[36m🔧 Using tool: {tool_name}\033[0m")
            # Indent each line of arguments
            for line in truncated.split("\n"):
                print(f"   \033[2m{line}\033[0m")

        return HookResult(action="continue")

    async def handle_tool_post(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display tool result with truncated output.

        Shows sub-agent tool results with indentation and agent name for clarity.

        Args:
            _event: Event name (tool:post) - unused
            data: Event data containing tool result (includes session_id from defaults)

        Returns:
            HookResult with action="continue"
        """
        tool_name = data.get("tool_name", "unknown")
        result = data.get("tool_response", data.get("result", {}))
        session_id = data.get("session_id")

        # Detect if this is a sub-agent's tool result
        agent_name = self._parse_agent_from_session_id(session_id)

        # Extract output from result (handle different result formats)
        if isinstance(result, dict):
            raw_output = result.get("output")

            # Check for bash-style output with returncode (special case for stdout/stderr handling)
            bash_output = raw_output if isinstance(raw_output, dict) else result
            if isinstance(bash_output, dict) and "returncode" in bash_output:
                stdout = bash_output.get("stdout", "")
                stderr = bash_output.get("stderr", "")
                returncode = bash_output.get("returncode", 0)
                success = returncode == 0

                # Smart stdout/stderr combining based on success
                if success:
                    output = stdout or stderr or "(no output)"
                else:
                    output = stdout
                    if stderr:
                        output = (
                            f"{output}\n[stderr]: {stderr}"
                            if output
                            else f"[stderr]: {stderr}"
                        )
                    output = output or "(no output)"
            else:
                # Generic handling - format nicely as JSON if dict/list
                success = result.get("success", True)
                output = self._format_for_display(
                    raw_output if raw_output is not None else result
                )
        else:
            # Not a dict - format generically
            output = self._format_for_display(result)
            success = True

        # Truncate output for display
        truncated = self._truncate_lines(output, self.show_tool_lines)

        # Choose icon based on success
        icon = "✅" if success else "❌"

        if agent_name:
            # Sub-agent tool result: status line cyan, 4-space indent, box drawing
            print(
                f"    \033[36m└─ {icon} [{agent_name}] Tool result: {tool_name}\033[0m"
            )
            # Indent each line of multi-line output
            indented = "\n".join(f"       {line}" for line in truncated.split("\n"))
            print(f"\033[2m{indented}\033[0m\n")
        else:
            # Parent tool result: status line cyan
            print(f"\033[36m{icon} Tool result: {tool_name}\033[0m")
            # Indent each line of multi-line output
            indented = "\n".join(f"   {line}" for line in truncated.split("\n"))
            print(f"\033[2m{indented}\033[0m\n")

        return HookResult(action="continue")

    def _format_for_display(self, value: Any) -> str:
        """Format any value for readable display.

        Detects dict/list structures and formats them as YAML-style for
        cleaner output (no quotes, no braces).

        Args:
            value: Any value to format

        Returns:
            Formatted string representation
        """
        if value is None:
            return "(none)"

        # Already a string - return as-is (preserves natural newlines)
        if isinstance(value, str):
            return value if value else "(empty)"

        # Dict or list - format as YAML-style (cleaner than JSON)
        if isinstance(value, (dict, list)):
            try:
                return self._to_yaml_style(value)
            except Exception:
                return str(value)

        # Anything else - string representation
        return str(value)

    def _to_yaml_style(self, value: Any, indent: int = 0) -> str:
        """Convert value to YAML-style string (without pyyaml dependency).

        Args:
            value: Value to format
            indent: Current indentation level

        Returns:
            YAML-style formatted string
        """
        prefix = "  " * indent

        if value is None:
            return "null"

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, str):
            # Multi-line strings get block style
            if "\n" in value:
                lines = value.split("\n")
                return "|\n" + "\n".join(f"{prefix}  {line}" for line in lines)
            # Only quote if truly ambiguous YAML - very minimal set
            # Allow: paths, globs (*), regex patterns, most normal strings
            if value and value[0] not in "-?:,[]{}#&!|>'\"%@`" and ": " not in value:
                return value
            return f'"{value}"'

        if isinstance(value, list):
            if not value:
                return "[]"
            lines = []
            for item in value:
                if isinstance(item, dict):
                    # Format dict items with dash, keys aligned
                    for i, (k, v) in enumerate(item.items()):
                        formatted_v = self._to_yaml_style(v, indent + 1)
                        if i == 0:
                            lines.append(f"{prefix}- {k}: {formatted_v}")
                        else:
                            lines.append(f"{prefix}  {k}: {formatted_v}")
                else:
                    formatted = self._to_yaml_style(item, indent + 1)
                    lines.append(f"{prefix}- {formatted}")
            return "\n".join(lines)

        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = []
            for k, v in value.items():
                if isinstance(v, (dict, list)) and v:
                    # Nested structure - put on next line, no extra indent
                    formatted = self._to_yaml_style(v, indent)
                    lines.append(f"{prefix}{k}:")
                    lines.append(formatted)
                else:
                    formatted = self._to_yaml_style(v, indent + 1)
                    lines.append(f"{prefix}{k}: {formatted}")
            return "\n".join(lines)

        return str(value)

    def _truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to max_lines with ellipsis.

        Handles both multi-line text and single-line output (like dicts).
        For single lines over 200 chars, truncates with character limit.

        Args:
            text: Text to truncate (may be any type despite type hint)
            max_lines: Maximum number of lines to show

        Returns:
            Truncated text with ellipsis if needed
        """
        # Defensive: ensure text is actually a string before any operations
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text:
            return "(empty)"

        lines = text.split("\n")

        # If it's a single line over 200 chars, truncate by character
        if len(lines) == 1 and len(text) > 200:
            return text[:200] + f"... ({len(text) - 200} more chars)"

        # Multi-line: truncate by line count
        if len(lines) <= max_lines:
            return text

        # Truncate and add indicator
        truncated = lines[:max_lines]
        remaining = len(lines) - max_lines
        truncated.append(f"... ({remaining} more lines)")
        return "\n".join(truncated)


def _flatten_reasoning_block(block: dict[str, Any]) -> str:
    """Flatten OpenAI reasoning block structures into plain text."""
    fragments: list[str] = []

    def _collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value:
                fragments.append(value)
            return
        if isinstance(value, dict):
            _collect(value.get("text"))
            _collect(value.get("thinking"))
            _collect(value.get("summary"))
            _collect(value.get("content"))
            return
        if isinstance(value, list):
            for item in value:
                _collect(item)
            return
        text_attr = getattr(value, "text", None)
        if isinstance(text_attr, str) and text_attr:
            fragments.append(text_attr)

    _collect(block.get("thinking"))
    _collect(block.get("text"))
    _collect(block.get("summary"))
    _collect(block.get("content"))

    return "\n".join(fragment for fragment in fragments if fragment)


def format_cost_usd(cost: Decimal | str | None) -> str:
    """Format cost for terminal display.

    None  → "?"       (no rate data — never show $0.00 for unknown)
    0     → "$0.00"   (known-free)
    ≥0.01 → "$X.XX"   (2 decimal places, e.g. "$0.09")
    <0.01 → 2 significant figures, e.g. "$0.0064" for $0.00639785

    Accepts Decimal or str — cost_usd values travel as strings through event
    dicts and a direct str call must not silently produce "?".
    """
    if cost is None:
        return "?"
    if isinstance(cost, str):
        try:
            cost = Decimal(cost)
        except Exception:
            return "?"
    if cost == Decimal("0"):
        return "$0.00"
    if cost < Decimal("0"):
        return "?"  # edge case: negative delta from contribution accounting
    if cost >= Decimal("0.01"):
        return f"${cost:.2f}"
    # Sub-cent: 2 significant figures. Do not use Decimal.as_tuple().exponent here —
    # intermediate Decimal arithmetic (e.g. tokens * rate / 1_000_000) stores many
    # trailing decimal places that would produce $0.000030000 instead of $0.00003.
    exp_floor = math.floor(math.log10(float(cost)))  # e.g. -3 for 0.0064, -4 for 0.0001
    decimal_places = -exp_floor + 1  # 2 sig figs
    result = f"${cost:.{decimal_places}f}"
    # Strip trailing zeros from computed Decimal precision (e.g. "$0.00400" → "$0.004").
    # Never strips past the decimal point.
    return result.rstrip("0") or "$0.00"


# Local copy of the cost-summing helper. Modules cannot depend on amplifier-foundation,
# so this cannot be imported from amplifier_foundation (public: sum_cost_usd).
# Keep in sync with the canonical version there: if you fix a bug here, fix it there too.
def _sum_cost_usd(contributions: list) -> Decimal | None:
    """Sum cost_usd from collect_contributions("session.cost") results.

    Returns None if no cost data is present. None != Decimal("0"):
    None means unknown (no rate data); 0 means known-free.
    Silently skips malformed values rather than raising.
    """
    total: Decimal | None = None
    for c in contributions:
        if c and isinstance(c, dict):
            cost = c.get("cost_usd")
            if cost is not None:
                if isinstance(cost, Decimal):
                    total = (total or Decimal("0")) + cost
                else:
                    try:
                        total = (total or Decimal("0")) + Decimal(str(cost))
                    except Exception:
                        pass  # malformed cost_usd value; skip and degrade gracefully
    return total


def _make_cost_handler(coordinator):
    """Create the orchestrator:complete handler and its state.

    Returns (handler_coroutine, state_dict) so tests can inspect state.
    The state dict has key 'prev_total'.
    """
    state: dict[str, Decimal | None] = {"prev_total": None}

    async def _on_orchestrator_complete(event: str, data: dict):
        # Skip sub-session orchestrator completions — only the root turn fires the cost summary.
        # Sub-session events propagate up through the hook bus; two forms must be excluded:
        #   (a) explicit session ID with "_" delimiter (e.g. "abc-def_agent-name"), or
        #   (b) session_id=None — a sub-session that omitted its ID from the payload.
        # Root sessions either omit session_id entirely or carry a plain UUID (no underscore).
        # The old guard `if session_id and "_" in session_id` was a truthiness check that
        # short-circuits to False when session_id is None, letting (b) slip through and
        # fire a spurious mid-turn summary while corrupting state["prev_total"].
        session_id = data.get("session_id")
        if "session_id" in data and session_id is None:
            return HookResult(action="continue")
        if session_id and "_" in session_id:
            return HookResult(action="continue")
        try:
            contributions = await coordinator.collect_contributions("session.cost")
            session_total = _sum_cost_usd(contributions)

            prev = state["prev_total"]
            if session_total is not None and prev is not None:
                turn_cost = session_total - prev
            else:
                turn_cost = (
                    session_total  # first turn: turn cost = session total so far
                )

            state["prev_total"] = session_total

            turn_str = format_cost_usd(turn_cost)
            session_str = format_cost_usd(session_total)
        except Exception:
            turn_str = "?"
            session_str = "?"

        print(f"\033[2m💰 Turn: {turn_str} | Session: {session_str}\033[0m", flush=True)

        return HookResult(action="continue")

    return _on_orchestrator_complete, state


__all__ = [
    "mount",
    "StreamingUIHooks",
    "format_cost_usd",
]
