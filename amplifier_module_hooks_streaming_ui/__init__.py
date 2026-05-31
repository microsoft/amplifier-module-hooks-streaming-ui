"""Streaming UI Hooks Module

Display streaming LLM output (thinking blocks, tool calls, and token usage) to console.
"""

# Amplifier module metadata
__amplifier_module_type__ = "hook"

import logging
import math
import os
import re
import signal
import sys
from decimal import Decimal
from typing import Any

from amplifier_core.models import HookResult
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Heading as _RichHeading
from rich.markdown import Markdown as _RichMarkdown
from rich.padding import Padding
from rich.styled import Styled
from rich.text import Text

logger = logging.getLogger(__name__)


# ─── Left-aligned Markdown ────────────────────────────────────────────────────
# Rich's default Markdown centers headings.  We always want left-aligned.
# Pattern mirrors amplifier-app-cli/amplifier_app_cli/console.py — defined
# locally so this module stays free of any app-layer dependency.


class _LeftAlignedHeading(_RichHeading):
    """Heading with left alignment — overrides Rich's default 'center'."""

    def __rich_console__(self, console: Console, options: Any) -> Any:  # type: ignore[override]
        text = self.text
        text.justify = "left"
        if self.tag == "h1":
            yield Text("")
            text.stylize("italic underline")
            yield text
            yield Text("")
        elif self.tag == "h2":
            yield Text("")
            text.stylize("bold")
            yield text
        else:
            text.stylize("dim")
            yield text


class Markdown(_RichMarkdown):
    """Markdown with left-aligned headings.

    Drop-in replacement for rich.markdown.Markdown used throughout this module
    so headings never centre-align, whether in the streaming Live preview, the
    final thinking render, or intermediate text blocks.
    """

    elements = {
        **_RichMarkdown.elements,
        "heading_open": _LeftAlignedHeading,
    }


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
    stream_tokens = ui_config.get("stream_tokens", False)

    # Determine overlay state up front so the atomic renderer can skip
    # thinking re-paints that the overlay already owns.
    _is_tty = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    overlay_active = _is_tty and stream_tokens

    # Create hook handlers
    hooks = StreamingUIHooks(
        show_thinking, show_tool_lines, show_token_usage, overlay_active=overlay_active
    )

    # Register atomic handlers (existing — unchanged in v2)
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
        _cost_handler, _ = _make_cost_handler(coordinator, hooks=hooks)
        coordinator.hooks.register(
            "orchestrator:complete", _cost_handler, name="streaming-ui-cost-summary"
        )

    # Register the deferred-flush handler on cleanup:render_end when the overlay
    # is active.  app-cli emits this event immediately after render_message paints
    # the final response, so Token Usage + cost land below the response text.
    if overlay_active:
        coordinator.hooks.register(
            "cleanup:render_end",
            hooks.handle_render_end,
            name="streaming-ui-render-end",
        )

    # --- v3 Transient Streaming Overlay --------------------------------------
    # Only register if stdout is a TTY. When piped or redirected, the CLI's
    # output is an API (`amplifier "x" > out.txt`); streamed tokens would be
    # unparseable noise. In non-TTY mode, the batch render at main.py fires
    # normally and output stays clean.
    #
    # v3 model: separate event channels for streaming lifecycle vs atomic.
    #   Overlay subscribes to llm:stream_* events (provider streaming lifecycle).
    #   Atomic renderer subscribes to content_block:* events (synthesized by
    #   loop-streaming from assembled response). No shared payload contract.
    #
    #   - Parent flavor: Rich Live(Markdown(buffer), transient=True). Live opens
    #     on llm:stream_block_start, updates on deltas, closes on
    #     llm:stream_block_end. For thinking blocks the overlay paints the
    #     permanent framed block immediately at block-end; the atomic renderer
    #     skips the re-paint (overlay_active flag). For text/response blocks
    #     the atomic renderer (and main.py) still own the final display.
    #   - Sub-agent flavor: per-session line buffer. Each delta accumulates;
    #     on newline, the complete line flushes atomically to stderr with
    #     cyan [agent] dim styling. Multiple parallel sub-agents produce
    #     non-interleaved prefixed lines. Atomic renderer paints bordered block.
    if _is_tty and stream_tokens:
        # --- v3 Transient overlay handlers -----------------------------------
        # Subscribes to llm:stream_* events (provider streaming lifecycle),
        # NOT content_block:* events (synthesized by loop-streaming).
        # This separation means the overlay and atomic renderer are on
        # independent event channels — no shared payload-field contract.
        _overlay = _make_streaming_overlay()
        coordinator.hooks.register(
            "llm:stream_block_start",
            _overlay["llm:stream_block_start"],
            name="streaming-ui-overlay-start",
        )
        coordinator.hooks.register(
            "llm:stream_block_delta",
            _overlay["llm:stream_block_delta"],
            name="streaming-ui-overlay-delta",
        )
        coordinator.hooks.register(
            "llm:stream_block_end",
            _overlay["llm:stream_block_end"],
            name="streaming-ui-overlay-end",
        )
        coordinator.hooks.register(
            "llm:stream_aborted",
            _overlay["llm:stream_aborted"],
            name="streaming-ui-overlay-aborted",
        )
        coordinator.hooks.register(
            "provider:retry",
            _overlay["provider:retry"],
            name="streaming-ui-overlay-retry",
        )
        coordinator.hooks.register(
            "prompt:submit",
            _overlay["prompt:submit"],
            name="streaming-ui-overlay-prompt-reset",
        )

    # Log successful mount
    logger.info(
        "Mounted hooks-streaming-ui (tty=%s, stream_tokens=%s) [v3]",
        _is_tty,
        stream_tokens,
    )

    return


class StreamingUIHooks:
    """Hooks for displaying streaming UI output."""

    def __init__(
        self,
        show_thinking: bool,
        show_tool_lines: int,
        show_token_usage: bool,
        overlay_active: bool = False,
    ):
        """Initialize streaming UI hooks.

        Args:
            show_thinking: Whether to display thinking blocks
            show_tool_lines: Number of lines to show for tool I/O
            show_token_usage: Whether to display token usage
            overlay_active: Whether the v3 transient streaming overlay is active.
                When True, the atomic renderer skips parent thinking re-paints
                because the overlay already painted them permanently at block-end.
                When False (default), the atomic renderer paints them as before.
        """
        self.show_thinking = show_thinking
        self.show_tool_lines = show_tool_lines
        self.show_token_usage = show_token_usage
        self.overlay_active = overlay_active
        self.thinking_blocks: dict[int, dict[str, Any]] = {}
        self.last_llm_info: dict | None = None
        # Deferred display state (overlay-active path only):
        # When overlay is on and the final response is a text block, Token Usage
        # and the turn cost line are stashed here instead of printed inline on
        # content_block:end / orchestrator:complete.  They flush on
        # cleanup:render_end (emitted by app-cli after render_message), so they
        # appear BELOW the rendered response rather than inserting above it.
        self._deferred_usage: tuple[str, str] | None = None  # (line1, line2)
        self._deferred_cost: str | None = None

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
        # Defensively discard any stale deferred state from the previous turn.
        # In normal operation content_block:end (final text) → orchestrator:complete
        # → cleanup:render_end flushes the stash before the next llm:response fires.
        # This guard protects against edge cases (e.g. aborted turns) where
        # render_end was never emitted.
        self._deferred_usage = None
        self._deferred_cost = None
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
                # Parent thinking: status line cyan.
                # Skip when the overlay is active — the overlay already shows a
                # live framed preview during streaming, so this status fires
                # AFTER the thinking block has been painted (at content_block:start
                # which arrives after llm:stream_block_end) and appears misplaced.
                if not self.overlay_active:
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
            # CHANGE B: When the overlay is active it already painted the
            # thinking block permanently at llm:stream_block_end (CHANGE A).
            # Skip the re-paint here to avoid a duplicate framed block.
            # The skip applies only to parent sessions (agent_name is None);
            # sub-agent thinking is still painted by the atomic renderer.
            overlay_owns_thinking = self.overlay_active and (agent_name is None)

            if not overlay_owns_thinking:
                # Extract thinking text from block
                thinking_text = (
                    block.get("thinking", "")
                    or block.get("text", "")
                    or _flatten_reasoning_block(block)
                )

                if thinking_text:
                    # Use the shared _thinking_renderable for both parent and
                    # sub-agent: full-width, dim+framed, left-aligned headings —
                    # identical to what the streaming Live preview shows, so the
                    # final-snap transition is not jarring.
                    out_console = Console(file=sys.stdout, highlight=False)
                    try:
                        render_width = out_console.size.width
                    except Exception:
                        render_width = 80
                    print()  # blank line before the block
                    out_console.print(
                        _thinking_renderable(
                            thinking_text,
                            width=render_width,
                            agent_name=agent_name,
                        )
                    )
                    print()  # blank line after the block

            # Always clean up tracking (whether painted or skipped)
            del self.thinking_blocks[block_index]

        # Display intermediate text blocks (P2 fix)
        # Only render text that accompanies tool calls (not the final response).
        # The final response (last block when stop_reason=end_turn) is rendered
        # by the main response path at full brightness.
        #
        # v3: no streaming suppression here. The overlay no longer paints
        # Markdown for text blocks, so there is no double-render risk.
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

            # Clear last_llm_info now — the info is already embedded in `header`.
            self.last_llm_info = None

            line1 = f"{indent}\033[2m│  {header}\033[0m"
            line2 = f"{indent}\033[2m└─ Input: {input_str}{cache_info} | Output: {output_str} | Total: {total_str}{cost_part}\033[0m"

            if self.overlay_active and block_type == "text":
                # Defer to cleanup:render_end (fires after render_message paints
                # the response).  This prevents usage from inserting above the
                # response — it will appear below it instead.
                self._deferred_usage = (line1, line2)
            else:
                # Inline print: overlay inactive, or last block is not a text
                # response (e.g. tool_use → render_message won't follow, so
                # there's no render_end to flush the stash).
                print()  # blank line separates Token Usage from preceding content
                print(line1)
                print(line2)

        return HookResult(action="continue")

    async def handle_render_end(self, _event: str, _data: dict[str, Any]) -> HookResult:
        """Flush deferred Token Usage + cost line after the response has rendered.

        app-cli emits cleanup:render_end immediately after render_message paints
        the final response.  By printing here, Token Usage and the turn cost line
        appear BELOW the response rather than inserting above it.

        Only registered when overlay_active=True (see mount()).  On the non-overlay
        path this handler is never registered so the stashes stay None.

        Resulting terminal order:
            Amplifier:
            [full markdown response]   ← painted by render_message
                                       ← blank line (this method)
            │  📊 Token Usage ...      ← flushed here
            └─ Input: ... | Output: ...
            💰 Turn: ... | Session: ... ← flushed here
        """
        if self._deferred_usage is not None or self._deferred_cost is not None:
            print()  # blank line between response and usage/cost
            if self._deferred_usage is not None:
                print(self._deferred_usage[0])
                print(self._deferred_usage[1])
                self._deferred_usage = None
            if self._deferred_cost is not None:
                print(self._deferred_cost, flush=True)
                self._deferred_cost = None
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


# =============================================================================
# Token-level streaming renderer
# =============================================================================
#
# Subscribes to the kernel-reserved delta events (content_block:delta,
# thinking:delta) plus our stream lifecycle events (content_block:stream_done,
# llm:stream_aborted, provider:retry, prompt:submit) and paints tokens to the
# terminal as they arrive.
#
# Output discipline (C11):
#   - Parent assistant text → stdout (matches the existing batch-render path
#     so `amplifier "x" > out.txt` keeps the same destination).
#   - Sub-agent text       → stderr, line-buffered with [agent] prefix.
#   - All thinking deltas  → stderr, dim styling (consistent with existing
#     thinking display in handle_content_block_end).
#   - All status markers   → stderr.
#
# Parallel sub-agent handling (C7, Invariant 10): sub-agent text is buffered
# per session_id and flushed atomically on newline or stream_done. Multiple
# sub-agents producing tokens at once interleave at line granularity, not
# per-token. The kernel hook bus's per-source FIFO (verified in spike S3)
# guarantees we never see a sub-agent's content_block:end before its last
# delta.
#
# ANSI swap (Invariant 12): on content_block:stream_done for the parent
# session, we cursor-up by the row count of Markdown(accumulated) (computed
# via console.render_lines, which matches wcwidth ground truth per spike S2),
# clear-down, and reprint as Markdown. Falls back to no swap (plain text
# stays) on capability-uncertain terminals or after SIGWINCH.
#
# Sanitizer (Invariant 9): model output is untrusted; treat it as such.
# Stateful per-session escape-sequence holder catches sequences split across
# delta boundaries. Unicode bidirectional control characters stripped
# explicitly (Trojan Source CVE-2021-42574).

_BIDI_CHARS = (
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE
    "\u2066"  # LEFT-TO-RIGHT ISOLATE
    "\u2067"  # RIGHT-TO-LEFT ISOLATE
    "\u2068"  # FIRST STRONG ISOLATE
    "\u2069"  # POP DIRECTIONAL ISOLATE
)
_BIDI_TRANS = {ord(c): None for c in _BIDI_CHARS}

# Complete escape sequences we strip: CSI (\x1b[...final), OSC (\x1b]...ST),
# and short Fp/Fs forms (\x1b<single final>).
_CSI_RE = re.compile(r"\x1b\[?[\x20-\x3F]*[\x40-\x7E]")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_FPFS_RE = re.compile(r"\x1b[@-Z\\-_]")


def _sanitize_delta(text: str, pending: str) -> tuple[str, str]:
    """Strip terminal control sequences and Unicode bidi controls from `text`,
    handling the case where an escape sequence is split across delta boundaries.

    `pending` is whatever incomplete escape was held over from the previous
    call. Returns (cleaned_text_to_emit, new_pending_to_hold).
    """
    full = pending + text

    # If the trailing tail of `full` looks like the start of an unterminated
    # escape sequence, hold from that point onward and emit only the prefix.
    new_pending = ""
    last_esc = full.rfind("\x1b")
    if last_esc >= 0:
        tail = full[last_esc:]
        # If tail matches a complete escape, no holding needed.
        if not (_CSI_RE.match(tail) or _OSC_RE.match(tail) or _FPFS_RE.match(tail)):
            # Incomplete escape — hold from \x1b onward.
            new_pending = tail
            full = full[:last_esc]

    # Strip every complete escape sequence we recognize.
    cleaned = _OSC_RE.sub("", full)
    cleaned = _CSI_RE.sub("", cleaned)
    cleaned = _FPFS_RE.sub("", cleaned)

    # Strip Unicode bidi controls.
    cleaned = cleaned.translate(_BIDI_TRANS)

    # Strip C0/C1 control characters except newline and tab.
    cleaned = "".join(
        c
        for c in cleaned
        if c in ("\n", "\t") or (0x20 <= ord(c) <= 0x7E) or ord(c) > 0x9F
    )

    return cleaned, new_pending


def _parse_agent(session_id: str | None) -> str | None:
    """Extract the agent name from a sub-session ID per W3C trace context:
    `{parent-span}-{child-span}_{agent-name}`. Returns None for the root
    session.
    """
    if not session_id or "_" not in session_id:
        return None
    parts = session_id.split("_", 1)
    return parts[1] if len(parts) == 2 else None


def _tail_buffer(buf: str, max_lines: int) -> str:
    """Return only the last max_lines lines of buf.

    Used as a fallback raw-line cap when rendered-height measurement is
    unavailable (e.g. mock consoles in tests).  The internal buffer keeps
    growing for correctness; only the slice shown to Live is trimmed.
    """
    if max_lines <= 0:
        return buf
    lines = buf.split("\n")
    if len(lines) <= max_lines:
        return buf
    return "\n".join(lines[-max_lines:])


def _fit_tail_to_height(
    buffer: str,
    budget_rows: int,
    render_fn: Any,
    console: Any,
) -> str:
    """Return the largest trailing slice of *buffer* whose RENDERED height
    (``render_fn(slice)`` measured on *console*) is <= *budget_rows*.

    Raw newline count is not a reliable proxy for rendered height: Rich
    Markdown expands plain text into more terminal rows through word-wrapping,
    list indentation, heading padding, and code-block borders.  Capping by
    raw lines alone lets long blocks overflow Rich's Live region → Live.stop()
    can only erase what's still on screen → scrolled-off rows survive as ghost
    text.

    Algorithm:
        1. Render the full buffer; if it fits, return it immediately.
        2. Otherwise start from the raw-line-cap tail as an upper bound and
           advance the start index by the measured excess on each iteration
           (aggressive drop keeps the loop short in practice).
        3. Fall back to :func:`_tail_buffer` if *console* doesn't implement
           ``render_lines`` (e.g. mock consoles in tests).

    Args:
        buffer:      Full accumulated streaming text.
        budget_rows: Maximum number of rendered terminal rows allowed.
        render_fn:   Callable ``(text: str) -> Rich renderable`` — e.g.
                     ``Markdown`` or ``lambda t: _thinking_renderable(t, width=w)``.
        console:     The Rich Console whose ``render_lines`` measures height.

    Returns:
        A trailing slice of *buffer* that renders within *budget_rows*.
    """
    if not buffer or budget_rows <= 0:
        return buffer

    lines = buffer.split("\n")
    n = len(lines)

    try:
        # Fast path: full buffer already fits.
        full_height = len(console.render_lines(render_fn(buffer), pad=False))
        if full_height <= budget_rows:
            return buffer

        # Start from where the raw-line cap would land (a reasonable upper bound
        # since rendered lines >= raw lines).  Then advance by the measured excess
        # to converge quickly without an O(n²) linear scan.
        start = max(0, n - budget_rows)
        while start < n - 1:
            tail = "\n".join(lines[start:])
            height = len(console.render_lines(render_fn(tail), pad=False))
            if height <= budget_rows:
                return tail
            # Advance by the overage — drops at least 1 line per iteration.
            start += max(1, height - budget_rows)

        # Last resort: single tail line.
        return lines[-1] if lines else buffer

    except Exception:
        # Console doesn't support render_lines (mock) or other failure.
        # Fall back to raw-line capping so existing tests are unaffected.
        return _tail_buffer(buffer, budget_rows)


def _thinking_renderable(
    content: str,
    *,
    width: int,
    agent_name: str | None = None,
) -> Any:
    """Return a Rich renderable for a thinking block.

    Renders as: dark-gray === / header / --- / dim markdown content / ===
    at the given width with left-aligned headings.  Suitable for both
    ``Live.update()`` during streaming and ``console.print()`` for the final
    atomic render, so both paths produce identical output.

    Args:
        content: Thinking block text (may be partial during streaming).
        width: Console width to use for the frame lines.
        agent_name: When set, adds 4-space indent and a ``[agent] Thinking:``
            header label; the content is also left-padded by 4 spaces.

    Returns:
        A Rich renderable (``Group``) suitable for ``Live.update()`` or
        ``console.print()``.
    """
    indent = "    " if agent_name else ""
    # Subtract indent width so the frame line doesn't overflow the terminal
    frame_width = max(10, width - (4 if agent_name else 0))
    header_label = f"[{agent_name}] Thinking:" if agent_name else "Thinking:"

    md = Markdown(content)
    dim_md: Any = Styled(md, "dim")
    if agent_name:
        # Indent the content block to align with the indented frame lines
        dim_md = Padding(dim_md, (0, 0, 0, 4))

    return Group(
        Text(indent + "=" * frame_width, style="bright_black"),
        Text(indent + header_label, style="bright_black"),
        Text(indent + "-" * frame_width, style="bright_black"),
        dim_md,
        Text(indent + "=" * frame_width, style="bright_black"),
    )


def _make_streaming_overlay():
    """v3 Transient Streaming Overlay.

    Per-block transient regions bounded by llm:stream_block_start and
    llm:stream_block_end events (provider streaming lifecycle channel).
    Completely separate from content_block:start/end (atomic renderer's
    channel, synthesized by loop-streaming). Two flavors based on session_id:

    Parent flavor (session_id has no underscore-agent suffix):
      - text / thinking blocks: Rich Live(Markdown(buffer), transient=True)
        opened at llm:stream_block_start, updated on deltas, closed at
        llm:stream_block_end. _tail_buffer() keeps the renderable bounded
        to terminal height so Rich's restore_cursor() clear is accurate.
        Final display handled by the atomic renderer via
        content_block:end (from loop-streaming with full assembled payload).
      - tool_use blocks: print "Building tool call: <name>..." placeholder
        at llm:stream_block_start. No Live region. The existing atomic flow
        paints the formatted tool box via tool:pre/tool:post events later.

    Sub-agent flavor (session_id matches `{parent}-{child}_{agent_name}`):
      - text / thinking blocks: per-session line buffer. Each delta accumulates
        into the buffer; on newline, the complete line flushes atomically to
        stderr with cyan [agent] dim styling. Multiple parallel sub-agents
        produce non-interleaved prefixed lines. The existing atomic renderer
        paints [agent] header + bordered block ADDITIVELY (decision D: don't
        clear streamed lines; let borders close around them).

    Sanitizer (invariant 9): per-session stateful, holds partial escape
    sequences across delta boundaries, strips ANSI control sequences and
    Unicode bidi controls (Trojan Source CVE-2021-42574).

    Returns a dict mapping event_name -> handler coroutine.
    """
    parent_console = Console(file=sys.stdout, highlight=False)

    # state[session_id] = {
    #     "agent": str | None,
    #     "blocks": {block_index: {
    #         "type": str,            # text | thinking | tool_use | ...
    #         "buffer": str,          # accumulated sanitized text
    #         "live": Live | None,    # parent flavor only
    #         "escape_pending": str,  # sanitizer carryover across deltas
    #         "name": str | None,     # tool_use block name
    #     }},
    # }
    state: dict[str, dict[str, Any]] = {}

    def _get_session(sid: str) -> dict[str, Any]:
        s = state.get(sid)
        if s is None:
            s = {"agent": _parse_agent(sid), "blocks": {}}
            state[sid] = s
        return s

    def _close_live(block: dict[str, Any]) -> None:
        live = block.get("live")
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass
            block["live"] = None

    def _reset_session(sid: str) -> None:
        s = state.get(sid)
        if s is None:
            return
        for block in s["blocks"].values():
            _close_live(block)
        s["blocks"] = {}

    async def _on_content_block_start(_event: str, data: dict[str, Any]) -> HookResult:
        sid = data.get("session_id") or ""
        idx = data.get("block_index")
        btype = data.get("block_type") or "text"
        if idx is None:
            return HookResult(action="continue")
        s = _get_session(sid)
        block: dict[str, Any] = {
            "type": btype,
            "buffer": "",
            "live": None,
            "escape_pending": "",
            "name": data.get("name"),
        }
        s["blocks"][idx] = block
        agent = s["agent"]

        if agent is None:
            # Parent flavor.
            if btype == "tool_use":
                # Decision A: placeholder, not args streaming.
                name = data.get("name") or "tool"
                try:
                    parent_console.print(
                        f"[dim]\U0001f527 Building tool call: {name}\u2026[/dim]"
                    )
                except BrokenPipeError:
                    pass
            elif btype in ("text", "thinking"):
                # Open Rich Live region. transient=True: Rich clears the
                # region on stop() via restore_cursor() which uses render-
                # height arithmetic. With _tail_buffer() keeping the Live
                # renderable bounded to terminal height, the clear is
                # always correct (no overflow into scrollback).
                try:
                    if btype == "thinking":
                        # Start with framed renderable so even the first
                        # frame shows "Thinking:" + dark-gray border.
                        try:
                            w = parent_console.size.width
                        except Exception:
                            w = 80
                        initial: Any = _thinking_renderable("", width=w)
                    else:
                        initial = Markdown("")
                    live = Live(
                        initial,
                        console=parent_console,
                        transient=True,
                        refresh_per_second=10,
                    )
                    live.start()
                    block["live"] = live
                except Exception:
                    # If Live can't start, fall back to no transient
                    # (delta handler will plain-print to stdout).
                    block["live"] = None
        # Sub-agent flavor: no setup needed at start. The line buffer
        # is initialized as part of the block dict above.
        return HookResult(action="continue")

    async def _on_delta(_event: str, data: dict[str, Any]) -> HookResult:
        """Handler for llm:stream_block_delta (all block types — text and thinking).
        block_type is read from the payload; if absent, falls back to the type
        recorded at llm:stream_block_start. Consumers route on block_type, not
        on event name."""
        text = data.get("text") or ""
        if not text:
            return HookResult(action="continue")
        sid = data.get("session_id") or ""
        idx = data.get("block_index")
        if idx is None:
            return HookResult(action="continue")
        s = _get_session(sid)
        block = s["blocks"].get(idx)
        if block is None:
            # Delta without a matching start (shouldn't happen but defensive).
            # Synthesize a minimal block entry; block_type comes from the
            # payload (present on every delta per the contract).
            block = {
                "type": data.get("block_type", "text"),
                "buffer": "",
                "live": None,
                "escape_pending": "",
                "name": None,
            }
            s["blocks"][idx] = block

        clean, block["escape_pending"] = _sanitize_delta(text, block["escape_pending"])
        if not clean:
            return HookResult(action="continue")

        block["buffer"] += clean
        agent = s["agent"]

        if agent is None:
            # Parent: update Live with progressive renderable.
            # Thinking blocks get the shared framed+dim renderable so the
            # streaming preview and the final atomic render look identical.
            # Text blocks get plain left-aligned Markdown (no frame).
            # Cap to terminal height minus a reserve so Live's
            # restore_cursor() arithmetic is always accurate.
            live = block.get("live")
            if live is not None:
                try:
                    btype = block.get("type", "text")
                    if btype == "thinking":
                        # Reserve 4 extra lines for the 4 frame lines
                        # (===, header, ---, ===) so the Live region stays
                        # within the terminal height including the frame.
                        try:
                            console_height = parent_console.size.height
                            console_width = parent_console.size.width
                        except Exception:
                            console_height = 24
                            console_width = 80
                        budget = max(5, console_height - 5 - 4)
                        tail = _fit_tail_to_height(
                            block["buffer"],
                            budget,
                            lambda t: _thinking_renderable(t, width=console_width),
                            parent_console,
                        )
                        live.update(_thinking_renderable(tail, width=console_width))
                    else:
                        try:
                            text_height = parent_console.size.height
                        except Exception:
                            text_height = 24
                        budget = max(5, text_height - 5)
                        tail = _fit_tail_to_height(
                            block["buffer"],
                            budget,
                            Markdown,
                            parent_console,
                        )
                        live.update(Markdown(tail))
                except Exception:
                    pass
            else:
                # No Live opened (synthesized block / tool_use / Live
                # failed to start). Plain-stdout fallback.
                try:
                    sys.stdout.write(clean)
                    sys.stdout.flush()
                except BrokenPipeError:
                    pass
        else:
            # Sub-agent: line-buffered. Flush each complete line as
            # an atomic write to stderr with [agent] cyan + dim styling.
            while "\n" in block["buffer"]:
                line, _, rest = block["buffer"].partition("\n")
                block["buffer"] = rest
                try:
                    sys.stderr.write(
                        f"    \033[36m[{agent}]\033[0m \033[2m{line}\033[0m\n"
                    )
                    sys.stderr.flush()
                except BrokenPipeError:
                    pass
        return HookResult(action="continue")

    async def _on_content_block_end(_event: str, data: dict[str, Any]) -> HookResult:
        sid = data.get("session_id") or ""
        idx = data.get("block_index")
        if idx is None:
            return HookResult(action="continue")
        s = _get_session(sid)
        block = s["blocks"].get(idx)
        if block is None:
            return HookResult(action="continue")

        agent = s["agent"]
        btype = block.get("type", "")
        buffer = block["buffer"]

        if agent is None:
            # Parent: close Live first (clears the transient capped preview).
            _close_live(block)

            # CHANGE A: For parent thinking blocks, immediately paint the
            # permanent framed version after the transient is cleared. This
            # resolves the thinking block in place so it stays on screen
            # while the response streams below it. The full accumulated
            # buffer is used (not tail-capped) so the complete thinking
            # scrolls naturally. The atomic renderer (handle_content_block_end
            # on StreamingUIHooks) sees overlay_active=True and skips the
            # re-paint, avoiding a duplicate framed block.
            #
            # Text/response blocks: no inline paint here — the atomic renderer
            # and main.py still own those final displays.
            # Tool_use blocks: no Live was opened; nothing to close. The
            # tool:pre/tool:post flow handles the formatted box.
            if btype == "thinking" and buffer.strip():
                try:
                    try:
                        w = parent_console.size.width
                    except Exception:
                        w = 80
                    parent_console.print(
                        _thinking_renderable(buffer, width=w, agent_name=None)
                    )
                except Exception:
                    pass
        else:
            # Sub-agent: flush any trailing partial line.
            tail = buffer
            block["buffer"] = ""
            if tail:
                try:
                    sys.stderr.write(
                        f"    \033[36m[{agent}]\033[0m \033[2m{tail}\033[0m\n"
                    )
                    sys.stderr.flush()
                except BrokenPipeError:
                    pass
        return HookResult(action="continue")

    async def _on_llm_stream_aborted(_event: str, data: dict[str, Any]) -> HookResult:
        sid = data.get("session_id") or ""
        s = state.get(sid)
        if s is None:
            return HookResult(action="continue")
        any_painted = any(block.get("buffer") for block in s["blocks"].values())
        for block in s["blocks"].values():
            _close_live(block)
        if any_painted:
            try:
                sys.stderr.write("\n\033[33m[stream interrupted]\033[0m\n")
                sys.stderr.flush()
            except BrokenPipeError:
                pass
        s["blocks"] = {}
        return HookResult(action="continue")

    async def _on_provider_retry(_event: str, data: dict[str, Any]) -> HookResult:
        sid = data.get("session_id") or ""
        s = state.get(sid)
        if s is None:
            return HookResult(action="continue")
        any_painted = any(block.get("buffer") for block in s["blocks"].values())
        for block in s["blocks"].values():
            _close_live(block)
        if any_painted:
            try:
                sys.stderr.write(
                    "\n\033[33m[retrying \u2014 previous partial output discarded]\033[0m\n"
                )
                sys.stderr.flush()
            except BrokenPipeError:
                pass
        s["blocks"] = {}
        return HookResult(action="continue")

    async def _on_prompt_submit(_event: str, data: dict[str, Any]) -> HookResult:
        sid = data.get("session_id") or ""
        _reset_session(sid)
        return HookResult(action="continue")

    return {
        "llm:stream_block_start": _on_content_block_start,
        "llm:stream_block_delta": _on_delta,
        "llm:stream_block_end": _on_content_block_end,
        "llm:stream_aborted": _on_llm_stream_aborted,
        "provider:retry": _on_provider_retry,
        "prompt:submit": _on_prompt_submit,
    }


def _make_cost_handler(coordinator, hooks=None):
    """Create the orchestrator:complete handler and its state.

    Returns (handler_coroutine, state_dict) so tests can inspect state.
    The state dict has key 'prev_total'.

    Args:
        coordinator: Amplifier coordinator for collect_contributions.
        hooks: Optional StreamingUIHooks instance.  When provided and
            hooks.overlay_active is True, the cost line is stashed on
            hooks._deferred_cost instead of printed inline.  It is then
            flushed by hooks.handle_render_end on cleanup:render_end
            (emitted by app-cli after render_message).
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

        cost_line = f"\033[2m💰 Turn: {turn_str} | Session: {session_str}\033[0m"

        if hooks is not None and hooks.overlay_active:
            # Defer to cleanup:render_end so the cost line appears after
            # render_message paints the final response (not before it).
            hooks._deferred_cost = cost_line
        else:
            print(cost_line, flush=True)

        return HookResult(action="continue")

    return _on_orchestrator_complete, state


__all__ = [
    "mount",
    "StreamingUIHooks",
    "format_cost_usd",
]
