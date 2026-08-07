"""Streaming UI Hooks Module

Display streaming LLM output (thinking blocks, tool calls, and token usage) to console.
"""

# Amplifier module metadata
__amplifier_module_type__ = "hook"

import logging
import math
import re
import sys
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from amplifier_core.models import HookResult
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock as _RichCodeBlock
from rich.markdown import Heading as _RichHeading
from rich.markdown import Markdown as _RichMarkdown
from rich.padding import Padding
from rich.rule import Rule
from rich.styled import Styled
from rich.syntax import Syntax
from rich.text import Text

logger = logging.getLogger(__name__)


# ─── THROTTLE/COALESCE spike (revertable, display-only) ─────────────────────
# While the user is composing a mid-turn steer, the pinned steering prompt
# and the streaming Live preview repaint independently, fighting each other
# for the terminal. COALESCE_S bounds how often the streaming preview
# physically repaints WHILE a steer is being composed -- output is never
# hidden (the buffer keeps accumulating every delta; only the on-screen
# refresh cadence is reduced). When no steer is being composed, the preview
# refreshes on every delta exactly as it always has.
COALESCE_S = 0.25


def should_refresh(
    is_composing: bool,
    now: float,
    last_refresh: float,
    coalesce_s: float = COALESCE_S,
    *,
    force: bool = False,
) -> bool:
    """Decide whether the streaming Live region should physically repaint now.

    Pure decision function -- no Rich/Live/terminal dependency -- so the
    coalesce policy is unit-testable in isolation from rendering concerns.

    Args:
        is_composing: True while the user has a mid-turn steer prompt open
            with non-empty draft text (``SteeringInputManager.is_composing``).
        now: Current monotonic timestamp (e.g. ``time.monotonic()``).
        last_refresh: Monotonic timestamp of the last physical repaint for
            this block (0.0 if never refreshed).
        coalesce_s: Minimum seconds between repaints while composing.
            Defaults to the module constant ``COALESCE_S`` (0.25s).
        force: When True, always returns True regardless of the other
            arguments. Used by the block-end flush path to guarantee the
            final buffered text is painted before the Live region closes.

    Returns:
        True if a physical repaint should happen now, False if the repaint
        should be skipped (the caller keeps accumulating into the buffer
        and will retry the decision on the next delta).
    """
    if force:
        return True
    if not is_composing:
        return True
    return (now - last_refresh) >= coalesce_s


def effective_composing(is_composing: bool, steering_active: bool) -> bool:
    """Broaden the throttle gate to cover the whole time a steering prompt
    is visible on screen, not only while its draft buffer is non-empty.

    ROOT CAUSE (Option B): a pinned steering prompt fights the streaming
    Live repaint for the terminal via prompt_toolkit's run_in_terminal for
    as long as the prompt is on screen -- which is the entire duration
    ``_composing_fn`` is registered (a turn is in flight), not just the
    moments the user has typed non-empty draft text. Gating ``should_refresh``
    on ``is_composing`` alone under-throttles: an empty-but-visible steering
    prompt still causes an atomic erase/redraw on every delta.

    Args:
        is_composing: Whether the steer draft buffer currently has text
            (``SteeringInputManager.is_composing()``).
        steering_active: Whether a steering prompt capability is registered
            at all for this turn (``hooks_instance._composing_fn is not
            None``) -- i.e. whether a pinned prompt COULD be on screen.

    Returns:
        True if the throttle/coalesce window should apply to this delta.
        False only when there is no steering prompt on screen at all
        (``steering_active`` is False) and the draft is empty -- this
        preserves the original per-delta smooth streaming outside of any
        steering context.
    """
    return is_composing or steering_active


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


# ─── Copy/paste-clean code blocks ────────────────────────────────────────
# Rich's stock CodeBlock renders via Syntax(..., padding=1), which writes a
# literal space character onto the left (and right) of every line -- a real
# character in the terminal's screen buffer that a mouse-drag/triple-click
# copy captures. amplifier-app-cli/amplifier_app_cli/console.py fixed this
# for the settled assistant-message render; this module has its own local
# Markdown subclass (see note above) so the same fix must be duplicated
# here, or streaming/intermediate previews keep the padded, un-copy-pasteable
# fences even after the settled render is clean.


class _CopyPasteCodeBlock(_RichCodeBlock):
    """Code block with no per-line whitespace padding, for clean copy/paste.

    Mirrors amplifier-app-cli/amplifier_app_cli/console.py:CopyPasteCodeBlock --
    keeps the block visually identifiable (background tint + syntax
    highlighting, plus a thin rule above/below) without baking padding
    characters into the code lines themselves.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        code = str(self.text).rstrip()
        yield Rule(style="dim", characters="\u2500")
        yield Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=0,
        )
        yield Rule(style="dim", characters="\u2500")


class Markdown(_RichMarkdown):
    """Markdown with left-aligned headings and copy/paste-clean code blocks.

    Drop-in replacement for rich.markdown.Markdown used throughout this module
    so headings never centre-align and fenced code has no injected padding,
    whether in the streaming Live preview, the final thinking render, or
    intermediate text blocks.
    """

    elements = {
        **_RichMarkdown.elements,
        "heading_open": _LeftAlignedHeading,
        "fence": _CopyPasteCodeBlock,  # ``` code blocks: no copy/paste padding
        "code_block": _CopyPasteCodeBlock,  # indented code blocks: same treatment
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
    spawn_tools = ui_config.get("spawn_tools", ["task", "delegate"])

    # Determine overlay state up front so the atomic renderer can skip
    # thinking re-paints that the overlay already owns.
    _is_tty = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    overlay_active = _is_tty and stream_tokens

    # Create hook handlers
    hooks = StreamingUIHooks(
        show_thinking,
        show_tool_lines,
        show_token_usage,
        overlay_active=overlay_active,
        spawn_tools=spawn_tools,
    )

    # THROTTLE/COALESCE spike (revertable, display-only): publish the
    # StreamingUIHooks instance as a capability so app-cli's
    # _execute_with_interrupt can reach it and wire up
    # hooks._composing_fn = manager.is_composing. Registered unconditionally
    # (cheap, no behavior change on its own) so app-cli's get_capability()
    # call finds it whenever this module version is mounted; an app-cli
    # built against an older module simply gets None back (back-compat).
    coordinator.register_capability("ui.streaming_hooks", hooks)

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
        _cost_handler, _cost_state = _make_cost_handler(coordinator, hooks=hooks)
        coordinator.hooks.register(
            "orchestrator:complete", _cost_handler, name="streaming-ui-cost-summary"
        )
        # Seed the per-turn cost baseline at the start of the first turn.  On a
        # resumed session app-cli restores prior spend as a history:<session_id>
        # contributor on the session.cost channel before the first turn runs;
        # without a baseline the first turn's delta would report the whole
        # cumulative total as *this* turn's cost.  Fresh sessions have no prior
        # contributor, so the baseline stays None and behavior is unchanged.
        _cost_seed_handler = _make_cost_seed_handler(coordinator, _cost_state)
        coordinator.hooks.register(
            "prompt:submit", _cost_seed_handler, name="streaming-ui-cost-seed"
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
        _overlay = _make_streaming_overlay(hooks)
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
        spawn_tools: tuple[str, ...] = ("task", "delegate"),
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
            spawn_tools: Tool names that spawn sub-agents.  Any tool whose name
                appears in this tuple triggers per-agent list tracking.
                Default: ("task", "delegate").  Configurable via
                ``ui.spawn_tools`` in the mount config.
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
        # Per-session set of block indices that the overlay already painted as
        # interleaved text (whisper/rail).  handle_content_block_end checks this
        # to suppress the deferred duplicate render (Part 4 guard).
        # Keyed by session_id (str); values are sets of int block indices.
        self._overlay_painted_text: dict[str, set[int]] = {}
        # _spawn_tools: names of tools that spawn sub-agents (config-driven).
        # Used to gate the static per-spawn marker printed in handle_tool_pre.
        self._spawn_tools: tuple[str, ...] = tuple(spawn_tools)
        # THROTTLE/COALESCE spike (revertable, display-only): settable from
        # outside (app-cli's _execute_with_interrupt) to a zero-arg callable
        # returning True while the user is composing a mid-turn steer
        # (SteeringInputManager.is_composing). None (default) means "not
        # composing" -- the streaming Live preview refreshes on every delta
        # exactly as it always has. See _make_streaming_overlay's _on_delta.
        self._composing_fn: Callable[[], bool] | None = None

    def set_composing_source(self, fn: Callable[[], bool] | None) -> None:
        """Register the mid-turn-steer "is composing" predicate.

        Public contract for the app layer (app-cli's _execute_with_interrupt)
        to wire up ``SteeringInputManager.is_composing`` without reaching into
        the private ``_composing_fn`` attribute. Callers obtain this hooks
        instance via ``coordinator.get_capability("ui.streaming_hooks")``.

        Args:
            fn: A zero-arg callable returning True while the user has a
                non-empty mid-turn steer draft open, or None to clear the
                source (the default "not composing" behavior).
        """
        self._composing_fn = fn

    # ── Formula helper ─────────────────────────────────────────────────────

    def _compute_total_input(self, usage: dict, provider: str | None = None) -> int:
        """Compute gross total input tokens.

        input_tokens already includes cache reads for all providers. OpenAI also
        includes cache writes, while other providers report cache creation
        separately and require it to be added.

        Args:
            usage: Usage dict from the event
            provider: Provider name, when known

        Returns:
            Gross total input token count
        """
        input_tokens = usage.get("input_tokens") or 0
        if (provider or "").lower().startswith("openai"):
            return input_tokens

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
                # Sub-agent thinking: emit attributed marker.
                sys.stderr.write(f"\n\033[36m🤔 [{agent_name}] Thinking...\033[0m\n")
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

    def _paint_interleaved_text(
        self,
        text: str,
        agent_name: str | None,
        *,
        omit_trailing_blank: bool = False,
        dim: bool = False,
    ) -> None:
        """Paint an interleaved text block using the appropriate renderable.

        Called both by handle_content_block_end (deferred/final path) and by the
        overlay's _on_content_block_start look-ahead (in-place path).  Extracted
        from the original inline block so both callers share identical output.

        Parent (agent_name is None): renders via ``_text_renderable`` — the same
        ``Amplifier:`` label + full-width ``Markdown`` body used by the streaming
        Live preview.  This makes the streaming preview and the settled paint
        byte-identical (no snap).  ``_text_renderable`` includes a leading blank
        line (``Text("")``); do NOT add a second leading blank.  One trailing
        blank line is added for separation.

        Sub-agent (agent_name is not None): full-width dimmed Markdown identical
        to the parent renderable, but with a dim-cyan ``[agent_name]`` label
        instead of ``Amplifier:``.  The spinner (if active) is paused around the
        print so the live region is not corrupted.

        Args:
            text:       The interleaved text to render.  Must be non-empty;
                        callers are responsible for the strip() guard.
            agent_name: Agent name for sub-agent attribution, or None for parent.
        """
        if agent_name is None:
            # Parent: uniform renderable identical to the streaming Live preview.
            # _text_renderable already includes a leading blank line (Text(""));
            # do NOT add a second one.  Add ONE trailing blank for separation.
            out = Console(file=sys.stdout, highlight=False)
            out.print(_text_renderable(text, dim=dim))
            if not omit_trailing_blank:
                # The final response is immediately followed by the Token Usage
                # panel, which supplies its OWN leading blank line. Emitting our
                # trailing blank too would stack into a double blank. Interleaved
                # asides keep the trailing blank because the following tool-call
                # placeholder has no leading blank of its own.
                print()
        else:
            # Sub-agent final result: full-width dimmed Markdown with a dim-cyan
            # [agent_name] label.  _text_renderable's leading Text("") provides
            # the blank-before; we add one trailing blank for separation.
            out = Console(file=sys.stdout, highlight=False)
            out.print(
                _text_renderable(
                    text,
                    dim=True,
                    label=f"[{agent_name}]",
                    label_style="dim cyan",
                )
            )
            print()  # Blank line after

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
            # The skip applies only to parent sessions (agent_name is None).
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

        # Paint text blocks.
        #
        # PARENT (agent_name is None): paints interleaved ASIDES only; skips the final.
        # app-cli's render_message is the SOLE owner of the final-response render —
        # this avoids the #256 double-render that the two-sided coordination caused.
        # The overlay's look-ahead stash (pending_text) handles INTERLEAVED asides:
        # it paints them IN PLACE at the start of the NEXT block, records the index
        # in self._overlay_painted_text, and the guard below suppresses the duplicate
        # here.  The FINAL text block (is_last_block=True) is skipped entirely — the
        # hook no longer owns it.
        #
        # SUB-AGENT (agent_name is not None):
        # ALL settled text blocks (intermediate asides AND the final) are
        # painted ATTRIBUTED via _paint_interleaved_text, which renders a
        # dim-cyan [agent_name] label above a full-width dimmed Markdown body.
        # Live token streaming stays suppressed (overlay _on_delta pass).
        if block_type == "text" and block.get("text", "").strip():
            text = block["text"]
            if agent_name is None:
                # PARENT (agent_name is None): paints interleaved ASIDES only.
                # The hook no longer owns the final-response render — app-cli's
                # render_message is the single owner (fixes #256 double-render).
                if is_last_block:
                    pass  # app-cli's render_message owns the final response (single owner; fixes #256 double-render)
                else:
                    # Interleaved aside: guard against overlay duplicate; then paint dimmed.
                    sid_key = session_id or ""
                    painted_set = self._overlay_painted_text.get(sid_key)
                    if painted_set is not None and block_index in painted_set:
                        # Overlay already rendered this block — suppress the duplicate.
                        painted_set.discard(block_index)
                        if not painted_set:
                            del self._overlay_painted_text[sid_key]
                    else:
                        # Interleaved asides are dimmed so they recede when scrolling back.
                        self._paint_interleaved_text(
                            text,
                            agent_name,
                            dim=True,
                        )
            else:
                # Sub-agent: render ALL settled text blocks (intermediate asides
                # AND the final), attributed + dimmed. Live token streaming stays
                # suppressed (handled in the overlay _on_delta); this only paints
                # COMPLETE blocks at content_block:end, so parallel sub-agents
                # produce coherent (non-interleaved-at-token-level) output.
                self._paint_interleaved_text(text, agent_name)

        # Display token usage after last block (if present and configured)
        if is_last_block and self.show_token_usage and usage:
            indent = "    " if agent_name else ""

            # Get raw token counts (guard against None values from model_dump())
            output_tokens = usage.get("output_tokens") or 0

            # Cache metrics (Anthropic splits input into cached/uncached buckets)
            # Support both Anthropic-SDK field names and amplifier-core Usage model names
            provider = (self.last_llm_info or {}).get("provider")
            is_openai = isinstance(provider, str) and provider.lower().startswith(
                "openai"
            )
            cache_read_input = usage.get("cache_read_input_tokens")
            cache_read_fallback = usage.get("cache_read_tokens")
            if is_openai:
                # OpenAI distinguishes measured zero from unavailable telemetry.
                cache_read = (
                    cache_read_input
                    if cache_read_input is not None
                    else cache_read_fallback or 0
                )
            else:
                # Preserve Anthropic's legacy value fallback: a zero raw field
                # falls through to a positive canonical alternate.
                cache_read = cache_read_input or cache_read_fallback or 0
            cache_create = (
                usage.get("cache_creation_input_tokens")
                or usage.get("cache_write_tokens")
                or 0
            )
            has_openai_cache_read = is_openai and (
                cache_read_input is not None or cache_read_fallback is not None
            )

            # Compute actual total input using helper (fixes double-count bug)
            total_input = self._compute_total_input(usage, provider)
            total_tokens = total_input + output_tokens

            # Format numbers with thousands separators
            input_str = f"{total_input:,}"
            output_str = f"{output_tokens:,}"
            total_str = f"{total_tokens:,}"

            # Build cache info string if caching is active
            cache_info = ""
            if cache_read > 0 or has_openai_cache_read:
                cache_pct = (
                    int((cache_read / total_input) * 100) if total_input > 0 else 0
                )
                cache_info = f" ({cache_pct}% cached)"
            else:
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
                # Defer to cleanup:render_end (fires after app-cli's render_message
                # paints the final response).  This ensures Token Usage appears BELOW
                # the response rather than inserting above it.
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

        is_spawn = tool_name in self._spawn_tools

        if agent_name:
            # Sub-agent tool call: direct print (no live panel to pause).
            print(f"\n    \033[36m┌─ 🔧 [{agent_name}] Using tool: {tool_name}\033[0m")
            # Indent each line of arguments
            for line in truncated.split("\n"):
                print(f"    \033[36m│\033[0m  \033[2m{line}\033[0m")
            if is_spawn:
                agent_label = (
                    (tool_input or {}).get("agent") or tool_name or "sub-agent"
                )
                print(f"\033[2m⏳ {agent_label} working…\033[0m")
        else:
            # Parent tool call.
            print(f"\n\033[36m🔧 Using tool: {tool_name}\033[0m")
            # Indent each line of arguments
            for line in truncated.split("\n"):
                print(f"   \033[2m{line}\033[0m")
            if is_spawn:
                agent_label = (
                    (tool_input or {}).get("agent") or tool_name or "sub-agent"
                )
                print(f"\033[2m⏳ {agent_label} working…\033[0m")

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
            # Sub-agent tool result: direct print (no live panel to pause).
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


def _text_renderable(
    content: str,
    *,
    dim: bool = False,
    label: str = "Amplifier:",
    label_style: str | None = None,
) -> Any:
    """Return a Rich renderable for a streaming text block.

    Renders as: a blank separator line, a label line, then full-width
    ``Markdown(content)`` as the body.

    Default behaviour (parent path — ``label="Amplifier:"`` unchanged):
      - ``label_style`` defaults to ``"dim green"`` when ``dim=True`` and
        ``"bold green"`` otherwise.
      - Every parent call site passes no new arguments, so parent output is
        byte-identical to before.

    Sub-agent path (``label="[agent_name]"``, ``label_style="dim cyan"``):
      - Produces full-width dimmed Markdown with the dim-cyan agent label,
        matching the parent's layout but with distinct styling.

    This is the **single uniform renderable** used for:

      - streaming preview: ``Live.update(_text_renderable(tail))`` during
        token streaming (transient — clears when block ends).
      - settled interleaved asides: ``_paint_interleaved_text`` for parent
        sessions, which calls ``Console.print(_text_renderable(text))``.
      - final response: painted by app-cli's ``render_message`` (sole owner,
        #256 fix); the hook no longer settle-paints the final text block.
      - sub-agent final result: ``_paint_interleaved_text`` with
        ``label="[agent_name]"`` and ``label_style="dim cyan"``.

    Because the streaming preview and settled asides all use this same
    renderable, there is NO snap at settle time — the output is byte-identical.

    The label lives INSIDE the transient Live region (it is NOT printed
    permanently during streaming), so it clears together with the streamed
    text when the block ends.  On settle (aside) it is printed once via
    ``Console.print``; on final it is NOT reprinted (app-cli owns it).

    Used for both ``Live.update()`` during streaming and the height
    measurement in :func:`_fit_tail_to_height`, so the budget arithmetic
    accounts for the two leading lines (blank + label).  The ``Markdown``
    body uses the full console width, which ``_fit_tail_to_height`` measures
    accurately via ``console.render_lines`` — no budget change needed.
    """
    if label_style is None:
        label_style = "dim green" if dim else "bold green"
    body: Any = Markdown(content)
    if dim:
        # Settled interleaved asides are dimmed so they don't compete with the
        # final response when scrolling back. Layout is byte-identical to the
        # bright version (same label, same full-width Markdown wrap) — only the
        # styling changes, so the streaming->settled transition is non-jarring
        # (a dim-down in place, not a reflow). The streaming preview and the
        # final response stay bright (dim=False); we can't tell aside from final
        # until the block ends, so the de-emphasis is applied only at settle.
        body = Styled(body, "dim")
    return Group(
        Text(""),
        Text(label, style=label_style),
        body,
    )


def _make_streaming_overlay(hooks_instance: "StreamingUIHooks"):
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
    # ROOT-CAUSE FIX (Option B): do NOT capture sys.stdout at mount time.
    # Rich's Console resolves `file=None` DYNAMICALLY via its `file` property
    # (falls back to sys.stdout on every write) instead of freezing whatever
    # sys.stdout WAS at construction time. Mounting happens before any turn's
    # patch_stdout(raw=True) is active, so binding `file=sys.stdout` here
    # captured the real terminal stdout permanently -- every later Live
    # repaint wrote straight to the terminal, bypassing prompt_toolkit's
    # StdoutProxy (installed by patch_stdout) and fighting the pinned
    # steering prompt. With `file=None` (the default), each print/repaint
    # re-reads sys.stdout at call time: during a turn that's the StdoutProxy
    # (routes through run_in_terminal, same atomic path as tool-block
    # print()s); outside a turn it's the real stdout, unchanged from before.
    parent_console = Console(highlight=False)

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
        # Part 5: clear pending interleaved-text stash so it doesn't leak
        # into the next turn (the FINAL text block becomes pending but never
        # has a next-block-start; reset clears it before the next turn).
        s.pop("pending_text", None)
        # Also clear the hooks-side painting record for this session so
        # handle_content_block_end doesn't see stale indices from a prior turn.
        hooks_instance._overlay_painted_text.pop(sid, None)

    async def _on_content_block_start(_event: str, data: dict[str, Any]) -> HookResult:
        sid = data.get("session_id") or ""
        idx = data.get("block_index")
        btype = data.get("block_type") or "text"
        if idx is None:
            return HookResult(action="continue")
        s = _get_session(sid)

        # Part 3 — look-ahead: drain any pending interleaved text stashed at
        # the end of the PREVIOUS text block.  We paint it NOW, in-order,
        # directly below its "Amplifier:" label, BEFORE printing this new
        # block's placeholder / Live region.  This keeps [text][tool_use]
        # rendering as: label → whisper/rail → "🔧 Building tool call" with no
        # orphaned label and no late-pop after the tool placeholder.
        #
        # Only applies to parent sessions (pending_text is only stored for
        # agent is None paths); sub-agent sessions never set pending_text.
        pending = s.pop("pending_text", None)
        if pending is not None and pending["buffer"].strip():
            # Look-ahead always drains an INTERLEAVED aside (a next block has
            # started), so it is always dimmed.
            hooks_instance._paint_interleaved_text(
                pending["buffer"], pending["agent_name"], dim=True
            )
            # Record in hooks-side set so handle_content_block_end skips it
            # (Part 4 guard — suppress the deferred duplicate render).
            painted = hooks_instance._overlay_painted_text.setdefault(sid, set())
            painted.add(pending["index"])

        block: dict[str, Any] = {
            "type": btype,
            "buffer": "",
            "live": None,
            "escape_pending": "",
            "name": data.get("name"),
            # THROTTLE/COALESCE spike: monotonic timestamp of the last
            # physical Live repaint for this block. 0.0 means "never
            # refreshed" -- should_refresh() treats that as due immediately.
            "last_refresh": 0.0,
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
                        # Text blocks lead with the bold-green "Amplifier:"
                        # label INSIDE the transient Live (see _text_renderable).
                        initial = _text_renderable("")
                    # THROTTLE/COALESCE spike: auto_refresh=False -- Rich's
                    # background refresh thread (the old refresh_per_second=10)
                    # repaints on its own schedule REGARDLESS of live.update()
                    # calls, so it cannot be throttled by skipping updates.
                    # Disabling it and driving live.update(..., refresh=...)
                    # ourselves in _on_delta is the only way to actually
                    # control repaint cadence while a steer is being composed.
                    live = Live(
                        initial,
                        console=parent_console,
                        transient=True,
                        auto_refresh=False,
                        # ROOT-CAUSE FIX (Option B): don't let Rich wrap
                        # console.file in its own FileProxy. With the dynamic
                        # Console (file=None, above) each write already
                        # re-resolves sys.stdout at call time; adding Rich's
                        # redirect_stdout FileProxy on top is an unnecessary
                        # indirection Rich would otherwise have to unwrap via
                        # rich_proxied_file. Keeping this False routes writes
                        # straight to the dynamically-resolved stdout (the
                        # StdoutProxy during a turn, real stdout otherwise).
                        redirect_stdout=False,
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
                "last_refresh": 0.0,
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
                    # THROTTLE/COALESCE: the buffer above has already
                    # accumulated `clean` regardless of what happens next --
                    # output is never hidden, only the physical repaint
                    # cadence is gated while a steering prompt is visible.
                    #
                    # ROOT-CAUSE FIX (Option B): gate on `effective_composing`,
                    # not raw `is_composing`. `is_composing` only reflects
                    # whether the draft buffer has text; but the pinned
                    # steering prompt fights the Live repaint over
                    # run_in_terminal for the WHOLE time it is on screen --
                    # i.e. for the whole turn `_composing_fn` is registered,
                    # not just while the user has typed something. Gating on
                    # `is_composing` alone under-throttles the empty-buffer,
                    # prompt-visible case.
                    steering_active = hooks_instance._composing_fn is not None
                    is_composing = False
                    if steering_active:
                        try:
                            is_composing = bool(hooks_instance._composing_fn())  # type: ignore[misc]
                        except Exception:
                            is_composing = False
                    now = time.monotonic()
                    do_refresh = should_refresh(
                        effective_composing(is_composing, steering_active),
                        now,
                        block.get("last_refresh", 0.0),
                    )

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
                        live.update(
                            _thinking_renderable(tail, width=console_width),
                            refresh=do_refresh,
                        )
                    else:
                        try:
                            text_height = parent_console.size.height
                        except Exception:
                            text_height = 24
                        # Reserve 2 lines for the blank separator + "Amplifier:"
                        # label that _text_renderable prepends, so the Live
                        # region (label + body) stays within terminal height.
                        budget = max(5, text_height - 5 - 2)
                        tail = _fit_tail_to_height(
                            block["buffer"],
                            budget,
                            _text_renderable,
                            parent_console,
                        )
                        live.update(_text_renderable(tail), refresh=do_refresh)

                    if do_refresh:
                        block["last_refresh"] = now
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
            # Change 1: sub-agent live stream suppressed.
            # Buffer has already accumulated above (block["buffer"] += clean).
            # Nothing is written to stderr — all live sub-agent output is held
            # until content_block:end where the final result is painted attributed.
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
            # THROTTLE/COALESCE spike: force a final repaint of any pending
            # buffered text BEFORE closing the Live region. While composing,
            # some deltas may have accumulated into block["buffer"] without
            # ever being physically repainted (coalesced). should_refresh(
            # force=True) always returns True, guaranteeing the last tokens
            # are shown at least momentarily even though transient=True
            # clears the region an instant later in _close_live().
            live = block.get("live")
            if live is not None and block.get("buffer"):
                try:
                    if btype == "thinking":
                        try:
                            console_height = parent_console.size.height
                            console_width = parent_console.size.width
                        except Exception:
                            console_height = 24
                            console_width = 80
                        # Tail-cap the forced final repaint the same way the
                        # streaming delta path does: reserve 4 frame lines
                        # (===, header, ---, ===) plus the shared 5-line margin
                        # so a large coalesced buffer cannot exceed terminal
                        # height when flushed at block-end.
                        budget = max(5, console_height - 5 - 4)
                        tail = _fit_tail_to_height(
                            block["buffer"],
                            budget,
                            lambda t: _thinking_renderable(t, width=console_width),
                            parent_console,
                        )
                        live.update(
                            _thinking_renderable(tail, width=console_width),
                            refresh=should_refresh(False, 0.0, 0.0, force=True),
                        )
                    else:
                        try:
                            text_height = parent_console.size.height
                        except Exception:
                            text_height = 24
                        # Reserve 2 lines for the blank separator + "Amplifier:"
                        # label that _text_renderable prepends, matching the
                        # streaming delta path budget arithmetic.
                        budget = max(5, text_height - 5 - 2)
                        tail = _fit_tail_to_height(
                            block["buffer"],
                            budget,
                            _text_renderable,
                            parent_console,
                        )
                        live.update(
                            _text_renderable(tail),
                            refresh=should_refresh(False, 0.0, 0.0, force=True),
                        )
                except Exception:
                    pass

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
            # Text/response blocks: no inline paint here — see Part 2 below.
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

            # Part 2 — look-ahead stash: for parent text blocks, don't paint
            # now.  We don't yet know whether there is a NEXT block (which
            # would mean this is interleaved, not the final response).  Stash
            # the finished buffer in the session state; _on_content_block_start
            # for the NEXT block will drain it, paint it in-place, and record
            # the index so handle_content_block_end skips the duplicate.
            #
            # If there IS no next block (final response / end-of-turn), the
            # stash is held here and cleaned up later:
            #   - handle_content_block_end SKIPS the final text block (#256 fix)
            #     — app-cli's render_message is the sole owner of the final paint.
            #     The stash is never drained for the final block (no next-block-start
            #     to trigger the look-ahead drain); it is discarded by _reset_session.
            #   - _reset_session (from _on_prompt_submit / _on_llm_stream_aborted
            #     / _on_provider_retry) clears any leftover stash without painting.
            if btype == "text" and buffer.strip():
                s["pending_text"] = {
                    "index": idx,
                    "buffer": buffer,
                    "agent_name": agent,  # None for parent sessions
                }
        else:
            # Sub-agent: clear buffer. Trailing flush suppressed (change 1) —
            # nothing was streamed to stderr, so no orphaned partial to flush.
            block["buffer"] = ""
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
        # Part 5: clear stash on abort so it doesn't leak into the next turn.
        s.pop("pending_text", None)
        hooks_instance._overlay_painted_text.pop(sid, None)
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
        # Part 5: clear stash on retry so it doesn't leak into the retried turn.
        s.pop("pending_text", None)
        hooks_instance._overlay_painted_text.pop(sid, None)
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


def _make_cost_seed_handler(coordinator, state):
    """Create the prompt:submit handler that seeds the per-turn cost baseline.

    The orchestrator:complete cost handler reports each turn's cost as the delta
    between the current session.cost total and state["prev_total"].  On the very
    first turn of a process prev_total is None, so it falls back to reporting the
    whole session total as the turn cost.  That fallback is correct for a *fresh*
    session (nothing came before) but wrong for a *resumed* one: app-cli restores
    prior spend by registering a history:<session_id> contributor on session.cost
    before the first turn runs, so the first post-resume total already includes
    everything spent in previous runs.  Without a baseline the first line after a
    resume shows Turn == Session (e.g. "Turn: $5.10 | Session: $5.10").

    This handler captures the pre-turn session.cost snapshot into
    state["prev_total"] once, at the start of the first turn this process handles,
    so the first orchestrator:complete computes a correct delta.  Turn 2+ baselines
    are maintained by the complete handler itself, so this only ever acts once.
    """

    async def _on_prompt_submit(event: str, data: dict):
        # Only seed the first turn; after that the complete handler owns prev_total.
        if state["prev_total"] is not None:
            return HookResult(action="continue")
        # Mirror the complete handler's sub-session guard: a sub-session prompt
        # must never seed the root baseline.  (Sub-sessions carry a session_id
        # with an "_" delimiter, or omit the id as None.)
        session_id = data.get("session_id")
        if "session_id" in data and session_id is None:
            return HookResult(action="continue")
        if session_id and "_" in session_id:
            return HookResult(action="continue")
        try:
            contributions = await coordinator.collect_contributions("session.cost")
            state["prev_total"] = _sum_cost_usd(contributions)
        except Exception:
            # Baseline seeding is best-effort display polish; never disrupt the turn.
            pass
        return HookResult(action="continue")

    return _on_prompt_submit


__all__ = [
    "mount",
    "StreamingUIHooks",
    "format_cost_usd",
]
