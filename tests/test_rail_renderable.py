"""Tests for _rail_renderable — the shared rail renderable for asides.

Covers:
  - Short content (1 line): ▍ on every line, no ▸
  - Long content (5+ lines): ▍ on every line, no ▸
  - Sub-agent (agent_name set): 4-space indent prefix
  - Empty/whitespace content: empty Group, no output
  - Content text appears in output
  - Rail colors with force_terminal (ANSI 256 codes 103/145)
"""

from __future__ import annotations

import io

import amplifier_module_hooks_streaming_ui as _mod
from rich.console import Console


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render(content: str, agent_name=None, width: int = 120) -> str:
    """Render _rail_renderable via a non-terminal Console and return the string."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False).print(
        _mod._rail_renderable(content, agent_name)
    )
    return buf.getvalue()


def _render_terminal(content: str, agent_name=None, width: int = 120) -> str:
    """Render _rail_renderable via a 256-colour terminal Console (for ANSI checks)."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=True, color_system="256").print(
        _mod._rail_renderable(content, agent_name)
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Rail glyph present on all content lines
# ---------------------------------------------------------------------------


class TestRailRenderable:
    def test_short_content_has_rail_glyph(self):
        """1-line content must render with ▍, never ▸."""
        out = _render("One line.")
        assert "▍" in out, f"Expected ▍ in output; got: {out!r}"
        assert "▸" not in out, "Whisper glyph ▸ must NEVER appear in _rail_renderable"

    def test_long_content_all_lines_have_rail_glyph(self):
        """5+ paragraph content: every non-blank line must carry ▍, never ▸."""
        text = "\n\n".join(f"Paragraph {i + 1} of analysis text." for i in range(5))
        out = _render(text)
        assert "▍" in out, f"Expected ▍ in long-content output; got: {out!r}"
        assert "▸" not in out, "Whisper glyph ▸ must NEVER appear"
        # Every non-blank line in output should contain ▍
        content_lines = [ln for ln in out.split("\n") if ln.strip()]
        for ln in content_lines:
            assert "▍" in ln, f"Expected ▍ on every content line; missing from: {ln!r}"

    def test_content_text_appears_in_output(self):
        """The actual content text must be present in the rendered output."""
        out = _render("Some important analysis text.")
        assert "important analysis" in out

    # ---------------------------------------------------------------------------
    # Sub-agent indentation
    # ---------------------------------------------------------------------------

    def test_sub_agent_has_4_space_indent(self):
        """agent_name set → every ▍ line is prefixed by exactly 4 spaces."""
        out = _render("Checking module structure.", agent_name="foundation:explorer")
        glyph_lines = [ln for ln in out.split("\n") if "▍" in ln]
        assert glyph_lines, "No ▍ lines found in sub-agent output"
        for ln in glyph_lines:
            assert ln.startswith("    "), (
                f"Sub-agent ▍ line must start with 4-space indent; got: {ln!r}"
            )

    def test_parent_has_no_4_space_indent(self):
        """agent_name=None → ▍ lines must NOT start with 4 spaces."""
        out = _render("Simple text.")
        glyph_lines = [ln for ln in out.split("\n") if "▍" in ln]
        for ln in glyph_lines:
            assert not ln.startswith("    "), (
                f"Parent ▍ line must not have 4-space indent; got: {ln!r}"
            )

    # ---------------------------------------------------------------------------
    # Empty / whitespace content
    # ---------------------------------------------------------------------------

    def test_empty_content_returns_empty_group(self):
        """Empty string → empty Group → no ▍ or ▸ in output."""
        out = _render("")
        assert "▍" not in out, "No ▍ expected for empty content"
        assert "▸" not in out

    def test_whitespace_only_returns_empty_group(self):
        """Whitespace-only string → empty Group → no ▍."""
        out = _render("   \n  \n ")
        assert "▍" not in out, "No ▍ expected for whitespace-only content"

    # ---------------------------------------------------------------------------
    # ANSI 256-color codes (require force_terminal)
    # ---------------------------------------------------------------------------

    def test_rail_glyph_uses_color_103(self):
        """▍ must be styled color(103) — ANSI-256 code 38;5;103."""
        out = _render_terminal("Hello rail.")
        assert "\033[38;5;103m" in out, (
            f"Expected ANSI-256 color 103 for ▍; output was: {out!r}"
        )

    def test_rail_text_uses_color_145(self):
        """Line text must be styled color(145) — ANSI-256 code 38;5;145."""
        out = _render_terminal("Hello rail.")
        assert "\033[38;5;145m" in out, (
            f"Expected ANSI-256 color 145 for text; output was: {out!r}"
        )
