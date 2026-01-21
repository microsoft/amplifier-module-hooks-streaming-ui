"""Streaming UI Hooks Module

Display streaming LLM output with hierarchical session tree showing:
- Parent sessions with spinner animation
- Sub-sessions/tasks with status indicators
- Box-drawing tree connectors
- Bold sweep animation for active children
"""

# Amplifier module metadata
__amplifier_module_type__ = "hook"

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from amplifier_core.models import HookResult
from rich.console import Console
from rich.markdown import Markdown

logger = logging.getLogger(__name__)

# Terminal escape codes
SPINNERS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STRIKE = "\033[9m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
UP = "\033[A"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


class TaskStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class SessionTask:
    """A task within a session."""
    name: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass 
class SessionNode:
    """A session in the hierarchy."""
    session_id: str
    name: str
    parent_id: str | None = None
    status: TaskStatus = TaskStatus.ACTIVE
    start_time: datetime = field(default_factory=datetime.now)
    tasks: list[SessionTask] = field(default_factory=list)
    children: list["SessionNode"] = field(default_factory=list)


class SessionTree:
    """Manages hierarchical session state."""
    
    def __init__(self):
        self.sessions: dict[str, SessionNode] = {}
        self.root_session_id: str | None = None
        self._lock = threading.Lock()
    
    def add_session(self, session_id: str, name: str, parent_id: str | None = None) -> SessionNode:
        """Add a new session to the tree."""
        with self._lock:
            node = SessionNode(session_id=session_id, name=name, parent_id=parent_id)
            self.sessions[session_id] = node
            
            if parent_id is None:
                self.root_session_id = session_id
            elif parent_id in self.sessions:
                self.sessions[parent_id].children.append(node)
            
            return node
    
    def get_or_create_session(self, session_id: str, name: str = "Session", parent_id: str | None = None) -> SessionNode:
        """Get existing session or create new one."""
        with self._lock:
            if session_id in self.sessions:
                return self.sessions[session_id]
        return self.add_session(session_id, name, parent_id)
    
    def add_task(self, session_id: str, task_name: str) -> SessionTask | None:
        """Add a task to a session."""
        with self._lock:
            if session_id not in self.sessions:
                return None
            task = SessionTask(name=task_name, status=TaskStatus.ACTIVE, start_time=datetime.now())
            self.sessions[session_id].tasks.append(task)
            return task
    
    def complete_task(self, session_id: str, task_name: str, success: bool = True):
        """Mark a task as complete."""
        with self._lock:
            if session_id not in self.sessions:
                return
            for task in self.sessions[session_id].tasks:
                if task.name == task_name and task.status == TaskStatus.ACTIVE:
                    task.status = TaskStatus.COMPLETE if success else TaskStatus.FAILED
                    task.end_time = datetime.now()
                    break
    
    def complete_session(self, session_id: str, success: bool = True):
        """Mark a session as complete."""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].status = TaskStatus.COMPLETE if success else TaskStatus.FAILED
    
    def get_root(self) -> SessionNode | None:
        """Get the root session."""
        with self._lock:
            if self.root_session_id:
                return self.sessions.get(self.root_session_id)
            return None


class TreeRenderer:
    """Renders the session tree with animations."""
    
    def __init__(self, tree: SessionTree):
        self.tree = tree
        self.frame = 0
        self._prev_lines = 0
        self._lock = threading.Lock()
        self._paused = False
    
    def pause(self):
        """Pause rendering (for coordinating with other output)."""
        with self._lock:
            self._paused = True
    
    def resume(self):
        """Resume rendering."""
        with self._lock:
            self._paused = False
    
    def is_paused(self) -> bool:
        """Check if rendering is paused."""
        with self._lock:
            return self._paused
    
    def _bold_sweep(self, text: str, frame: int) -> str:
        """Apply bold sweep animation to text."""
        if not text:
            return text
        pos = frame % len(text)
        return text[:pos] + BOLD + text[pos] + RESET + text[pos+1:]
    
    def _format_elapsed(self, start: datetime | None) -> str:
        """Format elapsed time."""
        if not start:
            return ""
        elapsed = datetime.now() - start
        total_seconds = int(elapsed.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        return f"[{minutes:02d}:{seconds:02d}]"
    
    def _render_task(self, task: SessionTask, is_last: bool, frame: int, indent: str = "") -> str:
        """Render a single task."""
        prefix = "└─" if is_last else "├─"
        elapsed = self._format_elapsed(task.start_time) if task.status == TaskStatus.ACTIVE else ""
        
        if task.status == TaskStatus.COMPLETE:
            # Checkmark + strikethrough task name
            return f"{indent}{prefix} {GREEN}✓{RESET} {STRIKE}{DIM}{task.name}{RESET}"
        elif task.status == TaskStatus.FAILED:
            return f"{indent}{prefix} {RED}✗{RESET} {STRIKE}{DIM}{task.name}{RESET}"
        elif task.status == TaskStatus.ACTIVE:
            styled_name = self._bold_sweep(task.name, frame)
            return f"{indent}{prefix} {CYAN}◐{RESET} {styled_name} {DIM}{elapsed}{RESET}"
        else:  # PENDING
            return f"{indent}{prefix} {DIM}○ {task.name}{RESET}"
    
    def _render_session(self, node: SessionNode, frame: int, indent: str = "", is_last: bool = True) -> list[str]:
        """Render a session and its children recursively."""
        lines = []
        
        # Session header with spinner if active
        if node.status == TaskStatus.ACTIVE:
            spinner = SPINNERS[frame % len(SPINNERS)]
            elapsed = self._format_elapsed(node.start_time)
            lines.append(f"{indent}{CYAN}{spinner}{RESET} {node.name} {DIM}{elapsed}{RESET}")
        elif node.status == TaskStatus.COMPLETE:
            # Checkmark + strikethrough for completed sessions
            lines.append(f"{indent}{GREEN}✓{RESET} {STRIKE}{DIM}{node.name}{RESET}")
        else:
            lines.append(f"{indent}{DIM}○ {node.name}{RESET}")
        
        # Child indent
        child_indent = indent + ("   " if is_last else "│  ")
        
        # Render tasks
        all_items = list(node.tasks) + list(node.children)
        for i, item in enumerate(all_items):
            is_last_item = (i == len(all_items) - 1)
            if isinstance(item, SessionTask):
                lines.append(self._render_task(item, is_last_item, frame, child_indent))
            else:
                lines.extend(self._render_session(item, frame, child_indent, is_last_item))
        
        return lines
    
    def render(self) -> list[str]:
        """Render the full tree."""
        with self._lock:
            if self._paused:
                return []
            self.frame += 1
            root = self.tree.get_root()
            if not root:
                return []
            return self._render_session(root, self.frame)
    
    def display(self):
        """Display the tree, clearing previous output."""
        with self._lock:
            if self._paused:
                return
        
        lines = self.render()
        if not lines:
            return
        
        # Move up and clear previous render
        if self._prev_lines > 0:
            sys.stderr.write(UP * self._prev_lines)
        
        for line in lines:
            sys.stderr.write(CLEAR_LINE + "\r" + line + "\n")
        sys.stderr.flush()
        
        self._prev_lines = len(lines)
    
    def clear(self):
        """Clear the tree display completely."""
        with self._lock:
            if self._prev_lines > 0:
                # Move up to start of tree
                sys.stderr.write(UP * self._prev_lines)
                # Clear each line
                for _ in range(self._prev_lines):
                    sys.stderr.write(CLEAR_LINE + "\r\n")
                # Move back up
                sys.stderr.write(UP * self._prev_lines)
                sys.stderr.flush()
                self._prev_lines = 0


async def mount(coordinator: Any, config: dict[str, Any]) -> None:
    """Mount streaming UI hooks module."""
    ui_config = config.get("ui", {})
    show_thinking = ui_config.get("show_thinking_stream", True)
    show_tool_lines = ui_config.get("show_tool_lines", 5)
    show_token_usage = ui_config.get("show_token_usage", True)
    show_tree = ui_config.get("show_tree", True)
    spinner_interval = ui_config.get("spinner_interval", 0.1)

    hooks = StreamingUIHooks(
        show_thinking=show_thinking,
        show_tool_lines=show_tool_lines,
        show_token_usage=show_token_usage,
        show_tree=show_tree,
        spinner_interval=spinner_interval,
    )

    # Register hooks
    coordinator.hooks.register("content_block:start", hooks.handle_content_block_start)
    coordinator.hooks.register("content_block:end", hooks.handle_content_block_end)
    coordinator.hooks.register("tool:pre", hooks.handle_tool_pre)
    coordinator.hooks.register("tool:post", hooks.handle_tool_post)
    coordinator.hooks.register("task:spawned", hooks.handle_task_spawned)
    coordinator.hooks.register("task:complete", hooks.handle_task_complete)

    logger.info("Mounted hooks-streaming-ui with session tree")
    return


class StreamingUIHooks:
    """Hooks for displaying streaming UI with hierarchical session tree."""

    def __init__(
        self,
        show_thinking: bool = True,
        show_tool_lines: int = 5,
        show_token_usage: bool = True,
        show_tree: bool = True,
        spinner_interval: float = 0.1,
    ):
        self.show_thinking = show_thinking
        self.show_tool_lines = show_tool_lines
        self.show_token_usage = show_token_usage
        self.show_tree = show_tree
        self.spinner_interval = spinner_interval
        
        self.thinking_blocks: dict[int, dict[str, Any]] = {}
        
        # Session tree
        self.tree = SessionTree()
        self.renderer = TreeRenderer(self.tree)
        
        # Animation thread
        self._running = False
        self._thread: threading.Thread | None = None
        self._output_lock = threading.Lock()
        
        # Session timing
        self.session_start: datetime | None = None

    def _ensure_session(self, session_id: str, parent_id: str | None = None):
        """Ensure session exists in tree."""
        if not self.session_start:
            self.session_start = datetime.now()
        
        # Parse agent name from session_id if it's a child
        name = "Session"
        if parent_id and "_" in session_id:
            parts = session_id.split("_", 1)
            if len(parts) == 2:
                name = parts[1]
        
        self.tree.get_or_create_session(session_id, name, parent_id)

    def _start_animation(self):
        """Start the animation thread."""
        if not self.show_tree:
            return
        with self._output_lock:
            if not self._running:
                self._running = True
                self.renderer.resume()
                self._thread = threading.Thread(target=self._animation_loop, daemon=True)
                self._thread.start()
            else:
                self.renderer.resume()

    def _stop_animation(self, clear: bool = False):
        """Stop the animation and optionally clear display."""
        with self._output_lock:
            self.renderer.pause()
            if clear:
                self.renderer.clear()

    def _animation_loop(self):
        """Background animation loop."""
        sys.stderr.write(HIDE_CURSOR)
        sys.stderr.flush()
        try:
            while self._running:
                if not self.renderer.is_paused():
                    self.renderer.display()
                time.sleep(self.spinner_interval)
        finally:
            sys.stderr.write(SHOW_CURSOR)
            sys.stderr.flush()

    def _format_elapsed(self) -> str:
        """Format elapsed time since session start."""
        if not self.session_start:
            return ""
        elapsed = datetime.now() - self.session_start
        total_seconds = int(elapsed.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _parse_session_info(self, data: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract session_id and parent_id from event data."""
        session_id = data.get("session_id")
        parent_id = data.get("parent_id")
        return session_id, parent_id

    def _output_content(self, content: str, newline: bool = True):
        """Output content with proper synchronization."""
        with self._output_lock:
            self.renderer.pause()
            self.renderer.clear()
            time.sleep(0.05)  # Brief pause to ensure clear completes
            sys.stdout.write(content)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()

    async def handle_content_block_start(
        self, _event: str, data: dict[str, Any]
    ) -> HookResult:
        """Detect thinking blocks and start animation."""
        session_id, parent_id = self._parse_session_info(data)
        if session_id:
            self._ensure_session(session_id, parent_id)
        
        block_type = data.get("block_type")
        block_index = data.get("block_index")

        if (
            block_type in {"thinking", "reasoning"}
            and self.show_thinking
            and block_index is not None
        ):
            self.thinking_blocks[block_index] = {"started": True, "session_id": session_id}
            
            # Add thinking as a task
            if session_id:
                self.tree.add_task(session_id, "Thinking")
            
            self._start_animation()

        return HookResult(action="continue")

    async def handle_content_block_end(
        self, _event: str, data: dict[str, Any]
    ) -> HookResult:
        """Display complete thinking block and token usage."""
        block_index = data.get("block_index")
        total_blocks = data.get("total_blocks")
        block = data.get("block", {})
        block_type = block.get("type")
        usage = data.get("usage")
        is_last_block = block_index == total_blocks - 1 if total_blocks else False

        session_id, _ = self._parse_session_info(data)
        agent_name = self._get_agent_name(session_id)

        # Complete thinking task
        if session_id and block_index in self.thinking_blocks:
            self.tree.complete_task(session_id, "Thinking")

        if (
            block_type in {"thinking", "reasoning"}
            and block_index is not None
            and block_index in self.thinking_blocks
        ):
            # Stop animation and clear tree before showing thinking content
            self._stop_animation(clear=True)
            
            thinking_text = (
                block.get("thinking", "")
                or block.get("text", "")
                or _flatten_reasoning_block(block)
            )

            if thinking_text:
                self._display_thinking(thinking_text, agent_name)
            del self.thinking_blocks[block_index]

        if is_last_block and self.show_token_usage and usage:
            self._stop_animation(clear=True)
            self._display_token_usage(usage, agent_name)

        return HookResult(action="continue")

    async def handle_tool_pre(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display tool invocation and add to tree."""
        tool_name = data.get("tool_name", "unknown")
        tool_input = data.get("tool_input", {})
        session_id, parent_id = self._parse_session_info(data)
        
        if session_id:
            self._ensure_session(session_id, parent_id)
            # Use more descriptive name for task tool
            if tool_name == "task":
                agent = tool_input.get("agent", "")
                short_agent = agent.split(":")[-1] if ":" in agent else (agent or "task")
                task_display = f"Delegating to {short_agent}"
            else:
                task_display = f"Tool: {tool_name}"
            self.tree.add_task(session_id, task_display)
        
        # Stop animation, clear, show tool info, then resume animation
        self._stop_animation(clear=True)
        
        agent_name = self._get_agent_name(session_id)
        input_str = self._format_for_display(tool_input)
        truncated = self._truncate_lines(input_str, self.show_tool_lines)

        with self._output_lock:
            if agent_name:
                print(f"\n    {CYAN}┌─ 🔧 [{agent_name}] Using tool: {tool_name}{RESET}")
                for line in truncated.split("\n"):
                    print(f"    {CYAN}│{RESET}  {DIM}{line}{RESET}")
            else:
                print(f"\n{CYAN}🔧 Using tool: {tool_name}{RESET}")
                for line in truncated.split("\n"):
                    print(f"   {DIM}{line}{RESET}")
            sys.stdout.flush()
        
        # Resume animation after output
        self._start_animation()

        return HookResult(action="continue")

    async def handle_tool_post(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Display tool result and update tree."""
        tool_name = data.get("tool_name", "unknown")
        result = data.get("tool_response", data.get("result", {}))
        session_id, _ = self._parse_session_info(data)
        
        # Complete tool task
        success = True
        if session_id:
            if isinstance(result, dict):
                success = result.get("success", True)
                if "returncode" in result:
                    success = result.get("returncode", 0) == 0
            # Match the task name used in tool:pre
            if tool_name == "task":
                tool_input = data.get("tool_input", {})
                agent = tool_input.get("agent", "")
                short_agent = agent.split(":")[-1] if ":" in agent else (agent or "task")
                task_display = f"Delegating to {short_agent}"
            else:
                task_display = f"Tool: {tool_name}"
            self.tree.complete_task(session_id, task_display, success)
        
        # Stop animation and clear for output
        self._stop_animation(clear=True)
        
        agent_name = self._get_agent_name(session_id)
        
        # Format result
        if isinstance(result, dict):
            raw_output = result.get("output")
            bash_output = raw_output if isinstance(raw_output, dict) else result
            if isinstance(bash_output, dict) and "returncode" in bash_output:
                stdout = bash_output.get("stdout", "")
                stderr = bash_output.get("stderr", "")
                returncode = bash_output.get("returncode", 0)
                success = returncode == 0
                if success:
                    output = stdout or stderr or "(no output)"
                else:
                    output = stdout
                    if stderr:
                        output = f"{output}\n[stderr]: {stderr}" if output else f"[stderr]: {stderr}"
                    output = output or "(no output)"
            else:
                success = result.get("success", True)
                output = self._format_for_display(raw_output if raw_output is not None else result)
        else:
            output = self._format_for_display(result)
            success = True

        truncated = self._truncate_lines(output, self.show_tool_lines)
        icon = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"

        with self._output_lock:
            if agent_name:
                print(f"    {CYAN}└─{RESET} {icon} [{agent_name}] Tool result: {tool_name}")
                indented = "\n".join(f"       {line}" for line in truncated.split("\n"))
                print(f"{DIM}{indented}{RESET}\n")
            else:
                print(f"{icon} Tool result: {tool_name}")
                indented = "\n".join(f"   {line}" for line in truncated.split("\n"))
                print(f"{DIM}{indented}{RESET}\n")
            sys.stdout.flush()

        return HookResult(action="continue")

    async def handle_task_spawned(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Handle sub-agent task being spawned."""
        child_session_id = data.get("child_session_id")
        parent_session_id = data.get("parent_session_id") or data.get("session_id")
        agent_name = data.get("agent", "Sub-task")
        instruction = data.get("instruction", "")
        
        # Create descriptive name: "agent: short description"
        display_name = self._format_agent_task_name(agent_name, instruction)
        
        if child_session_id:
            self.tree.add_session(child_session_id, display_name, parent_session_id)
        
        return HookResult(action="continue")
    
    def _format_agent_task_name(self, agent_name: str, instruction: str, max_len: int = 40) -> str:
        """Format agent task name with short description from instruction."""
        # Extract just the agent type (e.g., "explorer" from "foundation:explorer")
        short_agent = agent_name.split(":")[-1] if ":" in agent_name else agent_name
        
        if not instruction:
            return short_agent
        
        # Get first line/sentence as description
        desc = instruction.strip().split("\n")[0]
        
        # Truncate to first sentence if too long
        for sep in [". ", "! ", "? "]:
            if sep in desc:
                desc = desc.split(sep)[0] + sep[0]
                break
        
        # Truncate if still too long
        if len(desc) > max_len:
            desc = desc[:max_len-3].rstrip() + "..."
        
        return f"{short_agent}: {desc}"

    async def handle_task_complete(self, _event: str, data: dict[str, Any]) -> HookResult:
        """Handle sub-agent task completing."""
        session_id = data.get("session_id") or data.get("child_session_id")
        success = data.get("success", True)
        
        if session_id:
            self.tree.complete_session(session_id, success)
        
        return HookResult(action="continue")

    def _get_agent_name(self, session_id: str | None) -> str | None:
        """Extract agent name from session ID."""
        if not session_id:
            return None
        if "_" in session_id:
            parts = session_id.split("_", 1)
            if len(parts) == 2:
                return parts[1]
        return None

    def _display_thinking(self, text: str, agent_name: str | None):
        """Display formatted thinking block."""
        from io import StringIO
        
        with self._output_lock:
            if agent_name:
                print(f"\n    {DIM}{'=' * 56}{RESET}")
                print(f"    {DIM}[{agent_name}] Thinking:{RESET}")
                print(f"    {DIM}{'-' * 56}{RESET}")
                buffer = StringIO()
                temp_console = Console(file=buffer, highlight=False, width=52)
                temp_console.print(Markdown(text))
                rendered = buffer.getvalue()
                for line in rendered.rstrip().split("\n"):
                    print(f"    {DIM}{line}{RESET}")
                print(f"    {DIM}{'=' * 56}{RESET}\n")
            else:
                buffer = StringIO()
                temp_console = Console(file=buffer, highlight=False, width=60)
                temp_console.print(Markdown(text))
                rendered = buffer.getvalue()
                print(f"\n{DIM}{'=' * 60}{RESET}")
                print(f"{DIM}Thinking:{RESET}")
                print(f"{DIM}{'-' * 60}{RESET}")
                print(f"{DIM}{rendered.rstrip()}{RESET}")
                print(f"{DIM}{'=' * 60}{RESET}\n")
            sys.stdout.flush()

    def _display_token_usage(self, usage: dict, agent_name: str | None):
        """Display token usage with elapsed time."""
        with self._output_lock:
            indent = "    " if agent_name else ""
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            total_input = input_tokens + cache_read + cache_create
            total_tokens = total_input + output_tokens

            input_str = f"{total_input:,}"
            output_str = f"{output_tokens:,}"
            total_str = f"{total_tokens:,}"

            cache_info = ""
            if cache_read > 0 or cache_create > 0:
                cache_pct = int((cache_read / total_input) * 100) if total_input > 0 else 0
                cache_info = f" ({cache_pct}% cached)" if cache_read > 0 else " (caching...)"

            elapsed = self._format_elapsed()
            elapsed_str = f" | ⏱ {elapsed}" if elapsed else ""

            print(f"{indent}{DIM}│  📊 Token Usage{RESET}")
            print(f"{indent}{DIM}└─ Input: {input_str}{cache_info} | Output: {output_str} | Total: {total_str}{elapsed_str}{RESET}")
            sys.stdout.flush()

    def _format_for_display(self, value: Any) -> str:
        """Format any value for readable display."""
        if value is None:
            return "(none)"
        if isinstance(value, str):
            return value if value else "(empty)"
        if isinstance(value, (dict, list)):
            try:
                return self._to_yaml_style(value)
            except Exception:
                return str(value)
        return str(value)

    def _to_yaml_style(self, value: Any, indent: int = 0) -> str:
        """Convert value to YAML-style string."""
        prefix = "  " * indent
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            if "\n" in value:
                lines = value.split("\n")
                return "|\n" + "\n".join(f"{prefix}  {line}" for line in lines)
            if value and value[0] not in "-?:,[]{}#&!|>'\"%@`" and ": " not in value:
                return value
            return f'"{value}"'
        if isinstance(value, list):
            if not value:
                return "[]"
            lines = []
            for item in value:
                if isinstance(item, dict):
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
                    formatted = self._to_yaml_style(v, indent)
                    lines.append(f"{prefix}{k}:")
                    lines.append(formatted)
                else:
                    formatted = self._to_yaml_style(v, indent + 1)
                    lines.append(f"{prefix}{k}: {formatted}")
            return "\n".join(lines)
        return str(value)

    def _truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to max_lines with ellipsis."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text:
            return "(empty)"
        lines = text.split("\n")
        if len(lines) == 1 and len(text) > 200:
            return text[:200] + f"... ({len(text) - 200} more chars)"
        if len(lines) <= max_lines:
            return text
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


__all__ = ["mount", "StreamingUIHooks", "SessionTree", "TreeRenderer"]
