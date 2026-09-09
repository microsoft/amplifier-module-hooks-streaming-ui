# Streaming UI Hooks Module

Progressive display for thinking blocks, tool invocations, and token usage in the Amplifier console.

## Purpose

Display streaming LLM output (thinking blocks, tool calls, and token usage) to console with configurable formatting and truncation.

## Contract

### Inputs

- `content_block:start` events - Detect start of thinking blocks
- `content_block:delta` events - Track thinking block progress (not displayed)
- `content_block:end` events - Display complete thinking blocks
- `tool:pre` events - Display tool invocations
- `tool:post` events - Display tool results
- `llm:response` events - Display token usage statistics

### Outputs

- Formatted console output via print statements
- No data transformations or system state changes

### Side Effects

- Console output only (print statements to stdout)

### Dependencies

- `amplifier-core` - For events, coordinator, and HookResult types
- Standard library only (no external dependencies)

## Configuration

Configure via `profile.ui` section:

```toml
[ui]
show_thinking_stream = true  # Display thinking blocks (default: true)
show_tool_lines = 5          # Max lines to show for tool I/O (default: 5)
show_token_usage = true      # Display token usage after each turn (default: true)
```

## Features

- **Thinking Block Display**: Shows formatted thinking blocks with clear visual boundaries
- **Tool Invocation Display**: Shows tool name and truncated arguments
- **Tool Result Display**: Shows success/failure status with truncated output
- **Token Usage Display**: Shows input/output/total token counts after each LLM response
- **Configurable Truncation**: Limit tool I/O display to configured line count
- **Clean Formatting**: Visual separators and icons for better readability

## Events Hooked

| Event                 | Purpose                         | Action                              |
| --------------------- | ------------------------------- | ----------------------------------- |
| `content_block:start` | Detect thinking block start     | Display "Thinking..." indicator     |
| `content_block:end`   | Receive complete thinking block | Display formatted thinking content  |
| `tool:pre`            | Tool invocation                 | Display tool name and arguments     |
| `tool:post`           | Tool result                     | Display success/failure with output |
| `llm:response`        | LLM response received           | Display token usage statistics      |

## Display Format

### Thinking Blocks

```
🧠 Thinking...

============================================================
Thinking:
------------------------------------------------------------
[thinking content here]
============================================================
```

### Tool Invocations

```
🔧 Using tool: tool_name
   Arguments: {truncated arguments}

✅ Tool result: tool_name
   {truncated output}
```

### Token Usage

```
│  📊 Token Usage · 2026-09-09 11:30 UTC
└─ Input: 1,234 | Output: 567 | Total: 1,801
```

## Philosophy Compliance

✅ **Zero kernel changes** - Pure hooks implementation, no core modifications
✅ **Pure observability** - Only displays information, no behavior changes
✅ **Configuration via profile** - Uses standard profile.ui settings
✅ **Simple, focused implementation** - Single responsibility: console display
✅ **Modular design** - Self-contained module with clear contract

## Testing

Run tests with:

```bash
cd amplifier-module-hooks-streaming-ui
pytest tests/
```

## Regeneration Specification

This module can be fully regenerated from this README specification. Key invariants:

1. Mount function signature: `async def mount(coordinator, config)`
2. Hook registration on coordinator.hooks
3. Return HookResult with action="continue"
4. No state changes to system, only console output
5. Configuration from profile.ui section

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
