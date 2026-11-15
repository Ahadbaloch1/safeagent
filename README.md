# SafeAgent

A small, security-minded AI agent framework in Python. An AI agent is a language model that can use tools: it decides which tool to call, your code runs the tool, the result goes back to the model, and the loop repeats until the model gives a final answer. SafeAgent implements that loop from scratch, with the safety controls that real agents need, and it comes with a test suite and an evaluation harness.

It uses only the Python standard library, so there is nothing to install.

## Features

- **The agent loop:** model call, tool execution, result feedback, repeat, with a step limit
- **Real model support:** a provider for the Anthropic Messages API with tool use, retries on rate limits, and error handling
- **Offline demo model:** a scripted model that lets you run and test everything without an API key
- **Built-in tools:** calculator, list files, read file, search files, current time, and an optional write file
- **Security controls** (see below)
- **Tracing:** every model call and tool call can be saved as JSON lines
- **Evaluation harness:** run a file of tasks and grade the answers and tool use automatically
- **Chat mode** that keeps the conversation history
- **42 unit tests**, including tests of the exact API request and response format

## Security design

Agents act on text they read, so they need guardrails. SafeAgent includes:

| Control | What it does |
|---------|--------------|
| Workspace sandbox | Tools can only access one folder. `..`, absolute paths, and symlinks that point outside are blocked |
| Safe calculator | Uses a parsed expression tree and an allow-list. It never calls `eval`, and it limits the size of powers |
| Human approval | Dangerous tools (writing files) are not offered unless `--allow-write` is set, and each call asks for confirmation unless `--auto-approve` is set. Without an approval handler they are denied |
| Secret blocking | Files such as `.env`, `*.pem`, `*.key` and `id_rsa` cannot be read, searched, or written |
| Step limit | The loop stops after `--max-steps` so a confused model cannot run forever |
| Output limits | Tool output is truncated, and file reads have size limits, to protect the model's context |
| Untrusted tool results | The system prompt tells the model to treat tool output as data and never obey instructions found in files |
| Error containment | Bad arguments, unknown tools, and bugs inside a tool become error messages for the model and never crash the agent |
| No key leaks | API errors never include the API key |

These controls reduce risk but do not remove it. Prompt injection is an unsolved problem, so keep the workspace small, avoid putting secrets in it, and review what an agent is allowed to do.

## Requirements

- Python 3.8 or later
- An Anthropic API key only if you want to use a real model

## Usage

### Try it offline

```bash
python3 -m safeagent run "List files and then read report.md" --workspace examples/workspace
python3 -m safeagent eval --tasks examples/tasks.json --workspace examples/workspace
```

The offline model is a scripted demo. It understands only simple tasks (calculations, listing, reading and searching files, and the time) and it is not intelligent. It exists to show and test the agent loop.

### Use a real model

```bash
export ANTHROPIC_API_KEY="your-key"      # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
python3 -m safeagent run "Find the project deadline in the notes and tell me how many days remain until 31 December." \
    --provider anthropic --workspace examples/workspace
python3 -m safeagent chat --provider anthropic --workspace examples/workspace
```

The default model name is set in `safeagent/providers.py` (`DEFAULT_MODEL`). Change it with `--model` or the `ANTHROPIC_MODEL` environment variable to a model that your account can use. See the official Claude documentation for current model names.

The real-model path is covered by unit tests that replace the network with a fake transport, so the request and response format is tested, but I could not call the live API when building this project. Test it with your own key before relying on it.

### Allow the agent to write files

```bash
python3 -m safeagent run "Save a one-line summary of notes.txt into summary.txt" \
    --provider anthropic --workspace examples/workspace --allow-write
```

You will be asked to approve each write.

### Commands and options

| Command | Description |
|---------|-------------|
| `run "task"` | Run one task |
| `chat` | Interactive chat that remembers the conversation |
| `eval --tasks file.json` | Run a task file and grade the results (exit status 2 if any task fails) |
| `tools` | List the tools the agent would get |

| Option | Description |
|--------|-------------|
| `--provider` | `mock` (default) or `anthropic` |
| `--model` | Model name for the Anthropic provider |
| `--workspace` | The only folder the agent may access (default: current folder) |
| `--allow-write` | Give the agent a `write_file` tool |
| `--auto-approve` | Do not ask before dangerous tools run |
| `--max-steps` | Maximum loop steps (default: 8) |
| `--trace file.jsonl` | Save every event |
| `--quiet` | Do not print tool calls |

## Evaluation

A task file is a list of tasks. Each task has a prompt and checks:

```json
{"name": "two-step-chain", "prompt": "List files and then read report.md",
 "expect_all": ["Quarterly sales"], "expect_tools": ["list_files", "read_file"]}
```

Available checks: `expect_all` (text that must appear), `expect_regex`, `expect_tools`, `forbid_tools`, and `max_steps`. The included tasks check normal behaviour and security behaviour such as blocking path traversal. With the offline model all 7 tasks pass, which shows the harness works. To measure a real model, run the same command with `--provider anthropic` and add harder tasks.

## How it works

1. `agent.py` keeps the message history and runs the loop. It asks the provider for the next step, executes the requested tools (asking for approval for dangerous ones), and sends the results back as `tool_result` messages.
2. `providers.py` turns the loop into API calls. The Anthropic provider sends the `tools` list with JSON schemas, reads `tool_use` blocks from the response, and retries on rate limits.
3. `tools.py` holds the tool registry, which validates arguments against each tool's schema, the sandbox, and the built-in tools.
4. `evaluation.py` runs and grades task files.

### Adding your own tool

```python
from safeagent.tools import Tool, ToolRegistry, default_tools

registry = default_tools("examples/workspace")
registry.register(Tool(
    name="word_count",
    description="Count the words in a text.",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    func=lambda text: len(text.split()),
))
```

## Running the tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover the calculator (including injection attempts), argument validation, the sandbox (including symlinks), secret blocking, the agent loop (parallel tool calls, step limits, approvals, truncation, tracing, chat history), the Anthropic request and response format with a fake transport (retries, error handling, no key leaks), the offline model, the evaluation harness, and the command-line interface.

## Limitations

- It is a learning project and a starting point, not a production agent platform.
- The offline model is scripted and only demonstrates the loop.
- Tools run in your own process. For untrusted workloads, add operating-system level isolation such as containers.
- There is no streaming, no long-term memory, and conversation history is never summarized, so very long chats will exceed the model's context window.
- Prompt-injection defenses are mitigations and not guarantees.

## Possible improvements

- Add a web search or HTTP tool with a domain allow-list
- Summarize old messages to keep long chats within the context window
- Run tools in a container or a restricted subprocess
- Add providers for other model APIs
- Add cost and token tracking

## License

MIT License. See the `LICENSE` file.
