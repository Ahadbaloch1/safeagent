"""Command-line interface: run, chat, eval, tools."""

import argparse
import json
import os
import sys

from . import __version__
from .agent import Agent, Tracer
from .evaluation import load_tasks, run_eval
from .providers import DEFAULT_MODEL, AnthropicProvider, MockProvider, ProviderError
from .tools import default_tools


def build_parser():
    p = argparse.ArgumentParser(prog="safeagent", description="A small, security-minded AI agent")
    p.add_argument("--version", action="version", version="safeagent " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(parser):
        parser.add_argument("--provider", choices=["mock", "anthropic"], default="mock",
                            help="mock is an offline demo model; anthropic needs ANTHROPIC_API_KEY")
        parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
        parser.add_argument("--workspace", default=".", help="The only folder the agent may access")
        parser.add_argument("--allow-write", action="store_true", help="Give the agent a write_file tool")
        parser.add_argument("--auto-approve", action="store_true", help="Do not ask before dangerous tools run")
        parser.add_argument("--max-steps", type=int, default=8)
        parser.add_argument("--trace", help="Write every event to this JSON-lines file")
        parser.add_argument("--quiet", action="store_true", help="Do not print the tool calls")

    r = sub.add_parser("run", help="Run one task")
    r.add_argument("task")
    common(r)
    c = sub.add_parser("chat", help="Chat with the agent (it remembers the conversation)")
    common(c)
    e = sub.add_parser("eval", help="Run a task file and grade the results")
    e.add_argument("--tasks", required=True)
    common(e)
    t = sub.add_parser("tools", help="List the tools the agent would get")
    t.add_argument("--workspace", default=".")
    t.add_argument("--allow-write", action="store_true")
    return p


def make_provider(args):
    if args.provider == "mock":
        return MockProvider()
    return AnthropicProvider(os.environ.get("ANTHROPIC_API_KEY"), model=args.model)


def ask_permission(name, arguments):
    shown = json.dumps(arguments)[:300]
    try:
        return input("\nThe agent wants to run %s %s\nAllow? [y/N] " % (name, shown)).strip().lower() == "y"
    except EOFError:
        return False


def make_agent(args, max_steps=None, tracer=None):
    approve = (lambda name, arguments: True) if args.auto_approve else ask_permission
    return Agent(make_provider(args), default_tools(args.workspace, args.allow_write),
                 max_steps=max_steps or args.max_steps, approve=approve, tracer=tracer)


def _print_event(event):
    if event["type"] == "tool":
        preview = event["output"].replace("\n", " ")[:100]
        flag = " (error)" if event["is_error"] else ""
        print("  tool %s %s -> %s%s" % (event["name"], json.dumps(event["input"]), preview, flag))


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "tools":
            for tool in default_tools(args.workspace, args.allow_write).names():
                print(tool)
            return 0
        tracer = Tracer(getattr(args, "trace", None), None if args.quiet else _print_event)
        if args.command == "run":
            result = make_agent(args, tracer=tracer).run(args.task)
            print("\n" + result.final_text)
            print("\n[%s after %d step(s), %d tool call(s)]" % (result.stop_reason, result.steps, len(result.tool_calls)))
        elif args.command == "chat":
            agent, history = make_agent(args, tracer=tracer), []
            print("Chat started (workspace: %s). Type 'exit' to quit." % args.workspace)
            while True:
                try:
                    text = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in ("exit", "quit"):
                    break
                if text:
                    result = agent.run(text, history)
                    history = result.messages
                    print("\nagent> " + result.final_text)
        else:
            tasks = load_tasks(args.tasks)
            rows = run_eval(lambda steps: make_agent(args, steps), tasks)
            for row in rows:
                print("%s  %-28s steps=%d tools=%s%s" % ("PASS" if row["passed"] else "FAIL", row["name"], row["steps"],
                                                       ",".join(row["tools"]) or "-",
                                                       "  <- " + "; ".join(row["failures"]) if row["failures"] else ""))
            passed = sum(r["passed"] for r in rows)
            print("\n%d of %d tasks passed." % (passed, len(rows)))
            return 0 if passed == len(rows) else 2
    except (ProviderError, ValueError, OSError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    return 0
