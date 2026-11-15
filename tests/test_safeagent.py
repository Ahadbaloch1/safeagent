import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from safeagent import evaluation, providers, tools  # noqa: E402
from safeagent.agent import Agent, Tracer  # noqa: E402
from safeagent.cli import main  # noqa: E402
from safeagent.providers import (AnthropicProvider, MockProvider, ModelResponse, ProviderError,  # noqa: E402
                                 ToolCall, plan_task)

EXAMPLES = ROOT / "examples"


class CalculatorTests(unittest.TestCase):
    def test_valid_expressions(self):
        cases = {"(12 + 8) * 3": 60, "2 ** 10": 1024, "10 / 4": 2.5, "sqrt(16)": 4, "-3 + 5": 2,
                 "round(2.567, 1)": 2.6, "max(1, 7, 3)": 7, "7 // 2": 3, "7 % 4": 3}
        for expression, expected in cases.items():
            self.assertEqual(tools.calculate(expression), expected, expression)
        self.assertAlmostEqual(tools.calculate("pi"), math.pi, places=6)

    def test_dangerous_or_invalid_expressions_are_rejected(self):
        bad = ["__import__('os').system('ls')", "open('x')", "1/0", "2 ** 100000", "9 ** 9 ** 9", "x + 1",
               "'a' * 3", "True + 1", "(1).real", "lambda: 1", "1e400", "sqrt(-1)", "2 +", "[1, 2]",
               "1" * 300, "abs(1, 2, key=3)", "print(1)"]
        for expression in bad:
            with self.assertRaises(tools.ToolError, msg=expression):
                tools.calculate(expression)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = tools.ToolRegistry([
            tools.Tool("echo", "Echo", {"type": "object", "properties": {"text": {"type": "string"},
                                                                          "n": {"type": "integer"}},
                                         "required": ["text"]}, lambda text, n=1: text * n),
            tools.Tool("boom", "Fails", {"type": "object", "properties": {}}, lambda: 1 / 0)])

    def test_success_and_errors(self):
        self.assertEqual(self.registry.call("echo", {"text": "ab", "n": 2}), ("abab", False))
        self.assertTrue(self.registry.call("nope", {})[1])
        self.assertIn("Missing required", self.registry.call("echo", {})[0])
        self.assertIn("must be of type", self.registry.call("echo", {"text": 5})[0])
        self.assertIn("wrong type", self.registry.call("echo", {"text": "a", "n": True})[0])
        self.assertIn("Unknown argument", self.registry.call("echo", {"text": "a", "x": 1})[0])
        self.assertIn("Arguments must be an object", self.registry.call("echo", "text")[0])

    def test_tool_bug_does_not_crash(self):
        text, is_error = self.registry.call("boom", {})
        self.assertTrue(is_error)
        self.assertIn("ZeroDivisionError", text)

    def test_duplicate_names(self):
        with self.assertRaises(ValueError):
            self.registry.register(tools.Tool("echo", "again", {"type": "object"}, lambda: ""))


class BuiltinToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ws"
        self.root.mkdir()
        (self.root / "notes.txt").write_text("Budget approved\nDeadline: Friday\n", encoding="utf-8")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "deep.txt").write_text("another budget line\n", encoding="utf-8")
        (self.root / ".env").write_text("API_KEY=supersecret\n", encoding="utf-8")
        (self.root / "binary.bin").write_bytes(b"\xff\xfe\x00budget")
        (Path(self.tmp.name) / "outside.txt").write_text("outside data budget", encoding="utf-8")
        self.registry = tools.default_tools(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, name, **args):
        return self.registry.call(name, args)

    def test_list_and_read(self):
        text, err = self.call("list_files")
        self.assertFalse(err)
        self.assertIn("notes.txt", text)
        self.assertIn("sub/", text)
        self.assertIn("Deadline", self.call("read_file", path="notes.txt")[0])
        self.assertIn("Budget", self.call("read_file", path="sub/../notes.txt")[0])

    def test_sandbox_blocks_escape(self):
        for path in ["../outside.txt", "/etc/passwd", "sub/../../outside.txt"]:
            text, err = self.call("read_file", path=path)
            self.assertTrue(err, path)
            self.assertIn("outside the workspace", text)
        self.assertTrue(self.call("list_files", path="..")[1])

    def test_symlink_escape_is_blocked(self):
        link = self.root / "link.txt"
        try:
            link.symlink_to(Path(self.tmp.name) / "outside.txt")
        except OSError:
            self.skipTest("symlinks are not supported here")
        text, err = self.call("read_file", path="link.txt")
        self.assertTrue(err)
        self.assertIn("outside the workspace", text)
        self.assertNotIn("outside data", self.call("search_files", query="budget")[0])

    def test_sensitive_files_are_blocked(self):
        text, err = self.call("read_file", path=".env")
        self.assertTrue(err)
        self.assertNotIn("supersecret", text)
        self.assertNotIn("supersecret", self.call("search_files", query="API_KEY")[0])

    def test_search(self):
        text = self.call("search_files", query="BUDGET")[0]
        self.assertIn("notes.txt:1", text)
        self.assertIn("sub/deep.txt:1", text)
        self.assertNotIn("binary.bin", text)
        self.assertEqual(self.call("search_files", query="zzzzz")[0], "No matches found.")
        self.assertTrue(self.call("search_files", query="a")[1])

    def test_read_limits(self):
        (self.root / "long.txt").write_text("x" * 10000, encoding="utf-8")
        self.assertIn("truncated", self.call("read_file", path="long.txt", max_chars=100)[0])
        (self.root / "huge.txt").write_text("x" * (tools.MAX_READ_BYTES + 1), encoding="utf-8")
        self.assertIn("too large", self.call("read_file", path="huge.txt")[0])
        self.assertTrue(self.call("read_file", path="sub")[1])
        self.assertTrue(self.call("read_file", path="missing.txt")[1])

    def test_write_tool_only_exists_when_allowed(self):
        self.assertNotIn("write_file", self.registry.names())
        writable = tools.default_tools(self.root, allow_write=True)
        self.assertTrue(writable.get("write_file").dangerous)
        self.assertIn("Wrote 5", writable.call("write_file", {"path": "new/dir/a.txt", "content": "hello"})[0])
        self.assertEqual((self.root / "new/dir/a.txt").read_text(), "hello")
        self.assertTrue(writable.call("write_file", {"path": "../evil.txt", "content": "x"})[1])
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())
        self.assertTrue(writable.call("write_file", {"path": ".env", "content": "x"})[1])
        self.assertTrue(writable.call("write_file", {"path": "sub", "content": "x"})[1])

    def test_missing_workspace(self):
        with self.assertRaises(ValueError):
            tools.default_tools(Path(self.tmp.name) / "nope")

    def test_get_time(self):
        self.assertTrue(self.call("get_time")[0].endswith("UTC"))


class ScriptedProvider(providers.Provider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, messages, tools_):
        self.calls.append(json.loads(json.dumps(messages)))
        return self.responses.pop(0)


def call(name, tid="t1", **args):
    return ToolCall(tid, name, args)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.txt").write_text("hello", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, responses, **kwargs):
        provider = ScriptedProvider(responses)
        registry = tools.default_tools(self.root, allow_write=kwargs.pop("allow_write", False))
        return Agent(provider, registry, **kwargs), provider

    def test_tool_loop_passes_results_back(self):
        agent, provider = self.agent([ModelResponse("", [call("calculator", expression="6*7")], "tool_use"),
                                      ModelResponse("The answer is 42.")])
        result = agent.run("What is 6*7?")
        self.assertEqual(result.final_text, "The answer is 42.")
        self.assertEqual((result.stop_reason, result.steps), ("final", 2))
        last = provider.calls[1][-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"], [{"type": "tool_result", "tool_use_id": "t1", "content": "42"}])
        self.assertEqual(result.messages[-1]["role"], "assistant")

    def test_parallel_tool_calls(self):
        agent, provider = self.agent([ModelResponse("", [call("calculator", "a", expression="1+1"),
                                                         call("get_time", "b")], "tool_use"),
                                      ModelResponse("done")])
        agent.run("two things")
        ids = [b["tool_use_id"] for b in provider.calls[1][-1]["content"]]
        self.assertEqual(ids, ["a", "b"])

    def test_max_steps(self):
        agent, _ = self.agent([ModelResponse("", [call("get_time")], "tool_use") for _ in range(10)], max_steps=3)
        result = agent.run("loop forever")
        self.assertEqual((result.stop_reason, result.steps), ("max_steps", 3))
        self.assertEqual(len(result.tool_calls), 3)
        self.assertEqual(result.messages[-1]["role"], "assistant")

    def test_unknown_tool_and_errors_are_reported_to_the_model(self):
        agent, provider = self.agent([ModelResponse("", [call("hack_the_planet")], "tool_use"),
                                      ModelResponse("I could not do that.")])
        result = agent.run("do something")
        block = provider.calls[1][-1]["content"][0]
        self.assertTrue(block["is_error"])
        self.assertIn("Unknown tool", block["content"])
        self.assertTrue(result.tool_calls[0]["is_error"])

    def test_dangerous_tool_is_denied_without_approval(self):
        agent, _ = self.agent([ModelResponse("", [call("write_file", path="x.txt", content="hi")], "tool_use"),
                               ModelResponse("ok")], allow_write=True)
        result = agent.run("write a file")
        self.assertFalse((self.root / "x.txt").exists())
        self.assertTrue(result.tool_calls[0]["is_error"])

    def test_dangerous_tool_with_approval(self):
        seen = []
        agent, _ = self.agent([ModelResponse("", [call("write_file", path="x.txt", content="hi")], "tool_use"),
                               ModelResponse("ok")], allow_write=True,
                              approve=lambda name, args: seen.append((name, args)) or True)
        agent.run("write a file")
        self.assertEqual((self.root / "x.txt").read_text(), "hi")
        self.assertEqual(seen[0][0], "write_file")

    def test_user_can_reject(self):
        agent, _ = self.agent([ModelResponse("", [call("write_file", path="x.txt", content="hi")], "tool_use"),
                               ModelResponse("ok")], allow_write=True, approve=lambda n, a: False)
        agent.run("write a file")
        self.assertFalse((self.root / "x.txt").exists())

    def test_long_tool_output_is_truncated(self):
        (self.root / "big.txt").write_text("y" * 15000, encoding="utf-8")
        agent, provider = self.agent([ModelResponse("", [call("read_file", path="big.txt", max_chars=15000)], "tool_use"),
                                      ModelResponse("ok")], max_tool_output=500)
        agent.run("read it")
        content = provider.calls[1][-1]["content"][0]["content"]
        self.assertLess(len(content), 600)
        self.assertIn("truncated", content)

    def test_tracer_writes_json_lines(self):
        path = self.root / "trace.jsonl"
        agent, _ = self.agent([ModelResponse("", [call("get_time")], "tool_use"), ModelResponse("done")],
                              tracer=Tracer(str(path)))
        agent.run("what time is it")
        kinds = [json.loads(line)["type"] for line in path.read_text().splitlines()]
        self.assertEqual(kinds, ["task", "model", "tool", "model", "final"])

    def test_chat_history_continues(self):
        agent, provider = self.agent([ModelResponse("first"), ModelResponse("second")])
        first = agent.run("one")
        agent.run("two", first.messages)
        roles = [m["role"] for m in provider.calls[1]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_invalid_max_steps(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedProvider([]), tools.default_tools(self.root), max_steps=0)


class AnthropicProviderTests(unittest.TestCase):
    def make(self, replies, **kwargs):
        self.requests, queue = [], list(replies)

        def transport(url, headers, payload, timeout):
            self.requests.append((url, headers, json.loads(json.dumps(payload))))
            return queue.pop(0)
        self.sleeps = []
        return AnthropicProvider("sk-test-secret", transport=transport, sleep=self.sleeps.append, **kwargs)

    def test_request_format_and_text_response(self):
        provider = self.make([(200, {"content": [{"type": "text", "text": "Hi"}], "stop_reason": "end_turn"})],
                             model="my-model")
        tool_list = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
        response = provider.complete("be nice", [{"role": "user", "content": "hello"}], tool_list)
        url, headers, payload = self.requests[0]
        self.assertEqual(url, providers.API_URL)
        self.assertEqual(headers["x-api-key"], "sk-test-secret")
        self.assertEqual(headers["anthropic-version"], providers.API_VERSION)
        self.assertEqual((payload["model"], payload["system"], payload["tools"]), ("my-model", "be nice", tool_list))
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual((response.text, response.tool_calls), ("Hi", []))

    def test_tool_use_response(self):
        body = {"content": [{"type": "text", "text": "Let me check."},
                            {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {"expression": "1+1"}}],
                "stop_reason": "tool_use"}
        response = self.make([(200, body)]).complete("s", [{"role": "user", "content": "x"}], [])
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(response.tool_calls[0], ToolCall("toolu_1", "calculator", {"expression": "1+1"}))
        self.assertNotIn("tools", self.requests[0][2])

    def test_retries_on_rate_limit(self):
        ok = (200, {"content": [{"type": "text", "text": "ok"}]})
        provider = self.make([(429, {}), (529, {}), ok])
        self.assertEqual(provider.complete("s", [], []).text, "ok")
        self.assertEqual(self.sleeps, [1, 2])

    def test_gives_up_after_retries(self):
        provider = self.make([(500, {}), (500, {}), (500, {})])
        with self.assertRaises(ProviderError):
            provider.complete("s", [], [])
        self.assertEqual(len(self.requests), 3)

    def test_auth_error_does_not_leak_the_key(self):
        provider = self.make([(401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}})])
        with self.assertRaises(ProviderError) as ctx:
            provider.complete("s", [], [])
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("sk-test-secret", str(ctx.exception))
        self.assertEqual(len(self.requests), 1)  # not retried

    def test_malformed_response_and_missing_key(self):
        with self.assertRaises(ProviderError):
            self.make([(200, {"unexpected": True})]).complete("s", [], [])
        with self.assertRaises(ProviderError):
            AnthropicProvider("")

    def test_full_agent_loop_with_api_messages(self):
        tool_use = (200, {"content": [{"type": "tool_use", "id": "toolu_9", "name": "calculator",
                                       "input": {"expression": "6*7"}}], "stop_reason": "tool_use"})
        final = (200, {"content": [{"type": "text", "text": "42"}], "stop_reason": "end_turn"})
        provider = self.make([tool_use, final])
        with tempfile.TemporaryDirectory() as tmp:
            result = Agent(provider, tools.default_tools(tmp)).run("What is 6*7?")
        self.assertEqual(result.final_text, "42")
        second_request = self.requests[1][2]["messages"]
        self.assertEqual([m["role"] for m in second_request], ["user", "assistant", "user"])
        self.assertEqual(second_request[1]["content"][0]["type"], "tool_use")
        self.assertEqual(second_request[2]["content"], [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "42"}])


class MockProviderTests(unittest.TestCase):
    def test_plan_task(self):
        self.assertEqual(plan_task("Calculate (12 + 8) * 3"), [("calculator", {"expression": "(12 + 8) * 3"})])
        self.assertEqual(plan_task("what is 2^8?"), [("calculator", {"expression": "2**8"})])
        self.assertEqual([s[0] for s in plan_task("List files and then read Report.md")], ["list_files", "read_file"])
        self.assertEqual(plan_task("List the files in sub"), [("list_files", {"path": "sub"})])
        self.assertEqual(plan_task("search for budget"), [("search_files", {"query": "budget"})])
        self.assertEqual(plan_task("what time is it"), [("get_time", {})])
        self.assertEqual(plan_task("tell me a joke"), [])

    def test_mock_through_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Agent(MockProvider(), tools.default_tools(EXAMPLES / "workspace")).run("Read notes.txt")
            self.assertIn("deadline", result.final_text)
            unknown = Agent(MockProvider(), tools.default_tools(tmp)).run("tell me a joke")
            self.assertIn("offline demo model", unknown.final_text)
            self.assertEqual(unknown.tool_calls, [])


class EvaluationTests(unittest.TestCase):
    def test_example_tasks_pass_with_mock(self):
        tasks = evaluation.load_tasks(EXAMPLES / "tasks.json")
        rows = evaluation.run_eval(
            lambda steps: Agent(MockProvider(), tools.default_tools(EXAMPLES / "workspace"), max_steps=steps or 8), tasks)
        self.assertEqual(len(rows), len(tasks))
        self.assertTrue(all(r["passed"] for r in rows), [r for r in rows if not r["passed"]])

    def test_grading_detects_failures(self):
        result = evaluation.__dict__["grade"]({"expect_all": ["zebra"], "expect_tools": ["read_file"],
                                               "forbid_tools": ["calculator"], "expect_regex": ["^\\d+$"]},
                                              type("R", (), {"final_text": "apple", "stop_reason": "max_steps",
                                                             "tool_calls": [{"name": "calculator"}]})())
        self.assertEqual(len(result), 5)

    def test_invalid_task_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"not": "a list"}')
            with self.assertRaises(ValueError):
                evaluation.load_tasks(path)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.workspace = str(EXAMPLES / "workspace")

    def test_run_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = os.path.join(tmp, "trace.jsonl")
            code = main(["run", "Calculate 2 + 2", "--workspace", self.workspace, "--trace", trace, "--quiet"])
            self.assertEqual(code, 0)
            with open(trace, encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.readline())["type"], "task")

    def test_tools_listing_depends_on_flag(self):
        import contextlib
        import io
        for flag, expected in (([], False), (["--allow-write"], True)):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                main(["tools", "--workspace", self.workspace] + flag)
            self.assertEqual("write_file" in buffer.getvalue(), expected)

    def test_anthropic_without_key_is_an_error(self):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertEqual(main(["run", "hi", "--provider", "anthropic", "--workspace", self.workspace]), 1)
        finally:
            if saved:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_eval_exit_codes(self):
        self.assertEqual(main(["eval", "--tasks", str(EXAMPLES / "tasks.json"), "--workspace", self.workspace,
                               "--quiet"]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tasks.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"name": "impossible", "prompt": "Calculate 1 + 1", "expect_all": ["banana"]}], handle)
            self.assertEqual(main(["eval", "--tasks", path, "--workspace", self.workspace, "--quiet"]), 2)

    def test_bad_workspace(self):
        self.assertEqual(main(["run", "hi", "--workspace", "/definitely/not/here"]), 1)


if __name__ == "__main__":
    unittest.main()
