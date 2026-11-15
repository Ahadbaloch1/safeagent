"""A small evaluation harness: run tasks and grade the answers automatically."""

import json
import re


def load_tasks(path):
    with open(path, encoding="utf-8") as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, list) or not all(isinstance(t, dict) and "name" in t and "prompt" in t for t in tasks):
        raise ValueError("The task file must be a list of objects with 'name' and 'prompt'.")
    return tasks


def grade(task, result):
    """Return a list of failure messages (an empty list means the task passed)."""
    failures = []
    text = result.final_text.lower()
    for expected in task.get("expect_all", []):
        if expected.lower() not in text:
            failures.append("answer does not contain '%s'" % expected)
    for pattern in task.get("expect_regex", []):
        if not re.search(pattern, result.final_text, re.I | re.S):
            failures.append("answer does not match /%s/" % pattern)
    used = {call["name"] for call in result.tool_calls}
    for name in task.get("expect_tools", []):
        if name not in used:
            failures.append("tool '%s' was not used" % name)
    for name in task.get("forbid_tools", []):
        if name in used:
            failures.append("tool '%s' must not be used" % name)
    if result.stop_reason != "final":
        failures.append("agent stopped with '%s'" % result.stop_reason)
    return failures


def run_eval(agent_factory, tasks):
    """Run every task with a fresh agent. Returns a list of result dictionaries."""
    rows = []
    for task in tasks:
        agent = agent_factory(task.get("max_steps"))
        try:
            result = agent.run(task["prompt"])
            failures = grade(task, result)
            rows.append({"name": task["name"], "passed": not failures, "failures": failures,
                         "steps": result.steps, "tools": [c["name"] for c in result.tool_calls]})
        except Exception as exc:
            rows.append({"name": task["name"], "passed": False, "failures": ["error: %s" % exc],
                         "steps": 0, "tools": []})
    return rows
