"""Tool definitions: a registry, a sandboxed workspace, and the built-in tools."""

import ast
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

MAX_READ_BYTES = 200_000
MAX_WRITE_CHARS = 100_000
MAX_SEARCH_MATCHES = 50
SENSITIVE_NAME = re.compile(r"(^\.env($|\.)|^id_rsa|^id_ed25519|\.pem$|\.key$|\.p12$|^\.netrc$|credentials)", re.I)


class ToolError(Exception):
    """An error that is safe to show to the model."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable
    dangerous: bool = False

    def schema(self):
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


class ToolRegistry:
    def __init__(self, tools=()):
        self._tools = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool):
        if tool.name in self._tools:
            raise ValueError("Duplicate tool name: %s" % tool.name)
        self._tools[tool.name] = tool

    def names(self):
        return sorted(self._tools)

    def get(self, name):
        return self._tools.get(name)

    def schemas(self):
        return [self._tools[name].schema() for name in self.names()]

    def call(self, name, arguments):
        """Run a tool. Returns (text, is_error). Never raises for bad input from the model."""
        tool = self._tools.get(name)
        if tool is None:
            return "Unknown tool '%s'. Available tools: %s." % (name, ", ".join(self.names())), True
        try:
            args = self._validate(tool, arguments)
            return str(tool.func(**args)), False
        except ToolError as exc:
            return "Error: %s" % exc, True
        except Exception as exc:  # a bug in a tool must not crash the agent
            return "Error: the tool failed unexpectedly (%s)." % type(exc).__name__, True

    @staticmethod
    def _validate(tool, arguments):
        if not isinstance(arguments, dict):
            raise ToolError("Arguments must be an object.")
        schema = tool.input_schema
        properties = schema.get("properties", {})
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ToolError("Unknown argument(s): %s." % ", ".join(unknown))
        for key in schema.get("required", []):
            if key not in arguments:
                raise ToolError("Missing required argument '%s'." % key)
        for key, value in arguments.items():
            expected = _TYPES.get(properties[key].get("type"))
            if expected is None:
                continue
            if isinstance(value, bool) and properties[key].get("type") != "boolean":
                raise ToolError("Argument '%s' has the wrong type." % key)
            if not isinstance(value, expected):
                raise ToolError("Argument '%s' must be of type %s." % (key, properties[key]["type"]))
        return arguments


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
class Workspace:
    """Resolves paths and refuses anything outside the workspace folder (including via symlinks)."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("Workspace folder does not exist: %s" % root)

    def resolve(self, relative="."):
        try:
            target = (self.root / relative).resolve()
        except (OSError, RuntimeError, ValueError):
            raise ToolError("Invalid path.")
        if target != self.root and self.root not in target.parents:
            raise ToolError("Path is outside the workspace.")
        return target

    def relative(self, path):
        return str(path.relative_to(self.root)) or "."

    @staticmethod
    def check_not_sensitive(path):
        if SENSITIVE_NAME.search(path.name):
            raise ToolError("Access to '%s' is blocked because it may contain secrets." % path.name)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
_BIN_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _power(base, exponent):
    if abs(exponent) > 1000:
        raise ToolError("Exponent is too large.")
    if isinstance(base, int) and isinstance(exponent, int) and abs(base) > 1 and exponent > 0:
        if exponent * math.log2(abs(base)) > 4096:
            raise ToolError("Result would be too large.")
    return base ** exponent


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            return _power(left, right)
        operation = _BIN_OPS.get(type(node.op))
        if operation:
            return operation(left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
            and not node.keywords and len(node.args) <= 10:
        return _FUNCS[node.func.id](*[_eval(arg) for arg in node.args])
    raise ToolError("Only numbers, + - * / // % **, parentheses, pi, e, and sqrt/abs/round/min/max are allowed.")


def calculate(expression):
    if len(expression) > 200:
        raise ToolError("Expression is too long.")
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval"))
    except SyntaxError:
        raise ToolError("Could not parse the expression.")
    except ZeroDivisionError:
        raise ToolError("Division by zero.")
    except (ValueError, OverflowError, TypeError):
        raise ToolError("The expression has no valid numeric result.")
    if isinstance(result, float):
        if not math.isfinite(result):
            raise ToolError("The expression has no valid numeric result.")
        result = int(result) if result.is_integer() and abs(result) < 1e15 else float("%.12g" % result)
    return result


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------
def default_tools(workspace, allow_write=False):
    """Return a ToolRegistry with the built-in tools for a workspace folder."""
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

    def list_files(path="."):
        folder = ws.resolve(path)
        if not folder.is_dir():
            raise ToolError("'%s' is not a folder." % path)
        entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:200]
        lines = ["%s/" % p.name if p.is_dir() else "%s (%d bytes)" % (p.name, p.stat().st_size) for p in entries]
        return "\n".join(lines) or "(empty folder)"

    def read_file(path, max_chars=4000):
        target = ws.resolve(path)
        ws.check_not_sensitive(target)
        if not target.is_file():
            raise ToolError("'%s' is not a file." % path)
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError("File is too large (limit %d bytes)." % MAX_READ_BYTES)
        text = target.read_text(encoding="utf-8", errors="replace")
        max_chars = max(1, min(max_chars, 20000))
        return text if len(text) <= max_chars else text[:max_chars] + "\n...[file truncated]"

    def search_files(query, path="."):
        if len(query.strip()) < 2:
            raise ToolError("The query must have at least 2 characters.")
        start = ws.resolve(path)
        needle, matches = query.lower(), []
        for folder, dirs, files in os.walk(start):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for name in sorted(files):
                file = Path(folder) / name
                if SENSITIVE_NAME.search(name) or file.is_symlink() and ws.root not in file.resolve().parents:
                    continue
                try:
                    if file.stat().st_size > MAX_READ_BYTES:
                        continue
                    lines = file.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                for number, line in enumerate(lines, 1):
                    if needle in line.lower():
                        matches.append("%s:%d: %s" % (ws.relative(file), number, line.strip()[:200]))
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            return "\n".join(matches) + "\n...[more matches not shown]"
        return "\n".join(matches) or "No matches found."

    def get_time():
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    registry = ToolRegistry([
        Tool("calculator", "Evaluate an arithmetic expression. Use ** for powers, not ^.",
             {"type": "object", "properties": {"expression": {"type": "string", "description": "For example (12 + 8) * 3"}},
              "required": ["expression"]}, calculate),
        Tool("list_files", "List the files and folders inside a workspace folder.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Folder, default '.'"}}},
             list_files),
        Tool("read_file", "Read a text file from the workspace.",
             {"type": "object", "properties": {"path": {"type": "string"},
                                               "max_chars": {"type": "integer", "description": "Default 4000"}},
              "required": ["path"]}, read_file),
        Tool("search_files", "Search all text files in the workspace for a word or phrase (case-insensitive).",
             {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
              "required": ["query"]}, search_files),
        Tool("get_time", "Get the current date and time in UTC.", {"type": "object", "properties": {}}, get_time),
    ])

    if allow_write:
        def write_file(path, content):
            if len(content) > MAX_WRITE_CHARS:
                raise ToolError("Content is too large (limit %d characters)." % MAX_WRITE_CHARS)
            target = ws.resolve(path)
            ws.check_not_sensitive(target)
            if target == ws.root or target.is_dir():
                raise ToolError("'%s' is a folder." % path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return "Wrote %d characters to %s." % (len(content), ws.relative(target))

        registry.register(Tool(
            "write_file", "Create or overwrite a text file in the workspace. Requires human approval.",
            {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
             "required": ["path", "content"]}, write_file, dangerous=True))
    return registry
