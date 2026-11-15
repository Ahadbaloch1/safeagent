"""Model providers: an offline scripted demo model and the Anthropic Messages API."""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5-5"  # change it with --model or ANTHROPIC_MODEL
RETRY_STATUS = (429, 500, 502, 503, 529)


class ProviderError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class Provider:
    def complete(self, system, messages, tools):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic Messages API (standard library only)
# ---------------------------------------------------------------------------
def _urllib_transport(url, headers, payload, timeout):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (ValueError, OSError):
            body = {}
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError("Network error: %s" % getattr(exc, "reason", exc))
    except ValueError:
        raise ProviderError("The API returned a response that is not valid JSON.")


class AnthropicProvider(Provider):
    """Calls the Messages API with tool use. The transport can be replaced for testing."""

    def __init__(self, api_key, model=DEFAULT_MODEL, max_tokens=1024, transport=None,
                 sleep=time.sleep, max_retries=2, timeout=60):
        if not api_key:
            raise ProviderError("An API key is required. Set the ANTHROPIC_API_KEY environment variable.")
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self.max_retries = max_retries
        self.timeout = timeout

    def complete(self, system, messages, tools):
        payload = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
        if tools:
            payload["tools"] = tools
        headers = {"x-api-key": self._api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        for attempt in range(self.max_retries + 1):
            status, body = self._transport(API_URL, headers, payload, self.timeout)
            if status == 200:
                return self._parse(body)
            if status in RETRY_STATUS and attempt < self.max_retries:
                self._sleep(2 ** attempt)
                continue
            detail = body.get("error", {}).get("message", "") if isinstance(body, dict) else ""
            raise ProviderError("API error %s%s" % (status, ": " + detail if detail else ""))
        raise ProviderError("The API request failed.")

    @staticmethod
    def _parse(body):
        content = body.get("content") if isinstance(body, dict) else None
        if not isinstance(content, list):
            raise ProviderError("Unexpected response from the API.")
        text, calls = [], []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "text":
                text.append(block.get("text", ""))
            elif kind == "tool_use":
                calls.append(ToolCall(block.get("id", ""), block.get("name", ""), block.get("input") or {}))
        return ModelResponse("".join(text), calls, body.get("stop_reason") or "end_turn")


# ---------------------------------------------------------------------------
# Offline scripted demo model
# ---------------------------------------------------------------------------
_CLAUSE_SPLIT = re.compile(r"\s+(?:and then|then|and)\s+", re.I)
_CALC = re.compile(r"(?:calculate|compute|what is|what's)\s+([\d\s+\-*/().%^]*\d[\d\s+\-*/().%^]*)", re.I)
_LIST = re.compile(r"\b(?:list|show)\b.*\bfiles\b(?:\s+in\s+([\w./-]+))?", re.I)
_READ = re.compile(r"\b(?:read|open|summari[sz]e)\s+([\w./-]+\.\w+)", re.I)
_SEARCH = re.compile(r"\b(?:search|find|grep)\s+(?:for\s+)?['\"]?([\w-]{2,})['\"]?", re.I)


def plan_task(task):
    """Turn a simple task sentence into a list of (tool, arguments) steps."""
    plan = []
    for clause in _CLAUSE_SPLIT.split(task):
        clause = clause.strip().rstrip(".?!")
        if m := _CALC.search(clause):
            plan.append(("calculator", {"expression": m.group(1).strip().replace("^", "**")}))
        elif re.search(r"\b(time|date)\b", clause, re.I):
            plan.append(("get_time", {}))
        elif m := _LIST.search(clause):
            plan.append(("list_files", {"path": m.group(1) or "."}))
        elif m := _READ.search(clause):
            plan.append(("read_file", {"path": m.group(1)}))
        elif m := _SEARCH.search(clause):
            plan.append(("search_files", {"query": m.group(1)}))
    return plan


class MockProvider(Provider):
    """A deterministic, rule-based stand-in for a language model.

    It exists to demonstrate and test the agent loop without an API key. It understands a few
    simple task patterns and is NOT intelligent.
    """

    def complete(self, system, messages, tools):
        start = max(i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str))
        task = messages[start]["content"]
        plan = plan_task(task)
        later = messages[start + 1:]
        done = sum(1 for m in later if m["role"] == "assistant" and isinstance(m["content"], list)
                   for block in m["content"] if block.get("type") == "tool_use")
        if done < len(plan):
            name, arguments = plan[done]
            return ModelResponse("", [ToolCall("mock_%d" % (done + 1), name, arguments)], "tool_use")
        if not plan:
            return ModelResponse("I am the offline demo model and I only understand simple tasks such as "
                                 "calculations, listing, reading or searching files, and the time. Set "
                                 "ANTHROPIC_API_KEY and use --provider anthropic for real reasoning.")
        results = []
        for message in later:
            if message["role"] == "user" and isinstance(message["content"], list):
                results += [b["content"] for b in message["content"] if b.get("type") == "tool_result"]
        return ModelResponse("Done. Results:\n" + "\n---\n".join(str(r)[:500] for r in results))
