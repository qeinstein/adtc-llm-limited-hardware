"""Sparse-system backend: pinned+patched llama-server as a managed subprocess.

The frozen Qwen3.6 sparse runtime (K4/16 graph patch + bounded SSD-backed
executor) lives in the pinned llama.cpp tree, not in pip llama-cpp-python,
so the web UI and repro harness drive it over the server's OpenAI-compatible
HTTP API instead of in-process inference.

Stdlib only (subprocess + urllib): no new dependencies.

The runtime may emit a separate reasoning channel. The backend keeps that
channel separate from the final answer so the UI can place it in a collapsed
panel without contaminating answer text or conversation history. Streamed SSE deltas carry
``choices[0].delta.reasoning_content`` vs ``choices[0].delta.content``
(server-chat.cpp on the pin), and non-streamed replies carry both fields.
When the server returns merged text instead, :func:`split_thinking` falls back
to the model's ``<think>`` markers.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FREEZE_PATH = ROOT / "configs" / "final_runtime.json"
CACHE_CFG_PATH = ROOT / "probes" / "edge0_port" / "cache_config_k4.json"

LLAMA_DIR = Path(os.environ.get("ADTC_LLAMA_DIR", str(ROOT / "runtime" / "llama.cpp")))
BUILD_DIR = LLAMA_DIR / "build-native"
RUNTIME_PIN = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
PATCH_HEADER = ROOT / "probes" / "edge0_port" / "join4_phase6.h"
STAMP_PATH = LLAMA_DIR / ".edge0-stamp"


def _binary_candidates(name: str) -> list[Path]:
    """Return CMake single- and multi-config output locations."""
    return [
        BUILD_DIR / "bin" / name,
        BUILD_DIR / "bin" / f"{name}.exe",
        BUILD_DIR / "bin" / "Release" / name,
        BUILD_DIR / "bin" / "Release" / f"{name}.exe",
    ]


def _binary_path(name: str) -> Path:
    for candidate in _binary_candidates(name):
        if candidate.is_file():
            return candidate
    # Keep a deterministic path before the first build.  Windows generators
    # normally place the eventual executable under bin/Release/*.exe.
    return _binary_candidates(name)[1 if os.name == "nt" else 0]


def _expected_stamp() -> str:
    return RUNTIME_PIN + " " + hashlib.sha256(PATCH_HEADER.read_bytes()).hexdigest()


SERVER_BIN = _binary_path("llama-server")
CLI_BIN = _binary_path("llama-cli")
BENCH_BIN = _binary_path("llama-bench")

DEFAULT_PORT = int(os.environ.get("ADTC_SPARSE_PORT", "8421"))
DEFAULT_REASONING_BUDGET = 1024
QWEN_CHAT_TEMPLATE_KWARGS = '{"enable_thinking":true}'


def _native_reasoning_budget_message(message: str | None) -> str | None:
    """Return text to inject *before* Qwen's native thinking end tag.

    Pinned llama-server appends the template's detected end tag itself. Older
    Jamii Afya settings also supplied ``</think>``, producing two closing tags;
    the second could become the only visible ``content`` and then be stripped
    by the UI as formatting, leaving an apparently empty answer.
    """
    if not message:
        return message
    normalized = message.rstrip()
    while normalized.endswith("</think>"):
        normalized = normalized[:-len("</think>")].rstrip()
    return normalized


def load_freeze() -> dict:
    with open(FREEZE_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_built() -> bool:
    if not all(any(path.is_file() for path in _binary_candidates(name))
               for name in ("llama-server", "llama-cli", "llama-bench")):
        return False
    try:
        return STAMP_PATH.read_text(encoding="utf-8").strip() == _expected_stamp()
    except OSError:
        return False


def ensure_built() -> None:
    if is_built():
        return
    if os.name == "nt":
        # Native Windows has no guaranteed /bin/sh.  Keep the build entrypoint
        # in Python so PowerShell, cmd, and Git Bash all use the same recipe.
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_runtime.py")],
                       check=True)
    else:
        subprocess.run(["bash", str(ROOT / "scripts" / "build_runtime.sh")],
                       check=True)


def export_pins(tag: str, dest_dir: Path | None = None) -> Path:
    """Write the locked pin set for a budget tag (e.g. "3.0") to a file."""
    cfg = json.loads(CACHE_CFG_PATH.read_text(encoding="utf-8"))
    if tag not in cfg:
        raise KeyError(f"unknown pins tag {tag!r} (have {sorted(cfg)})")
    keys = cfg[tag]["pins_global"]
    assert len(keys) == cfg[tag]["npins"], (tag, len(keys))
    out_dir = dest_dir or (ROOT / "runtime" / "pins")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"pins_{tag}.txt"
    dest.write_text("\n".join(str(k) for k in keys) + "\n", encoding="utf-8")
    return dest


def build_env(arm: str, pins_path: str | None = None,
              profile: bool = False) -> dict[str, str]:
    """Deployment env for an arm ("resident" or "bounded_3gb"/"bounded_6gb").

    Pure function of the freeze file (testable without binaries/model).
    """
    freeze = load_freeze()
    env = {"GGML_MOE_K1": str(freeze["decode"]["k1"]),
           "GGML_MOE_K2": str(freeze["decode"]["k2"])}
    if arm == "resident":
        return env
    arms = freeze.get("arms", {})
    if arm not in arms or "slots" not in arms[arm]:
        raise KeyError(f"unknown bounded arm {arm!r}")
    if pins_path is None:
        raise ValueError("bounded arm requires pins_path (see export_pins)")
    # LLAMA_ARG_LAZY_MODE=on is REQUIRED, not optional: lazy mode `auto`
    # only lazy-marks tensors over 4 GiB while our Q2_K expert tensors are
    # ~84 MB, so the bounded executor needs explicit `on`. Server/CLI honor
    # it via common_arg; llama-bench's custom parser needs our compat patch
    # (join4_apply.py BENCH_LAZY_ENV_ANCHOR) to honor it too.
    env.update({"GGML_PHASE6_BOUNDED_CACHE": "1",
                "GGML_PHASE6_SLOTS": str(arms[arm]["slots"]),
                "GGML_PHASE6_ASYNC": "1",
                "GGML_PHASE6_PINS": pins_path,
                "LLAMA_ARG_LAZY_MODE": "on"})
    if profile:
        env["GGML_PHASE6_PROFILE"] = "1"
    return env


def server_cmd(model_path: str | Path, *, port: int = DEFAULT_PORT,
               n_ctx: int = 2048, threads: int = 4, poll: int = 0,
               host: str = "127.0.0.1",
               reasoning_budget: int = DEFAULT_REASONING_BUDGET,
               reasoning_budget_message: str | None = None) -> list[str]:
    """llama-server argv for the frozen config (pure, testable)."""
    if reasoning_budget < -1:
        raise ValueError("reasoning_budget must be -1 or greater")
    argv = [str(_binary_path("llama-server")), "-m", str(model_path),
            "--host", host, "--port", str(port), "-t", str(threads),
            "--poll", str(poll), "-c", str(n_ctx), "-ngl", "0",
            # Qwen3.6's native template emits <think> by default. Make the
            # private channel explicit and force llama-server to extract it
            # into reasoning_content instead of visible content.
            "--reasoning", "on", "--reasoning-format", "deepseek",
            "--chat-template-kwargs", QWEN_CHAT_TEMPLATE_KWARGS,
            "--reasoning-budget", str(reasoning_budget)]
    normalized_message = _native_reasoning_budget_message(reasoning_budget_message)
    if normalized_message:
        argv.extend(["--reasoning-budget-message", normalized_message])
    return argv


def split_thinking(text: str) -> tuple[str, str]:
    """Split merged model text into (thinking, answer) on <think> markers.

    Fallback for servers that do not populate reasoning_content. Returns
    ("", text) when no markers are present; unclosed <think> means the
    whole tail is still thinking.
    """
    start, end = text.find("<think>"), text.find("</think>")
    if start < 0:
        # Some chat templates emit only the closing marker in the visible
        # content stream. It is a formatting token, not part of the answer.
        return "", text.removeprefix("</think>").lstrip()
    if end < 0 or end < start:
        return text[start + len("<think>"):].strip(), ""
    thinking = text[start + len("<think>"):end].strip()
    answer = (text[:start] + text[end + len("</think>"):]).strip()
    return thinking, answer


class _StreamingThinkingParser:
    """Split legacy inline ``<think>`` output without losing chunk boundaries."""

    def __init__(self) -> None:
        self.mode = "text"
        self.pending = ""
        self.at_start = True
        self.leading = ""

    @staticmethod
    def _suffix_length(value: str, marker: str) -> int:
        # Keep a possible partial marker until the next network chunk arrives.
        for size in range(min(len(value), len(marker) - 1), 0, -1):
            if value.endswith(marker[:size]):
                return size
        return 0

    def feed(self, piece: str) -> Iterator[tuple[str, str]]:
        # Be tolerant of runtimes/templates that start the visible channel
        # with an orphan closing marker (``</think>answer``). Keep a partial
        # marker across network chunks, but do not hide a normal answer that
        # merely begins with a ``<`` character.
        if self.at_start:
            candidate = self.leading + piece
            close = "</think>"
            if close.startswith(candidate) and candidate != close:
                self.leading = candidate
                return
            if candidate.startswith(close):
                self.leading = ""
                self.at_start = False
                self.pending += candidate[len(close):]
            else:
                self.leading = ""
                self.at_start = False
                self.pending += candidate
        else:
            self.pending += piece
        marker = "<think>" if self.mode == "text" else "</think>"
        while self.pending:
            at = self.pending.find(marker)
            if at >= 0:
                if at:
                    yield self.mode, self.pending[:at]
                self.pending = self.pending[at + len(marker):]
                self.mode = "thinking" if self.mode == "text" else "text"
                marker = "<think>" if self.mode == "text" else "</think>"
                continue
            keep = self._suffix_length(self.pending, marker)
            emit = len(self.pending) - keep
            if emit:
                yield self.mode, self.pending[:emit]
                self.pending = self.pending[emit:]
            break

    def finish(self) -> Iterator[tuple[str, str]]:
        if self.leading:
            yield "text", self.leading
            self.leading = ""
        if self.pending:
            yield self.mode, self.pending
            self.pending = ""


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"server {exc.code}: {body}") from exc


def _post_sse(url: str, payload: dict, timeout: float):
    """Yield raw SSE data lines (dicts) until [DONE]."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"server {exc.code}: {body}") from exc
    with resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    return
                try:
                    yield json.loads(data.decode("utf-8"))
                except ValueError:
                    continue


def _decode_chat_stream(url: str, payload: dict, timeout: float):
    """Normalize llama-server SSE into reasoning, text, usage, and finish events.

    ``deepseek`` format normally separates ``reasoning_content`` and
    ``content``. The marker parser remains active for compatibility with
    runtimes that merge the channels, and also removes an orphan closing tag
    without mistaking it for a user-visible answer.
    """
    parser = _StreamingThinkingParser()
    structured_reasoning = False
    for ev in _post_sse(url, payload, timeout=timeout):
        if "usage" in ev or "timings" in ev:
            usage = dict(ev.get("usage") or {})
            if ev.get("timings"):
                usage["timings"] = ev["timings"]
            yield ("usage", usage)
            continue

        choice = (ev.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        reasoning = delta.get("reasoning_content") or ""
        if reasoning:
            if not structured_reasoning:
                for parsed_kind, parsed_piece in parser.finish():
                    yield (
                        "reasoning" if parsed_kind == "thinking" else parsed_kind,
                        parsed_piece,
                    )
                parser = _StreamingThinkingParser()
                structured_reasoning = True
            yield ("reasoning", str(reasoning))

        content = delta.get("content") or ""
        if content:
            for parsed_kind, parsed_piece in parser.feed(str(content)):
                yield (
                    "reasoning" if parsed_kind == "thinking" else parsed_kind,
                    parsed_piece,
                )

        # A terminal event can contain both the final content delta and the
        # finish reason, so parse the delta first.
        if choice.get("finish_reason"):
            yield ("finish", str(choice["finish_reason"]))

    for parsed_kind, parsed_piece in parser.finish():
        yield (
            "reasoning" if parsed_kind == "thinking" else parsed_kind,
            parsed_piece,
        )


class SparseServer:
    """Managed llama-server for the frozen sparse system."""

    def __init__(self, model_path: str | Path, *,
                 arm: str = "bounded_3gb",
                 port: int = DEFAULT_PORT,
                 n_ctx: int = 2048, threads: int = 4, poll: int = 0,
                 reasoning_budget: int | None = None,
                 timeout_s: float = 600.0) -> None:
        from src.config import get_runtime_config

        self.model_path = str(model_path)
        self.arm = arm
        self.port = port
        self.n_ctx = n_ctx
        self.threads = threads
        self.poll = poll
        self.reasoning_budget = (
            get_runtime_config().reasoning_budget
            if reasoning_budget is None else reasoning_budget
        )
        self.reasoning_budget_message = get_runtime_config().reasoning_budget_message
        if self.reasoning_budget < -1:
            raise ValueError("reasoning_budget must be -1 or greater")
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen | None = None
        self._log_file = None
        self.log_path = ROOT / "runtime" / f"llama-server-{self.port}.log"
        if arm != "resident":
            tag = {"bounded_3gb": "3.0", "bounded_4gb": "4.0",
                   "bounded_5gb": "5.0", "bounded_6gb": "6.0"}[arm]
            pins = export_pins(tag)
        else:
            pins = None
        self._env = dict(os.environ)
        self._env.update(build_env(arm, str(pins) if pins else None))
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self, wait_s: float = 300.0) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"model not found: {self.model_path} (run `make model` first)")
        ensure_built()
        if self.proc is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "ab", buffering=0)
        self._log_file.write(
            f"\n=== llama-server start {time.strftime('%Y-%m-%d %H:%M:%S %z')} "
            f"model={self.model_path} arm={self.arm} ctx={self.n_ctx} ===\n".encode()
        )
        try:
            popen_kwargs = {
                "env": self._env,
                "stdout": self._log_file,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            self.proc = subprocess.Popen(
                server_cmd(self.model_path, port=self.port, n_ctx=self.n_ctx,
                           threads=self.threads, poll=self.poll,
                           reasoning_budget=self.reasoning_budget,
                           reasoning_budget_message=self.reasoning_budget_message),
                **popen_kwargs)
            self.wait_ready(wait_s)
        except Exception:
            # Do not leave a dead child or a stale file descriptor behind.  In
            # particular, this lets the web app retry after a transient build,
            # port, or runtime-startup failure.
            self.stop()
            raise

    def _log_tail(self, lines: int = 80) -> str:
        """Return the latest child output without masking the original error."""
        try:
            if self._log_file is not None:
                self._log_file.flush()
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:]).strip()

    def wait_ready(self, wait_s: float = 300.0) -> None:
        deadline = time.time() + wait_s
        url = self.base_url + "/health"
        last = ""
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = self._log_tail()
                detail = f"\nLast runtime log output:\n{tail}" if tail else ""
                raise RuntimeError(
                    f"llama-server exited {self.proc.returncode} during startup; "
                    f"see {self.log_path}{detail}")
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last = repr(exc)
            time.sleep(1.0)
        tail = self._log_tail()
        detail = f"\nLast runtime log output:\n{tail}" if tail else ""
        raise TimeoutError(
            f"llama-server not ready in {wait_s}s ({last}); "
            f"see {self.log_path}{detail}")

    def rss_mb(self) -> float:
        """Current server-process RSS in MiB (0.0 if unavailable)."""
        if self.proc is None:
            return 0.0
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            pass
        try:
            import psutil

            return psutil.Process(self.proc.pid).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _payload(self, messages: list[dict], max_tokens: int,
                 temperature: float, top_p: float, stream: bool) -> dict:
        from src.config import get_generation_config

        generation = get_generation_config()
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p,
                   "top_k": generation.top_k,
                   "min_p": generation.min_p,
                   "presence_penalty": generation.presence_penalty,
                   "repeat_penalty": generation.repeat_penalty,
                   "stream": stream, "cache_prompt": True}
        # llama-server only sends usage/timing data for streamed completions
        # when the OpenAI-compatible option is explicitly enabled. Keeping
        # this in the request, rather than estimating tokens in the browser,
        # makes the displayed rate the runtime's actual decode rate.
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _continuation_payload(
        self, messages: list[dict], hidden_reasoning: str, max_tokens: int,
        temperature: float, top_p: float, stream: bool,
    ) -> dict:
        payload = self._payload(
            [
                *messages,
                {
                    "role": "assistant",
                    "reasoning_content": hidden_reasoning,
                    "content": "",
                },
            ],
            max_tokens, temperature, top_p, stream,
        )
        payload.update({
            "continue_final_message": "content",
            "chat_template_kwargs": {"enable_thinking": True},
        })
        return payload

    def _direct_payload(
        self, messages: list[dict], max_tokens: int, stream: bool,
    ) -> dict:
        payload = self._payload(
            messages, max_tokens, 0.7, 0.8, stream,
        )
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    @staticmethod
    def _response_channels(data: dict) -> tuple[str, str]:
        msg = (data.get("choices") or [{}])[0].get("message", {})
        hidden_reasoning = str(msg.get("reasoning_content", "") or "")
        text = str(msg.get("content", "") or "")
        if "<think>" in text or text.lstrip().startswith("</think>"):
            inline_reasoning, text = split_thinking(text)
            if not hidden_reasoning:
                hidden_reasoning = inline_reasoning
        return hidden_reasoning, text

    def chat(self, messages: list[dict], *, max_tokens: int | None = None,
             temperature: float | None = None, top_p: float | None = None,
             ) -> dict:
        """Non-streaming chat. Returns only the final text and usage."""
        from src.config import get_generation_config

        gen = get_generation_config()
        max_tokens = gen.max_tokens if max_tokens is None else max_tokens
        temperature = gen.temperature if temperature is None else temperature
        top_p = gen.top_p if top_p is None else top_p
        url = self.base_url + "/v1/chat/completions"
        data = _post_json(
            url,
            self._payload(messages, max_tokens, temperature, top_p, False),
            timeout=self.timeout_s,
        )
        hidden_reasoning, text = self._response_channels(data)
        if not text.strip() and hidden_reasoning:
            try:
                data = _post_json(
                    url,
                    self._continuation_payload(
                        messages, hidden_reasoning, max_tokens,
                        temperature, top_p, False,
                    ),
                    timeout=self.timeout_s,
                )
                _continued_reasoning, text = self._response_channels(data)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                text = ""
        if not text.strip():
            data = _post_json(
                url,
                self._direct_payload(
                    messages, max_tokens, False,
                ),
                timeout=self.timeout_s,
            )
            _direct_reasoning, text = self._response_channels(data)
        usage = dict(data.get("usage") or {})
        if data.get("timings"):
            usage["timings"] = data["timings"]
        return {"text": text.strip(), "usage": usage}

    def stream_chat_events(
        self, messages: list[dict], *, max_tokens: int | None = None,
        temperature: float | None = None, top_p: float | None = None,
    ) -> Iterator[tuple[str, str | dict[str, Any]]]:
        """Yield reasoning and final text as separate channels, plus metadata.

        If a model turn ends after producing only ``reasoning_content``, resume
        that same assistant turn in its content channel. This uses Qwen3.6's
        native assistant-prefill format, preserving the model's work without
        mixing it into the final answer. If a runtime cannot continue a
        structured assistant turn, make one final request in Qwen's officially
        supported direct-response mode instead of returning an empty answer.
        """
        from src.config import get_generation_config

        gen = get_generation_config()
        max_tokens = gen.max_tokens if max_tokens is None else max_tokens
        temperature = gen.temperature if temperature is None else temperature
        top_p = gen.top_p if top_p is None else top_p
        url = self.base_url + "/v1/chat/completions"
        primary = self._payload(messages, max_tokens, temperature, top_p, True)
        attempts: list[tuple[str, dict]] = [("primary", primary)]
        finish_reason = ""

        while attempts:
            attempt, payload = attempts.pop(0)
            hidden_parts: list[str] = []
            visible_parts: list[str] = []
            recovery_failed = False
            try:
                for kind, piece in _decode_chat_stream(
                    url, payload, timeout=self.timeout_s,
                ):
                    if kind == "reasoning":
                        hidden_parts.append(str(piece))
                        yield ("thinking", piece)
                    elif kind == "text":
                        visible_parts.append(str(piece))
                        yield ("text", piece)
                    elif kind == "usage":
                        yield ("usage", piece)
                    elif kind == "finish":
                        finish_reason = str(piece)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                if attempt == "primary":
                    raise
                recovery_failed = True

            if "".join(visible_parts).strip():
                break

            if attempt == "primary" and hidden_parts:
                # Continue the same assistant message immediately after its
                # hidden reasoning block. Qwen's template renders this as
                # <think>...reasoning...</think> followed by the content slot.
                continuation = self._continuation_payload(
                    messages, "".join(hidden_parts), max_tokens,
                    temperature, top_p, True,
                )
                attempts.append(("continuation", continuation))
                continue

            if attempt != "direct":
                # Official Qwen direct-response mode pre-closes the empty
                # thinking block in the chat template, so generated tokens are
                # unambiguously returned as content.
                direct = self._direct_payload(
                    messages, max_tokens, True,
                )
                attempts.append(("direct", direct))
                continue

            if recovery_failed:
                raise RuntimeError("model response recovery failed")
            break

        if finish_reason:
            yield ("finish", finish_reason)

    def stream_chat(self, messages: list[dict], *, max_tokens: int | None = None,
                    temperature: float | None = None, top_p: float | None = None,
                    ) -> Iterator[tuple[str, str]]:
        """Yield only final-text SSE events in arrival order."""
        for kind, piece in self.stream_chat_events(
            messages, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p,
        ):
            if kind == "text":
                yield kind, str(piece)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
