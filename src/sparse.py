"""Sparse-system backend: pinned+patched llama-server as a managed subprocess.

The frozen Qwen3.6 sparse runtime (K4/16 graph patch + bounded SSD-backed
executor) lives in the pinned llama.cpp tree, not in pip llama-cpp-python,
so the web UI and repro harness drive it over the server's OpenAI-compatible
HTTP API instead of in-process inference.

Stdlib only (subprocess + urllib): no new dependencies.

Thinking comes back structurally: streamed SSE deltas carry
``choices[0].delta.reasoning_content`` vs ``choices[0].delta.content``
(server-chat.cpp on the pin), and non-streamed replies carry
``message.reasoning_content`` + ``message.content``. When the server
returns merged text instead, :func:`split_thinking` falls back to the
model's ``<think>`` markers.
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
from pathlib import Path
from typing import Any, Iterator

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
               reasoning_budget: int = DEFAULT_REASONING_BUDGET) -> list[str]:
    """llama-server argv for the frozen config (pure, testable)."""
    if reasoning_budget < -1:
        raise ValueError("reasoning_budget must be -1 or greater")
    return [str(_binary_path("llama-server")), "-m", str(model_path), "--host", host,
            "--port", str(port), "-t", str(threads), "--poll", str(poll),
            "-c", str(n_ctx), "-ngl", "0", "--reasoning-budget",
            str(reasoning_budget)]


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
                           reasoning_budget=self.reasoning_budget),
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
        payload = {"messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p,
                   "stream": stream, "cache_prompt": True}
        # llama-server only sends usage/timing data for streamed completions
        # when the OpenAI-compatible option is explicitly enabled. Keeping
        # this in the request, rather than estimating tokens in the browser,
        # makes the displayed rate the runtime's actual decode rate.
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def chat(self, messages: list[dict], *, max_tokens: int | None = None,
             temperature: float | None = None, top_p: float | None = None,
             ) -> dict:
        """Non-streaming chat. Returns {thinking, text, usage}."""
        from src.config import get_generation_config

        gen = get_generation_config()
        max_tokens = gen.max_tokens if max_tokens is None else max_tokens
        temperature = gen.temperature if temperature is None else temperature
        top_p = gen.top_p if top_p is None else top_p
        data = _post_json(self.base_url + "/v1/chat/completions",
                          self._payload(messages, max_tokens, temperature,
                                        top_p, False),
                          timeout=self.timeout_s)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        thinking = msg.get("reasoning_content", "") or ""
        text = msg.get("content", "") or ""
        if not thinking and ("<think>" in text):
            thinking, text = split_thinking(text)
        usage = dict(data.get("usage") or {})
        if data.get("timings"):
            usage["timings"] = data["timings"]
        return {"thinking": thinking.strip(), "text": text.strip(),
                "usage": usage}

    def stream_chat_events(
        self, messages: list[dict], *, max_tokens: int | None = None,
        temperature: float | None = None, top_p: float | None = None,
    ) -> Iterator[tuple[str, str | dict[str, Any]]]:
        """Yield thinking/text deltas plus the server's final usage event."""
        from src.config import get_generation_config

        gen = get_generation_config()
        max_tokens = gen.max_tokens if max_tokens is None else max_tokens
        temperature = gen.temperature if temperature is None else temperature
        top_p = gen.top_p if top_p is None else top_p
        parser = _StreamingThinkingParser()
        structured_reasoning = False
        for ev in _post_sse(self.base_url + "/v1/chat/completions",
                            self._payload(messages, max_tokens, temperature,
                                          top_p, True),
                            timeout=self.timeout_s):
            if "usage" in ev or "timings" in ev:
                usage = dict(ev.get("usage") or {})
                if ev.get("timings"):
                    usage["timings"] = ev["timings"]
                yield ("usage", usage)
                continue
            delta = (ev.get("choices") or [{}])[0].get("delta", {})
            if delta.get("reasoning_content"):
                if not structured_reasoning:
                    yield from parser.finish()
                    structured_reasoning = True
                yield ("thinking", delta["reasoning_content"])
            if delta.get("content"):
                if structured_reasoning:
                    yield ("text", delta["content"])
                else:
                    yield from parser.feed(delta["content"])
        if not structured_reasoning:
            yield from parser.finish()

    def stream_chat(self, messages: list[dict], *, max_tokens: int | None = None,
                    temperature: float | None = None, top_p: float | None = None,
                    ) -> Iterator[tuple[str, str]]:
        """Yield ("thinking"|"text", piece) SSE events in arrival order."""
        for kind, piece in self.stream_chat_events(
            messages, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p,
        ):
            if kind in ("thinking", "text"):
                yield kind, str(piece)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
