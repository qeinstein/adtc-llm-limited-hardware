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

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
FREEZE_PATH = ROOT / "configs" / "final_runtime.json"
CACHE_CFG_PATH = ROOT / "probes" / "edge0_port" / "cache_config_k4.json"

LLAMA_DIR = Path(os.environ.get("ADTC_LLAMA_DIR", str(ROOT / "runtime" / "llama.cpp")))
BUILD_DIR = LLAMA_DIR / "build-native"
SERVER_BIN = BUILD_DIR / "bin" / "llama-server"
CLI_BIN = BUILD_DIR / "bin" / "llama-cli"
BENCH_BIN = BUILD_DIR / "bin" / "llama-bench"

DEFAULT_PORT = int(os.environ.get("ADTC_SPARSE_PORT", "8421"))


def load_freeze() -> dict:
    with open(FREEZE_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_built() -> bool:
    return SERVER_BIN.is_file() and CLI_BIN.is_file() and BENCH_BIN.is_file()


def ensure_built() -> None:
    if is_built():
        return
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
    # LLAMA_ARG_LAZY_MODE=on is REQUIRED, not optional: at the frozen pin,
    # lazy mode `auto` only lazy-marks tensors over 4 GiB, and our Q2_K
    # expert tensors are ~84 MB each — without explicit `on`, the loader
    # never fires the bounded-cache registration hook and init aborts
    # ("incomplete routed tensor registration"). Upstream common-arg env.
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
               host: str = "127.0.0.1") -> list[str]:
    """llama-server argv for the frozen config (pure, testable)."""
    return [str(SERVER_BIN), "-m", str(model_path), "--host", host,
            "--port", str(port), "-t", str(threads), "--poll", str(poll),
            "-c", str(n_ctx), "-ngl", "0"]


def split_thinking(text: str) -> tuple[str, str]:
    """Split merged model text into (thinking, answer) on <think> markers.

    Fallback for servers that do not populate reasoning_content. Returns
    ("", text) when no markers are present; unclosed <think> means the
    whole tail is still thinking.
    """
    start, end = text.find("<think>"), text.find("</think>")
    if start < 0:
        return "", text
    if end < 0 or end < start:
        return text[start + len("<think>"):].strip(), ""
    thinking = text[start + len("<think>"):end].strip()
    answer = (text[:start] + text[end + len("</think>"):]).strip()
    return thinking, answer


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
                 timeout_s: float = 600.0) -> None:
        self.model_path = str(model_path)
        self.arm = arm
        self.port = port
        self.n_ctx = n_ctx
        self.threads = threads
        self.poll = poll
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen | None = None
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
        log = open(ROOT / "runtime" / f"llama-server-{self.port}.log",
                   "ab", buffering=0)
        self.proc = subprocess.Popen(
            server_cmd(self.model_path, port=self.port, n_ctx=self.n_ctx,
                       threads=self.threads, poll=self.poll),
            env=self._env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        self.wait_ready(wait_s)

    def wait_ready(self, wait_s: float = 300.0) -> None:
        deadline = time.time() + wait_s
        url = self.base_url + "/health"
        last = ""
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited {self.proc.returncode} during startup; "
                    f"see runtime/llama-server-{self.port}.log")
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last = repr(exc)
            time.sleep(1.0)
        raise TimeoutError(f"llama-server not ready in {wait_s}s ({last})")

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
        return 0.0

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def _payload(self, messages: list[dict], max_tokens: int,
                 temperature: float, top_p: float, stream: bool) -> dict:
        return {"messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p,
                "stream": stream, "cache_prompt": True}

    def chat(self, messages: list[dict], *, max_tokens: int = 512,
             temperature: float = 0.3, top_p: float = 0.9,
             ) -> dict:
        """Non-streaming chat. Returns {thinking, text, usage}."""
        data = _post_json(self.base_url + "/v1/chat/completions",
                          self._payload(messages, max_tokens, temperature,
                                        top_p, False),
                          timeout=self.timeout_s)
        msg = (data.get("choices") or [{}])[0].get("message", {})
        thinking = msg.get("reasoning_content", "") or ""
        text = msg.get("content", "") or ""
        if not thinking and ("<think>" in text):
            thinking, text = split_thinking(text)
        return {"thinking": thinking.strip(), "text": text.strip(),
                "usage": data.get("usage", {})}

    def stream_chat(self, messages: list[dict], *, max_tokens: int = 512,
                    temperature: float = 0.3, top_p: float = 0.9,
                    ) -> Iterator[tuple[str, str]]:
        """Yield ("thinking"|"text", piece) SSE events in arrival order."""
        for ev in _post_sse(self.base_url + "/v1/chat/completions",
                            self._payload(messages, max_tokens, temperature,
                                          top_p, True),
                            timeout=self.timeout_s):
            delta = (ev.get("choices") or [{}])[0].get("delta", {})
            if delta.get("reasoning_content"):
                yield ("thinking", delta["reasoning_content"])
            if delta.get("content"):
                yield ("text", delta["content"])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
