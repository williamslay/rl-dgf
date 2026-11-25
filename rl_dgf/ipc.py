"""Unix domain socket IPC and subprocess lifecycle management for AFLRun RL."""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


_HEADER_STRUCT = struct.Struct(">I")
_MAX_MSG_BYTES = 16 * 1024 * 1024


class AFLRunIPCError(RuntimeError):
    """Raised when communication with or lifecycle of AFLRun fails."""


class AFLRunIPCClient:
    """Manages one AFLRun subprocess and its Unix domain socket connection."""

    def __init__(
        self,
        afl_fuzz_bin: str | Path,
        target_bin: str | Path,
        aflrun_dir: str | Path | None,
        seeds_dir: str | Path,
        work_dir: str | Path,
        target_args: list[str] | None = None,
        rl_seed: int | None = None,
        exec_timeout_ms: int | None = None,
        extra_env: dict[str, str] | None = None,
        keep_artifacts: bool = False,
        connect_timeout_s: float = 30.0,
    ) -> None:
        self.afl_fuzz_bin = Path(afl_fuzz_bin)
        self.target_bin = Path(target_bin)
        self.aflrun_dir = self._resolve_aflrun_dir(aflrun_dir)
        self.seeds_dir = Path(seeds_dir)
        self.work_dir = Path(work_dir)
        self.target_args = list(target_args or [])
        self.rl_seed = rl_seed
        self.exec_timeout_ms = exec_timeout_ms
        self.extra_env = dict(extra_env or {})
        self.keep_artifacts = keep_artifacts
        self.connect_timeout_s = connect_timeout_s

        self.out_dir = self.work_dir / "out"
        self.sock_path = self.work_dir / "rl.sock"
        self.log_path = self.work_dir / "afl_fuzz.log"

        self._server_sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None
        self._closed = False

    def start(self) -> dict[str, Any]:
        """Bind AF_UNIX server, start afl-fuzz --rl-sock, and return ready state."""
        if len(str(self.sock_path)) >= 104:
            raise AFLRunIPCError(
                f"Unix socket path exceeds safe 104-byte limit: {self.sock_path}"
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.sock_path.exists():
            self.sock_path.unlink()
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.sock_path))
        server.listen(1)
        server.settimeout(self.connect_timeout_s)
        self._server_sock = server

        cmd = [
            str(self.afl_fuzz_bin),
            "-d",
            "-i",
            str(self.seeds_dir),
            "-o",
            str(self.out_dir),
        ]
        if self.aflrun_dir is not None:
            cmd.extend(["--dir", str(self.aflrun_dir)])
        cmd.extend(["--rl-sock", str(self.sock_path)])
        if self.rl_seed is not None:
            cmd.extend(["--rl-seed", str(int(self.rl_seed))])
        if self.exec_timeout_ms is not None:
            cmd.extend(["-t", str(int(self.exec_timeout_ms))])
        cmd.extend(["--", str(self.target_bin), *self.target_args])

        env = os.environ.copy()
        env.setdefault("AFL_NO_UI", "1")
        env.setdefault("AFL_SKIP_CPUFREQ", "1")
        env.setdefault("AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES", "1")
        env.update(self.extra_env)

        self._log_file = open(self.log_path, "wb")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Poll accept briefly so early afl-fuzz crashes fail fast with log details.
        deadline = time.monotonic() + self.connect_timeout_s
        server.settimeout(0.25)
        conn: socket.socket | None = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                log_tail = self._read_log_tail()
                self.close()
                raise AFLRunIPCError(
                    f"afl-fuzz exited early (rc={self._proc.returncode if self._proc else '?'}) "
                    f"before connecting to RL socket. Log tail:\n{log_tail}"
                )
            try:
                conn, _ = server.accept()
                break
            except TimeoutError:
                continue

        if conn is None:
            log_tail = self._read_log_tail()
            self.close()
            raise AFLRunIPCError(
                f"Timed out waiting for afl-fuzz to connect on {self.sock_path}. "
                f"Log tail:\n{log_tail}"
            )

        conn.settimeout(self.connect_timeout_s)
        self._conn = conn
        ready = self._recv_json()
        if ready.get("type") != "ready" or ready.get("status") != "ok":
            self.close()
            raise AFLRunIPCError(f"Unexpected handshake from afl-fuzz: {ready}")
        return ready

    def step(self, seed_id: int, exec_budget: int) -> dict[str, Any]:
        """Execute one deterministic RL step on `seed_id` with `exec_budget`."""
        if self._conn is None or self._closed:
            raise AFLRunIPCError("Cannot step on closed AFLRunIPCClient.")
        payload = {
            "op": "step",
            "seed_id": int(seed_id),
            "exec_budget": int(exec_budget),
        }
        self._send_json(payload)
        reply = self._recv_json()
        if reply.get("status") != "ok":
            raise AFLRunIPCError(f"AFLRun step error: {reply}")
        return reply

    def close(self) -> None:
        """Stop afl-fuzz gracefully and release sockets, files, and temp dirs."""
        if self._closed:
            return
        self._closed = True

        if self._conn is not None:
            try:
                self._conn.settimeout(2.0)
                self._send_json({"op": "stop"})
                _ = self._recv_json()
            except OSError:
                pass
            except AFLRunIPCError:
                pass
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        if self._proc is not None:
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            self._proc = None

        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

        if self.sock_path.exists():
            try:
                self.sock_path.unlink()
            except OSError:
                pass

        if not self.keep_artifacts and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def _send_json(self, obj: dict[str, Any]) -> None:
        assert self._conn is not None
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        frame = _HEADER_STRUCT.pack(len(raw)) + raw
        self._conn.sendall(frame)

    def _recv_json(self) -> dict[str, Any]:
        assert self._conn is not None
        header = self._recv_exact(_HEADER_STRUCT.size)
        (length,) = _HEADER_STRUCT.unpack(header)
        if length == 0 or length > _MAX_MSG_BYTES:
            raise AFLRunIPCError(f"Invalid frame length from afl-fuzz: {length}")
        body = self._recv_exact(length)
        return json.loads(body.decode("utf-8"))

    def _recv_exact(self, n: int) -> bytes:
        assert self._conn is not None
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self._conn.recv(remaining)
            if not chunk:
                raise AFLRunIPCError("EOF on AFLRun Unix socket.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _resolve_aflrun_dir(aflrun_dir: str | Path | None) -> Path | None:
        if aflrun_dir is None:
            return None
        p = Path(aflrun_dir)
        if (p / "BBreachable.txt").exists():
            return p
        candidates = [c for c in p.iterdir() if c.is_dir() and (c / "BBreachable.txt").exists()] if p.is_dir() else []
        if len(candidates) == 1:
            return candidates[0]
        return p

    def _read_log_tail(self, max_bytes: int = 2048) -> str:
        if not self.log_path.exists():
            return "<no log>"
        data = self.log_path.read_bytes()
        return data[-max_bytes:].decode("utf-8", errors="replace")
