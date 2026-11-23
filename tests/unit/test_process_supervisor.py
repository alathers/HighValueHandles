"""Unit tests for lib.process_supervisor.

Real child processes (``sys.executable`` running tiny scripts) exercise the
behaviors that motivated the module: chatty servers must not deadlock on an
undrained pipe, early exits are detected fast, and terminate() kills the whole
process group (no orphaned grandchildren).
"""

import os
import socket
import sys
import textwrap
import time

import pytest

from lib.process_supervisor import (
    ManagedProcess,
    ProbeResult,
    default_log_dir,
    port_in_use,
    wait_until_ready,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still "exists"; treat reaped-or-zombie as dead via waitpid when possible.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def _script(tmp_path, body: str):
    path = tmp_path / "child.py"
    path.write_text(textwrap.dedent(body))
    return path


def _connect_probe(port: int):
    def probe() -> ProbeResult:
        return ProbeResult(port_in_use("127.0.0.1", port, timeout=0.2), "connection refused")

    return probe


def test_ready_despite_heavy_stdout(tmp_path):
    """256KB of startup output would fill an undrained PIPE; a log file never blocks."""
    port = _free_port()
    script = _script(tmp_path, f"""
        import socket, sys, time
        sys.stdout.write("x" * 262144)
        sys.stdout.flush()
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", {port})); s.listen(5)
        time.sleep(30)
    """)
    proc = ManagedProcess([sys.executable, str(script)], log_path=tmp_path / "logs" / "a.log")
    proc.spawn()
    try:
        result = wait_until_ready(_connect_probe(port), timeout=10, is_alive=proc.is_alive)
        assert result.ready, result.detail
        assert (tmp_path / "logs" / "a.log").stat().st_size >= 262144
    finally:
        assert proc.terminate(grace_seconds=5)
    assert proc.popen is None


def test_early_exit_detected_fast(tmp_path):
    script = _script(tmp_path, """
        import sys
        print("fatal: model not found in cache", flush=True)
        sys.exit(3)
    """)
    proc = ManagedProcess([sys.executable, str(script)], log_path=tmp_path / "b.log")
    proc.spawn()
    start = time.monotonic()
    result = wait_until_ready(
        lambda: ProbeResult(False, "connection refused"), timeout=60, is_alive=proc.is_alive
    )
    elapsed = time.monotonic() - start
    assert not result.ready
    assert result.detail == "process exited before becoming ready"
    assert elapsed < 2.0
    assert proc.returncode == 3
    assert "fatal: model not found in cache" in proc.tail_log()
    proc.terminate()


def test_terminate_kills_process_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = _script(tmp_path, f"""
        import subprocess, sys, time
        gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pid_file)!r}, "w").write(str(gc.pid))
        time.sleep(60)
    """)
    proc = ManagedProcess([sys.executable, str(script)], log_path=tmp_path / "c.log")
    proc.spawn()
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "grandchild never started"
        time.sleep(0.05)
    grandchild = int(pid_file.read_text())
    leader = proc.pid
    assert proc.terminate(grace_seconds=5)
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild), "grandchild was orphaned"
    assert not _alive(leader)


def _alive_by_signal(pid: int) -> bool:
    """Liveness of a process that is not our child (no waitpid possible)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_stragglers_after_leader_exit_get_sigkill(tmp_path, monkeypatch):
    """Leader exits (and is reaped by is_alive) but leaves a SIGTERM-ignoring member."""
    import lib.process_supervisor as ps

    monkeypatch.setattr(ps, "STRAGGLER_GRACE_SECONDS", 0.5)
    pid_file = tmp_path / "straggler.pid"
    script = _script(tmp_path, f"""
        import subprocess, sys
        code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        gc = subprocess.Popen([sys.executable, "-c", code])
        open({str(pid_file)!r}, "w").write(str(gc.pid))
    """)
    proc = ManagedProcess([sys.executable, str(script)], log_path=tmp_path / "s.log")
    proc.spawn()
    deadline = time.monotonic() + 10
    while proc.is_alive() or not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "leader never exited"
        time.sleep(0.05)
    straggler = int(pid_file.read_text())
    time.sleep(0.3)  # let the straggler install its SIGTERM handler
    assert proc.terminate(grace_seconds=1)
    deadline = time.monotonic() + 5
    while _alive_by_signal(straggler) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive_by_signal(straggler), "straggler survived"


def test_reaped_leader_with_empty_group_is_never_signalled(tmp_path, monkeypatch):
    """After the leader is reaped with no members left, its pid may be reused: no killpg."""
    import lib.process_supervisor as ps

    proc = ManagedProcess([sys.executable, "-c", "pass"], log_path=tmp_path / "r.log")
    proc.spawn()
    deadline = time.monotonic() + 10
    while proc.is_alive():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    sent = []
    real_killpg = os.killpg
    monkeypatch.setattr(ps.os, "killpg", lambda pgid, sig: (sent.append(sig), real_killpg(pgid, sig))[1])
    assert proc.terminate(grace_seconds=1)
    assert [s for s in sent if s != 0] == []


def test_terminate_idempotent(tmp_path):
    proc = ManagedProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"], log_path=tmp_path / "d.log"
    )
    assert proc.terminate() is True  # never spawned
    proc.spawn()
    assert proc.is_alive()
    assert proc.terminate(grace_seconds=5) is True
    assert proc.terminate() is True
    assert proc.pid == 0
    assert proc.returncode is None


def test_detach_leaves_child_running(tmp_path):
    proc = ManagedProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"], log_path=tmp_path / "e.log"
    )
    proc.spawn()
    popen = proc.popen
    proc.detach()
    try:
        assert proc.popen is None
        assert popen.poll() is None
    finally:
        popen.kill()
        popen.wait(5)


def test_spawn_missing_executable(tmp_path):
    proc = ManagedProcess(["/nonexistent/definitely-not-here"], log_path=tmp_path / "f.log")
    with pytest.raises(FileNotFoundError):
        proc.spawn()
    assert proc.popen is None


def test_wait_until_ready_backoff_and_timeout():
    now = [0.0]
    sleeps: list[float] = []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(round(seconds, 4))
        now[0] += seconds

    result = wait_until_ready(
        lambda: ProbeResult(False, "connection refused"), timeout=10, clock=clock, sleep=sleep
    )
    assert not result.ready
    assert result.detail == "timed out after 10s (connection refused)"
    assert sleeps[:6] == [0.25, 0.4, 0.64, 1.024, 1.6384, 2.0]
    assert sum(sleeps) == pytest.approx(10.0)


def test_wait_until_ready_returns_first_ready():
    results = iter([ProbeResult(False, "no"), ProbeResult(True, "yes", {"version": "1"})])
    result = wait_until_ready(lambda: next(results), timeout=5, sleep=lambda _s: None)
    assert result.ready
    assert result.payload == {"version": "1"}


def test_tail_log_missing_file(tmp_path):
    proc = ManagedProcess(["true"], log_path=tmp_path / "missing" / "never.log")
    assert proc.tail_log() == ""


def test_default_log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("BOURBON_LOG_DIR", str(tmp_path))
    assert default_log_dir() == tmp_path
    monkeypatch.delenv("BOURBON_LOG_DIR")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_log_dir().parts[-3:] == (".cache", "bourbon-processor", "logs")


def test_default_log_dir_honours_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.delenv("BOURBON_LOG_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_log_dir() == tmp_path / "bourbon-processor" / "logs"


def test_port_in_use():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert port_in_use("127.0.0.1", port) is True
    assert port_in_use("127.0.0.1", _free_port()) is False
