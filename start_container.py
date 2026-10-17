"""Start and supervise the two services in the Render container."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    children = []
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        backend = subprocess.Popen(
            [sys.executable, "-u", str(root / "backend" / "app.py")],
            cwd=root, env={**os.environ, "PORT": "8000", "HOST": "127.0.0.1"})
        children.append(backend)
        deadline = time.monotonic() + 60
        while not stopping:
            if backend.poll() is not None:
                raise RuntimeError(f"QC backend exited during startup ({backend.returncode})")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("QC backend did not become ready in 60 seconds")
            time.sleep(0.2)
        if stopping:
            return 0
        frontend = subprocess.Popen(["dotnet", "BlazorApp.dll"], cwd=root / "blazor")
        children.append(frontend)
        while not stopping:
            for child in children:
                if child.poll() is not None:
                    raise RuntimeError(f"Service PID {child.pid} exited ({child.returncode})")
            time.sleep(0.5)
        return 0
    except Exception as exc:
        print(f"[startup] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
