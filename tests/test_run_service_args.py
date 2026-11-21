import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_service.py"


def run(*args):
    command = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(command, capture_output=True, text=True)


def test_text_only_rejects_plate_worker():
    result = run("--text-only", "--embed-worker-python", "e",
                 "--plate-worker-python", "p", "--gallery", "g.npz")
    assert result.returncode == 1
    assert "cannot be used with --plate-worker-python" in result.stderr


def test_text_only_needs_gallery():
    result = run("--text-only", "--embed-worker-python", "e")
    assert result.returncode == 1
    assert "--text-only needs --gallery" in result.stderr


def test_gallery_check_comes_before_text_only_checks():
    result = run("--text-only", "--plate-worker-python", "p",
                 "--gallery", "g.npz")
    assert result.returncode == 1
    assert "--gallery needs --embed-worker-python" in result.stderr
