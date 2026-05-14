import shutil
import subprocess
from pathlib import Path


def test_frontend_audio_behavior_helpers_pass_node_tests():
    node = shutil.which("node")
    assert node, "node is required to run frontend audio behavior tests"

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, "--test", "tests/js/audio-behavior.test.mjs"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
