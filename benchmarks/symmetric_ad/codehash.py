"""Hash of the code a point's result depends on (for resumable matrices).

git HEAD sha plus a hash of the uncommitted changes under ``src`` and
``benchmarks`` (tracked diff and untracked files), so the whole library is
covered and an uncommitted edit invalidates a stored result.
"""

import hashlib
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PATHS = ("src", "benchmarks")


def _git(root, *args) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


def code_hash(root: Path = _ROOT) -> str:
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    h = hashlib.sha1(_git(root, "diff", "HEAD", "--", *_PATHS))
    for f in sorted(
        _git(root, "ls-files", "-o", "--exclude-standard", "--", *_PATHS)
        .decode()
        .splitlines()
    ):
        if "__pycache__" not in f:
            h.update(f.encode())
            h.update((Path(root) / f).read_bytes())
    return f"{head[:12]}-{h.hexdigest()[:12]}"
