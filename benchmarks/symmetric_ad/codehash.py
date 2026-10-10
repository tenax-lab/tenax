"""Hash of the code a point's result depends on (for resumable matrices)."""

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FILES = (
    "benchmarks/symmetric_ad/pipeline.py",
    "src/tenax/algorithms/_symmetric_ad_validation.py",
    "src/tenax/algorithms/ctm_layout.py",
)


def code_hash() -> str:
    h = hashlib.sha1()
    for f in _FILES:
        h.update((_ROOT / f).read_bytes())
    return h.hexdigest()[:12]
