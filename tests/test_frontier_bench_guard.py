import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from bench_ctm_frontier_grad import skip_reason


def test_skip_divisibility_n2():
    # N=2: all even D^2 shard; odd D^2 does not.
    assert skip_reason(10, 2, True) is None   # 100 % 2 == 0
    assert skip_reason(12, 2, True) is None   # 144 % 2 == 0
    assert skip_reason(11, 2, True) is not None  # 121 % 2 != 0


def test_skip_divisibility_n3():
    # N=3: 144 shards; 64 and 100 do not.
    assert skip_reason(12, 3, True) is None      # 144 % 3 == 0
    assert skip_reason(8, 3, True) is not None   # 64 % 3 != 0
    assert skip_reason(10, 3, True) is not None  # 100 % 3 != 0


def test_no_shard_never_skips():
    assert skip_reason(11, 2, False) is None
    assert skip_reason(8, 1, False) is None
