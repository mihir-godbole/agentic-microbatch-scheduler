import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available for testing"
)

from scheduler.cuda_utils import batch_stats


def test_batch_stats():
    lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    total, m = batch_stats(lens)
    assert total == 10
    assert m == 4
