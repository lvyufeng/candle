import pytest

import candle as torch



def _npu_available() -> bool:
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="NPU unavailable")
@pytest.mark.xfail(reason="Known issue: aclnnMul returns 561000 on current stack", strict=False)
def test_npu_mul_tensor_tensor_known_issue():
    a = torch.randn((4, 4), device="npu", dtype=torch.float32)
    b = torch.randn((4, 4), device="npu", dtype=torch.float32)
    _ = a * b


@pytest.mark.skipif(not _npu_available(), reason="NPU unavailable")
@pytest.mark.xfail(reason="Known issue: sum backward uses NPU mul path (aclnnMul 561000)", strict=False)
def test_npu_sum_backward_mul_known_issue():
    x = torch.randn((4, 4), device="npu", dtype=torch.float32)
    x.requires_grad = True
    y = x.sum()
    y.backward()
    assert x.grad is not None
