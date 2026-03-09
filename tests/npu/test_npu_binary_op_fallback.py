import pytest

import candle as torch


def _npu_available() -> bool:
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="NPU unavailable")
def test_npu_binary_ops_fallback_keep_npu_and_correctness():
    x = torch.randn((4, 4), device="npu", dtype=torch.float32)
    y = torch.randn((4, 4), device="npu", dtype=torch.float32)

    add_out = x + y
    sub_out = x - y
    mul_out = x * y

    assert add_out.device.type == "npu"
    assert sub_out.device.type == "npu"
    assert mul_out.device.type == "npu"

    torch.testing.assert_close(add_out.cpu(), x.cpu() + y.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sub_out.cpu(), x.cpu() - y.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mul_out.cpu(), x.cpu() * y.cpu(), rtol=1e-5, atol=1e-5)
