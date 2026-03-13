import candle as torch

from .helpers import run_training_core_parity_case


def test_normalize_forward_matches_torch_contract():
    import torch as real_torch

    result = run_training_core_parity_case(
        op_name="normalize",
        candle_fn=lambda x: torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12),
        torch_fn=lambda x: real_torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12),
        candle_inputs=lambda: (
            torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32),
        ),
        torch_inputs=lambda: (
            real_torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=real_torch.float32),
        ),
    )

    assert result["dtype_match"] is True
    assert result["shape_match"] is True
    assert result["value_match"] is True
