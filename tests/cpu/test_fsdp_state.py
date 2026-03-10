"""Tests for FSDPState hook orchestration (single-process)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class MockMeshInfo:
    def __init__(self):
        self.shard_mesh_rank = 0
        self.shard_mesh_size = 1
        self.shard_process_group = None
        class _M:
            def size(self, dim=0): return 1
            @property
            def ndim(self): return 1
        self.mesh = _M()


def _apply_fsdp_single_rank(module):
    """Apply FSDP to a module for single-rank testing."""
    from candle.distributed._composable.fsdp._fsdp_param import FSDPParam
    from candle.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
    from candle.distributed._composable.fsdp._fsdp_state import FSDPState

    mesh_info = MockMeshInfo()
    params = [(name, param) for name, param in module.named_parameters(recurse=False)]
    fsdp_params = [FSDPParam(p, module, name, mesh_info) for name, p in params]

    # Override unshard to single-rank version
    for fp in fsdp_params:
        fp.unshard = fp._unshard_single_rank

    group = FSDPParamGroup(fsdp_params, module, mesh_info)
    state = FSDPState(module, group, mesh_info, reshard_after_forward=True)
    module._fsdp_state = state
    return state


def test_fsdp_state_pre_forward_unshards():
    """Pre-forward hook should unshard parameters."""
    m = nn.Linear(8, 4)
    _apply_fsdp_single_rank(m)
    assert isinstance(m.weight, DTensor)
    x = torch.randn(2, 8)
    out = m(x)
    assert out.shape == (2, 4)


def test_fsdp_state_post_forward_reshards():
    """Post-forward hook should reshard parameters (restore DTensor)."""
    m = nn.Linear(8, 4)
    state = _apply_fsdp_single_rank(m)
    x = torch.randn(2, 8)
    out = m(x)
    assert isinstance(m.weight, DTensor), "Expected weight to be resharded to DTensor after forward"
