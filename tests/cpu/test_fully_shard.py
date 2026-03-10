"""Tests for fully_shard() API (single-process, world_size=1)."""
import candle as torch
import candle.nn as nn
from candle.distributed.tensor.dtensor import DTensor


class TwoLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MockMesh:
    """Mock DeviceMesh for single-process testing."""
    def __init__(self):
        self.device_type = "cpu"
        self._mesh_shape = (1,)
        self.mesh_dim_names = ("shard",)
        self._dim_groups = [None]
    def get_group(self, dim=0): return None
    def size(self, dim=0): return 1
    @property
    def ndim(self): return 1


def test_fully_shard_basic():
    """fully_shard should convert params to DTensor and attach state."""
    from candle.distributed._composable.fsdp import fully_shard
    model = TwoLayerMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)
    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)
    assert hasattr(model, '_fsdp_state')
    assert hasattr(model.fc1, '_fsdp_state')


def test_fully_shard_forward_works():
    """Forward pass should work after fully_shard."""
    from candle.distributed._composable.fsdp import fully_shard
    model = TwoLayerMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 8)


def test_fully_shard_mixin_injected():
    """fully_shard should inject FSDPModule mixin into module's MRO."""
    from candle.distributed._composable.fsdp import fully_shard, FSDPModule
    model = nn.Linear(8, 4)
    mesh = MockMesh()
    fully_shard(model, mesh=mesh)
    assert isinstance(model, FSDPModule)
    assert hasattr(model, 'fsdp_state')


def test_fully_shard_excludes_child_params():
    """Parent fully_shard should not re-shard params already managed by child."""
    from candle.distributed._composable.fsdp import fully_shard
    model = TwoLayerMLP(8)
    mesh = MockMesh()
    fully_shard(model.fc1, mesh=mesh)
    fully_shard(model.fc2, mesh=mesh)
    fully_shard(model, mesh=mesh)
    # model itself has no direct params (fc1/fc2 manage them)
    # The important thing is no double-sharding and no crash
