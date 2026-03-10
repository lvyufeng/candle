"""Tests for DTensor (lightweight metadata container)."""
import candle as torch
from candle._tensor import Tensor
from candle.distributed.tensor.placement import Shard, Replicate
from candle.distributed.tensor.dtensor import DTensor, DTensorSpec, TensorMeta


def _make_spec(placements, global_shape=(8, 4)):
    """Create a DTensorSpec with a mock mesh for unit testing."""
    class MockMesh:
        def size(self, dim=0):
            return 2

        @property
        def ndim(self):
            return 1

    mesh = MockMesh()
    meta = TensorMeta(shape=global_shape, stride=(4, 1), dtype=torch.float32)
    return DTensorSpec(mesh, placements, tensor_meta=meta)


def test_dtensor_is_tensor_subclass():
    assert issubclass(DTensor, Tensor)


def test_dtensor_creation():
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),), global_shape=(8, 4))
    dt = DTensor(local, spec)
    assert dt.shape == local.shape
    assert dt._local_tensor is local
    assert dt._spec is spec


def test_dtensor_to_local():
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    recovered = dt.to_local()
    assert recovered is local


def test_dtensor_placements_property():
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    assert dt.placements == (Shard(0),)


def test_dtensor_device_mesh_property():
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    assert dt.device_mesh is spec.mesh


def test_dtensor_spec_has_shard():
    spec_shard = _make_spec((Shard(0),))
    assert spec_shard.has_shard_placement()
    spec_rep = _make_spec((Replicate(),))
    assert not spec_rep.has_shard_placement()


def test_dtensor_has_torch_dispatch():
    assert hasattr(DTensor, '__torch_dispatch__')
    assert DTensor.__torch_dispatch__ is not Tensor.__torch_dispatch__


def test_dtensor_sharded_blocks_compute():
    """Direct compute on a sharded DTensor should raise RuntimeError."""
    local = torch.randn(4, 4)
    spec = _make_spec((Shard(0),))
    dt = DTensor(local, spec)
    b = torch.randn(4, 4)
    try:
        result = torch.add(dt, b)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "sharded" in str(e).lower() or "not supported" in str(e).lower()
