# FSDP2 MVP Design for Candle

## Overview

Implement PyTorch-compatible FSDP2 (`fully_shard()`) for Candle, targeting the NPU (Ascend/HCCL) backend. The design fully aligns with PyTorch's `torch.distributed.fsdp._fully_shard` API.

### Scope (MVP)

**In scope:**
- `__torch_function__` / `__torch_dispatch__` tensor subclass protocols
- `DeviceMesh` (1D)
- `DTensor` + `Placement` types (lightweight metadata container)
- `FSDPParam` / `FSDPParamGroup` (per-parameter sharding)
- `FSDPState` + hook-based forward/backward orchestration
- `fully_shard()` composable API with dynamic MRO injection
- `nn.Module` hook fixes (`prepend`, `with_kwargs`)

**Out of scope (future phases):**
- HSDP (2D mesh, `Replicate + Shard`)
- Mixed precision (`MixedPrecisionPolicy`)
- CPU offload (`CPUOffloadPolicy`)
- Communication/computation overlap (multi-stream)
- Buffer merging (single all-gather for parameter groups)
- Tensor parallelism composition (FSDP+TP)
- Distributed checkpoint / sharded state dict
- Forward/backward prefetching

### Validation Target

End-to-end training of a small model (GPT-2 small) on multi-card NPU, comparing loss curves against DDP baseline.

---

## Layer 0: Tensor Subclass Dispatch Protocols

### Problem

Candle's dispatch system has no `__torch_function__` or `__torch_dispatch__` protocols. The `Tensor` class is a plain class with no subclass interception. DTensor as a Tensor subclass would break — backend kernels would receive DTensor objects when they expect plain Tensors.

### Design

Add two interception points to Candle's dispatch flow, matching PyTorch's two-layer protocol:

```
User calls candle.add(a, b)
    |
    v
[NEW] __torch_function__ check    <-- Python API layer, before dispatcher
    |
    v
  Dispatcher: build keyset
    |
    v
  Autograd / Autocast / Functionalize
    |
    v
[NEW] __torch_dispatch__ check    <-- After autograd, before kernel
    |
    v
  Backend kernel execution
```

### 0.1 `__torch_function__` Protocol

**Location:** `_functional.py` layer (before `dispatch()`)

**Semantics:** When any candle function is called, check if any tensor argument is a subclass that overrides `__torch_function__`. If so, call the subclass implementation. The subclass can either handle the call entirely or return `NotImplemented` to fall through to normal dispatch.

```python
# candle/_functional.py — each function entry

def add(*args, **kwargs):
    result = _handle_torch_function(add, args, kwargs)
    if result is not NotImplemented:
        return result
    # existing logic
    return dispatch("add", None, *args, **kwargs)
```

Core helper:

```python
def _handle_torch_function(func, args, kwargs):
    """Check for __torch_function__ overrides on tensor subclasses."""
    # 1. Collect tensor types from all arguments
    types = _get_overloaded_types(args, kwargs)

    # 2. No subclasses -> fast return
    if not types:
        return NotImplemented

    # 3. Call subclasses in MRO priority order
    for cls in _sorted_by_mro(types):
        result = cls.__torch_function__(func, types, args, kwargs or {})
        if result is not NotImplemented:
            return result

    return NotImplemented


def _get_overloaded_types(args, kwargs):
    """Find tensor subclass types that override __torch_function__."""
    types = set()
    for arg in _iter_tensors(args, kwargs):
        cls = type(arg)
        if cls is not Tensor and hasattr(cls, '__torch_function__'):
            types.add(cls)
    return types
```

Default implementation on Tensor base class:

```python
class Tensor:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented  # Don't intercept, use normal dispatch
```

### 0.2 `__torch_dispatch__` Protocol

**Location:** Inside `dispatch_with_keyset()`, triggered by the `Python` dispatch key.

**Prerequisite — Fix dispatch key priority order:**

Current order has `Python` key BEFORE `Autograd`, but PyTorch places it AFTER. This must be corrected so `__torch_dispatch__` is called after autograd recording.

```python
# candle/_dispatch/keys.py — corrected priority order

_DISPATCH_KEY_ORDER = [
    # 1. Pre-processing
    DispatchKey.BackendSelect,
    DispatchKey.Pipeline,

    # 2. Transform layers
    DispatchKey.Functionalize,
    DispatchKey.Autocast,
    DispatchKey.ADInplaceOrView,

    # 3. Autograd layer
    DispatchKey.AutogradOther,
    DispatchKey.AutogradCPU,
    DispatchKey.AutogradNPU,
    DispatchKey.AutogradCUDA,
    DispatchKey.AutogradXPU,
    DispatchKey.AutogradMeta,

    # 4. [MOVED] Python key — after Autograd, before Backend
    DispatchKey.Python,

    # 5. Composite keys
    DispatchKey.CompositeImplicitAutograd,
    DispatchKey.CompositeExplicitAutograd,

    # 6. Backend keys
    DispatchKey.Meta,
    DispatchKey.NPU,
    DispatchKey.CUDA,
    DispatchKey.CPU,
]
```

**Keyset construction — detect subclasses:**

```python
def _compute_dispatch_keyset(tensors):
    keyset = ...  # existing logic

    # [NEW] Detect tensor subclasses with __torch_dispatch__
    for t in tensors:
        if type(t) is not Tensor and hasattr(type(t), '__torch_dispatch__'):
            keyset = keyset.add(DispatchKey.Python)
            # Remove Python from fallthrough set
            break

    return keyset
```

**Dispatch handling — Python key kernel:**

```python
def dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs):
    # ... existing autograd / autocast / functionalize handling ...

    # [NEW] Python dispatch key -> __torch_dispatch__
    if keyset.has(DispatchKey.Python):
        tensors = _extract_tensors(args, kwargs)
        types = _get_overloaded_types_dispatch(tensors)
        if types:
            for cls in _sorted_by_mro(types):
                result = cls.__torch_dispatch__(
                    func,        # op reference (e.g. "aten::add")
                    types,       # set of tensor types involved
                    args,
                    kwargs or {}
                )
                if result is not NotImplemented:
                    return result

    # Normal backend kernel execution
    kernel, key = _kernel_for_entry(entry, _key_order(keyset))
    return kernel(*args, **impl_kwargs)
```

Default implementation on Tensor base class:

```python
class Tensor:
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return NotImplemented  # Don't intercept, use normal dispatch
```

### 0.3 Key Semantic Guarantees

1. `__torch_function__` is called BEFORE entering the dispatcher — subclasses can bypass schema validation and all dispatch logic.
2. `__torch_dispatch__` is called AFTER autograd recording — subclasses receive autograd-wrapped tensors, and return values are correctly tracked.
3. MRO priority: when multiple subclass types are present, the most derived class gets called first.
4. `NotImplemented` fallthrough: if all subclasses return `NotImplemented`, normal dispatch proceeds.

---

## Layer 1: DeviceMesh (1D MVP)

**Location:** `candle/distributed/device_mesh.py`

```python
class DeviceMesh:
    """Multi-dimensional device topology abstraction.

    MVP: 1D mesh only (pure FSDP). API aligned with PyTorch for 2D+ extension.

    Usage:
        mesh = DeviceMesh("npu", (8,), mesh_dim_names=("shard",))
    """

    def __init__(self, device_type, mesh_shape, *, mesh_dim_names=None):
        self.device_type = device_type           # "npu" / "cpu"
        self.mesh = _build_mesh_tensor(mesh_shape)  # rank arrangement
        self.mesh_dim_names = mesh_dim_names
        self._dim_groups: list[ProcessGroup] = []
        self._init_process_groups()

    def get_group(self, mesh_dim=0) -> ProcessGroup:
        """Get the ProcessGroup for a mesh dimension."""
        return self._dim_groups[mesh_dim]

    def size(self, mesh_dim=0) -> int:
        """Number of devices along a mesh dimension."""
        return self.mesh.shape[mesh_dim]

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    def _init_process_groups(self):
        """Create ProcessGroups per mesh dimension.

        1D MVP: reuse the global WORLD process group.
        Future 2D+: create sub-groups per dimension.
        """
        ...
```

**Design notes:**
- `DeviceMesh` is a thin wrapper over `ProcessGroup` with multi-dimensional indexing.
- 1D mesh directly reuses `dist.group.WORLD`.
- API matches `torch.distributed.device_mesh.DeviceMesh` exactly.

---

## Layer 2: DTensor + Placement Types

### 2.1 Placement Types

**Location:** `candle/distributed/tensor/placement.py`

```python
class Placement:
    """Base class for tensor placement strategies."""
    pass

class Shard(Placement):
    """Tensor is sharded along a dimension across the mesh."""
    def __init__(self, dim: int = 0):
        self.dim = dim

class Replicate(Placement):
    """Tensor is replicated across all ranks in the mesh."""
    pass

class Partial(Placement):
    """Tensor has pending reduction (e.g., gradient before reduce-scatter)."""
    def __init__(self, reduce_op="sum"):
        self.reduce_op = reduce_op
```

### 2.2 DTensor

**Location:** `candle/distributed/tensor/dtensor.py`

DTensor is a **lightweight metadata container** — it wraps a local tensor shard with placement/mesh information. FSDP manages communication manually; DTensor does NOT perform automatic redistribution.

```python
class DTensorSpec:
    """Metadata describing how a tensor is distributed."""
    def __init__(self, mesh, placements, tensor_meta=None):
        self.mesh = mesh                     # DeviceMesh
        self.placements = tuple(placements)  # (Shard(0),) etc.
        self.tensor_meta = tensor_meta       # TensorMeta (global shape/stride/dtype)

    def has_shard_placement(self):
        return any(isinstance(p, Shard) for p in self.placements)


class TensorMeta:
    """Global tensor metadata (shape as if not sharded)."""
    def __init__(self, shape, stride, dtype):
        self.shape = shape
        self.stride = stride
        self.dtype = dtype


class DTensor(Tensor):
    """Distributed Tensor — sharded parameter container for FSDP2.

    Wraps a local tensor shard with placement metadata.
    Not intended for direct computation when sharded.
    """

    def __init__(self, local_tensor, spec, *, requires_grad=False):
        super().__init__(
            local_tensor._storage,
            local_tensor.shape,
            local_tensor.stride,
            local_tensor.offset,
            requires_grad=requires_grad,
        )
        self._local_tensor = local_tensor
        self._spec = spec

    @property
    def placements(self):
        return self._spec.placements

    @property
    def device_mesh(self):
        return self._spec.mesh

    @staticmethod
    def from_local(local_tensor, device_mesh, placements):
        """Construct DTensor from a local shard. (PyTorch-aligned API)"""
        tensor_meta = TensorMeta(
            shape=_compute_global_shape(local_tensor.shape, device_mesh, placements),
            stride=_compute_global_stride(local_tensor.stride, device_mesh, placements),
            dtype=local_tensor.dtype,
        )
        spec = DTensorSpec(device_mesh, placements, tensor_meta)
        return DTensor(local_tensor, spec, requires_grad=local_tensor.requires_grad)

    def to_local(self):
        """Extract the local tensor shard (unwrap DTensor)."""
        return self._local_tensor

    # ── Dispatch protocols ──

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # MVP: delegate everything to __torch_dispatch__
        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        specs = _extract_specs(args)

        # Sharded DTensors must not be computed on directly.
        # FSDP unshards to plain Tensors before forward/backward.
        for spec in specs:
            if spec.has_shard_placement():
                raise RuntimeError(
                    f"{func}: direct compute on sharded DTensor is not supported. "
                    f"Ensure fully_shard() hooks are properly registered."
                )

        # Replicate placement: execute on local tensor directly
        unwrapped = _unwrap_dtensors(args)
        result = func(*unwrapped, **(kwargs or {}))
        return _rewrap_result(result, specs)
```

### 2.3 DTensor Dispatch Strategy for FSDP2 MVP

| DTensor State | When | `__torch_dispatch__` Behavior |
|---------------|------|-------------------------------|
| **Sharded** | Parameter storage state | **Raise error** (should not be computed on directly) |
| **Replicate** | HSDP replicate group (future) | Execute on local tensor |
| **Unsharded** | During forward/backward | Not a DTensor — plain Tensor, no dispatch interception |

In the FSDP2 flow, `__torch_dispatch__` is rarely invoked because FSDP's pre-forward hook unshards DTensor parameters into plain Tensors before `module.forward()` runs.

---

## Layer 3: FSDPParam + FSDPParamGroup

### 3.1 FSDPParam — Per-Parameter Shard Lifecycle

**Location:** `candle/distributed/_composable/fsdp/_fsdp_param.py`

```python
class ShardedState(Enum):
    SHARDED = auto()      # Only sharded shard in device memory
    UNSHARDED = auto()    # Both sharded + unsharded in memory


class FSDPParam:
    """Manages shard/unshard lifecycle for a single parameter."""

    def __init__(self, param, module, param_name, mesh_info):
        self._module = module
        self._param_name = param_name
        self._mesh_info = mesh_info
        self._sharded_state = ShardedState.SHARDED

        # Original parameter metadata
        self._orig_shape = param.shape
        self._orig_dtype = param.dtype
        self._shard_dim = 0  # MVP: fixed dim-0 sharding

        # Perform initial sharding
        self._sharded_param = self._init_shard(param)  # DTensor
        self._unsharded_param = None                     # Plain Tensor after unshard

        # Replace module parameter with sharded DTensor
        setattr(module, param_name, self._sharded_param)

    def _init_shard(self, param):
        """Shard a full parameter into local shard, wrap as DTensor."""
        rank = self._mesh_info.shard_mesh_rank
        world_size = self._mesh_info.shard_mesh_size

        chunks = _chunk_tensor(param, world_size, dim=self._shard_dim)
        local_shard = chunks[rank].contiguous()

        return DTensor.from_local(
            local_shard,
            self._mesh_info.mesh,
            placements=(Shard(self._shard_dim),),
        )

    def unshard(self):
        """All-gather: reconstruct full parameter from shards."""
        if self._sharded_state == ShardedState.UNSHARDED:
            return

        local_tensor = self._sharded_param.to_local()
        world_size = self._mesh_info.shard_mesh_size
        output = _alloc_unsharded_buffer(local_tensor, world_size, self._shard_dim)

        pg = self._mesh_info.shard_process_group
        dist.all_gather_into_tensor(output, local_tensor, group=pg)

        self._unsharded_param = output.reshape(self._orig_shape)
        self._unsharded_param.requires_grad = self._sharded_param.requires_grad

        # Swap module parameter to plain Tensor for forward/backward
        setattr(self._module, self._param_name, self._unsharded_param)
        self._sharded_state = ShardedState.UNSHARDED

    def reshard(self):
        """Free unsharded parameter, restore sharded DTensor on module."""
        if self._sharded_state == ShardedState.SHARDED:
            return

        setattr(self._module, self._param_name, self._sharded_param)
        self._unsharded_param = None
        self._sharded_state = ShardedState.SHARDED

    def reduce_scatter_grad(self):
        """Reduce-scatter: unsharded gradient -> sharded gradient."""
        grad = self._unsharded_param.grad
        if grad is None:
            return

        shard_size = self._sharded_param.to_local().shape
        reduced_grad = empty(shard_size, dtype=grad.dtype, device=grad.device)

        pg = self._mesh_info.shard_process_group
        dist.reduce_scatter_tensor(reduced_grad, grad, group=pg)

        self._sharded_param.to_local().grad = reduced_grad
```

### 3.2 FSDPParamGroup — Batched Communication

**Location:** `candle/distributed/_composable/fsdp/_fsdp_param_group.py`

```python
class FSDPParamGroup:
    """Groups parameters from one fully_shard() call for batched collectives."""

    def __init__(self, fsdp_params, module, mesh_info):
        self.fsdp_params = fsdp_params      # list[FSDPParam]
        self.module = module
        self.mesh_info = mesh_info
        self._is_unsharded = False

    def unshard(self):
        if self._is_unsharded:
            return
        # MVP: per-parameter all-gather
        # TODO: merge into single all-gather (concatenate shards into one buffer)
        for p in self.fsdp_params:
            p.unshard()
        self._is_unsharded = True

    def reshard(self):
        if not self._is_unsharded:
            return
        for p in self.fsdp_params:
            p.reshard()
        self._is_unsharded = False

    def reduce_scatter_grads(self):
        # MVP: per-parameter reduce-scatter
        # TODO: merge into single reduce-scatter
        for p in self.fsdp_params:
            p.reduce_scatter_grad()

    def pre_forward(self):
        self.unshard()

    def post_forward(self):
        self.reshard()

    def pre_backward(self):
        self.unshard()

    def post_backward(self):
        self.reduce_scatter_grads()
        self.reshard()
```

**MVP simplifications:**
- Per-parameter all-gather / reduce-scatter (no buffer merging)
- No communication/computation overlap (single stream)
- Fixed dim-0 sharding

---

## Layer 4: FSDPState + Hook Orchestration

### 4.1 FSDPState

**Location:** `candle/distributed/_composable/fsdp/_fsdp_state.py`

```python
class FSDPState:
    """Manages FSDP hook lifecycle for a module."""

    def __init__(self, module, param_group, mesh_info, reshard_after_forward):
        self.module = module
        self.param_group = param_group
        self.mesh_info = mesh_info
        self.reshard_after_forward = reshard_after_forward
        self._is_root = None                  # Lazily identified
        self._pre_backward_hook_handles = []
        self._grad_count = 0
        self._total_managed_params = len(param_group.fsdp_params)

        # Register forward hooks
        self._pre_fwd_handle = module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_fwd_handle = module.register_forward_hook(
            self._post_forward, prepend=False
        )

    # ── Forward ──────────────────────────────────

    def _pre_forward(self, module, args, kwargs):
        """Pre-forward: all-gather to reconstruct full parameters."""
        if self._is_root is None:
            self._lazy_init_root()
        self.param_group.pre_forward()
        return args, kwargs

    def _post_forward(self, module, args, output):
        """Post-forward: register backward hooks + reshard."""
        self._register_pre_backward_hooks(output)
        self._register_post_backward_hooks()

        if self.reshard_after_forward:
            self.param_group.post_forward()
        return output

    # ── Backward ─────────────────────────────────

    def _register_pre_backward_hooks(self, output):
        """Register hooks on forward output tensors -> trigger unshard on backward entry."""
        tensors = _extract_tensors_from_output(output)
        for t in tensors:
            if t.requires_grad:
                handle = t.register_hook(self._pre_backward)
                self._pre_backward_hook_handles.append(handle)

    def _pre_backward(self, grad):
        """Backward reaches this module: re-all-gather parameters."""
        self.param_group.pre_backward()
        return grad

    def _register_post_backward_hooks(self):
        """Register gradient hooks on parameters -> trigger reduce-scatter when all grads ready."""
        self._grad_count = 0
        for fsdp_param in self.param_group.fsdp_params:
            unsharded = fsdp_param._unsharded_param
            if unsharded is not None and unsharded.requires_grad:
                unsharded.register_hook(
                    self._make_post_backward_hook(fsdp_param)
                )

    def _make_post_backward_hook(self, fsdp_param):
        def hook(grad):
            self._grad_count += 1
            if self._grad_count == self._total_managed_params:
                self.param_group.post_backward()
                self._grad_count = 0
            return grad
        return hook

    # ── Root identification ──────────────────────

    def _lazy_init_root(self):
        """Identify root on first forward (outermost fully_shard module)."""
        self._is_root = not any(
            hasattr(m, '_fsdp_state')
            for m in _get_parent_modules(self.module)
        )
        if self._is_root:
            self.reshard_after_forward = False
```

### 4.2 Forward/Backward Flow

```
FORWARD:
  pre_forward_hook  -> all-gather (unshard all params in group)
  module.forward()  -> compute with plain Tensors (no DTensor dispatch)
  post_forward_hook -> register backward hooks on output tensors
                    -> register gradient hooks on parameters
                    -> reshard (free unsharded memory, non-root only)

BACKWARD:
  output tensor hook  -> all-gather (re-unshard params for grad computation)
  autograd backward   -> compute gradients using unsharded params
  param gradient hook -> count completed grads
                      -> when all grads ready: reduce-scatter + reshard
```

---

## Layer 5: `fully_shard()` API + FSDPModule Mixin

### 5.1 FSDPModule Mixin

**Location:** `candle/distributed/_composable/fsdp/_fsdp_api.py`

```python
class FSDPModule:
    """Mixin injected into module's MRO by fully_shard()."""

    @property
    def fsdp_state(self) -> FSDPState:
        return self._fsdp_state

    def set_reshard_after_forward(self, value):
        self._fsdp_state.reshard_after_forward = value

    def set_modules_to_forward_prefetch(self, modules):
        pass  # MVP no-op, reserved for overlap optimization

    def set_modules_to_backward_prefetch(self, modules):
        pass  # MVP no-op, reserved for overlap optimization
```

### 5.2 `fully_shard()` Entry Point

**Location:** `candle/distributed/_composable/fsdp/__init__.py`

```python
def fully_shard(module, *, mesh, reshard_after_forward=None):
    """
    Apply FSDP2 to a module. PyTorch-compatible API.

    Must be called bottom-up on the model:
        mesh = DeviceMesh("npu", (world_size,))
        fully_shard(model.encoder, mesh=mesh)
        fully_shard(model.decoder, mesh=mesh)
        fully_shard(model, mesh=mesh)  # root

    Args:
        module: nn.Module to shard
        mesh: DeviceMesh (1D for FSDP)
        reshard_after_forward: bool or None
            True  = free unsharded params after forward (saves memory)
            False = keep unsharded params (avoids re-all-gather in backward)
            None  = auto (True for non-root, False for root)

    Returns:
        The same module, with FSDP state and hooks attached.
    """
    # 1. Build mesh info
    mesh_info = FSDPMeshInfo(mesh)

    # 2. Collect directly-owned parameters (exclude child fully_shard params)
    managed_params = _get_managed_params(module)
    if not managed_params:
        return module

    # 3. Create FSDPParam for each parameter (performs initial sharding)
    fsdp_params = [
        FSDPParam(param, module, name, mesh_info)
        for name, param in managed_params
    ]

    # 4. Group parameters for batched communication
    param_group = FSDPParamGroup(fsdp_params, module, mesh_info)

    # 5. Reshard strategy: default True, root overrides to False in lazy_init
    if reshard_after_forward is None:
        reshard_after_forward = True

    # 6. Create FSDPState and register hooks
    state = FSDPState(module, param_group, mesh_info, reshard_after_forward)
    module._fsdp_state = state

    # 7. Inject FSDPModule mixin into MRO
    _inject_fsdp_mixin(module)

    return module


def _inject_fsdp_mixin(module):
    """Dynamically insert FSDPModule into module's class MRO."""
    cls = type(module)
    if FSDPModule not in cls.__mro__:
        new_cls = type(f"FSDP_{cls.__name__}", (FSDPModule, cls), {})
        module.__class__ = new_cls


def _get_managed_params(module):
    """Collect directly-owned parameters, excluding those managed by child fully_shard calls."""
    child_fsdp_params = set()
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, '_fsdp_state'):
            for p in child.parameters():
                child_fsdp_params.add(id(p))

    managed = []
    for name, param in module.named_parameters(recurse=False):
        if id(param) not in child_fsdp_params:
            managed.append((name, param))
    return managed
```

---

## Layer 6: nn.Module Hook Fixes

### Required Changes

FSDP2 depends on `prepend=True` and `with_kwargs=True` for forward hooks. Currently these parameters are accepted but ignored.

**File:** `candle/nn/module.py`

### 6.1 Hook Registration with `prepend` Support

```python
def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
    handle = _RemovableHandle(self._forward_pre_hooks_dict)
    self._forward_pre_hooks_dict[handle.id] = (hook, with_kwargs)
    if prepend:
        self._forward_pre_hooks_dict.move_to_end(handle.id, last=False)
    return handle

def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False):
    handle = _RemovableHandle(self._forward_hooks_dict)
    self._forward_hooks_dict[handle.id] = (hook, with_kwargs)
    if prepend:
        self._forward_hooks_dict.move_to_end(handle.id, last=False)
    return handle
```

### 6.2 `__call__` with `with_kwargs` Support

```python
def __call__(self, *args, **kwargs):
    # Pre-forward hooks
    for hook_entry in self._forward_pre_hooks_dict.values():
        hook, with_kwargs = hook_entry
        if with_kwargs:
            result = hook(self, args, kwargs)
            if result is not None:
                args, kwargs = result
        else:
            result = hook(self, args)
            if result is not None:
                if isinstance(result, tuple):
                    args = result
                else:
                    args = (result,)

    output = self.forward(*args, **kwargs)

    # Post-forward hooks
    for hook_entry in self._forward_hooks_dict.values():
        hook, with_kwargs = hook_entry
        if with_kwargs:
            result = hook(self, (args, kwargs), output)
        else:
            result = hook(self, args, output)
        if result is not None:
            output = result

    return output
```

**Note:** The hook dict value format changes from `hook` to `(hook, with_kwargs)` tuple. Existing code that registers hooks without `with_kwargs` needs to be backward-compatible — default `with_kwargs=False`.

---

## Implementation Order

Bottom-up, each layer builds on the previous:

| Step | Component | Files | Depends On |
|------|-----------|-------|------------|
| 1 | nn.Module hook fixes (`prepend`, `with_kwargs`) | `nn/module.py` | — |
| 2 | `__torch_function__` protocol | `_functional.py`, `_tensor.py` | — |
| 3 | `__torch_dispatch__` protocol + key reorder | `_dispatch/keys.py`, `_dispatch/dispatcher.py`, `_tensor.py` | — |
| 4 | `Placement` types | `distributed/tensor/placement.py` | — |
| 5 | `DeviceMesh` (1D) | `distributed/device_mesh.py` | ProcessGroup (existing) |
| 6 | `DTensor` | `distributed/tensor/dtensor.py` | Placement, DeviceMesh, `__torch_function__`, `__torch_dispatch__` |
| 7 | `FSDPParam` | `distributed/_composable/fsdp/_fsdp_param.py` | DTensor, DeviceMesh, collectives (existing) |
| 8 | `FSDPParamGroup` | `distributed/_composable/fsdp/_fsdp_param_group.py` | FSDPParam |
| 9 | `FSDPState` + hook orchestration | `distributed/_composable/fsdp/_fsdp_state.py` | FSDPParamGroup, module hooks |
| 10 | `fully_shard()` + `FSDPModule` | `distributed/_composable/fsdp/__init__.py` | FSDPState |
| 11 | End-to-end validation | `tests/`, `examples/` | All above |

Steps 1-3 can be parallelized. Steps 4-5 can be parallelized. Steps 7-8 are sequential.

---

## File Layout

```
src/candle/
├── _tensor.py                          # Add __torch_function__, __torch_dispatch__
├── _functional.py                      # Add _handle_torch_function() calls
├── _dispatch/
│   ├── keys.py                         # Reorder Python key priority
│   └── dispatcher.py                   # Add __torch_dispatch__ handling
├── nn/
│   └── module.py                       # Fix prepend + with_kwargs
└── distributed/
    ├── device_mesh.py                  # DeviceMesh (replace stub)
    ├── tensor/
    │   ├── __init__.py                 # Re-export DTensor, Placement
    │   ├── placement.py                # Shard, Replicate, Partial
    │   └── dtensor.py                  # DTensor class
    └── _composable/
        └── fsdp/
            ├── __init__.py             # fully_shard(), FSDPModule
            ├── _fsdp_param.py          # FSDPParam
            ├── _fsdp_param_group.py    # FSDPParamGroup
            ├── _fsdp_state.py          # FSDPState + hooks
            └── _fsdp_common.py         # FSDPMeshInfo, helpers
```

---

## Risks and Open Questions

1. **Autograd compatibility**: The post-backward hook relies on gradient hooks counting to `_total_managed_params`. If some parameters don't receive gradients (unused in forward), the count will never reach the threshold. Need a mechanism to detect unused parameters (similar to DDP's `find_unused_parameters`).

2. **Parameter mutation semantics**: `setattr(module, param_name, tensor)` swaps between DTensor and plain Tensor. Need to verify this works correctly with candle's `nn.Module.named_parameters()` and optimizer parameter groups.

3. **Optimizer integration**: After reduce-scatter, sharded gradients live on `self._sharded_param.to_local().grad`. The optimizer must operate on the sharded parameters (DTensors), not the unsharded ones. Need to verify optimizer step works with DTensor parameters.

4. **Memory management**: MVP doesn't resize storage (PyTorch uses `alloc_storage` / `free_storage`). Unsharded params are simply set to `None` on reshard. This may not immediately free device memory in all cases.

5. **`all_gather_into_tensor` / `reduce_scatter_tensor`**: Need to verify these exist in candle's distributed API. If only `all_gather` (list-based) is available, need to add the tensor-based variants.
