# NPU Execution Flow Alignment with torch_npu

**Date**: 2026-03-22
**Status**: Draft
**Goal**: Reduce candle NPU per-op submit cost from ~137µs to ~55µs, aligning with torch_npu's ~57µs on 910B3 + CANN 8.5.0

---

## Background

### Measured gap (torch.add, 64×64 float32, 910B3)

| | candle | torch_npu |
|---|---|---|
| Single op e2e (w/ sync) | 193µs | 57µs |
| Per-op submit (chained) | ~137µs | ~22µs |
| synchronize() | ~25µs | ~35µs |

### torch_npu execution model (verified by measurement)

torch_npu's `torch.add` is **submit-to-stream**: it calls aclnn, enqueues work on the CANN stream, and returns immediately. The NPU executes asynchronously. `synchronize()` blocks until the stream drains.

This is NOT a worker-thread async queue. The CANN stream itself is the queue. Both candle and torch_npu already use this model — the gap is purely in per-op Python/Cython overhead before and after the aclnn submit.

### candle fast_binary_op breakdown (bench_fast_path_profile.py)

```
Stage                          Median
─────────────────────────────────────
fast_binary_op total           137µs
  aclnn kernel (3×create+exec)  72µs   ← rebuild descriptors every op
  alloc_device                  16µs   ← CachingAllocator.malloc
  Tensor.__init__               16µs   ← TensorImpl object construction
  npu_typed_storage_from_ptr    14µs   ← FastNPUStorage + FastTypedStorage
  a.storage() × 2 + other       19µs   ← Python method calls
```

### torch_npu per-op breakdown (inferred from profiling + source analysis)

```
torch.add call                 ~22µs (chained) / ~32µs (first)
  aclnnAdd submit               ~10µs  (aclCreateTensor cached in GE engine)
  NPUStorageImpl construction    ~8µs  (C++ alloc + StorageImpl)
  other C++ overhead             ~4µs
```

---

## Design Principles

1. **No new fast-path bypasses**: every optimization operates within the existing correct architecture (allocator → storage → tensor → aclnn). No shortcuts that skip memory safety or device management.
2. **Small, independently verifiable PRs**: each PR has a measurable benchmark delta. A PR that shows no improvement is a sign of a problem, not merged as-is.
3. **Cython-only**: no C++ dependencies. All hot-path code stays in `.pyx` files.
4. **Correctness before speed**: if a risk item (see Risk section) conflicts with a performance gain, correctness wins.

---

## Five PRs

### PR-1: aclTensor Descriptor Cache

**Target**: eliminate 3× `aclCreateTensor` + 3× `aclDestroyTensor` per op
**Expected savings**: ~30µs (aclnn time 72µs → ~42µs)

#### Design

Add `TensorDescCache` cdef class to `_aclnn_ffi.pyx`:

```
cdef class TensorDescCache:
    # LRU dict: key → aclTensor handle (uintptr_t)
    # key = (data_ptr, shape_tuple, stride_tuple, dtype_code, fmt)
    cdef dict _cache
    cdef int _max_size  # default 64
```

**Cache key**: `(data_ptr: int, shape: tuple, stride: tuple, dtype_code: int, fmt: int)`

`data_ptr` is the **actual data pointer** (base_ptr + storage_offset × itemsize), not the storage base ptr. This ensures sliced views (`a[2:]`) with the same base ptr but different offset are treated as distinct entries.

**Why stride in key**: non-contiguous tensors (transpose, slice) have different stride. Same data_ptr + different stride = different aclTensor.

**Invalidation**: when the allocator frees a block, it calls `desc_cache.invalidate(base_ptr)`. This removes all entries whose `data_ptr` falls within the freed block's address range `[base_ptr, base_ptr + size)`. This is the only invalidation mechanism needed — the allocator owns all device memory lifetimes.

**Scope**: cache **input** tensors only. Output tensor ptr is freshly allocated each op; it cannot be cached without `out=` API (future work).

**Cache size**: bounded at 64 entries (LRU eviction). Typical workloads reuse the same set of tensors in a loop, so 64 is sufficient.

**Thread safety**: candle holds the GIL for all Python/Cython code. No additional locking needed.

**Integration point**: in `_aclnn_ffi.pyx`'s `binary_op_with_alpha` (and similar), replace:
```python
aclCreateTensor(shape, stride, dtype, ptr)  # every call
```
with:
```python
cache.get_or_create(ptr, shape, stride, dtype, fmt)
```

**Risk**: stride-based invalidation. An in-place op changes tensor contents but not ptr/shape/stride — cache correctly serves the new data since aclTensor only holds metadata (ptr + layout), not data. An op that changes a view's stride (e.g., `.contiguous()` producing a new allocation) changes ptr, so it naturally misses the cache.

#### Files changed
- `src/candle/_cython/_aclnn_ffi.pyx`: add `TensorDescCache`, integrate into hot path
- `src/candle/_backends/npu/allocator.py` or `_allocator.pyx`: call `desc_cache.invalidate(ptr)` in `free()`
- `tests/npu/`: add cache hit/miss tests, invalidation-on-free test

---

### PR-2: Direct Device Pointer Access

**Target**: eliminate 2× `a.storage().data_ptr()` Python method calls
**Expected savings**: ~8µs

#### Design

`fast_binary_op` currently calls `a.storage()` (Python method → returns new storage view object) then `.data_ptr()` (another Python call). Both are avoidable because `FastNPUStorage._device_ptr` is a `cdef public int64_t` — directly accessible from Cython with zero overhead.

Change `fast_binary_op` in `_npu_ops.pyx`:
```python
# Before
a_ptr = a.storage().data_ptr()
b_ptr = b.storage().data_ptr()

# After (Cython direct access)
a_ptr = a._impl._storage._untyped._device_ptr
b_ptr = b._impl._storage._untyped._device_ptr
```

**Risk**: `DummyStorage` used in unit tests does not have `_device_ptr`. Fix: expose `_device_ptr` property on `DummyStorage` that returns `0` or raises a clear error, so tests keep working without a slow fallback.

**Note**: this does NOT bypass storage — it reads the same field that `storage().data_ptr()` would return, just without the Python method call overhead.

#### Files changed
- `src/candle/_cython/_npu_ops.pyx`: replace `.storage().data_ptr()` calls
- `src/candle/_storage.py` or test helpers: expose `_device_ptr` on `DummyStorage`
- `tests/npu/`: verify existing tests still pass

---

### PR-3: C-level Output Tensor Construction

**Target**: eliminate Python `__init__` overhead for output tensor
**Expected savings**: ~30µs (storage_from_ptr 14µs + Tensor.__init__ 16µs)

#### Design

Add `cy_make_npu_tensor` cdef function to `_storage.pyx` (or `_npu_ops.pyx`):

```cython
cdef object cy_make_npu_tensor(
    int64_t device_ptr, int64_t n_elements,
    object dtype, object device,
    tuple shape, tuple stride):
    """Construct NPU Tensor entirely in Cython — zero Python __init__ calls."""
    # 1. FastNPUStorage.__cinit__ (already cdef, fast)
    untyped = FastNPUStorage(device_ptr, n_elements * itemsize, device)
    # 2. FastTypedStorage.__cinit__ (already cdef, fast)
    typed = FastTypedStorage(untyped, dtype, n_elements)
    # 3. TensorImpl construction via Cython-accessible init path
    impl = TensorImpl.__new__(TensorImpl)
    impl._set_shape(shape)
    impl._set_stride(stride)
    impl._set_device_from_obj(device)
    impl._set_dtype_from_obj(dtype)
    impl._storage = typed
    impl.requires_grad = False
    impl._update_dispatch_keys()
    # 4. Tensor.__new__ + assign _impl
    t = Tensor.__new__(Tensor)
    t._impl = impl
    return t
```

`TensorImpl` is already a cdef class with `_set_shape`, `_set_stride` etc. as cdef methods. `Tensor.__new__` skips `__init__`. The result is semantically identical to `Tensor(storage, shape, stride)` but without the Python dispatch overhead.

**Pre-condition**: verify that `TensorImpl` exposes `_set_shape`, `_set_stride`, `_set_device_from_obj`, `_set_dtype_from_obj`, and `_update_dispatch_keys` as cdef methods before implementing. If any are missing, add them as part of this PR.

**Risk**: if `Tensor.__init__` does additional setup beyond what `cy_make_npu_tensor` replicates, the produced tensor will be broken. Mitigation: add a test that compares a tensor produced by `cy_make_npu_tensor` against one produced by the normal constructor for all observable properties (shape, stride, dtype, device, data_ptr, requires_grad, grad_fn).

#### Files changed
- `src/candle/_cython/_storage.pyx` or `_npu_ops.pyx`: add `cy_make_npu_tensor`
- `src/candle/_cython/_npu_ops.pyx`: replace `npu_typed_storage_from_ptr` + `Tensor(...)` with `cy_make_npu_tensor`
- `tests/npu/`: add constructor equivalence test

---

### PR-4: Allocator `_drain_pending` Hot Path

**Target**: reduce `_drain_pending` overhead from ~10µs to ~3µs
**Expected savings**: ~7µs (contributes to alloc_device 16µs → ~9µs)

#### Design

`_drain_pending` currently iterates a Python `list` of `(event, block)` tuples, calling `_event_complete(event)` (which imports runtime and calls `query_event`) for each entry. In the common steady-state case the list has 0–2 entries.

Optimizations:
1. **Inline `_event_complete` as cdef**: remove the `from candle._backends.npu import runtime` import inside the loop (it's cached but still costs a dict lookup). Use a module-level cached reference.
2. **Replace `list` with a C array of structs**: for ≤64 pending events, use a fixed-size C array of `(event_ptr: uintptr_t, block_ptr: void*)` in `FastNpuAllocator`, eliminating Python list allocation and tuple unpacking. If the array fills (> 64 entries), overflow to a Python list for correctness — this case is rare and the slow path is acceptable.
3. **Skip drain when list is empty**: `if not self._pending_events: return` (already done; verify it's at cdef level).

This keeps the same event-based free semantics as torch_npu's `NPUCachingAllocator` — no behavioral change.

#### Files changed
- `src/candle/_cython/_allocator.pyx`: optimize `_drain_pending`, inline `_event_complete`
- `tests/npu/`: allocator correctness tests must continue to pass

---

### PR-5: Workspace Static Reuse

**Target**: eliminate workspace allocation from main allocator pool
**Expected savings**: ~5µs (removes one malloc+free cycle for workspace)

#### Design

torch_npu uses a dedicated `NPUWorkspaceAllocator` separate from the main caching allocator. `unsafe_empty_workspace(size)` returns a tensor backed by a static, growing buffer. For `aclnnAdd`, workspace size is typically 0 — no allocation needed.

Add per-device workspace buffer to `_aclnn_ffi.pyx`:

```cython
cdef int64_t[8] _ws_ptr      # per device (max 8 devices)
cdef int64_t[8] _ws_size

cdef int64_t _get_workspace(int device_id, int64_t needed, object runtime):
    """Return a workspace ptr of at least `needed` bytes.
    If needed == 0, returns 0 (NULL) without any allocation.
    Reuses existing buffer if large enough; reallocates otherwise.
    """
    if needed == 0:
        return 0
    if _ws_size[device_id] >= needed:
        return _ws_ptr[device_id]
    # reallocate: free old, malloc new via runtime allocator
    if _ws_ptr[device_id] != 0:
        runtime.acl.rt.free(<void*>_ws_ptr[device_id])
    _ws_ptr[device_id] = runtime.acl.rt.malloc(needed)
    _ws_size[device_id] = needed
    return _ws_ptr[device_id]
```

This buffer is NOT tracked by the caching allocator — exactly matching torch_npu's `NPUWorkspaceAllocator` behavior. It is device-global and grows monotonically (like PyTorch's workspace).

**Integration**: replace workspace allocation in `binary_op_with_alpha` (and other aclnn callers) with `_get_workspace(device_id, ws_size, runtime)`.

**Risk**: if a kernel truly requires workspace > 0, the old path used the caching allocator (with event-tracking). The new path uses `aclrtMalloc` directly, which is fine for workspace (short-lived, never shared). Add an assertion that workspace is freed after kernel execution (already guaranteed since workspace is reused, not accumulated).

**Cleanup**: register an `atexit` handler (or `__dealloc__` on a cdef class) to `aclrtFree` the workspace buffer on process exit, matching torch_npu's `NPUWorkspaceAllocator` destructor behavior.

#### Files changed
- `src/candle/_cython/_aclnn_ffi.pyx`: add `_get_workspace`, replace workspace malloc in all op callers
- `tests/npu/`: add test for op that requires workspace > 0 (e.g., matmul)

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| descriptor cache: stride change invalidation | High | key includes full stride tuple; ptr change (new alloc) is sufficient for output tensors |
| descriptor cache: format mismatch | Medium | key includes fmt; current candle always uses ACL_FORMAT_ND (correct for ND tensors) |
| PR-3: Tensor.__new__ missing init | Medium | add equivalence test comparing all observable Tensor properties |
| PR-5: workspace with size > 0 | Low | common ops (add/mul/sub/div) have ws=0; tested with matmul which has ws>0 |
| allocator: event pool exhaustion under PR-4 | Low | event pool falls back to `create_event`; same as before |

---

## Success Criteria

Each PR merged only if `bench_fast_path_profile.py` shows regression-free improvement:

| After PR | Expected fast_binary_op median |
|---|---|
| baseline (current) | 137µs |
| PR-1 (descriptor cache) | ~107µs |
| PR-2 (direct ptr) | ~99µs |
| PR-3 (C-level tensor) | ~69µs |
| PR-4 (allocator drain) | ~62µs |
| PR-5 (workspace reuse) | ~57µs |
| torch_npu reference | ~57µs |

---

## PR Ordering Notes

PRs 1–5 are logically independent (each targets a different bottleneck) but the recommended merge order is 1→2→3→4→5 because:
- PR-2 validates direct cdef attribute access on `FastNPUStorage`, which PR-3 also relies on
- PR-1 (descriptor cache) has the largest single impact; merging it first gives the most visible benchmark signal to validate the approach before committing to the rest
- PR-4 and PR-5 are lower-risk and can be merged in either order

Any PR can be skipped or deferred without blocking others.

---

## Out of Scope

- Worker-thread async queue: torch_npu does NOT use one; CANN stream is the queue
- PyTorch ATen dispatcher registration: candle remains independent of PyTorch runtime
- NCHW / non-ND format support: separate feature, not part of this alignment work
- `out=` parameter API: follow-on optimization, unlocks output descriptor caching
