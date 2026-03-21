"""Candle's C extension layer — Cython accelerators for hot paths.

Runtime-critical modules are imported directly by their Python wrappers
(`_dispatch.dispatcher`, `_tensor`, `_device`, `autograd.engine`, etc.).
This package intentionally avoids eagerly importing autograd-heavy modules in
`__init__` because doing so creates circular-import bootstrapping issues during
`import candle`.

The runtime core requires the following compiled modules to exist:
- `_tensor_impl`
- `_dispatcher_core`
- `_device`
- `_dtype`
"""

_HAS_CYTHON_DISPATCH = False
_HAS_CYTHON_ALLOCATOR = False
_HAS_CYTHON_STORAGE = False
_HAS_CYTHON_NPU_OPS = False
_HAS_CYTHON_ACLNN_FFI = False
_HAS_CYTHON_TENSOR_IMPL = False
_HAS_CYTHON_DISPATCHER_CORE = False
_HAS_CYTHON_DEVICE = False
_HAS_CYTHON_DTYPE = False
_HAS_CYTHON_AUTOGRAD_NODE = False
_HAS_CYTHON_AUTOGRAD_GRAPH = False
_HAS_CYTHON_AUTOGRAD_ENGINE = False
_HAS_CYTHON_AUTOGRAD_FUNCTION = False
_HAS_CYTHON_AUTOGRAD_OPS = False
_HAS_CYTHON_FAST_OPS = False

# Non-core helper flags are informational only. They should not be used as
# package-level hard dependencies because their import surfaces differ from the
# direct consumer modules that load them.
try:
    from . import _dispatch as _dispatch_mod  # noqa: F401
    _HAS_CYTHON_DISPATCH = True
except ImportError:
    pass

try:
    from ._allocator import FastNpuAllocator  # noqa: F401
    _HAS_CYTHON_ALLOCATOR = True
except ImportError:
    pass

try:
    from . import _storage as _storage_mod  # noqa: F401
    _HAS_CYTHON_STORAGE = True
except ImportError:
    pass

try:
    from ._npu_ops import fast_binary_op  # noqa: F401
    _HAS_CYTHON_NPU_OPS = True
except ImportError:
    pass

try:
    from ._aclnn_ffi import (  # noqa: F401
        init as aclnn_ffi_init,
        create_tensor, destroy_tensor,
        create_scalar, destroy_scalar,
        create_int_array, destroy_int_array,
        destroy_executor, resolve_op, execute,
        binary_op_with_alpha, binary_op_no_alpha,
    )
    _HAS_CYTHON_ACLNN_FFI = True
except ImportError:
    pass

# Runtime-core availability flags: import only the leaf extension modules that
# do not themselves recurse back through candle._cython.__init__.
try:
    from ._tensor_impl import TensorImpl, _VersionCounterProxy  # noqa: F401
    _HAS_CYTHON_TENSOR_IMPL = True
except ImportError as exc:
    raise ImportError(
        "Failed to import candle._cython._tensor_impl. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    ) from exc

try:
    from ._dispatcher_core import cy_dispatch_full, cy_dispatch_with_keyset_fast  # noqa: F401
    _HAS_CYTHON_DISPATCHER_CORE = True
except ImportError as exc:
    raise ImportError(
        "Failed to import candle._cython._dispatcher_core. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    ) from exc

try:
    from ._device import FastDevice  # noqa: F401
    _HAS_CYTHON_DEVICE = True
except ImportError as exc:
    raise ImportError(
        "Failed to import candle._cython._device. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    ) from exc

try:
    from ._dtype import FastDType  # noqa: F401
    _HAS_CYTHON_DTYPE = True
except ImportError as exc:
    raise ImportError(
        "Failed to import candle._cython._dtype. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    ) from exc

try:
    from . import _fast_ops as _fast_ops_mod  # noqa: F401
    _HAS_CYTHON_FAST_OPS = True
except ImportError:
    pass

# Autograd-heavy modules are intentionally NOT eagerly imported here; they are
# imported directly by their Python wrappers to avoid circular bootstrapping.
