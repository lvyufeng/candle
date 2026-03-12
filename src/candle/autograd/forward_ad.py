import threading
from collections import namedtuple


_UnpackedDualTensor = namedtuple("_UnpackedDualTensor", ["primal", "tangent"])


class UnpackedDualTensor(_UnpackedDualTensor):
    pass


_STATE = threading.local()
_JVP_RULES = {}


def _level_stack():
    stack = getattr(_STATE, "levels", None)
    if stack is None:
        stack = []
        _STATE.levels = stack
    return stack


def _current_level():
    stack = _level_stack()
    if not stack:
        return -1
    return stack[-1]


def register_jvp(op_name, fn):
    _JVP_RULES[op_name] = fn


def get_jvp(op_name):
    return _JVP_RULES.get(op_name)


def enter_dual_level():
    stack = _level_stack()
    level = stack[-1] + 1 if stack else 0
    stack.append(level)
    return level


def exit_dual_level(*, level=None):
    stack = _level_stack()
    if not stack:
        raise RuntimeError(
            "Trying to exit a forward AD level but no level is currently active."
        )
    current = stack[-1]
    target = current if level is None else level
    if target != current:
        raise RuntimeError(
            "Trying to exit a forward AD level that was not the last one that was created."
        )
    stack.pop()


class dual_level:
    def __enter__(self):
        self.level = enter_dual_level()
        return self.level

    def __exit__(self, exc_type, exc, tb):
        exit_dual_level(level=self.level)
        return False


def _validate_tangent(tensor, tangent):
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(
            f"Expected primal to be floating point or complex, but got: {tensor.dtype}"
        )
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(
            f"Expected tangent to be floating point or complex, but got: {tangent.dtype}"
        )
    if tensor.shape != tangent.shape:
        raise RuntimeError(
            f"Expected tangent to have the same shape as primal, but got: {tangent.shape}"
        )
    if tensor.dtype != tangent.dtype:
        raise RuntimeError(
            f"Expected tangent to have the same dtype as primal, but got: {tangent.dtype}"
        )


def make_dual(tensor, tangent, *, level=None):
    if level is None:
        level = _current_level()
    if level < 0:
        raise RuntimeError(
            "Trying to create a dual Tensor for forward AD but no level exists, make sure to enter_dual_level() first."
        )
    _validate_tangent(tensor, tangent)
    tensor._fw_set(level, tangent)
    return tensor


def unpack_dual(tensor, *, level=None):
    if level is None:
        level = _current_level()
    if level < 0:
        return UnpackedDualTensor(tensor, None)
    return UnpackedDualTensor(tensor, tensor._fw_get(level))


__all__ = [
    "UnpackedDualTensor",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
    "dual_level",
    "register_jvp",
    "get_jvp",
]


register_jvp("add", lambda x, y, *, _tangents: _tangents[0] + _tangents[1])
