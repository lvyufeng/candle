"""Tests for nn.Module hook prepend and with_kwargs support."""
import candle as torch
import candle.nn as nn


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_forward_pre_hook_prepend():
    order = []
    m = SimpleModule()
    m.register_forward_pre_hook(lambda mod, inp: order.append("first"))
    m.register_forward_pre_hook(lambda mod, inp: order.append("prepended"), prepend=True)
    x = torch.randn(2, 4)
    m(x)
    assert order == ["prepended", "first"], f"Expected prepended hook first, got {order}"


def test_forward_hook_prepend():
    order = []
    m = SimpleModule()
    m.register_forward_hook(lambda mod, inp, out: order.append("first"))
    m.register_forward_hook(lambda mod, inp, out: order.append("prepended"), prepend=True)
    x = torch.randn(2, 4)
    m(x)
    assert order == ["prepended", "first"], f"Expected prepended hook first, got {order}"


def test_forward_pre_hook_with_kwargs():
    received = {}
    def hook(mod, args, kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return args, kwargs
    m = SimpleModule()
    m.register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(2, 4)
    m(x)
    assert "args" in received
    assert "kwargs" in received


def test_forward_pre_hook_with_kwargs_can_modify():
    def hook(mod, args, kwargs):
        new_input = torch.zeros_like(args[0])
        return (new_input,), kwargs
    m = SimpleModule()
    m.register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(2, 4)
    out = m(x)
    assert out is not None


def test_backward_compatible_hooks():
    called = [False]
    m = SimpleModule()
    m.register_forward_pre_hook(lambda mod, inp: called.__setitem__(0, True))
    x = torch.randn(2, 4)
    m(x)
    assert called[0]


def test_hook_removal():
    order = []
    m = SimpleModule()
    h1 = m.register_forward_pre_hook(lambda mod, inp: order.append("h1"))
    h2 = m.register_forward_pre_hook(lambda mod, inp: order.append("h2"))
    h1.remove()
    x = torch.randn(2, 4)
    m(x)
    assert order == ["h2"], f"Expected only h2, got {order}"


def test_forward_hook_with_kwargs():
    """Post-forward hooks with with_kwargs=True should receive (module, args, kwargs, output)."""
    received = {}
    def hook(mod, args, kwargs, output):
        received["args"] = args
        received["kwargs"] = kwargs
        received["output"] = output

    m = SimpleModule()
    m.register_forward_hook(hook, with_kwargs=True)
    x = torch.randn(2, 4)
    m(x)
    assert "args" in received
    assert "kwargs" in received
    assert "output" in received
