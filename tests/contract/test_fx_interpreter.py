"""Tests for candle.fx.Interpreter — per-node graph executor."""
import operator
import candle as torch
from candle import nn


def test_interpreter_import():
    from candle.fx import Interpreter
    assert Interpreter is not None


def test_interpreter_identity():
    from candle.fx import Graph, GraphModule, Interpreter
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(nn.Module(), g)
    interp = Interpreter(gm)
    t = torch.tensor([1.0, 2.0])
    result = interp.run(t)
    assert result is t


def test_interpreter_add():
    from candle.fx import Graph, GraphModule, Interpreter
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    gm = GraphModule(nn.Module(), g)
    interp = Interpreter(gm)
    result = interp.run(3, 4)
    assert result == 7


def test_interpreter_call_module():
    from candle.fx import Graph, GraphModule, Interpreter
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 3, bias=False)
        def forward(self, x):
            return self.linear(x)
    root = Root()
    g = Graph()
    x = g.placeholder("x")
    linear = g.call_module("linear", (x,))
    out = g.output(linear)
    gm = GraphModule(root, g)
    interp = Interpreter(gm)
    t = torch.randn(1, 2)
    result = interp.run(t)
    assert result.shape == (1, 3)


def test_interpreter_get_attr():
    from candle.fx import Graph, GraphModule, Interpreter
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.val = nn.Parameter(torch.tensor([42.0]))
        def forward(self, x):
            return x
    root = Root()
    g = Graph()
    v = g.get_attr("val")
    out = g.output(v)
    gm = GraphModule(root, g)
    interp = Interpreter(gm)
    result = interp.run()
    assert float(result[0]) == 42.0


def test_interpreter_call_method():
    from candle.fx import Graph, GraphModule, Interpreter
    g = Graph()
    x = g.placeholder("x")
    view = g.call_method("reshape", (x, 2, 3))
    out = g.output(view)
    gm = GraphModule(nn.Module(), g)
    interp = Interpreter(gm)
    t = torch.randn(6)
    result = interp.run(t)
    assert result.shape == (2, 3)


def test_interpreter_custom_call_function():
    """Subclass can override call_function for custom behavior."""
    from candle.fx import Graph, GraphModule, Interpreter
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    gm = GraphModule(nn.Module(), g)
    call_log = []
    class LoggingInterpreter(Interpreter):
        def call_function(self, target, args, kwargs):
            call_log.append(target.__name__)
            return super().call_function(target, args, kwargs)
    interp = LoggingInterpreter(gm)
    result = interp.run(10, 20)
    assert result == 30
    assert call_log == ["add"]


def test_interpreter_multiple_ops():
    from candle.fx import Graph, GraphModule, Interpreter
    g = Graph()
    x = g.placeholder("x")
    neg = g.call_function(operator.neg, (x,))
    double = g.call_function(operator.mul, (neg, 2))
    out = g.output(double)
    gm = GraphModule(nn.Module(), g)
    interp = Interpreter(gm)
    result = interp.run(5)
    assert result == -10
