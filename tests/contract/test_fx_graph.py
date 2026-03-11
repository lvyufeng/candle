"""Tests for candle.fx.Graph — the container for FX graph IR nodes."""
import operator


def test_graph_import():
    from candle.fx import Graph
    assert Graph is not None


def test_graph_create_placeholder():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    assert x.op == "placeholder"
    assert x.target == "x"
    assert x.name == "x"


def test_graph_create_call_function():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    assert add.op == "call_function"
    assert add.target is operator.add
    assert add.args == (x, y)


def test_graph_create_call_method():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    view = g.call_method("view", (x, 2, 3))
    assert view.op == "call_method"
    assert view.target == "view"
    assert view.args == (x, 2, 3)


def test_graph_create_call_module():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    out = g.call_module("linear", (x,))
    assert out.op == "call_module"
    assert out.target == "linear"


def test_graph_create_get_attr():
    from candle.fx import Graph
    g = Graph()
    w = g.get_attr("linear.weight")
    assert w.op == "get_attr"
    assert w.target == "linear.weight"


def test_graph_create_output():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    assert out.op == "output"
    assert out.target == "output"
    assert out.args == (x,)


def test_graph_name_uniquification():
    from candle.fx import Graph
    g = Graph()
    a = g.placeholder("x")
    b = g.placeholder("x")
    c = g.placeholder("x")
    assert a.name == "x"
    assert b.name == "x_1"
    assert c.name == "x_2"


def test_graph_name_from_call_function():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    n = g.call_function(operator.add, (x, x))
    assert n.name == "add"


def test_graph_nodes_iteration():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    nodes = list(g.nodes)
    assert nodes == [x, y, add, out]


def test_graph_nodes_len():
    from candle.fx import Graph
    g = Graph()
    assert len(g.nodes) == 0
    g.placeholder("x")
    assert len(g.nodes) == 1
    g.placeholder("y")
    assert len(g.nodes) == 2


def test_graph_erase_node():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    out = g.output(x)
    g.erase_node(y)
    assert y not in list(g.nodes)
    assert list(g.nodes) == [x, out]


def test_graph_erase_node_with_users_raises():
    from candle.fx import Graph
    import pytest
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    with pytest.raises(RuntimeError, match="user"):
        g.erase_node(x)


def test_graph_inserting_before():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    with g.inserting_before(out):
        y = g.call_function(operator.neg, (x,))
    nodes = list(g.nodes)
    assert nodes == [x, y, out]


def test_graph_inserting_after():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    with g.inserting_after(x):
        y = g.call_function(operator.neg, (x,))
    nodes = list(g.nodes)
    assert nodes == [x, y, out]


def test_graph_eliminate_dead_code():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    dead = g.call_function(operator.neg, (x,))
    out = g.output(x)
    removed = g.eliminate_dead_code()
    assert removed is True
    assert dead not in list(g.nodes)
    assert list(g.nodes) == [x, out]


def test_graph_str():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    s = str(g)
    assert "placeholder" in s
    assert "output" in s


def test_graph_multiple_outputs():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    out = g.output((x, y))
    assert out.args == ((x, y),)


def test_graph_node_copy():
    from candle.fx import Graph
    g1 = Graph()
    x1 = g1.placeholder("x")
    g2 = Graph()
    val_map = {}
    x2 = g2.node_copy(x1, lambda n: val_map[n])
    val_map[x1] = x2
    assert x2.op == "placeholder"
    assert x2.name == "x"
    assert x2.graph is g2


def test_graph_nodes_reversed():
    from candle.fx import Graph
    g = Graph()
    a = g.placeholder("a")
    b = g.placeholder("b")
    c = g.placeholder("c")
    assert list(reversed(g.nodes)) == [c, b, a]