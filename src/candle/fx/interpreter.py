"""candle.fx.interpreter -- Interpreter for per-node graph execution.

The :class:`Interpreter` walks a :class:`~candle.fx.graph.Graph` node by node,
evaluating each operation against concrete values.  Subclasses can override
individual opcode methods (e.g. :meth:`call_function`) to inject custom
behaviour such as logging, profiling, or shape propagation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple, TYPE_CHECKING

from candle.fx.node import Node, _map_arg

if TYPE_CHECKING:
    from candle.fx.graph import Graph
    from candle.nn.module import Module


class Interpreter:
    """Execute an FX :class:`Graph` node-by-node with concrete values.

    Parameters
    ----------
    module : Module
        The module (typically a :class:`GraphModule`) whose graph is executed.
    garbage_collect_values : bool
        If *True*, intermediate values are removed from :attr:`env` as soon as
        all their consumers have been evaluated.  This reduces peak memory.
    graph : Graph, optional
        The graph to interpret.  If *None*, obtained from ``module._graph``
        or ``module.graph``.
    """

    def __init__(
        self,
        module: "Module",
        garbage_collect_values: bool = True,
        graph: Optional["Graph"] = None,
    ) -> None:
        self.module = module
        self.garbage_collect_values = garbage_collect_values

        if graph is not None:
            self.graph = graph
        elif hasattr(module, "_graph"):
            self.graph = module._graph
        elif hasattr(module, "graph"):
            self.graph = module.graph
        else:
            raise ValueError(
                "No graph found. Pass a graph explicitly or use a GraphModule."
            )

        self.env: Dict[Node, Any] = {}
        self.args_iter: Iterator[Any] = iter(())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, *args: Any, initial_env: Optional[Dict[Node, Any]] = None) -> Any:
        """Execute the graph with the given positional arguments.

        Parameters
        ----------
        *args
            Concrete values consumed by ``placeholder`` nodes, in order.
        initial_env : dict, optional
            Pre-populated environment mapping nodes to values.

        Returns
        -------
        Any
            The value produced by the ``output`` node.
        """
        self.env = dict(initial_env) if initial_env else {}
        self.args_iter = iter(args)

        # Pre-compute the number of remaining users for GC.
        if self.garbage_collect_values:
            # Count how many nodes consume each node.
            node_to_last_use: Dict[Node, int] = {}
            nodes_list = list(self.graph.nodes)
            for idx, node in enumerate(nodes_list):
                for input_node in node.all_input_nodes:
                    node_to_last_use[input_node] = idx

        for idx, node in enumerate(list(self.graph.nodes)):
            result = self.run_node(node)
            self.env[node] = result

            if node.op == "output":
                return result

            # Garbage collect values no longer needed.
            if self.garbage_collect_values:
                for input_node in node.all_input_nodes:
                    if (
                        input_node in node_to_last_use
                        and node_to_last_use[input_node] <= idx
                        and input_node in self.env
                    ):
                        del self.env[input_node]

        return None

    # ------------------------------------------------------------------
    # Node dispatch
    # ------------------------------------------------------------------

    def run_node(self, n: Node) -> Any:
        """Evaluate a single node by dispatching to the appropriate handler.

        Parameters
        ----------
        n : Node
            The node to evaluate.

        Returns
        -------
        Any
            The result of evaluating the node.
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        op = n.op
        if op == "placeholder":
            return self.placeholder(n.target, args, kwargs)
        if op == "get_attr":
            return self.get_attr(n.target, args, kwargs)
        if op == "call_function":
            return self.call_function(n.target, args, kwargs)
        if op == "call_method":
            return self.call_method(n.target, args, kwargs)
        if op == "call_module":
            return self.call_module(n.target, args, kwargs)
        if op == "output":
            return self.output(n.target, args, kwargs)
        raise RuntimeError(f"Unknown opcode: {op!r}")

    # ------------------------------------------------------------------
    # Per-opcode handlers (override in subclasses)
    # ------------------------------------------------------------------

    def placeholder(self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Handle a ``placeholder`` node: consume the next positional argument."""
        return next(self.args_iter)

    def get_attr(self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Handle a ``get_attr`` node: fetch an attribute from the module."""
        return self.fetch_attr(str(target))

    def call_function(self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Handle a ``call_function`` node: call a free function."""
        return target(*args, **kwargs)

    def call_method(self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Handle a ``call_method`` node: call a method on the first argument."""
        return getattr(args[0], str(target))(*args[1:], **kwargs)

    def call_module(self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Handle a ``call_module`` node: invoke a submodule."""
        submod = self.fetch_attr(str(target))
        return submod(*args, **kwargs)

    def output(self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Handle an ``output`` node: return the output value."""
        return args[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def fetch_args_kwargs_from_env(self, n: Node) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Resolve Node references in *n*'s args/kwargs to concrete values.

        Parameters
        ----------
        n : Node
            The node whose arguments should be resolved.

        Returns
        -------
        tuple[tuple, dict]
            ``(args, kwargs)`` with all Node references replaced by their
            values from :attr:`env`.
        """
        def _load(node: Node) -> Any:
            return self.env[node]

        args = _map_arg(n.args, _load)
        kwargs = _map_arg(n.kwargs, _load)
        # Ensure args is a tuple and kwargs is a dict.
        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(kwargs, dict):
            kwargs = {}
        return args, kwargs

    def fetch_attr(self, target: str) -> Any:
        """Navigate the module hierarchy to fetch an attribute.

        Parameters
        ----------
        target : str
            Dot-separated attribute path (e.g. ``"layer.weight"``).

        Returns
        -------
        Any
            The attribute value.
        """
        atoms = target.split(".")
        attr = self.module
        for atom in atoms:
            if not hasattr(attr, atom):
                raise AttributeError(
                    f"Module {type(attr).__name__!r} has no attribute {atom!r}"
                )
            attr = getattr(attr, atom)
        return attr
