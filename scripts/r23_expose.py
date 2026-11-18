"""Round 23 part LW: make the numeric constants inside a catalog strategy's function into keyword parameters.

expose(e) rewrites e.fn's source: every int / float literal in the body becomes a keyword argument __c<k> with the
literal as default, except 0, 1 (and their float forms), values below 1e-6 in size, exponents (x ** 2), indices and
slices, and keyword arguments that are not tunable (axis, ddof, dtype, decimals). verified(e, fn, bars) checks that
the rewritten function gives exactly the original output at its defaults; if not, the original is kept.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd

SKIP_KW = {"axis", "ddof", "dtype", "decimals", "copy", "inplace"}


class _Rewriter(ast.NodeTransformer):
    def __init__(self):
        self.consts = []
        self._block = 0

    def _skip(self, node):
        self._block += 1
        self.generic_visit(node)
        self._block -= 1
        return node

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        self._block += 1
        node.slice = self.visit(node.slice)
        self._block -= 1
        return node

    def visit_BinOp(self, node):
        node.left = self.visit(node.left)
        if isinstance(node.op, ast.Pow):
            self._block += 1
            node.right = self.visit(node.right)
            self._block -= 1
        else:
            node.right = self.visit(node.right)
        return node

    def visit_keyword(self, node):
        if node.arg in SKIP_KW:
            return self._skip(node)
        node.value = self.visit(node.value)
        return node

    def visit_FunctionDef(self, node):                                     # nested defs keep their constants
        return self._skip(node) if self._outer_done else self._visit_outer(node)

    def visit_Lambda(self, node):
        return self._skip(node)

    def visit_Constant(self, node):
        v = node.value
        if self._block or isinstance(v, bool) or not isinstance(v, (int, float)):
            return node
        if v in (0, 1) or abs(v) < 1e-6:
            return node
        name = f"__c{len(self.consts)}"
        self.consts.append((name, v))
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

    _outer_done = False

    def _visit_outer(self, node):
        self._outer_done = True
        node.decorator_list = []
        node.body = [self.visit(b) for b in node.body]
        return node


def expose(e):
    """Returns (fn, consts) where consts is a list of (name, default). On any failure returns (e.fn, [])."""
    try:
        src = textwrap.dedent(inspect.getsource(e.fn))
        tree = ast.parse(src)
        fdef = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        rw = _Rewriter()
        fdef = rw._visit_outer(fdef)
        if not rw.consts:
            return e.fn, []
        for name, v in rw.consts:
            fdef.args.kwonlyargs.append(ast.arg(arg=name))
            fdef.args.kw_defaults.append(ast.Constant(value=v))
        mod = ast.Module(body=[fdef], type_ignores=[])
        ast.fix_missing_locations(mod)
        g = dict(e.fn.__globals__)
        exec(compile(mod, f"<exposed {e.id}>", "exec"), g)
        return g[fdef.name], rw.consts
    except Exception:                                                      # noqa: BLE001
        return e.fn, []


def _same(a, b):
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            return False
        for c in a.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            if x.dtype.kind in "fc" or y.dtype.kind in "fc":
                if not np.allclose(np.asarray(x, float), np.asarray(y, float), equal_nan=True, rtol=0, atol=0):
                    return False
            elif not (x == y).all():
                return False
        return True
    return False


def verified(e, fn, bars):
    try:
        a = e.fn(bars.copy(), **dict(e.params))
        b = fn(bars.copy(), **dict(e.params))
        return _same(a, b)
    except Exception:                                                      # noqa: BLE001
        return False
