from __future__ import annotations

import math
from typing import Sequence

import operator as op
from functools import reduce

import aiger_bv as BV


BVExpr = BV.UnsignedBVExpr
Number = "float | int"


def masked_outputs(outputs, key_name: str):
    size = min_bits(len(outputs))
    key = BV.uatom(size, key_name)

    for idx, output in enumerate(outputs):
        mask = (key == idx).repeat(output.size)
        yield output & mask


def mux(outputs: Sequence[BVExpr], *, key_name: str = "mux") -> BVExpr:
    """Create multiplexer to select between outputs."""
    return reduce(op.or_, masked_outputs(outputs, key_name))


def at_most_one(exprs) -> BVExpr:
    """Returns an expression that is true if at most one expression is true"""
    pairs = []
    for index in range(len(exprs)):
        for e2 in exprs[index+1:]:
            pairs.append(~(exprs[index] & e2))
    return reduce(op.and_, pairs)


def min_bits(x: Number) -> int:
    """Returns minimum number of bits to represent x."""
    return int(math.ceil(math.log(x+1, 2)))


def empty_circuit() -> BV.AIGBV:
    """Returns a circuit without input or output"""
    # TODO this should be easier
    return BV.source(wordlen=1, value=1, name="dummy", signed=False) \
           >> BV.sink(1, inputs=['dummy'])


def atom(n: Number, name: str) -> BVExpr:
    return BV.uatom(min_bits(n), name)


def par_compose(seq: BV.AIGBV) -> BV.AIGBV:
    """Takes parallel composition of a iterable of AIGBVs."""
    return reduce(op.or_, seq)


def min_op(lhs, rhs):
    return BV.ite(lhs < rhs, lhs, rhs)


def max_op(lhs, rhs):
    return BV.ite(lhs > rhs, lhs, rhs)


__all__ = ['mux', 'min_bits', 'par_compose', 'atom', 'min_op', 'max_op']
