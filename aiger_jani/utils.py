from __future__ import annotations

import math
from typing import Sequence, Iterable

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


def min_bits(x: Number) -> int:
    """Returns minimum number of bits to represent x."""
    return int(math.ceil(math.log(x)))


def atom(n: Number, name: str) -> BVExpr:
    return BV.uatom(min_bits(n), name)


def par_compose(seq: BV.AIGBV) -> BV.AIGBV:
    """Takes parallel composition of a iterable of AIGBVs."""
    return reduce(op.or_, seq)


__all__ = ['mux', 'min_bits', 'par_compose', 'atom']
