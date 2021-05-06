from pytest import approx

import aiger_bv as BV
from aiger_coins import infer


import aiger_jani
import aiger_jani.translation


def test_smoke():
    pass


def test_minimdp():
    circ = aiger_jani.translation.translate_file("tests/minimdp.jani")
    assert circ.outputs == {'main-x', 'main-y'}

    circ <<= BV.source(1, 0, 'edge', False)
    circ >>= BV.sink(2, ['main-y'])
    circ >>= (BV.uatom(2, 'main-x') == 3).aigbv

    assert infer.prob(circ.unroll(1)) == approx(0)
    assert infer.prob(circ.unroll(2, only_last_outputs=True)) == approx(1/4)
    assert infer.prob(circ.unroll(3, only_last_outputs=True)) == approx(1/3)
