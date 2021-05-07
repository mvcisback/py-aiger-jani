from pytest import approx

import aiger_bv as BV
from aiger_coins import infer


import aiger_jani
import aiger_jani.translation


def test_minimdp():
    x, y = BV.uatom(2, 'main-x'), BV.uatom(2, 'main-y')
    circ = aiger_jani.translation.translate_file("tests/minimdp.jani")
    assert circ.outputs == {'main-x', 'main-y'}

    # Fix edge and check probability of ending on x=3 given valid run.
    query = circ << BV.source(1, 0, 'edge', False)
    query >>= BV.sink(2, ['main-y'])
    query >>= (BV.uatom(2, 'main-x') == 3).aigbv

    assert infer.prob(query.unroll(1, only_last_outputs=True)) == approx(0)
    assert infer.prob(query.unroll(2, only_last_outputs=True)) == approx(1/4)
    assert infer.prob(query.unroll(3, only_last_outputs=True)) == approx(1/3)

    # Randomize edge and check probability of ending on x=y given valid run.
    query = circ.randomize({'edge': {0: 0.5, 1: 0.5}})
    query >>= (x == y).aigbv

    assert infer.prob(query.unroll(1, only_last_outputs=True)) == approx(1/4)
    assert infer.prob(query.unroll(2, only_last_outputs=True)) == approx(1/4)
    assert infer.prob(query.unroll(3, only_last_outputs=True)) == approx(13/38)
