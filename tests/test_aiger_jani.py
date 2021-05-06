import aiger_bv as BV
from aiger_coins import infer

import attr
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
    
    # TODO: Fix py-aiger-coins to remove this.
    circ = attr.evolve(circ, coin_biases=tuple(map(float, circ.coin_biases)))
    assert infer.prob(circ.unroll(1)) == 0
    assert infer.prob(circ.unroll(2, only_last_outputs=True)) == 1/4
    # TODO: probability traversal.
    # assert infer.prob(circ.unroll(3, only_last_outputs=True)) == 0

