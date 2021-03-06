from pytest import approx

import aiger_bv as BV
from aiger_coins import infer

from aiger_jani import translate_file


def test_minimdp():
    x, y = BV.uatom(2, 'main-x'), BV.uatom(2, 'main-y')
    circ = translate_file("tests/minimdp.jani")
    assert circ.outputs == {'main-x', 'main-y'}

    # Fix edge and check probability of ending on x=3 given valid run.
    # TODO this currently only works with one.
    query = circ << BV.source(2, 0, 'edge', False)
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


def test_die():
    circ = translate_file("tests/jani_files/die.jani")
    assert circ.outputs == {'global-s', 'global-d'}


def test_plaingrid():
    circ = translate_file("tests/jani_files/grid.jani")
    assert circ.outputs == {'global-x', 'global-y', 'red', 'station', 'exit'}


def test_obstacleflat():
    circ = translate_file("tests/jani_files/obstacle-flat-nonslip.jani")
    assert circ.outputs == {'global-ax', 'global-ay', "global-start"}
