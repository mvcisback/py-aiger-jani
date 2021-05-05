import aiger_bv as BV

import aiger_jani
import aiger_jani.translation


def test_smoke():
    pass


def test_minimdp():
    circ = aiger_jani.translation.translate_file("tests/minimdp.jani")
    assert circ.outputs == {'main-x', 'main-y'}
