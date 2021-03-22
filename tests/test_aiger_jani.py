import aiger_jani
import aiger_jani.translation

def test_smoke():
    pass

def test_minimdp():
    aiger_jani.translation.translate_file("tests/minimdp.jani")