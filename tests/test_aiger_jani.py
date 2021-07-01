from pytest import approx

import aiger_bv as BV
from aiger_coins import infer

from aiger_jani import translate_file


def test_minimdp():
    x, y = BV.atom(4, 'main-x'), BV.atom(4, 'main-y')
    circ = translate_file("tests/minimdp.jani")
    assert circ.outputs == {'main-x', 'main-y'}

    # Fix edge and check probability of ending on x=3 given valid run.
    # TODO this currently only works with one.
    query = circ << BV.source(2, 0, 'edge', False)
    query >>= BV.sink(4, ['main-y'])
    query >>= (BV.uatom(4, 'main-x') == 3).aigbv

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


cardinal = {"north":
            {'north': True, 'south': False, 'east': False, 'west': False},
            "south":
            {'north': False, 'south': True, 'east': False, 'west': False},
            "west":
            {'north': False, 'south': False, 'east': False, 'west': True},
            "east":
            {'north': False, 'south': False, 'east': True, 'west': False}}


def test_plaingrid():
    circ = translate_file("tests/jani_files/grid.jani",
                          action_deterministic=True)
    all_labels = {"red", "station", "exit"}

    def get_next_loc(step):
        # Result holds path from simulation
        assert result[step][0]["_valid_input"]
        return result[step][0]['global-x'], result[step][0]['global-y']

    def get_current_labels(step):
        # Results holds path from simulation
        return set([x for x in all_labels if result[step][0][x]])

    assert circ.inputs == {"north", "south", "west", "east"}
    assert circ.outputs == {'global-x', 'global-y', 'red',
                            'station', 'exit', '_valid_input'}

    result = circ.simulate([
        cardinal["north"],
        cardinal["north"],
        cardinal["south"],
        cardinal["south"],
        cardinal["south"],
        cardinal["south"],
        cardinal["east"],
        cardinal["east"],
        cardinal["east"],
        cardinal["east"],
        cardinal["east"],
        cardinal["north"],
        cardinal["north"],
        cardinal["north"],
        cardinal["north"],
    ])

    # Labels (before executing action)
    assert get_current_labels(0) == {"station"}
    # Next state
    assert get_next_loc(0) == (0, 0)

    # Labels (before executing action)
    assert get_current_labels(1) == {"station"}
    # Next state
    assert get_next_loc(1) == (0, 0)

    # Labels (before executing action)
    assert get_current_labels(2) == {"station"}
    # Next state
    assert get_next_loc(2) == (0, 1)

    # Labels (before executing action)
    assert get_current_labels(3) == set()
    # Next state
    assert get_next_loc(3) == (0, 2)

    # Labels (before executing action)
    assert get_current_labels(4) == set()
    # Next state
    assert get_next_loc(4) == (0, 3)

    # Labels (before executing action)
    assert get_current_labels(5) == set()
    # Next state
    assert get_next_loc(5) == (0, 4)

    # Labels (before executing action)
    assert get_current_labels(6) == set()
    # Next state
    assert get_next_loc(6) == (1, 4)

    # Labels (before executing action)
    assert get_current_labels(7) == {"station"}
    # Next state
    assert get_next_loc(7) == (2, 4)

    # Labels (before executing action)
    assert get_current_labels(8) == set()
    # Next state
    assert get_next_loc(8) == (3, 4)

    # Labels (before executing action)
    assert get_current_labels(9) == set()
    # Next state
    assert get_next_loc(9) == (4, 4)

    # Labels (before executing action)
    assert get_current_labels(10) == {"exit"}
    # Next state
    assert get_next_loc(10) == (4, 4)

    # Labels (before executing action)
    assert get_current_labels(11) == {"exit"}
    # Next state
    assert get_next_loc(11) == (4, 3)

    # Labels (before executing action)
    assert get_current_labels(12) == set()
    # Next state
    assert get_next_loc(12) == (4, 2)

    # Labels (before executing action)
    assert get_current_labels(13) == set()
    # Next state
    assert get_next_loc(13) == (4, 1)

    # Labels (before executing action)
    assert get_current_labels(14) == {"red"}
    # Next state
    assert get_next_loc(14) == (4, 0)

    two_actions = {'north': True, 'south': True, 'east': False, 'west': False}
    result = circ.simulate([two_actions])
    assert not result[0][0]["_valid_input"]

    no_actions = {'north': False, 'south': False, 'east': False, 'west': False}
    result = circ.simulate([no_actions])
    assert not result[0][0]["_valid_input"]


def test_plaingrid_extended():
    circ = translate_file("tests/jani_files/grid_two.jani",
                          action_deterministic=True)
    assert circ.outputs == {'global-x', 'global-y', 'red',
                            'station', 'exit', '_valid_input'}


def test_boundedgrid():
    circ = translate_file("tests/jani_files/bounded_grid.jani",
                          action_deterministic=True)
    assert circ.inputs == {"north", "south", "west", "east"}
    assert circ.outputs == {'global-x', 'global-y', 'red',
                            'station', 'exit', '_valid_input'}


def test_obstacleflat():
    circ = translate_file("tests/jani_files/obstacle-flat-nonslip.jani",
                          action_deterministic=True)
    assert circ.outputs == {'global-ax', 'global-ay',
                            "global-start", "goal", "notbad", "traps",
                            '_valid_input'}
