from __future__ import annotations

import json
import numpy as np
import aiger_bv as BV
import aiger_coins as C
import aiger_discrete
from bidict import bidict
from fractions import Fraction


class JaniIntegerVariable:
    """
    Integer variables in JANI
    """
    def __init__(self, name, lower_bound : int, upper_bound : int):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class JaniScope:
    def __init__(self):
        self._variables = {}
        self._aigvars = {}
        self._local = False

    def add_bounded_int_variable(self, name, lower_bound : int, upper_bound : int):
        """
        Add bounded integer variable to the scope
        :param name: name of the variable
        :param lower_bound: lower bound on the variable range (must be an int for now)
        :param upper_bound: upper bound on the variable range (must be an int for now)
        :return:
        """
        if name in self._variables:
            raise ValueError(f"Variable with name {name} already exists in scope.")
        self._variables[name] = JaniIntegerVariable(name, lower_bound, upper_bound)
        self._aigvars[name]  = BV.atom(int(np.ceil(np.log(upper_bound - lower_bound))), name)

    def make_local_scope_copy(self) -> JaniScope:
        if self._local:
            raise ValueError("You should not copy local scopes")
        local_scope = JaniScope()
        local_scope._variables = self._variables.copy()
        local_scope._aigvars = self._aigvars.copy()
        local_scope._local = True
        return local_scope

    def get_aig_variable(self, name : str):
        return self._aigvars[name]


class AutomatonContext:
    def __init__(self, aut_name, scope, locations):
        self._aut_name = aut_name
        self.scope = scope
        self.locations = locations
        self._actions = {"" : 0}
        self._actions_cnt = {"" : 0}
        self._distributions = {}

    def register_distribution(self, probs):
        """
        Register a distribution
        :param probs: The distribution as a list of probabilities
        :return: A pcirc that generates a binary-encoded output sel that corresponds to sampling from the distribution.
        """
        dist = tuple(probs)
        if dist not in self._distributions:
            name = f"{self._aut_name}_c{len(self._distributions)}"
            atom = BV.uatom(int(np.ceil(np.log(len(probs)))),name).with_output("sel")
            lookup = bidict({index: f'sel-{index}' for index in range(len(probs))})  # invertable dictionary.
            encoder = aiger_discrete.Encoding(decode=lookup.get, encode=lookup.inv.get)
            func = aiger_discrete.from_aigbv(
                atom.aigbv, input_encodings={name: encoder}
            )
            self._distributions[dist] = C.pcirc(func).randomize({name : { f"sel-{index}": prob for index, prob in enumerate(probs)}})
        return self._distributions[dist]

    def register_action(self, action : str):
        """
        :param action: The name of the action
        :return: The action index and the internal non-det index
        """
        if action not in self._actions:
            self._actions[action] = len(self._actions)
        self._actions_cnt[action] = self._actions_cnt.get(action, 0) + 1
        return self._actions[action], self._actions_cnt[action]




def _translate_constants(data : dict, scope : JaniScope):
    for _ in data:
        raise NotImplementedError("Constants are not supported yet.")

def _translate_variables(data : dict, scope : JaniScope):
    for v in data:
        if v["type"]["kind"] != "bounded":
            raise ValueError("Only bounded variables are supported")
        if v["type"]["base"] != "int":
            raise ValueError("Only integer-typed variables are supported")
        #TODO could be a constant expression, which is not yet supported
        lower_bound = v["type"]["lower-bound"]
        #TODO could be a constant expression, which is not yet supported
        upper_bound = v["type"]["upper-bound"]
        scope.add_bounded_int_variable(v["name"], lower_bound, upper_bound)



_BINARY_AEX_OPERATORS = frozenset(["+", "-"])
_BINARY_AEX_OP_MAP = {"+" : lambda x, y: x + y,
                      "-" : lambda x, y: x - y}
_BINARY_BOOL_OPERATORS = frozenset(["≤","≥"])
_BINARY_BOOL_OP_MAP = {"≤" : lambda x, y: x <= y ,
                       "≥" : lambda x, y: x >= y}



def _translate_expression(data : dict, scope : JaniScope):
    if isinstance(data, int):
        # TODO replace the constant two here.
        return BV.uatom(2, data)
    if isinstance(data, str):
        return scope.get_aig_variable(data)
    if "op" not in data:
        raise ValueError(f"{str(data)} is expected to have an operator")

    op = data["op"]
    if op in _BINARY_BOOL_OPERATORS:
        lexpr = _translate_expression(data["left"], scope)
        rexpr = _translate_expression(data["right"], scope)
        assert lexpr.size == rexpr.size
        return _BINARY_BOOL_OP_MAP[op](lexpr, rexpr)
    if op in _BINARY_AEX_OPERATORS:
        lexpr = _translate_expression(data["left"], scope)
        rexpr = _translate_expression(data["right"], scope)
        assert lexpr.size == rexpr.size
        return _BINARY_AEX_OP_MAP[op](lexpr, rexpr)
    else:
        raise NotImplementedError(f"{str(data)} not supported")


def _parse_prob(data):
    if isinstance(data, float):
        return Fraction(data)
    else:
        NotImplementedError(f"We only support constant probs given as floating point numbers, but got {data}")


def _selector(selector_bits, outputs, index = 0):
    if index == len(outputs) - 1:
        return outputs[index]
    return BV.ite(selector_bits == BV.uatom(1,index), outputs[index], _selector(selector_bits, outputs, index + 1))


def _translate_edges(data : dict, ctx ):
    max_internal_nondet_bits = int(np.ceil(np.log(len(data))))
    max_action_bits = int(np.ceil(np.log(len(data) + 1))) # +1 due to reserving space for silent act

    select_edge_expr = []
    edge_circuits =  []
    for edge in data:
        #TODO add location handling
        assert edge["location"] == "l"
        act, ndet = ctx.register_action(edge["action"])
        select_edge_expr.append(BV.uatom(max_action_bits, 'act') == BV.uatom(max_action_bits, act) & \
                           BV.uatom(max_internal_nondet_bits, 'nd') == BV.uatom(max_internal_nondet_bits, ndet)

        if "guard" in edge:
            guard_expr = _translate_expression(edge["guard"]["exp"], ctx.scope)
        else:
            #TODO add true
            pass
        if len(edge["destinations"]) == 1:
            pass
        else:
            probs = []
            for d in edge["destinations"]:
                probs.append(_parse_prob(d["probability"]["exp"]))
            prob_input = ctx.register_distribution(probs)
            #TODO consider what to do with the additional output of prob_input

        vars_written_to = set()
        for d in edge["destinations"]:
            for a in d["assignments"]:
                vars_written_to.add(a["ref"])
        destinations = []
        for index, d in enumerate(edge["destinations"]):
            assert d["location"] == "l"
            updates = {}
            #TODO add location handling
            for assignment in d["assignments"]:
                var_primed = assignment["ref"] + "'" + "-" + str(index)
                updates[var_primed] = _translate_expression(assignment["value"], ctx.scope).with_output(var_primed)
            for var in vars_written_to:
                var_primed = var + "'" + "-" + str(index)
                if var_primed not in updates:
                    updates[var_primed] = ctx.scope.get_aig_variable(var).with_output(var_primed)

            # CONCAT:
            updates = list(updates.values())
            update = updates[0].aigbv
            for up in updates[1:]:
                update = update | up.aigbv
            destinations.append(update)

        edge_circuit = destinations[0]
        for d in destinations[1:]:
            edge_circuit = edge_circuit | d

        vars_written_to = list(vars_written_to)

        if len(destinations) > 1:
            #TODO replace 2
            print("Construct selector")
            selector = _selector(BV.uatom(int(np.ceil(np.log(len(probs)))),'sel'),
                                   [BV.uatom(2, vars_written_to[0] + "'" + "-" + str(index)) for index in range(len(destinations))]).with_output(vars_written_to[0]).aigbv
            for var in vars_written_to[1:]:
                #TODO replace 2
                selector = selector | _selector(BV.uatom(int(np.ceil(np.log(len(probs)))),'sel'),
                                   [BV.uatom(2, var + "'" + "-" + str(index)) for index in range(len(destinations))]).with_output(var).aigbv
            edge_circuit = edge_circuit >> selector
            edge_circuit = prob_input >> edge_circuit

        edge_circuits.append(edge_circuit)


def _create_automaton_context(data : dict, scope : JaniScope):
    locations = {}
    for l in data["locations"]:
        locations[l["name"]] = False

    for l in data["initial-locations"]:
        if l not in locations:
            raise ValueError("Location {l} is unknown")
        locations[l] = True


    return AutomatonContext(data["name"], scope, locations)


def _translate_automaton(data : dict, scope : JaniScope):
    ctx = _create_automaton_context(data, scope)
    _translate_variables(data["variables"], scope)
    _translate_edges(data["edges"], ctx)


def translate_file(path):
    with open(path) as f:
        jani_enc = json.load(f)
    translate_jani(jani_enc)


def translate_jani(data : json):
    global_scope = JaniScope()
    _translate_constants(data["constants"], global_scope)
    _translate_variables(data["variables"], global_scope)
    for aut in data["automata"]:
        _translate_automaton(aut, global_scope.make_local_scope_copy())




