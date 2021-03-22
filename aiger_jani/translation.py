from __future__ import annotations

import json
import numpy as np
import aiger_bv


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
        self._aigvars[name]  = aiger_bv.atom(int(np.ceil(np.log(upper_bound - lower_bound))), name)

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



_BINARY_OPERATORS = frozenset(["≤","≥"])


def _translate_expression(data : dict, scope : JaniScope):
    if isinstance(data, int):
        return aiger_bv.SignedBVExpr(data)
    if isinstance(data, str):
        return scope.get_aig_variable(data)
    if "op" not in data:
        raise ValueError(f"{str(data)} is expected to have an operator")

    op = data["op"]
    if op in _BINARY_OPERATORS:
        lexpr = _translate_expression(data["left"], scope)
        rexpr = _translate_expression(data["right"], scope)
    else:
        raise NotImplementedError(f"{str(data)} not supported")


def _translate_edges(data : dict, scope : JaniScope):
    for edge in data:
        guard_expr = _translate_expression(edge["guard"]["exp"], scope)


def _translate_automaton(data : dict, scope : JaniScope):
    _translate_variables(data["variables"], scope)
    _translate_edges(data["edges"], scope)


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




