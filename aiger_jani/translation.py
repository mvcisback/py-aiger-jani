from __future__ import annotations

import json
import math
import operator as ops
from functools import reduce
from typing import Any, Sequence

import attr
import aiger_bv as BV
import aiger_coins as C
import aiger_discrete as D
from bidict import bidict
from fractions import Fraction

from aiger_jani.utils import atom, mux, min_bits, par_compose


# Lists all methods in public API.
__all__ = []


BVExpr = BV.UnsignedBVExpr


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniIntegerVariable:
    name: str
    lower_bound: int  # TODO: maybe tuple instead?
    upper_bound: int
    is_local: bool
    # TODO: Include initial value here.


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniScope:
    _local: bool = False
    _aigvars: dict[str, BVExpr] = attr.ib(factory=dict)
    _variables: dict[str, JaniIntegerVariable] = attr.ib(factory=dict)

    def add_bounded_int_variable(
            self, 
            name: str, 
            lower_bound : int,
            upper_bound : int) -> None:
        """
        Add bounded integer variable to the scope
        :param name: name of the variable.
        :param lower_bound: lower bound on the variable range (must be
           an int for now).
        :param upper_bound: upper bound on the variable range (must be
           an int for now).
        :return:n
        """
        if name in self._variables:
            raise ValueError(
                f"Variable with name {name} already exists in scope."
            )

        self._variables[name] = JaniIntegerVariable(
            name, lower_bound, upper_bound, self._local)
        self._aigvars[name] = BV.atom(
            wordlen=min_bits(upper_bound - lower_bound), 
            val=name
        )

    def make_local_scope_copy(self) -> JaniScope:
        if self._local:
            raise ValueError("You should not copy local scopes")

        return JaniScope(
            local=True,
            aigvars=self._aigvars.copy(),
            variables=self._variables.copy(),
        )

    def get_aig_variable(self, name : str) -> BV.UnsignedBVExpr:
        return self._aigvars[name]

    def is_local_var(self, name : str) -> bool:
        return self._variables[name].is_local

    @property
    def variables(self):
        # TODO: should this be frozenset?
        return list(self._variables.values())


Probs = Sequence[float]


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class AutomatonContext:
    _aut_name: str
    scope: JaniScope
    locations: dict[Any, bool]  # TODO: tighen signature.
    _actions: dict[str, int] = attr.ib(factory=lambda: {"": 0})
    _actions_cnt: dict[str, int] = attr.ib(factory=lambda: {"": 0})
    _distributions: dict[probs, C.PCirc] = attr.ib(factory=dict)

    def register_distribution(self, probs):
        """
        Register a distribution
        :param probs: The distribution as a list of probabilities
        :return: A pcirc that generates a binary-encoded output sel
          that corresponds to sampling from the distribution.
        """
        dist = tuple(probs)
        if dist not in self._distributions:
            name = f"{self._aut_name}_c{len(self._distributions)}"
            sel = atom(len(probs), name).with_output("sel")
            lookup = bidict({idx: f'sel-{idx}' for idx in range(len(probs))})
            encoder = D.Encoding(
                decode=lookup.get, 
                encode=lookup.inv.get
            )
            func = D.from_aigbv(sel.aigbv, input_encodings={name: encoder})

            name2prob = {name : { f"sel-{index}": prob for index, prob in enumerate(probs)}}
            coins_id = f"{self._aut_name}_c{len(self._distributions)}"

            self._distributions[dist] = C.pcirc(func) \
                                         .randomize(name2prob) \
                                         .with_coins_id(coins_id)
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
    if data:
        raise NotImplementedError("Constants are not supported yet.")


def _translate_variables(data : dict, scope : JaniScope):
    for v in data:
        if "name" not in v:
            raise ValueError("Variable not named")
        name = v["name"]
        if v["type"]["kind"] != "bounded":
            raise ValueError("Only bounded variables are supported")
        if v["type"]["base"] != "int":
            raise ValueError("Only integer-typed variables are supported")
        #TODO could be a constant expression, which is not yet supported
        lower_bound = v["type"]["lower-bound"]
        if lower_bound != 0:
            raise NotImplementedError("Variable {name} has a non-zero lower bound. This is currently not supported")
        #TODO could be a constant expression, which is not yet supported
        upper_bound = v["type"]["upper-bound"]
        scope.add_bounded_int_variable(v["name"], lower_bound, upper_bound)


BINARY_AEX_OPS = {"+" : ops.add, "-" : ops.sub}
BINARY_BOOL_OPS = {"≤" : ops.le, "≥" : ops.ge}
BINARY_OPS = BINARY_AEX_OPS | BINARY_BOOL_OPS


def _translate_expression(data : dict, scope : JaniScope):
    """
    Takes an expression in JANI json, returns a circuit.
    :param data:  the expression AST
    :param scope: The scope with the variable definitions.
    :return: An expression in py-aiger-bv
    """
    if isinstance(data, int):
        # TODO replace the constant two here.
        return BV.uatom(2, data)
    if isinstance(data, str):
        return scope.get_aig_variable(data)

    if "op" not in data:
        raise ValueError(f"{data} is expected to have an operator")

    try:
        op = BINARY_OPS[data["op"]]
    except KeyError:
        raise NotImplementedError(f"{data} not supported")

    return op(
        _translate_expression(data["left"], scope),
        _translate_expression(data["right"], scope),
    )


def _parse_prob(data):
    """
    Parses a probability

    :param data: A json JANI representation of the expression that gives the probability.
    :return: A fraction
    """
    if isinstance(data, float):
        return Fraction(data)
    else:
        NotImplementedError(f"We only support constant probs given as floating point numbers, but got {data}")


def _translate_destinations(data : dict, ctx : AutomatonContext) -> set[str]:
    """

    :param data: Describes the destinations.
    :param ctx:
    :return:
    """
    # Create coin flipping part.
    if len(data) == 1:
        pass
    else:
        probs = []
        for d in data:
            probs.append(_parse_prob(d["probability"]["exp"]))
        prob_input = ctx.register_distribution(probs)
        # TODO consider what to do with the additional output of prob_input

    vars_written_to = set()
    for d in data:
        for a in d["assignments"]:
            vars_written_to.add(a["ref"])

    destinations = []
    for index, d in enumerate(data):
        assert d["location"] == "l"
        updates = {}
        # TODO add location handling
        for assignment in d["assignments"]:
            var_primed = assignment["ref"]
            if len(data) > 1:  # TODO: why make this case special?
                var_primed += f"-{index}"

            val = assignment["value"]
            updates[var_primed] = _translate_expression(val, ctx.scope) \
                .with_output(var_primed)

        for var in vars_written_to:
            var_primed = var
            if len(data) > 1:
                var_primed += f"-{index}"  #TODO: why make this case special.
            if var_primed not in updates:
                updates[var_primed] = ctx.scope \
                                         .get_aig_variable(var) \
                                         .with_output(var_primed)

        update = par_compose(updates.values())
        destinations.append(update)

    edge_circuit = par_compose(destinations)

    vars_written_to = list(vars_written_to)
    if len(destinations) > 1:

        indices = range(len(destinations))

        def selectors():
            for var in vars_written_to:
                size = ctx.scope.get_aig_variable(var).size
                outputs= [BV.uatom(size, f"{var}-{idx}") for idx in indices]
                yield mux(outputs, key_name='sel').with_output(var).aigbv

        edge_circuit >>= par_compose(selectors())
        edge_circuit <<= prob_input

    return edge_circuit, vars_written_to


def _translate_edges(data : dict, ctx : AutomatonContext ):
    #TODO handle notion of actions
    # max_internal_nondet_bits = int(np.ceil(np.log(len(data))))
    # max_action_bits = int(np.ceil(np.log(len(data) + 1))) # +1 due to reserving space for silent act
    #
    # select_edge_expr = []
    edge_circuits =  []
    for edge_index, edge in enumerate(data):
        #TODO add location handling
        assert edge["location"] == "l"

        edge_circuit, vars_written_to = _translate_destinations(edge["destinations"], ctx)
        if "guard" in edge:
            guard_expr = _translate_expression(edge["guard"]["exp"], ctx.scope).with_output("enabled")
            edge_circuit = edge_circuit | guard_expr.aigbv

        # Make sure that the edge circuit treats all variables as outputs
        # Additionally, mark whether global variables have been written to.
        for v in ctx.scope.variables:
            if not v.is_local:
                edge_circuit |= BV.source(
                    wordlen=1,
                    value=int(v.name in vars_written_to),
                    name=f'{v.name}-mod', 
                    signed=False,
                )

            if v.name not in vars_written_to:
                edge_circuit |= ctx.scope.get_aig_variable(v.name) \
                                         .with_output(v.name) \
                                         .aigbv

        # Rename outputs such that we can later merge them.
        relabels = {o : f"{o}-{edge_index}" for o in edge_circuit.outputs}
        edge_circuits.append(edge_circuit['o', relabels])

    def selectors():
        indices = range(len(edge_circuits))
        for v in ctx.scope.variables:
            size = ctx.scope.get_aig_variable(v.name).size
            outputs = [BV.uatom(size, f"{v.name}-{idx}") for idx in indices]
            yield mux(outputs, key_name='edge').with_output(v.name).aigbv

    aut_circuit = par_compose(edge_circuits)
    return aut_circuit >> par_compose(selectors())


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
    # TODO: Apply feedback loops to make sequential circuit.
    ctx = _create_automaton_context(data, scope)
    _translate_variables(data["variables"], scope)
    return _translate_edges(data["edges"], ctx)


def translate_file(path):
    with open(path) as f:
        jani_enc = json.load(f)
    return translate_jani(jani_enc)


def translate_jani(data : json):
    global_scope = JaniScope()
    _translate_constants(data["constants"], global_scope)
    _translate_variables(data["variables"], global_scope)
    if len(data["automata"]) != 1:
        # TODO
        raise NotImplementedError("Only support monolithic jani.")
    aut, *_ = data["automata"]
    return _translate_automaton(aut, global_scope.make_local_scope_copy())
