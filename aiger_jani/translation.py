from __future__ import annotations

import json
import math
import operator as ops
from typing import Any, Sequence

import attr
import aiger_bv as BV
import aiger_coins as C
import aiger_discrete as D
from bidict import bidict
from fractions import Fraction

from aiger_jani.utils import atom, mux, par_compose, min_op, max_op, min_bits

BVExpr = BV.UnsignedBVExpr


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniVariable:
    pass  # TODO move stuff here


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniBooleanVariable(JaniVariable):
    name: str
    is_local: bool
    initial: bool


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniIntegerVariable(JaniVariable):
    name: str
    lower_bound: int  # TODO: maybe tuple instead?
    upper_bound: int
    is_local: bool
    initial: int


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniConstant:
    pass


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniIntegerConstant(JaniConstant):
    name: str
    value: int
    # TODO: Initial values can be expressions...
    # or not have an initial value at all.


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniRealConstant(JaniConstant):
    name: str
    value: float
    # TODO: Initial values can be expressions...
    # or not have an initial value at all.


@attr.s(auto_attribs=True, auto_detect=True, frozen=True)
class JaniScope:
    _local: bool = False
    _aigvars: dict[str, BVExpr] = attr.ib(factory=dict)
    _variables: dict[str, JaniVariable] = attr.ib(factory=dict)
    _constants: dict[str, JaniConstant] = attr.ib(factory=dict)
    _transient_vars: dict[str, JaniVariable] = attr.ib(factory=dict)

    def add_boolean_variable(self, name: str, init: bool, is_transient: bool):
        if name in self._variables:
            raise ValueError(
                f"Variable with name {name} already exists in scope.")
        variable = JaniBooleanVariable(name, self._local, init)
        if not is_transient:
            self._variables[name] = variable
            self._aigvars[name] = atom(1, name)
        else:
            pass # TODO support transient variables

    def add_bounded_int_variable(self, name: str, lower_bound: int,
                                 upper_bound: int, init: int, is_transient: bool) -> None:
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
                f"Variable with name {name} already exists in scope.")

        variable = JaniIntegerVariable(
            name, lower_bound, upper_bound, self._local, init
        )
        if not is_transient:
            self._variables[name] = variable
            self._aigvars[name] = atom(upper_bound - lower_bound, name)
        else:
            pass # TODO support transient variables


    def add_constant(self, name: str, tp: str, value: str) -> None:
        assert tp in ["real", "int"], f"Got type {tp}"
        if name in self._constants:
            raise ValueError(
                f"Constant with name {name} already exists in scope.")
        if tp == "int":
            self._constants[name] = JaniIntegerConstant(name, value)
        elif tp == "real":
            self._constants[name] = JaniRealConstant(name, value)
        else:
            assert False

        # TODO We current do not do anything useful with constants,
        # We just register them to provide a proper not supported message.

    def make_local_scope_copy(self) -> JaniScope:
        if self._local:
            raise ValueError("You should not copy local scopes")

        return JaniScope(
            local=True,
            aigvars=self._aigvars.copy(),
            variables=self._variables.copy(),
        )

    def get_aig_variable(self, name: str) -> BV.UnsignedBVExpr:
        return self._aigvars[name]

    def is_local_var(self, name: str) -> bool:
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
    _distributions: dict[Probs, C.PCirc] = attr.ib(factory=dict)

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
            encoder = D.Encoding(decode=lookup.get, encode=lookup.inv.get)
            func = D.from_aigbv(sel.aigbv, input_encodings={name: encoder})

            name2prob = {
                name:
                {f"sel-{index}": prob
                 for index, prob in enumerate(probs)}
            }
            coins_id = f"{self._aut_name}_c{len(self._distributions)}"

            self._distributions[dist] = C.pcirc(func) \
                                         .randomize(name2prob) \
                                         .with_coins_id(coins_id)
        return self._distributions[dist]

    def register_action(self, action: str):
        """
        :param action: The name of the action
        :return: The action index and the internal non-det index
        """
        if action not in self._actions:
            self._actions[action] = len(self._actions)
        self._actions_cnt[action] = self._actions_cnt.get(action, 0) + 1
        return self._actions[action], self._actions_cnt[action]


def _translate_constants(data: dict, scope: JaniScope):
    for v in data:
        if "name" not in v:
            raise ValueError("Constant not named")
        name = v["name"]
        if v["type"] not in ["int", "real"]:
            raise NotImplementedError("Only integer and real-valued constants are supported")
        type = v["type"]
        if "value" not in v:
            raise NotImplementedError("We currently expect a fixed value for constants")
        value = v["value"]
        scope.add_constant(name, type, value)


def _translate_variables(data: dict, scope: JaniScope):
    for v in data:
        if "name" not in v:
            raise ValueError("Variable not named")
        name = v["name"]
        if isinstance(v["type"], str):
            if v["type"] != "bool":
                raise ValueError(
                    f"Booleans are the only plain data type supported, but got {v['type']} for variable {name} (bounded inters are extended types).")
            if 'initial-value' not in v:
                raise NotImplementedError(
                    f"Variable {name} must have an initial value."
                )
            init = v['initial-value']
            if "transient" in v:
                is_transient = v["transient"]
            else:
                is_transient = False
            scope.add_boolean_variable(name, init, is_transient)
        elif isinstance(v["type"], dict):
            if v["type"]["kind"] != "bounded":
                raise ValueError(
                    "Bounded integers are the only extended variable type supported")
            if v["type"]["base"] != "int":
                raise ValueError("Only integer-typed variables are supported")
            # TODO could be a constant expression, which is not yet supported.
            lower_bound = v["type"]["lower-bound"]
            if lower_bound != 0:
                raise NotImplementedError(
                    f"Variable {name} has a non-zero lower bound."
                    "This is currently not supported")
            # TODO could be a constant expression, which is not yet supported
            upper_bound = v["type"]["upper-bound"]
            if 'initial-value' not in v:
                raise NotImplementedError(
                    f"Variable {name} must have an initial value."
                )
            init = v['initial-value']
            if "transient" in v:
                is_transient = v["transient"]
            else:
                is_transient = False
            scope.add_bounded_int_variable(
                name, lower_bound, upper_bound, init, is_transient)
        else:
            raise ValueError(
                f"Type of variable {name} must be a base type or an extended type")


BINARY_AEX_OPS = {"+": ops.add, "-": ops.sub, "min": min_op, "max": max_op}
BINARY_BOOL_OPS = {"≤": ops.le,
                   "≥": ops.ge,
                   "=": ops.eq,
                   "<": ops.lt,
                   ">": ops.gt,
                   "∧": ops.and_}
BINARY_OPS = BINARY_AEX_OPS | BINARY_BOOL_OPS
UNARY_OPS = {"¬": ops.inv, "-": ops.neg}


def _translate_expression(data: dict, scope: JaniScope):
    """
    Takes an expression in JANI json, returns a circuit.
    :param data:  the expression AST
    :param scope: The scope with the variable definitions.
    :return: An expression in py-aiger-bv
    """
    if isinstance(data, bool):
        return BV.uatom(1, 1 if data else 0)
    if isinstance(data, int):
        if data == 0:
            return BV.uatom(1, data)
        nr_bits = min_bits(data)
        return BV.uatom(nr_bits, data)
    if isinstance(data, str):
        return scope.get_aig_variable(data)

    if "op" not in data:
        raise ValueError(f"{data} is expected to have an operator")
    if "right" in data:
        try:
            op = BINARY_OPS[data["op"]]
        except KeyError:
            raise NotImplementedError(f"Operator {data['op']} not supported")
        left_subexpr = _translate_expression(data["left"], scope)
        right_subexpr = _translate_expression(data["right"], scope)

        # Match size
        if left_subexpr.size < right_subexpr.size:
            left_subexpr = left_subexpr.resize(right_subexpr.size)
        elif right_subexpr.size < left_subexpr.size:
            right_subexpr = right_subexpr.resize(left_subexpr.size)

        return op(left_subexpr, right_subexpr)
    else:
        try:
            op = UNARY_OPS[data["op"]]
        except KeyError:
            raise NotImplementedError(f"Operator {data['op']} not supported")

        subexpr = _translate_expression(data["exp"], scope)
        return op(subexpr)


def _parse_prob(data) -> Fraction:
    """
    Parses a probability

    :param data: A json JANI representation of the expression that
      gives the probability.
    :return: A fraction
    """
    if isinstance(data, float):
        return Fraction(data)
    elif isinstance(data, int):
        return Fraction(data)
    elif isinstance(data, dict) and "op" in data:
        return Fraction(data["left"], data["right"])
    else:
        raise NotImplementedError(
            "We only support constant probs given as floating point"
            f"numbers, but got {data}")


def _translate_destinations(data: dict, ctx: AutomatonContext) -> set[str]:
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
            if "probability" not in d:
                probs.append(1)
            else:
                probability = d["probability"]["exp"]
                probs.append(_parse_prob(probability))
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
                .with_output(var_primed) \
                .resize(ctx.scope.get_aig_variable(var_primed).size) \
                .aigbv

        for var in vars_written_to:
            var_primed = var
            if len(data) > 1:
                var_primed += f"-{index}"  # TODO: why make this case special.
            if var_primed not in updates:
                updates[var_primed] = ctx.scope \
                                         .get_aig_variable(var) \
                                         .with_output(var_primed) \
                                         .aigbv

        update = par_compose(updates.values())
        destinations.append(update)

    edge_circuit = par_compose(destinations)

    vars_written_to = list(vars_written_to)
    if len(destinations) > 1:
        indices = range(len(destinations))

        def selectors():
            for var in vars_written_to:
                size = ctx.scope.get_aig_variable(var).size
                outputs = [BV.uatom(size, f"{var}-{idx}") for idx in indices]
                yield mux(outputs, key_name='sel').with_output(var).aigbv

        edge_circuit >>= par_compose(selectors())
        edge_circuit <<= prob_input
    else:
        edge_circuit = C.pcirc(edge_circuit)  # Deterministic pcirc.

    return edge_circuit, vars_written_to


def _translate_edges(data: dict, ctx: AutomatonContext):
    # TODO handle notion of actions
    # max_internal_nondet_bits = int(np.ceil(np.log(len(data))))
    # max_action_bits = int(np.ceil(np.log(len(data) + 1))) # +1 due
    # to reserving space for silent act
    #
    # select_edge_expr = []
    edge_circuits = []
    edge_expr = atom(len(data), 'edge')
    for edge_index, edge in enumerate(data):
        # TODO add location handling
        assert edge["location"] == "l"

        edge_circuit, vars_written_to = _translate_destinations(
            edge["destinations"], ctx)

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
        relabels = {o: f"{o}-{edge_index}" for o in edge_circuit.outputs}
        edge_circuits.append(edge_circuit['o', relabels])

    def selectors():
        indices = range(len(edge_circuits))
        for v in ctx.scope.variables:
            size = ctx.scope.get_aig_variable(v.name).size
            outputs = [BV.uatom(size, f"{v.name}-{idx}") for idx in indices]
            yield mux(outputs, key_name='edge').with_output(v.name).aigbv

    edge_circuits_composed = par_compose(edge_circuits)
    selectors_composed = par_compose(selectors())
    update = edge_circuits_composed >> selectors_composed

    # Add guards
    edge_expr = atom(len(edge_circuits), 'edge')
    for edge_idx, edge in enumerate(data):
        if "guard" not in edge:
            continue
        guard_expr = _translate_expression(edge["guard"]["exp"], ctx.scope)
        update = update.assume((edge_expr != edge_idx) | guard_expr)

    return update


def _create_automaton_context(data: dict, scope: JaniScope):
    locations = {loc['name']: False for loc in data['locations']}
    initial_locations = {loc: True for loc in data['initial-locations']}

    if unknown_locations := set(initial_locations) - set(locations):
        raise ValueError(f'Locations {unknown_locations} are unknown.')

    locations.update(initial_locations)
    return AutomatonContext(data["name"], scope, locations)


def _translate_automaton(data: dict, scope: JaniScope):
    # TODO: Apply feedback loops to make sequential circuit.
    ctx = _create_automaton_context(data, scope)
    if "variables" in data:
        _translate_variables(data["variables"], scope)
    update = _translate_edges(data["edges"], ctx)
    wires, relabels = [], {}
    for var in ctx.scope.variables:
        name = f'{ctx._aut_name}-{var.name}'
        wires.append({
            'input': var.name,
            'output': var.name,
            'init': var.initial,
            'latch': name,
            'keep_output': True,
        })
        relabels[var.name] = name
    return update.loopback(*wires)['o', relabels]


def translate_file(path):
    with open(path) as f:
        jani_enc = json.load(f)
    return translate_jani(jani_enc)


def translate_jani(data: json):
    global_scope = JaniScope()
    if "constants" in data:
        _translate_constants(data["constants"], global_scope)
    if "variables" in data:
        _translate_variables(data["variables"], global_scope)
    if len(data["automata"]) != 1:
        # TODO
        raise NotImplementedError("Only support monolithic jani.")
    aut, *_ = data["automata"]
    # TODO: make global variables sequential (latches).
    return _translate_automaton(aut, global_scope.make_local_scope_copy())


__all__ = ['translate_file', 'translate_jani']
