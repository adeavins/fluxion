from collections import defaultdict
from itertools import count

import numpy

from .symbolic import *
from .field import *
from .steppers import *

from .tools import multimethod, Sequence


class IntegrationResults:

    def __init__(self, data):
        self.data = data


def linspace(start, stop, points):
    return numpy.linspace(start, stop, points)


def check_equation(eq):
    lhs, rhs = eq.args

    # check lhs

    assert isinstance(lhs, Differential)

    field, pdimension = lhs.args
    assert isinstance(field, UnknownField)
    assert isinstance(pdimension, PropagationDimension)

    # check rhs

    vs = used_variables(rhs)
    assert vs['propagation_dimensions'].issubset(set([pdimension]))
    assert (
        'transverse_dimensions' not in vs
        or vs['transverse_dimensions'].issubset(set(field.dimensions)))

    return field, pdimension, vs


def sample_field(field, pdim):
    return field


def sample(results, stepper, ufield_snapshot, samplers, events):
    # TODO: must be rewritten as a pure function
    for pdim_val, keys in events:
        interp_field = as_field(stepper.interpolate_at(pdim_val), template=ufield_snapshot)
        for key in keys:
            results[key]['pvalue'].append(pdim_val)
            result = samplers[key](interp_field, pdim_val)
            result = as_field(result)
            results[key]['mean'].append(result)


def integrate(eq, initial_field, pdim_start, seed=None, samplers={}):

    stepper_gen = RK4Stepper(step=0.02)

    # assert that the equation has a required form:
    ufield, pdimension, vs = check_equation(eq)

    tdimensions = [d for d in ufield.dimensions if d != pdimension]
    ufield_snapshot = ufield.without_dimensions(pdimension)
    field = as_field(initial_field, template=ufield_snapshot)

    sequences = {key: val[1] for key, val in samplers.items()}
    samplers = {key: val[0] for key, val in samplers.items()}

    seq = Sequence(sequences)

    initial_samples = seq.pop_events_until(pdim_start)

    results = {key: dict(pvalue=[], mean=[]) for key in samplers}

    # Stepper only deals with arrays, not fields
    deriv_func = CallableExpr(eq.args[1], ufield, pdimension)
    stepper = stepper_gen.initialize(pdim_start, field.data, deriv_func)

    sample(results, stepper, ufield_snapshot, samplers, initial_samples)

    while True:

        if seq.empty():
            break

        # propagate a step forward
        stepper.step()

        pdim_val = stepper.pdim

        to_sample = seq.pop_events_until(pdim_val)

        if len(to_sample) > 0:
            print(pdim_val)

        sample(results, stepper, ufield_snapshot, samplers, to_sample)

    for key in results:
        generic_field = find_generic_field(results[key]['mean'], ufield.dimensions)
        results[key] = join_fields(
            results[key]['mean'], pdimension, results[key]['pvalue'], generic_field)

    return results
