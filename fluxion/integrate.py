from collections import defaultdict
from itertools import count

import numpy

from .symbolic import *
from .field import *

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


class EulerStepper:

    def __init__(self, step=0.1):
        self.step = step

    def initialize(self, pdim_start, initial_field, func):
        return _EulerStepper(pdim_start, initial_field, func, self)


class _EulerStepper:

    def __init__(self, pdim_start, initial_field, func, params):
        self.pdim_prev = pdim_start
        self.pdim = pdim_start

        self.field_prev = initial_field
        self.field = initial_field

        self.func = func
        self.params = params

    def step(self):
        new_field = self.field + self.func((self.field, self.pdim)) * self.params.step

        self.pdim += self.params.step
        self.field_prev = self.field
        self.field = new_field

    def interpolate_at(self, pdim):
        assert self.pdim_prev <= pdim <= self.pdim
        if self.pdim_prev == self.pdim:
            return self.field
        return (
            self.field_prev
            + (pdim - self.pdim_prev) / (self.pdim - self.pdim_prev)
                * (self.field - self.field_prev))



def sample_field(field, pdim):
    return field


class CallableExpr:

    def __init__(self, expr, ufield, pdim):

        vs = used_variables(expr)
        self.differentials = vs['differentials']

        self.expr = expr
        self.param_symbols = [ufield, pdim]
        self.ufield_dimensions = tuple(dim for dim in ufield.dimensions if dim != pdim)

    def __call__(self, param_arrays):

        field_arr, pdim_value = param_arrays
        ufield, pdim = self.param_symbols

        to_sub = {
            ufield: Field('_ufield_arr', *self.ufield_dimensions, data=field_arr),
            pdim: Field('_pdim_val', data=pdim_value)
        }

        for diff in self.differentials:
            field = diff.args[0]
            variables = diff.args[1:]

            # Here we assume that the target dimensions are
            # - uniform
            # - periodic
            # so we can use FFT to calculate derivatives

            powers = defaultdict(lambda: 0)
            for variable in variables:
                powers[variable] += 1

            arr = field_arr
            axes = {variable: ufield.dimensions.index(variable) for variable in powers}
            arr = numpy.fft.fftn(arr, axes=list(axes.values()))
            for variable, power in powers.items():
                xs = variable.grid
                ks = numpy.fft.fftfreq(xs.size, xs[1] - xs[0]) * 2 * numpy.pi
                arr *= ((1j * ks)**power).reshape(ks.size, *([1] * (arr.ndim - axes[variable] - 1)))
            arr = numpy.fft.ifftn(arr, axes=list(axes.values()))

            to_sub[diff] = Field('_diff_arr', *self.ufield_dimensions, data=arr)

        return as_array(substitute(self.expr, to_sub))


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

    # FIXME: set to a small value to make the soliton example converge
    # really need to implement a decent stepper
    stepper_gen = EulerStepper(step=0.0001)

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
        print(pdim_val)

        to_sample = seq.pop_events_until(pdim_val)

        sample(results, stepper, ufield_snapshot, samplers, to_sample)

    for key in results:
        generic_field = find_generic_field(results[key]['mean'], ufield.dimensions)
        results[key] = join_fields(
            results[key]['mean'], pdimension, results[key]['pvalue'], generic_field)

    print(results)
    return results


def plot(results):
    raise NotImplementedError
