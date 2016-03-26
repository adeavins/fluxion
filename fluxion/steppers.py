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
            dtype = field_arr.dtype
            axes = {variable: ufield.dimensions.index(variable) for variable in powers}
            arr = numpy.fft.fftn(arr, axes=list(axes.values()))
            for variable, power in powers.items():
                xs = variable.grid
                ks = numpy.fft.fftfreq(xs.size, xs[1] - xs[0]) * 2 * numpy.pi
                arr *= ((-1j * ks)**power).reshape(ks.size, *([1] * (arr.ndim - axes[variable] - 1)))
            arr = numpy.fft.ifftn(arr, axes=list(axes.values()))
            arr = arr.astype(dtype) # resetting back to real numbers if necessary

            to_sub[diff] = Field('_diff_arr', *self.ufield_dimensions, data=arr)

        return as_array(substitute(self.expr, to_sub))


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


class RK4Stepper:

    def __init__(self, step=0.1):
        self.step = step

    def create(self, eq):
        pass

    def initialize(self, pdim_start, initial_field, func):
        return _RK4Stepper(pdim_start, initial_field, func, self)


class _RK4Stepper:

    def __init__(self, pdim_start, initial_field, func, params):
        self.pdim_prev = pdim_start
        self.pdim = pdim_start

        self.field_prev = initial_field
        self.field = initial_field
        self.step_times_deriv_prev = None
        self.step_times_deriv = None

        self.func = func
        self.params = params

        deriv_func = CallableExpr(eq.args[1], ufield, pdimension)

    def step(self):

        step = self.params.step
        pdim = self.pdim
        f = lambda f, p: self.func((f, p))

        self.step_times_deriv_prev = step * f(self.field, pdim)

        k1 = self.step_times_deriv_prev
        k2 = step * f(self.field + k1 / 2, pdim + step / 2)
        k3 = step * f(self.field + k2 / 2, pdim + step / 2)
        k4 = step * f(self.field + k3, pdim + step)

        self.step_times_deriv = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        new_field = self.field + self.step_times_deriv

        self.pdim += self.params.step
        self.field_prev = self.field
        self.field = new_field

    def interpolate_at(self, pdim):
        assert self.pdim_prev <= pdim <= self.pdim
        if self.pdim_prev == self.pdim:
            return self.field

        h = self.pdim - self.pdim_prev
        t = (pdim - self.pdim_prev) / h
        y = self.field
        yp = self.field_prev
        f = self.step_times_deriv
        fp = self.step_times_deriv_prev

        # Third-order approximation
        return (
            (1 - t) * yp + t * y
            + t * (t - 1) * ((1 - 2 * t) * (y - yp) + (t - 1) * fp + t * f))
