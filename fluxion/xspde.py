from __future__ import division

r"""
eXtensible Stochastic Partial Differential Equation solver
==========================================================

*Peter D Drummond*

*Swinburne University of Technology*


Uses interaction picture for linear parts.
Four algorithms:

1) Euler for stochastic
2) RK2  for stochastic
3) MP for stochastic: See Refs below
4) RK4(Ballagh et al, Otago University).

Solves:

.. math::

    d\mathbf{a} =
        \mathbf{A}(\mathbf{a},t) dt
        + B(\mathbf{a},t) d\mathbf{w}
        + C \mathbf{a} dt
        + D(\nabla^2) \mathbf{a} dt,

References:

* Mortimer, Drummond: J. Comp. Phys. 93, 144 (1991)
* Werner, Drummond: J. Comp. Phys. 132, 312 (1997)
* Caradoc-Davies, PhD thesis, Uni. of Otago (NZ, 2000)
"""

import functools
import operator

import numpy


def expm(a):
    """
    Slower than scipy.linalg.expm version, but does not require scipy.
    """
    w, v = numpy.linalg.eig(a)
    res = v.dot(numpy.diag(numpy.exp(w))).dot(numpy.linalg.inv(v))
    return res.astype(a.dtype)


def product(l):
    return functools.reduce(operator.mul, l, 1)


class Lattice:
    """
    Rectangular uniform lattice for transverse dimensions.

    :param shape: a tuple with the number of lattice points in each dimension.
        Defaults to a 0-dimensional lattice.
    :param box: a tuple with the length of the lattice in each dimension.
        If ``None``, will be set equal to ``shape``.

    .. py:attribute:: shape

    .. py:attribute:: box

    .. py:attribute:: ndim

        The number of lattice dimensions.

    .. py:attribute:: dxs

        A list of lattice step sizes for each dimension.

    .. py:attribute:: xs

        A list of arrays of lattice point coordinates for each dimension.

    .. py:attribute:: dV

        The volume of a lattice cell.

    .. py:attribute:: V

        The total lattice volume.

    .. py:attribute:: size

        The total number of lattice points.
    """

    def __init__(self, shape=(), box=None):
        if box is None:
            box = shape

        if isinstance(shape, int):
            shape = (shape,)
            box = (box,)

        self.ndim = len(shape)
        self.shape = shape
        self.box = box

        self.dxs = [l / n for l, n in zip(box, shape)]
        self.xs = [
            numpy.linspace(-l / 2 + dx / 2, l / 2 - dx / 2, n)
            for l, n, dx in zip(box, shape, self.dxs)]

        self.dV = product(self.dxs)
        self.V = product(self.box)
        self.size = product(self.shape)

        self._fft_scale = (self.dV / self.size)**0.5

    def to_kspace(self, arr):
        """
        Apply a scaled FFT to the array whose last dimensions coincide with the lattice shape.
        The FFT is scaled such that the squared absolute value in each point of the result
        is equal to the population of the corresponding mode.
        """
        fft_axes = tuple(range(arr.ndim - self.ndim, arr.ndim))
        return numpy.fft.fftn(arr, axes=fft_axes) * self._fft_scale

    def to_xspace(self, arr):
        """
        Apply a scaled IFFT to the array whose last dimensions coincide with the lattice shape.
        The scale matches that of :py:meth:`to_kspace`
        (that is, ``to_xspace(to_kspace(arr)) == arr``).
        """
        fft_axes = tuple(range(arr.ndim - self.ndim, arr.ndim))
        return numpy.fft.ifftn(arr, axes=fft_axes) / self._fft_scale

    def __repr__(self):
        return "Lattice(shape={shape}, box={box})".format(shape=self.shape, box=self.box)

    def __hash__(self):
        return hash(repr(self))


class Equation:
    r"""
    Defines an equation of the form

    .. math::

        d\mathbf{a} =
            \mathbf{A}(\mathbf{a},t) dt
            + B(\mathbf{a},t) d\mathbf{w}
            + C \mathbf{a} dt
            + D(\nabla^2) \mathbf{a} dt,

    where dot products assume Hadamard multiplication over spatial dimensions and:

    :math:`\mathbf{a}`: vector field (shape: ``components [X spatial dimensions]``);
    :math:`\mathbf{A}`: drift (vector function of `a`);
    :math:`B`: noise matrix;
    :math:`C`: constant matrix of linear coefficients;
    :math:`D`: vector of spatical derivatives.

    The vector field will be supplied with an additional first dimension of trajectories,
    and ``A`` and ``B`` results should be batched over it.

    :param components: the number of components of :math:`\mathbf{a}`.
    :param noise_components: the number of components of :math:`d\mathbf{w}`.
    :param complex_noise: if ``True``, :math:`d\mathbf{w}` will be complex.
    """

    def __init__(self, components=1, noise_components=0, complex_noise=False):
        self.components = components
        self.noise_components = noise_components
        self.complex_noise = complex_noise

    def drift(self, lattice, a, t):
        r"""
        Returns :math:`\mathbf{A}(\mathbf{a}, t)`.
        The result must be the same shape as :math:`\mathbf{a}`.
        """
        return numpy.zeros(a.shape)

    def noise_matrix(self, lattice, a, t):
        r"""
        Returns :math:`B(\mathbf{a}, t)`.
        The result must have the shape
        ``trajectories X components X noise components [X spatial dimensions]``.
        """
        trajectories = a.shape[0]
        components = a.shape[1]
        spatial_dimensions = a.shape[2:]
        return numpy.zeros(
            (trajectories, components, self.noise_components)
            + tuple(spatial_dimensions))

    def linear_coefficients(self):
        """
        Returns :math:`C`.
        The result must have the shape ``components X components``.
        """
        return numpy.zeros((self.components, self.components))

    def spatial_derivatives(self, nabla_squared):
        """
        Returns :math:`D`.
        The result must have the shape: ``components [X spatial dimensions]``.
        """
        return numpy.zeros_like(nabla_squared)

    def noise(self, lattice, a, t, dw):
        r"""
        A convenience function that returns :math:`B d\mathbf{w}`.
        If not defined, calculated using :py:meth:`noise_matrix`.
        """
        b = self.noise_matrix(lattice, a, t)
        return numpy.einsum('ijk...,ik...->ij...', b, dw)

    def deriv(self, lattice, a, t, dt, dw):
        r"""
        A convenience function that returns :math:`A dt + B d\mathbf{w}`.
        If not defined, calculated using :py:meth:`drift` and :py:meth:`noise`.
        """
        res = self.drift(lattice, a, t) * dt
        if dw is not None:
            res = res + self.noise(lattice, a, t, dw)
        return res


def get_ksquared(lattice):
    """
    Returns an array of squared lengths of k vectors for the given lattice.
    """
    ks = [
        2 * numpy.pi * numpy.fft.fftfreq(size, length / size)
        for size, length in zip(lattice.shape, lattice.box)]

    if len(lattice.shape) > 1:
        full_ks = numpy.meshgrid(*ks, indexing='ij')
    else:
        full_ks = ks

    return sum(full_k ** 2 for full_k in full_ks)


class Stepper:
    """
    Interaction picture stepper.
    Prepares propagators for the linear parts of the equation in advance.
    """

    def __init__(self, eqn, lattice, dt, ip_prop_coeff=1):
        ksquared = get_ksquared(lattice)
        c = eqn.linear_coefficients()
        d = eqn.spatial_derivatives(-ksquared)

        c = numpy.array(c)
        c = c.reshape(eqn.components, eqn.components)

        d = numpy.array(d)
        d = d.reshape(eqn.components, *lattice.shape)

        self.lattice = lattice
        self.x_prop = expm(dt * ip_prop_coeff * c)
        self.k_prop = numpy.exp(dt * ip_prop_coeff * d)
        self.dt = dt
        self.eqn = eqn

    def prop_ip(self, a):
        # x_prop: components X components
        # k_prop: components X spatial dimensions
        # a: trajectories X components X spatial dimensions
        a = numpy.einsum('ij,kj...->ki...', self.x_prop, a)
        ka = numpy.fft.fftn(a, axes=range(2, a.ndim))
        ka *= self.k_prop
        a = numpy.fft.ifftn(ka, axes=range(2, a.ndim))
        return a


class Euler(Stepper):

    extrapolation_order = 1

    def __call__(self, a, t, dw):
        a = a + self.eqn.deriv(self.lattice, a, t, self.dt, dw)
        a = self.prop_ip(a)
        return a


class RK2IP(Stepper):

    extrapolation_order = 2

    def __call__(self, a, t, dw):
        am = self.prop_ip(a)
        d1 = self.prop_ip(self.eqn.deriv(self.lattice, a, t, self.dt, dw))
        d2 = self.eqn.deriv(self.lattice, am + d1, t + self.dt, self.dt, dw)
        a = am + (d1 + d2) / 2
        return a


class Midpoint(Stepper):

    extrapolation_order = 2

    def __init__(self, eqn, lattice, dt, iterations=4):
        Stepper.__init__(self, eqn, lattice, dt, ip_prop_coeff=0.5)
        self.iterations = iterations

    def __call__(self, a, t, dw):
        am = self.prop_ip(a)
        at = am
        for i in range(self.iterations):
            d1 = self.eqn.deriv(self.lattice, at, t + self.dt / 2, self.dt, dw) / 2
            at = am + d1
        a = self.prop_ip(at + d1)
        return a


class RK4IP(Stepper):

    extrapolation_order = 4

    def __init__(self, eqn, lattice, dt):
        Stepper.__init__(self, eqn, lattice, dt, ip_prop_coeff=0.5)

    def __call__(self, a, t, dw):
        dt = self.dt
        deriv = self.eqn.deriv
        prop_ip = self.prop_ip
        lattice = self.lattice

        am = prop_ip(a)
        d1 = prop_ip(deriv(lattice, a, t, dt, dw) / 2)
        d2 = deriv(lattice, am + d1, t + dt / 2, dt, dw) / 2
        d3 = deriv(lattice, am + d2, t + dt / 2, dt, dw) / 2
        d4 = deriv(lattice, prop_ip(am + 2 * d3), t + dt, dt, dw) / 2
        a = prop_ip(am + (d1 + 2 * (d2 + d3)) / 3) + d4/3
        return a


STEPPER_CLASSES = dict(
    euler=Euler,
    rk2ip=RK2IP,
    midpoint=Midpoint,
    rk4ip=RK4IP)


class Noise:
    """
    Generates Wiener process increments.
    Returns the same noise for two single steps as for one double step.
    """

    def __init__(self, eqn, lattice, trajectories, dt, seed, double_step=False):
        self.double_step = double_step
        self.rng = numpy.random.RandomState(seed=seed)
        self.complex_noise = eqn.complex_noise
        self.noise_shape = (trajectories, eqn.noise_components) + lattice.shape
        self.noise_scale = (
            1 / lattice.dV**0.5
            * (dt / (2 if double_step else 1))**0.5
            * (0.5**0.5 if eqn.complex_noise else 1))

    def _gen(self):
        if self.complex_noise:
            return (
                self.rng.normal(size=self.noise_shape, scale=self.noise_scale)
                + 1j * self.rng.normal(size=self.noise_shape, scale=self.noise_scale))
        else:
            return self.rng.normal(size=self.noise_shape, scale=self.noise_scale)

    def __call__(self):
        if self.double_step:
            dw1 = self._gen()
            dw2 = self._gen()
            return dw1 + dw2
        else:
            return self._gen()


def _integrate_ensemble(lattice, tlattice, a, stepper, noise_gen=None, observe=None):
    """
    Integrate a single ensemble of trajectories.
    """

    if observe is None:
        observe = lambda lattice, a, t: {}

    if noise_gen is None:
        noise_gen = lambda: None

    results = []
    for t_start in tlattice.obs_times[:-1]:
        results.append(observe(lattice, a, t_start))
        for i in range(tlattice.steps_per_observation):
            t = t_start + i * tlattice.step_dt
            dw = noise_gen()
            a = stepper(a, t, dw)
    results.append(observe(lattice, a, tlattice.obs_times[-1]))

    results = {key:numpy.array([r[key] for r in results]) for key in results[0]}

    return a, results


def join_ensemble_results(results_list):
    """
    Calculate mean and sampling error estimate for results from multiple ensembles.
    """
    results = {}
    ensembles = len(results_list)
    for key in results_list[0]:
        vals = numpy.concatenate([r[key].reshape(1, *r[key].shape) for r in results_list])
        results[key] = dict(mean=vals.mean(0))
        if ensembles > 1:
            results[key]['sampling_err'] = vals.std(0) / (ensembles - 1)**0.5
    return results


class TimeLattice:

    def __init__(self, interval, observations, steps_per_observation):
        t_start, t_end = interval
        self.obs_times = numpy.linspace(t_start, t_end, observations + 1)
        self.steps_per_observation = steps_per_observation
        self.step_dt = (self.obs_times[1] - self.obs_times[0]) / steps_per_observation


def integrate_ensembles(
        eqn, a, lattice, interval, stepper_cls, seed,
        algorithm_params={}, observe=None, double_step=False,
        observations=1, steps_per_observation=2):

    tlattice = TimeLattice(
        interval, observations, steps_per_observation // (2 if double_step else 1))
    stepper = stepper_cls(eqn, lattice, tlattice.step_dt, **algorithm_params)

    ensembles = a.shape[0]
    ensemble_size = a.shape[1]

    rng = numpy.random.RandomState(seed=seed)

    as_final = []
    ensemble_results = []
    for ensemble_num in range(ensembles):
        print("Starting ensemble", ensemble_num)
        a_ensemble = a[ensemble_num]
        if eqn.noise_components > 0:
            noise_gen = Noise(
                eqn, lattice, ensemble_size, tlattice.step_dt,
                rng.randint(2**32), double_step=double_step)
        else:
            noise_gen = None
        a_ensemble_final, results = _integrate_ensemble(
            lattice, tlattice, a_ensemble, stepper, noise_gen=noise_gen, observe=observe)
        as_final.append(a_ensemble_final)
        ensemble_results.append(results)

    a_final = numpy.concatenate(as_final)
    results = join_ensemble_results(ensemble_results)
    results['times'] = tlattice.obs_times

    return a_final, results


def calculate_step_error(coarse_results, fine_results, extrapolate=False, extrapolation_order=1):
    """
    Calculate the step error based on results with normal and half time step.
    Possibly extrapolate the results based on the given extrapolation order.
    """
    results = {}
    for key in coarse_results:
        if key == 'times':
            results['times'] = coarse_results['times']
            continue
        cr = coarse_results[key]
        fr = fine_results[key]
        results[key] = dict(fr)
        step_err = cr['mean'] - fr['mean']
        if extrapolate:
            extrapolated_step_err = step_err / (2**extrapolation_order - 1)
            results[key]['mean'] -= extrapolated_step_err
            results[key]['step_err'] = numpy.abs(extrapolated_step_err)
        else:
            results[key]['step_err'] = numpy.abs(step_err)
    return results


def integrate(
        eqn, a, lattice, interval, observe=None,
        algorithm='rk4ip', algorithm_params={},
        seed=None, observations=1, steps_per_observation=1, ensembles=10,
        convergence_check=False, extrapolate=False):
    """
    Integrate an equation and return the final state and the observed quantities.

    :param eqn: a :py:class:`Equation` object.
    :param a: an array with the initial state for the propagation.
        Must have the shape ``(trajectories, components, *spatial_dimensions)``.
    :param lattice: a :py:class:`Lattice` obejct.
    :param interval: a pair (list or tuple) of the initial and final values
        of the propagation dimension (time).
    :param observe: if specified, this function will be called ``observations`` times
        during propagation, plus once at the start to collect the observables.
        See the requirements for this function below.
    :param algorithm: the name of the integration algorithm to use.
        Supported ones: ``"euler"``, ``"rk2ip"``, ``"midpoint"``, ``"rk4ip"``.
    :param algorithm_params: a dictionary with additional parameters for the integration algorithm.
        ``"midpoint"``: ``iterations`` defines the number of iterations taken.
    :param seed: a random seed for the noise (if any).
    :param observations: the number of observations to make during propagation
        (not counting the initial one).
    :param steps_per_observation: the number of time steps taken between observations.
    :param ensembles: the number of ensembles used to calculate the sampling error.
    :param convergence_check: if ``True``, runs an additional integration with halved time step
        and estimates the step error for all of the observables.
    :param extrapolate: if ``True`` (and ``convergence_check`` is ``True``),
        attempts to find the exact value of the observable by extrapolating the data
        from full and halved time step integration.
        Must be used with caution when integrating a stochastic equation.

    :returns: a pair of the final state of the propagated vector and a dictionary of the
        collected observables. The dictionary has the structure
        ``{'times': ..., 'observable1': {'mean': ..., 'step_err': ..., 'sampling_err': ...}, ...}``.
        The key ``times`` that contains the array of observation times.
        The key ``step_err`` is only present if ``convergence_check`` was ``True``.


    .. py:function:: observe(lattice, a, t)

        :param lattice: the :py:class:`Lattice` obejct passed to :py:func:`integrate`.
        :param a: a part of the current state vector with the first dimension
            being the stochastic trajectories.
        :param t: current time.

        Each call of this function will receive a part of the whole state and must
        only return the mean value; the integrator will handle the calculation
        of the sampling error.

        It must have the signature ``(lattice, a, t)`` and will be passed the lattice,
        the current state and the current time and must return a dictionary
    """

    trajectories = a.shape[0]
    components = a.shape[1]
    spatial_dimensions = a.shape[2:]
    assert trajectories % ensembles == 0
    assert components == eqn.components
    assert tuple(spatial_dimensions) == lattice.shape
    a = a.reshape(ensembles, trajectories // ensembles, components, *spatial_dimensions)
    if seed is None:
        seed = numpy.random.randint(2**32)

    stepper_cls = STEPPER_CLASSES[algorithm]

    integration_args = (eqn, a, lattice, interval, stepper_cls, seed)
    coarse_params = dict(
        observations=observations, steps_per_observation=steps_per_observation * 2,
        observe=observe, algorithm_params=algorithm_params, double_step=True)
    fine_params = dict(coarse_params)
    fine_params['double_step'] = False

    if convergence_check:
        a_final, fine_results = integrate_ensembles(*integration_args, **fine_params)
        _, coarse_results = integrate_ensembles(*integration_args, **coarse_params)
        results = calculate_step_error(
            coarse_results, fine_results,
            extrapolate=extrapolate, extrapolation_order=stepper_cls.extrapolation_order)
    else:
        a_final, results = integrate_ensembles(*integration_args, **coarse_params)

    a_final = a_final.reshape(trajectories, components, *spatial_dimensions)

    return a_final, results
