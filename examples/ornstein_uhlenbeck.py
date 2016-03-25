"""
An example of an ODE with integer dimensions:
a multi-dimensional Ornstein-Uhlenbeck process.

Currently, without the noise --- to be added later.

The equation is

    dx/dt = E - G x,

where `x` and `E` are vectors, and `G` is a square matrix.
The solution is

    x = (I - exp(-G t)) G^(-1) E + exp(-G t) x(0),

where `I` is the identity matrix, and `exp` is the matrix exponent.
"""

from fluxion import *

t = PropagationDimension('t')
i = TransverseIntegerDimension('component', 1, 3)
x = UnknownField('x', i, t, kind=COMPLEX)

E = as_field([1, 0.2, 1], i)
G = as_field([[0.1j, 0, 0], [0, 0.2, 0], [0.3j, 0, 0.3]], i, i)

eq = Eq(diff(x, t), E - G @ x)

x0 = (1, 2, 3)
results = integrate(eq, x0, 0)

plot(results)
