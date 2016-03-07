"""
Integrates an ODE

dy/dx = 2 x y^2

For y(x0) = y0 the solution is

y = 1 / (1 / y0 + x0^2 - x^2)
"""

from fluxion import *

x = PropagationDimension('x')
y = UnknownField('y', x, kind=REAL)

eq = Eq(diff(y, x), 2 * x * y**2)

results = integrate(eq, -0.5, 1, samplers=dict(y=(sample_field, linspace(1, 10, 100))))

plot(results)
