import numpy

import sympy
from sympy import Symbol, init_printing
from sympy.core.function import UndefinedFunction
from sympy.core.cache import cacheit

from functools import lru_cache, partial

init_printing()


class TransverseDimension(Symbol):

    def __new_stage2__(cls, name, start, stop, points):
        # Calling __xnew__ which is not cached (as opposed to __new__)
        obj = super(TransverseDimension, cls).__xnew__(cls, name, real=True)
        obj.params = (start, stop, points)
        obj.grid = numpy.linspace(start, stop, points, endpoint=False)
        return obj

    def __new__(cls, name, *args, **kwds):
        obj = TransverseDimension.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    # Conforming to the interface of Symbol
    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        return (Symbol._hashable_content(self), self.params)


class TransverseIntegerDimension(Symbol):

    def __new_stage2__(cls, name, start, stop):
        # Calling __xnew__ which is not cached (as opposed to __new__)
        obj = super(TransverseIntegerDimension, cls).__xnew__(cls, name, integer=True)
        obj.params = (start, stop)
        obj.grid = numpy.arange(start, stop + 1)
        return obj

    def __new__(cls, name, *args, **kwds):
        obj = TransverseIntegerDimension.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    # Conforming to the interface of Symbol
    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        return (Symbol._hashable_content(self), self.params)


class PropagationDimension(Symbol):

    def __new__(cls, name):
        obj = super(PropagationDimension, cls).__new__(cls, name, real=True)
        return obj


class __UnknownField(Symbol):

    def __new_stage2__(cls, name, *dimensions, complex=True):
        # Calling __xnew__ which is not cached (as opposed to __new__)
        obj = super(UnknownField, cls).__xnew__(cls, name, complex=complex)
        obj.params = (dimensions, complex)
        obj.dimensions = dimensions
        return obj

    def __new__(cls, name, *args, **kwds):
        obj = UnknownField.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    # Conforming to the interface of Symbol
    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        return (Symbol._hashable_content(self), self.params)

    def __call__(self, *args):
        return unknown_field_class(self.name, self.dimensions)(*args)

    def diff(self, *args):
        return unknown_field_class(self.name, self.dimensions)(*self.dimensions).diff(*args)


@cacheit
def unknown_field_class(name, dimensions):
    pass


class __UnknownFieldWithArgs(UndefinedFunction):

    def __new__(cls, name, *dimensions):
        obj = super(UnknownFieldWithArgs, cls).__new__(cls, name)
        print(obj, type(obj))
        obj.dimensions = dimensions
        return obj

    def __init__(self, *args):
        UndefinedFunction.__init__(self, *args)


class UnknownFieldFunction(sympy.Function):

    pass


def UnknownField(name, *dimensions):
    pass


def test_invariant(expr):
    print("Testing invariant for", srepr(expr))
    assert(expr.args == () or expr.func(*expr.args) == expr)


def variables(expr):
    if expr.func == UnknownField:
        var_list = expr.dimensions
    elif expr.func in (TransverseDimension, TransverseIntegerDimension):
        var_list = (expr,)
    else:
        var_list = ()

    found_vars = set(var_list)
    for arg in expr.args:
        found_vars |= variables(arg)

    return found_vars


def as_array(obj, *variables):
    pass


def as_field(obj, *variables):
    pass


if __name__ == '__main__':

    print("Creating functions")
    f = sympy.Function('f')
    g = sympy.Function('f')
    """
    a = Symbol('a')
    print(f is g)
    print(f(a) is g(a))
    print(type(f(a)).__mro__)
    print(f + 3 + 4)
    """
    exit()

    """
    Dimension:
    - carries additional attributes
    - behaves like Symbol in sympy algorithms, but the equality is checked
      based on the name _and_ a set of parameters
    - is printed like Symbol

    UnknownField:
    - carries additional attributes; in particular, a list of dependent Dimension objects
    - the equality is checked based on the name _and_ the set of parameters
    - in the exression, behaves as an unknown function depending on the specified dimension objects
      (in particular, can be differentiated over them)
    - it can be applied like a Function object; the resulting object still carries around
      the original object's set of parameters (dimensions and others)
    - during the application some arbitrary error-checking is executed
      (e.g. to check that integer dimensions are used for integer dimensions etc)
    - OPTIONAL: partial application is supported, e.g. phi(i, x, t) can be applied as
      phi(3 - i) which will be identical to phi(3 - i, x, t)
    - OPTIONAL: a field not applied to anything, or applied to its "native" dimensions is printed
      just like its name
    """


    Ngrid = 128
    L = 50

    i = TransverseIntegerDimension('i', 1, 2)
    x = TransverseDimension('x', 0, L, Ngrid)
    y = TransverseDimension('y', 0, L, Ngrid)
    t = PropagationDimension('t')

    psi = UnknownField('psi', i, x, t)
    phi = UnknownField('psi', i, x, t)

    """
    assert psi is phi
    assert psi(i,x,t) is phi(i,x,t)
    assert isinstance(phi, Symbol)
    assert isinstance(phi(i,x,t), AppliedUndef)
    """

    """
    # need to be able to:
    psi0 = as_field(x**2) # creates a 1D field psi0(x) = x**2
    psi0 = as_field(x**2, x, y) # creates a 2D field psi0(x, y) = x**2
    psi0 = as_field(1) # creates a 0D field?
    psi0 = as_field(1, x, y) # creates a 2D field psi0(x, y) = 1

    psi = field(x)
    psi0 = as_field(psi, x, y) # creates a 2D field psi0(x, y) = psi(x)

    arr = as_array(x**2) # creates a 1D array
    arr = as_array(x**2, x, y) # creates a 2D array (technically, can be achieved by numpy means)
    arr = as_array(psi0) # creates a 2D array
    arr = as_array(1, x, y) # creates a 2D array
    arr = as_array(1) # creates a 0D array
    """

