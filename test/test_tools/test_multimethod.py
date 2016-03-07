import pytest

from fluxion.tools.multimethod import Multimethod, multimethod


# Tests for the implicit method creation

@multimethod
def existing_method(a, b):
    return 0

@multimethod
def existing_method(a: int, b):
    return 1

def test_implicit_Multimethod():
    assert existing_method(None, None) == 0
    assert existing_method(1, None) == 1


def test_implicit_supported_types():

    class Dummy: pass

    with pytest.raises(TypeError):
        m = multimethod(Dummy)


# Multimethod() cannot inspect current scope, only the globals
# so we're defining this global function for the test below
def nonmethod():
    pass

def test_implicit_existing_nonmethod():

    def f():
        pass

    f.__name__ = 'nonmethod'

    with pytest.raises(TypeError):
        m = multimethod(f)


# Class hierarchy for tests

class A: pass

class B(A): pass

class C(A): pass

class D(C): pass

class E(B, D): pass


def test_no_suitable_methods():

    mm = Multimethod()

    @mm.method
    def func(a: B, b): return 0

    # There is no method defined for the first argument of type C or any of its bases
    with pytest.raises(TypeError):
        func(C(), A())


def test_same_signature():

    mm = Multimethod()

    @mm.method
    def func(a: B, b): return 0

    def func2(a: B, b): return 0

    # func2 has the same argument types as the already registered func
    with pytest.raises(TypeError):
        mm.method(func2)


def test_method_outside_hierarchy():

    # Check that if there are registered methods with types outside of the hierarchy
    # the passed arguments belong to, they are correctly ignored

    mm = Multimethod()

    @mm.method
    def func(a: A): return 0

    @mm.method
    def func(a: B): return 1

    @mm.method
    def func(a: C): return 2

    # In this case, C from the method above is neither a subclass nor a superclass of B
    assert mm(B()) == 1


def test_ambiguity_detected():

    mm = Multimethod()

    @mm.method
    def func(a: A, b: B): return 0

    def func2(a: B, b: A): return 1

    # func and func2 have ambiguous signatures: which one should be called for mm(B(), B())?
    with pytest.raises(TypeError):
        mm.method(func2)


    # Change order of the method declaration
    # in order to cover a branch in the ambiguity detection function

    mm = Multimethod()

    @mm.method
    def func(a: B, b: A): return 0

    def func2(a: A, b: B): return 1

    # func and func2 have ambiguous signatures: which one should be called for mm(B(), B())?
    with pytest.raises(TypeError):
        mm.method(func2)


def test_ambiguity_resolved():

    mm = Multimethod()

    @mm.method
    def func(a: A, b: B): return 0

    # In order to resolve the further ambiguity, we're defining a specific method
    @mm.method
    def func(a: B, b: B): return 1

    # Now this method can be registered
    @mm.method
    def func(a: B, b: A): return 2

    assert func(A(), B()) == 0
    assert func(B(), B()) == 1
    assert func(B(), A()) == 2


def test_nonpositional_arguments():

    mm = Multimethod()

    def func(a, *args): return 0

    with pytest.raises(ValueError):
        mm.method(func)

    def func(a, **kwds): return 0

    with pytest.raises(ValueError):
        mm.method(func)


def test_dispatch_order():

    mm = Multimethod()

    @mm.method
    def func(a: B): return 0

    @mm.method
    def func(a: D): return 1

    # B goes before D in the MRO for E,
    # so the method registered for B is more specific,
    # and, therefore, will get called
    assert func(E()) == 0
