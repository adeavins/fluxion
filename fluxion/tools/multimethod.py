"""
An implementation of multimethods dispatched based on the argument types (Julia-like).

TODO: perhaps, it is worth replacing with https://github.com/mrocklin/multipledispatch
"""

import inspect
import itertools
from functools import lru_cache


__all__ = ['Multimethod', 'multimethod']


def get_annotations(func):
    sig = inspect.signature(func)
    has_nonpositional_params = any(
        param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in sig.parameters.values())
    if has_nonpositional_params:
        raise ValueError("Only functions with positional parameters are supported")

    argtypes = tuple(
        param.annotation if param.annotation is not param.empty else object
        for param in sig.parameters.values())

    return argtypes


def cmp_generality(reference_tp, tp):
    """
    Answers the question: how general is ``tp`` as compared to ``reference_tp``?
    """
    if tp == reference_tp:
        return 0

    if issubclass(reference_tp, tp):
        return reference_tp.__mro__.index(tp)

    if issubclass(tp, reference_tp):
        return -tp.__mro__.index(reference_tp)

    return None


def cmp_generalities(reference_types, types):
    return [cmp_generality(reference_tp, tp) for reference_tp, tp in zip(reference_types, types)]


def all_comparable(generality_list):
    return not any(generality is None for generality in generality_list)


def all_general(generality_list):
    # Assumes that ``all_comparable`` is ``True``
    return not any(generality < 0 for generality in generality_list)


def is_uncertain(generality_list):
    # Assumes that ``all_comparable`` is ``True``
    has_specific = False
    has_general = False
    for generality in generality_list:
        if generality > 0:
            if has_specific:
                return True
            has_general = True
        elif generality < 0:
            if has_general:
                return True
            has_specific = True

    return False


def disambiguator(reference_types, types, generality_list):
    # Assumes that ``all_comparable`` is ``True``
    return tuple(
        reference_tp if generality > 0 else tp
        for reference_tp, tp, generality in zip(reference_types, types, generality_list))


def argmin(iterable, key=lambda x: x):
    return min(enumerate(iterable), key=lambda x: key(x[1]))[0]


class Dispatcher:

    def __init__(self):
        self._methods = {}

    def _register(self, func, argtypes):

        nargs = len(argtypes)
        if nargs not in self._methods:
            self._methods[nargs] = {}

        methods = self._methods[nargs]

        if argtypes in methods:
            raise TypeError(
                "A method for the signature " + str(repr(argtypes)) + " is already registered")

        for m_argtypes in methods:

            method_generality = cmp_generalities(argtypes, m_argtypes)

            if not all_comparable(method_generality):
                continue

            if is_uncertain(method_generality):
                d_argtypes = disambiguator(argtypes, m_argtypes, method_generality)
                if d_argtypes not in methods:
                    raise TypeError((
                        "Ambiguous multimethods for signatures {sig1} and {sig2}; "
                        + "resolve by defining a method with the signature {d_sig} "
                        + "at least before the second one").format(
                        sig1=m_argtypes, sig2=argtypes, d_sig=d_argtypes))

        self._methods[nargs][argtypes] = func
        self._dispatch.cache_clear()

    def register(self, func):
        annotations = get_annotations(func)
        type_groups = []
        for annotation in annotations:
            if isinstance(annotation, type):
                type_groups.append((annotation,))
            elif isinstance(annotation, (tuple, list, set)):
                type_groups.append(annotation)
            else:
                raise ValueError("Unsupported annotation type: " + str(type(annotation)))

        for argtypes in itertools.product(*type_groups):
            self._register(func, argtypes)

    def dispatch(self, args):
        argtypes = tuple(type(arg) for arg in args)
        method = self._dispatch(argtypes)
        return method(*args)

    @lru_cache(maxsize=128)
    def _dispatch(self, argtypes):

        all_methods = self._methods[len(argtypes)]

        suitable_methods = []
        suitable_methods_generalities = []

        for m_argtypes in all_methods:
            generality_list = cmp_generalities(argtypes, m_argtypes)
            if all_comparable(generality_list) and all_general(generality_list):
                suitable_methods.append(all_methods[m_argtypes])
                suitable_methods_generalities.append(generality_list)

        if len(suitable_methods) == 0:
            raise TypeError("No suitable methods for types " + str(repr(argtypes)))

        return suitable_methods[argmin(suitable_methods_generalities)]


class Multimethod:

    def __init__(self):
        self._dispatcher = Dispatcher()

    def method(self, func):
        self._dispatcher.register(func)
        return self

    def __call__(self, *args):
        return self._dispatcher.dispatch(args)


def multimethod(func):
    """
    Implicit multimethod creation, mimicking Julia behavior.
    On creation, the function name is searched in the global namespace,
    and, if it is a multimethod, ``func`` will be registered in it,
    otherwise a new multimethod is created.

    .. warning::

        Objects in the inner scopes won't be picked up, and will be just silently replaced.
    """

    if not inspect.isfunction(func):
        raise TypeError("Only a function can be made a multimethod")

    if func.__name__ in func.__globals__:
        existing_obj = func.__globals__[func.__name__]
        if isinstance(existing_obj, Multimethod):
            existing_obj.method(func)
            return existing_obj
        else:
            raise TypeError(
                "An object with the name " + func.__name__ + " already exists in this namespace, "
                "and it is not a multimethod")
    else:
        mm = Multimethod()
        mm.method(func)
        return mm
