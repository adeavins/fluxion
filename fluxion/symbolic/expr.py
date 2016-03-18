import numpy

from .assumptions import validate_assumptions

"""
Conventions for any Expr subclass:
* all Expr objects are immutable
* if len(args) > 0:
    - obj == eval(repr(obj))
    - obj == eval(as_tree(obj))
    - obj == type(obj)(*obj.args)
    - expr1 == expr2 if and only if type(expr1) is type(expr2) and expr1.args == expr2.args
    - hash is calculated based on the class
* if len(args) == 0:
    - it is equivalent to being a subclass of ExprLeaf
    - expr1 == expr2 if and only if type(expr1) is type(expr2) and expr1.content == expr2.content
"""

class Expr:

    def __add__(self, other):
        return Add(self, as_expr(other))

    def __radd__(self, other):
        return Add(as_expr(other), self)

    def __sub__(self, other):
        return Sub(self, as_expr(other))

    def __rsub__(self, other):
        return Sub(as_expr(other), self)

    def __mul__(self, other):
        return Mul(self, as_expr(other))

    def __rmul__(self, other):
        return Mul(as_expr(other), self)

    def __truediv__(self, other):
        return Div(self, as_expr(other))

    def __rtruediv__(self, other):
        return Div(as_expr(other), self)

    def __pow__(self, other):
        return Pow(self, as_expr(other))

    def __rpow__(self, other):
        return Pow(as_expr(other), self)

    def __call__(self, *exprs):
        return Apply(self, *tuple(as_expr(expr) for expr in exprs))


class ExprNode(Expr):

    def __init__(self, *args):
        super(ExprNode, self).__init__()
        self.args = args

    def __repr__(self):
        return type(self).__name__ + repr(self.args)


class Add(ExprNode):
    pass


class Sub(ExprNode):
    pass


class Mul(ExprNode):
    pass


class Div(ExprNode):
    pass


class Pow(ExprNode):
    pass


class Differential(ExprNode):
    pass


class Eq(ExprNode):
    pass


def diff(x, y):
    return Differential(x, y)


def as_expr(obj):
    if isinstance(obj, Expr):
        return obj

    if isinstance(obj, int):
        return Integer(obj)

    if isinstance(obj, float):
        return Float(obj)

    if isinstance(obj, complex):
        return Complex(obj)

    raise NotImplementedError(
        "Can't wrap an object of type " + str(type(obj)) + " in an expression")


class ExprLeaf(Expr):

    def __init__(self):
        super(ExprLeaf, self).__init__()

    def __eq__(self, other):
        if self is other:
            return True

        return type(self) == type(other) and self._canonical_args() == other._canonical_args()

    def __hash__(self):
        return hash((type(self), self._canonical_args()))

    def __repr__(self):
        args, kwds = self._canonical_args()

        if args is not None:
            arg_str = ', '.join(repr(arg) for arg in args)
        else:
            arg_str = ''

        if kwds is not None:
            kwds_str = ', '.join(key + '=' + repr(val) for key, val in kwds.items())
        else:
            kwds_str = ''

        return (
            type(self).__name__ + '('
            + arg_str
            + (', ' + kwds_str if kwds_str else '')
            + ')')


class TypedExprLeaf(ExprLeaf):

    def __init__(self, **assumptions):
        super(TypedExprLeaf, self).__init__()
        self.__assumptions = validate_assumptions(assumptions)

    def assumptions(self):
        return self.__assumptions

    def _canonical_args(self):
        return None, self.__assumptions


class Symbol(TypedExprLeaf):

    def __init__(self, name, **assumptions):
        super(Symbol, self).__init__(**assumptions)
        self.__name = name
        self.__assumptions = self.assumptions()

    def _canonical_args(self):
        return (self.__name,), self.__assumptions


class Function(ExprLeaf):

    def __init__(self, name, nargs):
        super(Function, self).__init__()
        self.__name = name
        self.__nargs = nargs

    def _canonical_args(self):
        return (self.__name, self.__nargs), None

    def propagate_assumptions(self, *args):
        pass

    def evaluate(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        assert len(args) == self.__nargs
        # and, probably, other checks
        return Apply(self, *args)


class Apply(ExprNode):
    pass


class Abs(Function):
    def __init__(self):
        super().__init__('abs', 1)

    def evaluate(self, x):
        return numpy.abs(x)


def abs(x):
    return Apply(Abs(), x)


class Cosh(Function):

    def __init__(self):
        super().__init__('abs', 1)

    def evaluate(self, x):
        return numpy.cosh(x)


def cosh(x):
    return Apply(Cosh(), x)


class Scalar(TypedExprLeaf):

    def __init__(self, value, **assumptions):
        super(Scalar, self).__init__(**assumptions)
        self.value = value

    def _canonical_args(self):
        return (self.value,), self.assumptions()


class Integer(Scalar):

    def __init__(self, value):
        super(Integer, self).__init__(value, integer=True)

    def _canonical_args(self):
        return (self.value,), None


class Real(Scalar):

    def __init__(self, value):
        super(Real, self).__init__(value, real=True)

    def _canonical_args(self):
        return (self.value,), None


class Complex(Scalar):

    def __init__(self, value):
        super(Complex, self).__init__(value, complex=True)

    def _canonical_args(self):
        return (self.value,), None


ZERO = Integer(0)

