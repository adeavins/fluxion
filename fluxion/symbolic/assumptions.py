class FactExpr:

    def __not__(self):
        return Not(self)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


class Fact(FactExpr):

    def __init__(self, name):
        self.name = name

    def __rshift__(self, fact):
        assert type(fact) == Fact
        return Implies(self, fact)

    def __eq__(self, fact_expr):
        return Alias(self, fact_expr)

    def __xor__(self, fact):
        assert type(fact) == Fact
        return Exclusive(self, fact)


class Rule:
    pass


class Implies(Rule):

    def __init__(self, premise, conclusion):
        self.premise = premise
        self.conclusion = conclusion


class Alias(Rule):

    def __init__(self, alias, expr):
        self.alias = alias
        self.expr = expr


class Exclusive(Rule):

    def __init__(self, fact1, fact2):
        self.fact1 = fact1
        self.fact2 = fact2


class Not(FactExpr):

    def __int__(self, x):
        self.args = (x,)


class And(FactExpr):

    def __int__(self, x, y):
        self.args = (x, y)


class Or(FactExpr):

    def __int__(self, x, y):
        self.args = (x, y)



"""
    'integer        ->  rational',
    'rational       ->  real',
    'real           ->  complex',
    'imaginary      ->  complex',
    'complex        ->  commutative',

    'real           ==  negative | zero | positive',

    'negative       ==  nonpositive & nonzero',
    'positive       ==  nonnegative & nonzero',
    'zero           ==  nonnegative & nonpositive',

    'nonpositive    ==  real & !positive',
    'nonnegative    ==  real & !negative',

    'zero           ->  finite',

    'imaginary      ->  !real',

    'infinite       ->  !finite',
    'noninteger     ==  real & !integer',
    'nonzero        ==  real & !zero',
"""


positive = Fact('positive')
# ...
"""
rules = [
    positive >> real,
    positive ^ zero,
    zero >> integer,
    integer >> real,
    real >> complex,
    complex >> commutative,

    nonpositive == real and not positive,
    negative == not positive and not zero,
    nonnegative == real and not negative,
    imaginary == not real,
    nonzero == real and not zero,
]
"""

def _validate_assumptions(assumptions):




    implies('positive', 'real')
    exclusive('positive', 'zero')
    implies('zero', 'integer')
    implies('integer', 'real')
    implies('real', 'complex')

    alias('nonpositive', '!positive')
    alias('negative', '!positive && !zero')
    alias('nonnegative', '!negative')
    alias('imaginary', '!real')

    return assumptions



def only_one(*seq):
    return sum(bool(x) for x in seq) == 1


def validate_assumptions(assumptions):
    is_integer = assumptions.get('integer', None)
    is_real = assumptions.get('real', None)
    is_complex = assumptions.get('complex', None)
    #assert only_one(is_integer, is_real, is_complex)

    if is_integer:
        assert (is_real is None or is_real is True) and (is_complex is None or is_complex is True)
        return dict(integer=True, real=True, complex=True)

    if is_real:
        assert (is_complex is None or is_complex is True)
        return dict(integer=None, real=True, complex=True)

    if is_complex:
        return dict(integer=None, real=None, complex=True)


    """
    implies('integer', 'real')
    implies('real', 'complex')
    exclusive('real', 'imaginary')
    exclusive('positive', 'nonnegative')
    exclusive('negative', 'nonpositive')
    """

    return assumptions
