""" Probability monad """
import random
from collections import Counter, namedtuple
from math import log, exp
import operator
import functools
import itertools

import sympy
import rfutils
from rfutils.compat import *

INF = float('inf')
_SENTINEL = object()

def keep_calling_forever(f):
    return iter(f, _SENTINEL)

# safelog : Float -> Float
def safelog(x):
    try:
        return log(x)
    except ValueError:
        return -INF

# logaddexp : Float x Float -> Float
def logaddexp(one, two):
    return safelog(exp(one) + exp(two))

# logsumexp : [Float] -> Float
def logsumexp(xs):
    return safelog(sum(map(exp, xs)))

# reduce_by_key : (a x a -> a) x [(b, a)] -> {b -> a}
def reduce_by_key(f, keys_and_values):
    d = {}
    for k, v in keys_and_values:
        if k in d:
            d[k] = f(d[k], v)
        else:
            d[k] = v
    return d

def lazy_product_map(f, xs):
    """ equivalent to itertools.product(*map(f, xs)), but does not hold the values
    resulting from map(f, xs) in memory. xs must be a sequence. """
    if not xs:
        yield []
    else:
        x = xs[0]
        for result in f(x):
            for rest in lazy_product_map(f, xs[1:]):
                yield [result] + rest

class Monad(object):
    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.values) + ")"

    def __rshift__(self, f):
        return self.bind(f)

    def __add__(self, bindee_without_arg):
        return self.bind(lambda _: bindee_without_arg())

    # lift : (a -> b) -> (m a -> m b)
    @classmethod
    def lift(cls, f):
        @functools.wraps(f)
        def wrapper(a):
            return a.bind(cls.lift_ret(f))
        return wrapper

    # lift_ret : (a -> b) -> a -> m b
    @classmethod
    def lift_ret(cls, f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            return cls.ret(f(*a, **k))
        return wrapper

    @property
    def mzero(self):
        return type(self)(self.zero)

    @classmethod
    def guard(cls, truth):
        if truth:
            return cls.ret(_SENTINEL) # irrelevant value
        else:
            return cls(cls.zero) # construct mzero

class Amb(Monad):
    def __init__(self, values):
        self.values = values

    zero = []

    def sample(self):
        return next(iter(self))

    def bind(self, f):
        return Amb(rfutils.flatmap(f, self.values))

    @classmethod
    def ret(cls, x):
        return cls([x])

    def __iter__(self):
        return iter(self.values)

    # mapM : (a -> Amb b) x [a] -> Amb [b]
    @classmethod
    def mapM(cls, f, *xss):
        return Amb(itertools.product(*map(f, *xss)))

    # filterM : (a -> Amb Bool) x [a] -> Amb [a]
    @classmethod
    def filterM(cls, f, xs):
        return cls(itertools.compress(xs, mask) for mask in cls.mapM(f, xs))

    # reduceM : (a x a -> Amb a) x [a] -> Amb [a]
    @classmethod
    def reduceM(cls, f, xs, initial=None):
        def do_it(acc, xs):
            if not xs:
                yield acc
            else:
                x = xs[0]
                xs = xs[1:]
                for new_acc in nf(acc, x):
                    for res in do_it(new_acc, xs):
                        yield res
        xs = tuple(xs)
        if initial is None:
            return cls(do_it(xs[0], xs[1:]))
        else:
            return cls(do_it(initial, xs))

    def conditional(self, f=None, normalized=True):
        if f is None:
            f = lambda x: x

        class CDict(dict):
            def __missing__(d, key):
                samples = (y for x, y in map(f, self.values) if x == key)
                d[key] = Amb(samples)
                return d[key]

        return CDict()

def Samples(rf):
    return Amb(keep_calling_forever(rf))

Field = namedtuple('Field', ['add', 'sum', 'mul', 'div', 'zero', 'one'])
p_space = Field(operator.add, sum, operator.mul, operator.truediv, 0, 1)
log_space = Field(logaddexp, logsumexp, operator.add, operator.sub, -INF, 0)

class Enumeration(Monad):
    def __init__(self,
                 values,
                 marginalized=False,
                 normalized=False):
        self.marginalized = marginalized
        self.normalized = normalized
        self.values = values
        if isinstance(values, dict):
            self.marginalized = True
            self.values = values.items()
            self._dict = values
        else:
            self.values = values
            self._dict = None

    field = log_space
    zero = []

    def bind(self, f):
        mul = self.field.mul
        def gen():
            for x, p_x in self.values:
                for y, p_y in f(x):
                    yield y, mul(p_y, p_x)
        return type(self)(gen()).marginalize().normalize()

    # return : a -> Enum a
    @classmethod
    def ret(cls, x):
        return cls(
            [(x, cls.field.one)],
            normalized=True,
            marginalized=True,
        )
    
    def marginalize(self):
        if self.marginalized:
            return self
        else:
            # add together probabilities of equal values
            result = reduce_by_key(self.field.add, self.values)
            # remove zero probability values
            zero = self.field.zero
            result = {k:v for k, v in result.items() if v != zero}
            return type(self)(
                result,
                marginalized=True,
                normalized=self.normalized,
            )

    def normalize(self):
        if self.normalized:
            return self
        else:
            enumeration = list(self)
            Z = self.field.sum(p for _, p in enumeration)
            div = self.field.div
            result = [(thing, div(p, Z)) for thing, p in enumeration]
            return type(self)(
                result,
                marginalized=self.marginalized,
                normalized=True,
            )

    def __iter__(self):
        return iter(self.values)

    @property
    def dict(self):
        if self._dict:
            return self._dict
        else:
            self._dict = dict(self.values)
            return self._dict

    def __getitem__(self, key):
        return self.dict[key]

    @classmethod
    def mapM(cls, ef, *xss):
        mul = cls.field.mul
        one = cls.field.one
        def gen():
            for sequence in itertools.product(*map(ef, *xss)):
                seq = []
                p = one
                for thing, p_thing in sequence:
                    seq.append(thing)
                    p = mul(p, p_thing)
                yield tuple(seq), p
        return cls(gen()).marginalize().normalize()

    @classmethod
    def reduceM(cls, ef, xs, initial=None):
        mul = cls.field.mul
        one = cls.field.one
        def do_it(acc, xs):
            if not xs:
                yield (acc, one)
            else:
                the_car = xs[0]
                the_cdr = xs[1:]
                for new_acc, p in ef(acc, the_car):
                    for res, p_res in do_it(new_acc, the_cdr):
                        yield res, mul(p, p_res)
        xs = tuple(xs)
        if initial is None:
            result = do_it(xs[0], xs[1:])
        else:
            result = do_it(initial, xs)
        return cls(result).marginalize().normalize()

    def expectation(self, f):
        return sum(f(v)*exp(lp) for v, lp in self.values)

    def entropy(self):
        return -sum(exp(logp)*logp for _, logp in self.normalize()) / log(2)

    def conditional(self, f=None, normalized=True):
        if f is None:
            f = lambda x: x

        add = self.field.add
        d = {}
        for value, p in self.values:
            condition, outcome = f(value)
            if condition in d:
                if outcome in d[condition]:
                    d[condition][outcome] = add(d[condition][outcome], p)
                else:
                    d[condition][outcome] = p
            else:
                d[condition] = {outcome: p}
        cls = type(self)
        if normalized:
            return {
                k : cls(v).normalize()
                for k, v in d.items()
            }
        else:
            return {k: cls(v) for k, v in d.items()}

    @classmethod
    def flip(cls, p):
        def gen():
            if p > 0:
                yield True, log(p)
            if p < 1:
                yield False, log(1-p)
        return cls(gen(), marginalized=True, normalized=True)

        
class PSpaceEnumeration(Enumeration):
    field = p_space

    @classmethod
    def flip(cls, p):
        def gen():
            yield True, p
            yield False, 1 - p
        return cls(gen(), marginalized=True, normalized=True)


class SymbolicEnumeration(PSpaceEnumeration):

    def marginalize(self):
        result = super().marginalize()
        new_result = {k:sympy.simplify(v) for k, v in result.values}
        return type(result)(
            new_result,
            marginalized=True,
            normalized=result.normalized
        )

def UniformEnumeration(xs):
    xs = list(xs)
    N = len(xs)
    return Enumeration([(x, -log(N)) for x in xs])

def UniformSamples(xs):
    return Samples(lambda: random.choice(xs))

def enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return Enumeration(f(*a, **k))
    return wrapper

def pspace_enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return PSpaceEnumeration(f(*a, **k))
    return wrapper

def uniform_enumerator(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return UniformEnumeration(f(*a, **k))
    return wrapper

uniform = uniform_enumerator
deterministic = Enumeration.lift_ret
certainly = Enumeration.ret

def sampler(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        return Samples(lambda: f(*a, **k))
    return wrapper

def enumeration_from_samples(samples, num_samples):
    counts = Counter(itertools.islice(samples, None, num_samples))
    return Enumeration((k, log(v)) for k, v in counts.items()).normalize()

def enumeration_from_sampling_function(f, num_samples):
    samples = iter(f, _SENTINEL)
    return enumeration_from_samples(samples, num_samples)

def approx_enumerator(num_samples):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            sample_f = lambda: f(*a, **k)
            return enumeration_from_sampling_function(sample_f, num_samples)
        return wrapper
    return decorator

# enum_flip :: Float -> Enum Bool
@enumerator
def enum_flip(p):
    if p > 0:
        yield True, log(p)
    if p < 1:
        yield False, log(1-p)

@pspace_enumerator
def pspace_flip(p):
    if p > 0:
        yield True, p
    elif p < 1:
        yield False, 1 - p
