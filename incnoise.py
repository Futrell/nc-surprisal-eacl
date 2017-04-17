from math import log, exp
import itertools
import random
import functools
from collections import Counter

import rfutils
from pmonad import *

from math import log2

# "Control characters"
HALT = "!H"
START = "!S"
ERASED = "!E"
DEFAULT_NUM_SAMPLES = 10000

def buildup(lst, start=0, end=None):
    """ Given an iterable [a, b, ...] generate tuples [], [a], [a, b], ... 
    starting with the tuple of length `start` and ending with the list of length
    `end`. """
    for i in range(start, len(lst)+1 if end is None else end):
        yield tuple(lst[:i])

# take_alternating :: Bool x [a] -> [a]
def take_alternating(start, xs):
    include = start
    for x in xs:
        if include:
            yield x
        include = not include

# take_even :: [a] -> [a]
def take_even(xs):
    return take_alternating(True, xs)

# take_odd :: [a] -> [a]
def take_odd(xs):
    return take_alternating(False, xs)

# replicate :: [a] -> [a]            
def replicate(xs, n):
    for x in xs:
        for _ in range(n):
            yield x


####### Character-level noise functions ############

def switch_letter(letter):
    return {'A': 'B', 'B': 'A', 'a': 'b', 'b': 'a'}[letter]

def switch_case(letter):
    if letter.islower():
        return letter.upper()
    elif letter.isupper():
        return letter.lower()
    else:
        raise ValueError

@enumerator
def switching_noise(letter, p):
    yield switch_letter(letter), log(p)
    yield switch_case(letter), log(p)
    yield letter, log(1 - 2*p)

def sample_bit_noise(bit, p):
    if sample_flip(p):
        return str(1 - int(bit))
    else:
        return bit

# bit_noise :: Char x Float -> Enum Char
def enum_bit_noise(bit, p):
    return Enumeration.flip(p).bind(
        Enumeration.lift_ret(lambda b: str(1 - int(bit)) if b else bit)
    )

def enum_bit_noise_by_word(bits, p):
    bits_enum = Enumeration.mapM(lambda bit: enum_bit_noise(bit, p), bits)
    return Enumeration.lift("".join)(bits_enum)

# maybe_erase :: a x Float -> Enum (Maybe a)
def maybe_erase(x, p):
    return Enumeration.flip(p).bind(
        Enumeration.lift_ret(lambda b: ERASED if b else x)
    )

####### Sequence-level noise functions ############

# successive_noise :: [a] x ([a] -> Enum [a]) -> Enum [a]
@enumerator
def successive_noise(iterable, noise):
    @enumerator
    def apply_noise(acc, x):
        for noisy_new_acc, p in noise(acc + (x,)):
            yield noisy_new_acc, p
    return Enumeration.reduceM(apply_noise, iterable, initial=())

# successive_noise :: ([a] -> Enum [a]) -> ([a] -> Enum [a])
def successive_noise_by_symbol(noise, *a, **k):
    def noisy(xs):
        return successive_noise(
            xs,
            lambda xs: Enumeration.mapM(lambda x: noise(x, *a, **k), xs)
        )
    return noisy

def test_successive_noise():
    @enumerator
    def noise(x):
        yield "A(%s)" % x, log(3/4)
        yield "B(%s)" % x, log(1/4)
        
    noise_f = lambda acc: Enumeration.mapM(noise, acc)
    sequence = tuple('xy')
    assert dict(successive_noise(sequence, noise_f)) == {
        ('A(A(x))', 'A(y)'): -0.8630462173553427,
        ('A(A(x))', 'B(y)'): -1.9616585060234524,
        ('A(B(x))', 'A(y)'): -1.9616585060234524,
        ('A(B(x))', 'B(y)'): -3.0602707946915624,
        ('B(A(x))', 'A(y)'): -1.9616585060234524,
        ('B(A(x))', 'B(y)'): -3.060270794691562,
        ('B(B(x))', 'A(y)'): -3.0602707946915624,
        ('B(B(x))', 'B(y)'): -4.1588830833596715
    }

    @enumerator
    def noise(x):
        yield x+1, log(3/4)
        yield x-1, log(1/4)
        
    # now noise_f will have a reference to the new noise function
    
    sequence = (1, 2)
    assert dict(successive_noise(sequence, noise_f)) == {
        (-1, 1): -4.1588830833596715,
        (-1, 3): -3.0602707946915624,
        (1, 1): -2.367123614131617,
        (1, 3): -1.2685113254635072,
        (3, 1): -1.9616585060234524,
        (3, 3): -0.8630462173553427
    }


# successive_erasure_noise :: [a] x Float -> Enum [Maybe a]
def successive_erasure_noise(xs, p):
    return successive_noise(xs, lambda xs: sequence_erasure_noise(xs, p))

def sample_successive_erasure_noise(xs, p):
    def gen():
        for x in xs:
            if sample_flip(p):
                yield ERASED
            else:
                yield x
    return tuple(gen())

def approx_successive_erasure_noise(xs, p, num_samples=DEFAULT_NUM_SAMPLES):
    return enumeration_from_sampling_function(
        lambda: sample_successive_erasure_noise(xs, p),
        num_samples
    )

# successive_deletion_noise :: [a] x Float -> Enum [a]
def successive_deletion_noise(xs, p):
    return successive_noise(xs, lambda xs: sequence_deletion_noise(xs, p))

# successive_bit_noise :: [a] x Float -> Enum [a]
def successive_bit_noise(xs, p):
    return successive_noise(xs, lambda xs: sequence_bit_noise(xs, p))

def sample_successive_bit_noise(xs, p):
    so_far = ()
    for x in xs:
        so_far = sample_sequence_bit_noise(so_far + (x,), p)
    return so_far

def approx_successive_bit_noise(xs, p, num_samples=DEFAULT_NUM_SAMPLES):
    return enumeration_from_sampling_function(
        lambda: sample_successive_bit_noise(xs, p),
        num_samples
    )    
    
# sequence_erasure_noise :: [a] x Float -> Enum [Maybe a]
def sequence_erasure_noise(xs, p):
    return Enumeration.mapM(lambda x: maybe_erase(x, p), xs)

# sequence_deletion_noise :: [a] x Float -> Enum [a]
def sequence_deletion_noise(xs, p):
    @enumerator
    def maybe_delete(acc, x):
        return Enumeration.flip(p).bind(
            Enumeration.lift_ret(lambda b: acc if b else acc + (x,))
        )
    return Enumeration.reduceM(maybe_delete, xs, initial=())

# sequence_bit_noise :: [a] x Float -> Enum [a]
def sequence_bit_noise(xs, p):
    return Enumeration.mapM(lambda x: enum_bit_noise_by_word(x, p), xs)

def sample_sequence_bit_noise(xs, p):
    return tuple("".join(sample_bit_noise(c, p) for c in x) for x in xs)

def approx_sequence_bit_noise(xs, p, num_samples=DEFAULT_NUM_SAMPLES):
    return enumeration_from_sampling_function(
        lambda: sample_sequence_bit_noise(xs, p),
        num_samples
    )

# noisy_channel_prefix_tree :: Enum [a] x ([a] -> Enum [b]) -> PT [a] a
def noisy_channel_prefix_tree(lang, noise):
    assert isinstance(lang, Enumeration)
    mul = lang.field.mul
    npt = noisy_prefix_tree(lang, noise)
    def traverse_from(real_prefix, perceived_prefix, p_so_far):
        for value, p_value in npt[perceived_prefix]:
            yield (real_prefix, value), mul(p_so_far, p_value)
            if value != HALT:
                new_perceived_prefix = perceived_prefix + (value,)
                new_real_prefix = real_prefix + (value,)
                for noisy_prefix, p_noise in noise(new_perceived_prefix):
                    new_p_so_far = mul(mul(p_so_far, p_value), p_noise)
                    yield from traverse_from(
                        new_real_prefix,
                        noisy_prefix,
                        new_p_so_far
                    )
    return type(lang)(traverse_from((), (), lang.field.one)).conditional()

# lang_from_prefix_tree :: PT [a] a -> Enum [a]
def lang_from_prefix_tree(prefix_tree):
    def traverse_from(prefix, logp_prefix):
        for value, p_value in prefix_tree[prefix]:
            p = ring.mul(p_prefix, p_value)
            if value == HALT:
                yield prefix, p
            elif prefix + (value,) in prefix_tree:
                yield from traverse_from(prefix + (value,), p)
    ring = prefix_tree[rfutils.first(prefix_tree.keys())].field
    return traverse_from((), ring.one)

def test_noisy_channel_prefix_tree():
    lang = UniformEnumeration(['AA', 'Bb', 'aA', 'bb'])
    p = .1
    noise = lambda xs: successive_erasure_noise(xs, p)
    ncf = noisy_channel_prefix_tree(lang, noise)

    assert is_close(ncf[('A',)]['A'], log(19/20))
    assert is_close(ncf[('A',)]['b'], log(1/20))
    assert is_close(ncf[('B',)]['A'], log(1/20))
    assert is_close(ncf[('B',)]['b'], log(19/20))

def monadic_noisy_prefix_tree(lang, noise):
    # Amazingly, this does the same thing as noisy_prefix_tree!
    joint = lang >> (lambda s:
            certainly(tuple(s) + (HALT,)) >> (lambda s:
            uniform(buildup(s, start=1)) >> (lambda prefix:
            certainly((prefix[:-1], prefix[-1])) >> (lambda pair:
            noise(pair[0]) >> (lambda noisy_prefix:
            certainly((noisy_prefix, pair[-1])))))))
    return joint.conditional()

# noisy_prefix_tree :: Enum [a] x ([a] -> Enum [b]) -> PT [b] a
def noisy_prefix_tree(lang, noise):
    """ Convert a joint probability distribution into a noisy prefix tree
    probability distribution, where contexts have had a noise e.f. applied
    to them. The noise e.f. must operate over sequences. """
    assert isinstance(lang, Enumeration)
    d = {}
    add = lang.field.add
    mul = lang.field.mul
    for string, p in lang:
        string = list(string) + [HALT]
        prefixes = buildup(string, start=1)
        for prefix in prefixes:
            *context, x = prefix
            context = tuple(context)
            for noisy_context, p_noise in noise(context):
                if noisy_context in d:
                    if x in d[noisy_context]:
                        d[noisy_context][x] = add(
                            d[noisy_context][x],
                            mul(p, p_noise)
                        )
                    else:
                        d[noisy_context][x] = mul(p, p_noise)
                else:
                    d[noisy_context] = {x: mul(p, p_noise)}

    for prefix, prefix_distro in d.items():
        d[prefix] = type(lang)(prefix_distro).normalize()

    return d

def test_noisy_prefix_tree():
    lang = UniformEnumeration(['AA', 'Bb', 'aA', 'bb'])
    noise = lambda xs: successive_noise(
        xs,
        lambda xs: Enumeration.mapM(lambda x: switching_noise(x, .1), xs)
    )
    nf = noisy_prefix_tree(lang, noise)

    # p_C(A|A) = \frac{\sum_w p_L(A|w) p_N(A|w) p_L(w)}
    #                 {\sum_w          p_N(A|w) p_L(w)}

    # Say p_L(w) = 1/4,
    # p_L(A|A) = 1, p_L(A|a) = 1, p_L(A|B) = 0, p_L(A|b) = 0
    # p_N(w'|w) = 1/10 where w' != w
    # p_N(w|w) = 8/10

    # p_C(A|A) = (1 * 8/10 * 1/4 + 1 * 1/10 * 1/4) / (1/4)
    # = 9/10
    assert all(is_close(logp, log(1/4)) for logp in nf[()].dict.values())
    assert is_close(nf[('A',)]['A'], log(9/10))
    assert is_close(nf[('A',)]['b'], log(1/10))
    assert is_close(nf[('A', 'A')][HALT], log(1))

    p = .1
    noise = lambda xs: sequence_erasure_noise(xs, p)
    nf = noisy_prefix_tree(lang, noise)
    assert all(logp == log(1/4) for logp in nf[()].dict.values())
    assert is_close(nf[(ERASED,)]['A'], log(1/2))
    assert is_close(nf[(ERASED,)]['b'], log(1/2))

def is_close(x, y, eps=10**-5):
    return abs(x-y) < eps

# prefix_tree :: Enum [a] -> PT [a] a
def prefix_tree(lang):
    """ Convert a joint probability distribution into a prefix tree probability 
    distribution. """
    assert isinstance(lang, Enumeration)
    add = lang.field.add
    d = {}
    for string, p in lang:
        string = list(string) + [HALT]
        prefixes = buildup(string, start=1)
        for prefix in prefixes:
            *context, x = prefix
            context = tuple(context)
            if context in d:
                if x in context:
                    d[context][x] = add(d[context][x], p)
                else:
                    d[context][x] = p
            else:
                d[context] = {x: p}
        
    # Normalize the prefixes
    for prefix, prefix_distro in d.items():
        d[prefix] = type(lang)(prefix_distro).normalize()

    return d

def test_prefix_tree():
    d = prefix_tree(Enumeration([
        ('abc', log(.4)),
        ('acb', log(.4)),
        ('acd', log(.2))
    ]))
    d_dict = {k:dict(v) for k, v in d.items()}
    assert d_dict == {
        (): {'a': 0.0},
        ('a',): {'b': -0.4054651081081645, 'c': -1.0986122886681098},
        ('a', 'b'): {'c': 0.0},
        ('a', 'b', 'c'): {HALT: 0.0},
        ('a', 'c'): {'b': -0.4054651081081645, 'd': -1.0986122886681098},
        ('a', 'c', 'b'): {HALT: 0.0},
        ('a', 'c', 'd'): {HALT: 0.0}
    }


# cost :: Float -> Float
def cost(p):
    return -log2(p)

# internal_string_cost :: Enum [a] x [a] x ([a] -> Enum [a]) -> Float
def internal_string_cost(lang, string, noise):
    assert lang.field == log_space
    noise = rfutils.memoize(noise)
    pt = noisy_prefix_tree(lang, noise)
    return sum(internal_symbol_costs(pt, string, noise))

# internal_symbol_costs :: PT [b] a x [a] x ([a] -> Enum [b]) -> [Float]
def internal_symbol_costs(pt, string, noise):
    prefixes = buildup(list(string) + [HALT], start=1)
    for prefix in prefixes:
        *context, x = prefix
        context = tuple(context)
        # TODO this requires that the noise function enumerate exactly
        # the same things as in previous calls --
        # as a stopgap, try memoizing the random enumerator?
        for noisy_context, logp_noise in noise(context):
            yield exp(logp_noise) * cost(exp(pt[tuple(noisy_context)][x]))

# internal_lang_cost :: Enum [a] x ([a] -> Enum [b]) -> Float
def internal_lang_cost(lang, noise):
    """ The expected average surprisal of each word given noisy representation 
    of the prefix. """
    noise = rfutils.memoize(noise)
    pt = noisy_prefix_tree(lang, noise)
    return lang.expectation(lambda s: sum(internal_symbol_costs(pt, s, noise)))

def verbose_internal_lang_cost(lang, noise):
    assert lang.field == log_space
    pt = noisy_prefix_tree(lang, noise)
    def string_costs():
        for s, logp_s in lang:
            print("p_L(%s) = %s" % (s, exp(logp_s)))
            s = list(s)
            the_cost = 0
            prefixes = buildup(list(s) + [HALT], start=1)
            for prefix in prefixes:
                *context, x = prefix
                for noisy_context, logp_noise in noise(context):
                    part_cost = cost(exp(pt[tuple(noisy_context)][x]))
                    print("p_N(%s | %s) = %s" % (noisy_context, context, exp(logp_noise)))
                    print("p(%s | %s) = %s" % (x, noisy_context, 2**(-part_cost)))
                    the_cost += exp(logp_s) * exp(logp_noise) * part_cost
            yield the_cost
    return sum(string_costs())

# external_symbol_costs :: PT [a] a x [a] x ([a] -> Enum [b]) -> [Float]
def external_symbol_costs(pt, string, noise):
    prefixes = buildup(list(string) + [HALT], start=1)
    for prefix in prefixes:
        *context, x = prefix
        yield cost(exp(pt[tuple(context)][x]))

# external_lang_cost :: Enum a x (a -> Enum b) -> Float
def external_lang_cost(lang, noise):
    """ The expected average surprisal of each word as generated by a noisy
    producer. """
    assert lang.field == log_space
    pt = noisy_channel_prefix_tree(lang, noise)
    return lang.expectation(lambda s: sum(external_symbol_costs(pt, s, noise)))

def sample_flip(p):
    return random.random() < p

p = .1
enoise = lambda xs: successive_erasure_noise(xs, p)
dnoise = lambda xs: successive_deletion_noise(xs, p)
snoise = successive_noise_by_symbol(switching_noise, p)
bnoise = lambda xs: successive_bit_noise(xs, p)

if __name__ == '__main__':
    import nose
    nose.runmodule()
