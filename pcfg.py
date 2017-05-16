""" Probabilistic context-free rewriting systems in the probability monad """
from collections import namedtuple, Counter
from math import log, exp
import functools
import operator

import rfutils
import pyrsistent as pyr

from pmonad import *

Rule = namedtuple('Rule', ['lhs', 'rhs'])
concatenate = operator.add

def make_pcfg(monad, rules, start='S', combine=concatenate):
    rewrites = process_rules(monad, rules)
    return PCFG(monad, rewrites, start, monad.lift_ret(combine))

def make_bounded_pcfg(monad, rules, bound, start='S', combine=concatenate):
    rewrites = process_rules(monad, rules)
    return BoundedPCFG(monad, rewrites, start, monad.lift_ret(combine), bound)

def process_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        if rule.lhs in d:
            d[rule.lhs].append((rule.rhs, prob))
        else:
            d[rule.lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

# PCFG : m x (a -> m [a]) x a x ([a] x [a] -> m [a])
class PCFG(object):
    def __init__(self, monad, rewrites, start, combine):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.combine = combine

    # rewrite_nonterminal : a -> Enum [a]
    def rewrite_nonterminal(self, symbol):
        return self.rewrites[symbol]

    # is_nonterminal : a -> Bool
    def is_nonterminal(self, symbol):
        return symbol in self.rewrites

    # rewrite_symbol : a -> Enum [a]
    def rewrite_symbol(self, symbol):
        if self.is_nonterminal(symbol):
            return self.rewrite_nonterminal(symbol) >> self.expand_string
        else:
            return self.monad.ret((symbol,))

    # expand_string : [a] -> Enum [a]
    def expand_string(self, string):
        return self.monad.mapM(self.rewrite_symbol, string) >> (lambda symbols:
            self.monad.reduceM(self.combine, symbols)
        )

    def distribution(self):
        return self.rewrite_symbol(self.start)

class BoundedPCFG(PCFG):
    def __init__(self, monad, rewrites, start, combine, bound):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.combine = combine
        self.bound = bound

    def rewrite_symbol(self, symbol, history):
        if self.is_nonterminal(symbol):
            result = self.rewrite_nonterminal(symbol)
            new_history = history.add(symbol)
            bound_ok = new_history.count(symbol) <= self.bound + 1
            return self.monad.guard(bound_ok) >> (
                lambda _: self.rewrite_nonterminal(symbol) >> (
                lambda s: self.expand_string(s, new_history)))
        else:
            return self.monad.ret((symbol,))

    def expand_string(self, string, history):
        result = self.monad.mapM(lambda s: self.rewrite_symbol(s, history), string)
        return result.bind(lambda s: self.monad.reduceM(self.combine, s, initial=()))

    def distribution(self):
        return self.rewrite_symbol(self.start, pyr.pbag([]))

def test_pcfg():
    from math import log, exp
    r1 = Rule('S', ('NP', 'VP'))
    r2 = Rule('NP', ('D', 'N'))
    r3 = Rule('VP', ('V', 'NP'))
    r4 = Rule('VP', ('V',))
    rules = [(r1, 0), (r2, 0), (r3, log(.25)), (r4, log(.75))]
    pcfg = make_pcfg(Enumeration, rules)
    enum = pcfg.distribution()
    assert enum.dict[('D', 'N', 'V')] == log(.75)
    assert enum.dict[('D', 'N', 'V', 'D', 'N')] == log(.25)
    assert sum(map(exp, enum.dict.values())) == 1

def test_bounded_pcfg():
    from math import log, exp
    r1 = Rule('S', ('a', 'S', 'b'))
    r2 = Rule('S', ())
    rules = [(r1, log(1/2)), (r2, log(1/2))]
    
    pcfg = make_bounded_pcfg(Enumeration, rules, 1)
    enum = pcfg.distribution()
    assert enum.dict[('a', 'b')] == log(1/2)
    assert enum.dict[()] == log(1/2)

    pcfg = make_bounded_pcfg(Enumeration, rules, 2)
    enum = pcfg.distribution()
    assert enum.dict[()] == log(1/2)
    assert enum.dict[('a', 'b')] == log(1/4)
    assert enum.dict[('a', 'a', 'b', 'b')] == log(1/4)

    pcfg = make_bounded_pcfg(Enumeration, rules, 3)
    enum = pcfg.distribution()
    assert enum.dict[()] == log(1/2)
    assert enum.dict[('a', 'b')] == log(1/4)
    assert enum.dict[('a', 'a', 'b', 'b')] == log(1/8)
    assert enum.dict[('a', 'a', 'a', 'b', 'b', 'b')] == log(1/8)
    
def _dont_test_pmcfg():
    from math import log
    r1 = Rule('S', (('NP', 'VP'),))
    r2 = Rule('NP', (('D', 'N'),))
    r3 = Rule('NPR', (('D', 'N'), ('RP',)))
    r4 = Rule('VP', (('V',),))
    r5 = Rule('S', (('NPR', 'VP', 'NPR0'),))
    r6 = Rule('D', (('the',),))
    r7 = Rule('D', (('a',),))
    r8 = Rule('N', (('cat',),))
    r9 = Rule('N', (('dog',),))
    r10 = Rule('V', (('jumped',),))
    r11 = Rule('V', (('cried',),))
    r12 = Rule('RP', (('that I saw yesterday',),))
    r13 = Rule('RP', (('that belongs to Bob',),))

    rules = [
        (r1, log(3/4)),
        (r2, 0),
        (r3, 0),
        (r4, 0),
        (r5, log(1/4)),
        (r6, log(1/2)),
        (r7, log(1/2)),
        (r8, log(1/2)),
        (r9, log(1/2)),
        (r10, log(1/2)),
        (r11, log(1/2)),
        (r12, log(1/3)),
        (r13, log(2/3))
    ]


if __name__ == '__main__':
    import nose
    nose.runmodule()
