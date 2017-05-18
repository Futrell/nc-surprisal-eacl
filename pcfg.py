""" Probabilistic context-free rewriting systems in the probability monad """
from collections import namedtuple, Counter
from math import log, exp
import functools
import operator

import rfutils
import pyrsistent as pyr

from pmonad import *

Rule = namedtuple('Rule', ['lhs', 'rhs'])
def concatenate(sequences):
    return sum(sequences, ())

def make_pcfg(monad, rules, start='S'):
    rewrites = process_pcfg_rules(monad, rules)
    return PCFG(monad, rewrites, start)

def make_bounded_pcfg(monad, rules, bound, start='S'):
    rewrites = process_pcfg_rules(monad, rules)
    return BoundedPCFG(monad, rewrites, start, bound)

def process_pcfg_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        if rule.lhs in d:
            d[rule.lhs].append((rule.rhs, prob))
        else:
            d[rule.lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

def process_pcfrs_rules(monad, rules):
    d = {}
    for rule, prob in rules:
        new_lhs = (rule.lhs, len(rule.rhs))
        if rule.lhs in d:
            d[new_lhs].append((rule.rhs, prob))
        else:
            d[new_lhs] = [(rule.rhs, prob)]
    for k, v in d.items():
        d[k] = monad(v).normalize()
    return d

def make_pcfrs(monad, rules, start='S'):
    rewrites = process_pcfrs_rules(monad, rules)
    return PCFRS(monad, rewrites, start)

# PCFG : m x (a -> m [a]) x a x ([a] x [a] -> m [a])
class PCFG(object):
    def __init__(self, monad, rewrites, start):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.combine = self.monad.lift_ret(concatenate)

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
        """ Expand a string of symbols into distributions over rewrites,
        then combine the rewrites into all the possible resulting strings using
        the (possibly probabilitic) combination function. """
        return self.monad.mapM(self.rewrite_symbol, string) >> self.combine

    def distribution(self):
        return self.rewrite_symbol(self.start)

class BoundedPCFG(PCFG):
    """ PCFRS where a symbol can only be rewritten n times recursively. """
    def __init__(self, monad, rewrites, start, bound):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start
        self.combine = self.monad.lift_ret(concatenate)
        self.bound = bound

    def rewrite_symbol(self, symbol, history):
        if self.is_nonterminal(symbol):
            return self.monad.guard(history.count(symbol) <= self.bound) >> (
                lambda _: self.rewrite_nonterminal(symbol) >> (
                lambda s: self.expand_string(s, history.add(symbol))))
        else:
            return self.monad.ret((symbol,))

    def expand_string(self, string, history):
        rewrite = lambda s: self.rewrite_symbol(s, history)
        return self.monad.mapM(rewrite, string) >> self.combine

    def distribution(self):
        return self.rewrite_symbol(self.start, pyr.pbag([]))

def process_indexed_string(string):
    symbols = []
    part_of = []
    seen = {}
    for i, part in enumerate(string):
        if isinstance(part, str):
            symbols.append((part, 1))
            part_of.append(i)
        else:
            symbol, index, num_blocks = part
            symbols.append(symbol)
            if (symbol, index) in seen:
                part_of.append(seen[symbol, index])
            else:
                part_of.append(i)
                seen[symbol, index] = i
    return symbol, part_of
            
def put_into_indices(self, symbols, indices):
    seen = Counter()
    def gen():
        for index in indices:
            yield symbols[index][seen[index]]
            seen[index] += 1
    return tuple(gen())

class PCFRS(PCFG):
    # rules have format (symbol, num_blocks) -> (blocks)
    def __init__(self, monad, rewrites, start):
        self.monad = monad
        self.rewrites = rewrites
        self.start = start

    def expand_string(self, string):
        symbols, indices = process_indexed_string(string)
        return self.monad.mapM(self.rewrite_symbol, symbols) >> (
            lambda s: self.monad.ret(concatenate(put_into_indices(s, indices))))

    def distribution(self):
        return self.rewrite_nonterminal((self.start, 1))

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
    
def test_pcfrs():
    from math import log
    r1 = Rule('S', (('NP', 'VP'),))
    r2 = Rule('NP', (('D', 'N'),))
    r3 = Rule('NPR', (('D', 'N'), ('RP',)))
    r4 = Rule('VP', (('V',),))
    r5 = Rule('S', ((('NPR', 0, 0), 'VP', ('NPR', 0, 1)),))
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

    pcfrs = make_pcfrs(Enumeration, rules)
    return pcfrs


if __name__ == '__main__':
    import nose
    nose.runmodule()
