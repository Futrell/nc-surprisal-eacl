import functools
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import sympy


from pmonad import *
import incnoise
import infotrees
import pcfg

EPSILON = 10 ** -4

def is_close(x, y, tol):
    return abs(x - y) < tol

def embedding_lang(p_mod, p_rc, p_src, num_levels):
    rules = [
        (pcfg.Rule('S', ('NP', 'V')), 1),
        (pcfg.Rule('NP', ('N',)), 1 - p_mod),
        (pcfg.Rule('NP', ('N', 'RC')), p_mod * p_rc),
        (pcfg.Rule('NP', ('N', 'PP')), p_mod * (1 - p_rc)),
        (pcfg.Rule('PP', ('P', 'NP')), 1),
        (pcfg.Rule('RC', ('C', 'NP', 'V')), 1 - p_src),
        (pcfg.Rule('RC', ('C', 'V', 'NP')), p_src)
    ]
    grammar = pcfg.make_bounded_pcfg(SymbolicEnumeration, rules, num_levels)
    return grammar.distribution()

def _embedding_lang(p_mod, p_rc, p_src, num_levels):
    # Old and broken!
    N, V, C, P = "NVCP"

    enum = SymbolicEnumeration
    
    def add_distractor(s):
        insertion_index = len(s) - s[::-1].index(N)
        return s[:insertion_index] + (P, N) + s[insertion_index:]

    def rc_dist(s):
        insertion_index = len(s) - s[::-1].index(N) 
        def add_src(s):
            return s[:insertion_index] + (C, V, N) + s[insertion_index:]
        def add_orc(s):
            return s[:insertion_index] + (C, N, V) + s[insertion_index:]
        return enum.flip(p_src).bind(
            lambda b: enum.ret(add_src(s)) if b else enum.ret(add_orc(s))
        )

    def rc_rule(s):
        return enum.flip(p_rc).bind(lambda b: rc_dist(s) if b else enum.ret(s))

    def mod_rule(s):
        return enum.flip(p_mod).bind( # if flip(p_mod), then ...
            lambda mod: enum.ret(s) if not mod else enum.flip(p_rc).bind(
            lambda rc: rc_dist(s) if rc else enum.ret(add_distractor(s))
            )
        )

    s = enum.ret((N, V))
    for _ in range(num_levels):
        s = s.bind(mod_rule)
    return s


def verb_forgetting_grid(eps=.01):
    ms = [.25, .5, .75]
    rs = list(np.arange(eps, 1-eps, eps))
    ss = list(np.arange(eps, 1-eps, eps))
    es = [.1, .2, .3]

    conditions = itertools.product(ms, rs, ss, es)
    def gen():
        for m, r, s, e in conditions:
            (corr1, incorr1), (corr2, incorr2) = verb_forgetting_conditions(m=m,
                                                                            r=r,
                                                                            s=s,
                                                                            e=e)
            yield m, r, s, e, ((corr1 < incorr1), (corr2 < incorr2))
    df = pd.DataFrame(gen())
    df.columns = ['m', 'r', 's', 'e', 'vcont']
    return df

def verb_forgetting_plot(df):
    df['d'] = df['e']
    grid = sns.FacetGrid(
        df[(df['d'] < .4)],
        col='d',
        row='m',
        hue='vcont',
        legend_out=True,
        margin_titles=True,
        palette='colorblind'
    )
    def do(*a, **k):
        plt.scatter(*a, s=150, marker='s', **k)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
    grid.map(do, 's', 'r')
    
def verb_forgetting_conditions(num_levels=2,
                               m=sympy.Symbol('m'),
                               r=sympy.Symbol('r'),
                               s=sympy.Symbol('s'),
                               e=sympy.Symbol('e')):

    SE = SymbolicEnumeration

    # create the language
    lang = embedding_lang(m, r, s, num_levels).marginalize()

    # the noise is sequence erasure noise, here in the SE monad
    def symbol_noise(x):
        # flip e, if true then erase else don't
        return SE.flip(e).bind( 
            SE.lift_ret(lambda b: 'E' if b else x)
        )

    def erasure_noise(s):
        return SE.mapM(symbol_noise, s)

    def deletion_noise(s):
        return SE.mapM(symbol_noise, s).bind(
            SE.lift_ret(lambda s: tuple(x for x in s if x != 'E'))
        )

    # apply the noise function to each sentence.
    # this might be slow.
    npt = incnoise.noisy_prefix_tree(lang, deletion_noise)

    prefixes = [
        tuple("N")
        + tuple("CN") * (num_levels - k)
        + tuple("V") * (num_levels - k)
        for k in reversed(range(num_levels))
    ]

    def gen():
        for i in range(num_levels):
            noisy_prefix_distro = deletion_noise(prefixes[i]).marginalize()

            cost_correct = 0
            cost_incorrect = 0
            for noisy_prefix, p in noisy_prefix_distro.dict.items():
                p_correct = npt[noisy_prefix].dict.get('V', 0)
                p_incorrect = npt[noisy_prefix].dict.get(incnoise.HALT, 0)
                cost_correct -= p * sympy.log(p_correct)
                cost_incorrect -= p * sympy.log(p_incorrect)
            cost_correct = cost_correct.simplify()
            cost_incorrect = cost_incorrect.simplify()
            yield cost_correct, cost_incorrect

    return tuple(gen())

def tree_cost(t, ks, noise=None, p=.1, num_samples=10000, sparsity=1):
    lang = enumeration_from_sampling_function(
        lambda: infotrees.sample_string_for_tree(t, ks, sparsity=sparsity),
        num_samples
    )
    if noise is None:
        noise = lambda s: incnoise.approx_successive_bit_noise(
            s,
            p,
            num_samples
        )
    return incnoise.internal_lang_cost(lang, noise)

def bit_noise(p):
    return lambda s: incnoise.successive_bit_noise(s, p)

def approx_bit_noise(p, num_samples=10**5):
    return lambda s: incnoise.approx_successive_bit_noise(
        s,
        p,
        num_samples
    )

def erasure_noise(p):
    return lambda s: incnoise.successive_erasure_noise(s, p)

def approx_erasure_noise(p, num_samples=10**5):
    return lambda s: incnoise.approx_successive_erasure_noise(
        s,
        p,
        num_samples
    )

no_noise = deterministic(lambda s: s)
complete_noise = deterministic(lambda s: ())

def marking_lang(marking, num_samples=10000, **kwds):
    return enumeration_from_sampling_function(
        lambda: infotrees.sample_two_arg(marking=marking, **kwds),
        num_samples
    )

hm_lang = functools.partial(marking_lang, 'hm')
dm_lang = functools.partial(marking_lang, 'dm', hms_mi=0, hmo_mi=0)
null_lang = functools.partial(marking_lang, 'hm', hms_mi=0, hmo_mi=0)

def test_marking_lang():
    lang = marking_lang(
        'hm',
        num_samples=10**5,
        hs_mi=1,
        ho_mi=1,
        hms_mi=0,
        hmo_mi=0,
    )
    # Upper bound on entropy is 3
    # There are the cases 00,0,0 and 1,1,1 where the last bit
    # does not come into play; these have probability 1/2,
    # therefore the total entropy is 1 + 1 + 1/2
    assert is_close(lang.entropy(), 2.5, .2)

    # Make sure s-marking and o-marking are symmetrical
    lang1 = marking_lang(
        'hm',
        num_samples=10**5,
        hms_mi=1,
        hmo_mi=0,
    )
    lang2 = marking_lang(
        'hm',
        num_samples=10**5,
        hms_mi=0,
        hmo_mi=1,
    )
    assert is_close(lang1.entropy(), lang2.entropy(), 0.001)

    # These entropies should be equal to 4.
    # The extra marking on S but not O (or vice versa) eliminates the
    # confusable case 00,0,0. Also, there is the extra bit of hmo/s_mi
    # itself. Therefore the total entropy is 1+1+1+1.
    assert is_close(lang1.entropy(), 4, .1)

    # If you add a bit of MI shared between H, S, and O, you add 1 bit
    # to the total entropy:
    lang3 = marking_lang(
        'hm',
        num_samples=10**5,
        hms_mi=1,
        hmo_mi=0,
        hso_mi=1,
    )
    assert is_close(lang3.entropy(), lang1.entropy() + 1, 0.001)

    # Similarly if you add self-information to the head or to the subject
    lang4 = marking_lang(
        'hm',
        num_samples=10**5,
        head_self_info=1,
        hms_mi=1,
        hmo_mi=0,
        hso_mi=0,
    )
    assert is_close(lang4.entropy(), lang1.entropy() + 1, 0.001)

    lang5 = marking_lang(
        'hm',
        num_samples=10**5,
        subj_self_info=1,
        hms_mi=1,
        hmo_mi=0,
        hso_mi=0,
    )
    assert is_close(lang5.entropy(), lang1.entropy() + 1, 0.001)

    # But if you add bit to the object, ambiguity is reintroduced.
    # Previously we could identify the subject by noting it had length 2,
    # while the object had length 1. If they are both length 2, then
    # there are cases where the subject and object are identical,
    # and therefore the shuffling bit is lost in 1/4 of cases.
    lang6 = marking_lang(
        'hm',
        num_samples=10**5,
        obj_self_info=1,
        hms_mi=1,
        hmo_mi=0,
        hso_mi=0,
    )
    assert is_close(lang6.entropy(), lang1.entropy() + 3/4, 0.01)

    # in lang6, the head consists of ABC, where A is the first bit
    # of the subject, B is the first bit of the object, and C is
    # the second bit of the subject. The second bit of the object
    # is random. Therefore, the subject is entirely predictable
    # given the head, whereas the object has conditional entropy 1.

    # Let's think about a more symmetrical language:
    lang7 = marking_lang(
        'hm',
        num_samples=10**5,
        hms_mi=1,
        hmo_mi=1,
    )
    assert is_close(lang7.entropy(), lang1.entropy() + 3/4, 0.01)    

    # The subject and object are predictable from the head, but their
    # order is not in 3/4 of cases, therefore the entropy of lang7 should
    # be equal to the entropy of the head plus 3/4.
    head_ent = lang7.bind(lambda s: certainly(s[0])).entropy()
    assert is_close(lang7.entropy(), head_ent + 3/4, .01)

    # Suppose we delete the head. Then the remaining entropy is just the
    # entropy of the head, because the subject and object are no longer
    # distinguishable, so the bit that was used to shuffle them is obscured.
    no_head_ent = lang7.bind(lambda s: certainly(s[1:])).entropy()
    assert is_close(head_ent, no_head_ent, 0.01)

    # Suppose we delete only the affixes on the head. Then the remaining
    # entropy is the "weird cost" of predicting A or B, with A and B known,
    # but without knowing which is which.
    nomark_lang = lang7.bind(lambda s: certainly((s[0][:-2], s[1], s[2])))
    nomark_cost = nomark_lang.entropy() + 2

def local_nonlocal(**kwds):
    t_local = nx.DiGraph([(0, 1), (0, 2), (2, 3), (3, 4)])
    t_nonlocal = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 4)])

    local_cost = tree_cost(t_local, [0, 1], **kwds)
    nonlocal_cost = tree_cost(t_nonlocal, [0, 1], **kwds)
    return local_cost, nonlocal_cost

def test_local_nonlocal():
    local_cost, nonlocal_cost = local_nonlocal()
    assert local_cost < nonlocal_cost

@uniform_enumerator            
def copy_lang():
    return ['aa', 'AA', 'bb', 'BB']

@uniform_enumerator
def sparse_second_lang():
    return ['AA', 'Bb', 'aA', 'bb']

@uniform_enumerator
def sparse_first_lang():
    return ['AA', 'bB', 'Aa', 'bb']

def test_mi_basics():
    t = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    lang_with_mi_0 = enumeration_from_sampling_function(
        lambda: infotrees.sample_string_for_tree(t, [2]),
        1000
    )
    # Total entropy of lang_with_mi_0 is 8 bits
    # Cost with no noise should be slightly less than 8:
    lang_with_mi_0_cost_no_noise = incnoise.internal_lang_cost(
        lang_with_mi_0,
        no_noise
    )
    assert 7.7 < lang_with_mi_0_cost_no_noise < 8

    # Noise should not affect cost, because the bits are all independent.
    lang_with_mi_0_cost_erasure_noise = incnoise.internal_lang_cost(
        lang_with_mi_0,
        lambda s: incnoise.successive_erasure_noise(s, .1)
    )
    assert 7.7 < lang_with_mi_0_cost_erasure_noise < 8

    lang_with_mi_0_cost_bit_noise = incnoise.internal_lang_cost(
        lang_with_mi_0,
        lambda s: incnoise.approx_successive_bit_noise(s, .1, 10000)
    )
    assert 7.7 < lang_with_mi_0_cost_erasure_noise < 8        

    # Now let's consider a lang where heads and dependents share 1 bit.
    lang_with_mi_1 = enumeration_from_sampling_function(
        lambda: infotrees.sample_string_for_tree(t, [0, 1]),
        1000
    )

    # With no noise, we should have cost much less than the 0-mi lang
    lang_with_mi_1_cost_no_noise = incnoise.internal_lang_cost(
        lang_with_mi_1,
        no_noise,
    )
    assert 3.9 < lang_with_mi_1_cost_no_noise < 4

    # With noise, the cost should go up considerably.
    lang_with_mi_1_cost_erasure_noise = incnoise.internal_lang_cost(
        lang_with_mi_1,
        lambda s: incnoise.successive_erasure_noise(s, .1)
    )
    assert lang_with_mi_1_cost_erasure_noise > 4
    # Should be something like 4.3

    # Bit-flipping noise should be worse
    lang_with_mi_1_cost_bit_noise = incnoise.internal_lang_cost(
        lang_with_mi_1,
        lambda s: incnoise.approx_successive_bit_noise(s, .1, 10000)
    )
    assert lang_with_mi_1_cost_bit_noise > 5
    # Should be something like 5.4




if __name__ == '__main__':
    import nose
    nose.runmodule()
