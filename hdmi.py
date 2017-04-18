""" head-dep mi

Investigating mutual information between heads and dependents, etc. 

Is there greater mutual information between heads and dependents than between 
arbitrary pairs of words? Does this mutual information persist to grandparents?
How does it compare to mutual information among sisters? Or among adjacent 
pairs? Given w1 and w2 with distance d between them, is the MI between w1 and
w2 different depending on the syntactic relationship between w1 and w2: 
head-dep, grandparent-dep, sister, etc? Does MI between words fall off with
increasing distance as a power law, as does MI between letters 
(Lin & Tegmark, 2016; Wentian Li, 1990)?

Do these relationships vary over languages? Do more morphologically complex
languages have more mutual information? Does this quantity vary with word order
freedom?

Bentz & Alikaniotis (on arxiv) find that MLE entropy estimates for unigram word
entropy converge at something like 7 x 10^6 tokens. So we should be able to do 
this in corpora where there are 70K pairs of the type we are interested in. This
 means counting edges, which is just word count - sentence count (because there
 are N-1 edges per sentence).

We should be able to get stable estimates below the Bentz & Alikaniotis line by:
(1) Using POS tags rather than wordforms, and
(2) using the PYM estimator.

Optimally we would do our own convergence analysis, but maybe that won't happen.

We will estimate mutual information using the PYM estimator as:
I(X;Y) = H(X) + H(Y) - H(X,Y)
This is the first quasi-Bayesian estimator from archer2013bayesian.

Which UD corpora are large enough?

"""
from collections import Counter, defaultdict
import itertools
import bisect
import random
from math import log, exp
import sys

import pandas as pd
from rfutils import sliding, mean, count
import cliqs.depgraph as depgraph
import cliqs.corpora as corpora
import cliqs.conditioning as cond

import entropy

def pair_processor(rel_f1, rel_f2, f1, f2):
    def get_pair_attrs(s, n):
        return (f1(s, rel_f1(n)), f2(s, rel_f2(s, n)))
    return get_pair_attrs

def identity(x):
    return x

hdw = pair_processor(identity, depgraph.head_of, cond.get_word, cond.get_word)
hdp = pair_processor(identity, depgraph.head_of, cond.get_pos, cond.get_pos)
gdw = pair_processor(
    identity,
    lambda s, n: depgraph.transitive_head_of(s, n, 2),
    cond.get_word,
    cond.get_word
)
gdp = pair_processor(
    identity,
    lambda s, n: depgraph.transitive_head_of(s, n, 2),
    cond.get_pos,
    cond.get_pos
)

def adjacents(f, sentences, i):
    for s in sentences:
        for n in sorted(s.nodes())[:-i]:
            if n != 0:
                yield f(s, n), f(s, n+i)

def ss(f, sentences):
    for s in sentences:
        for n in s.nodes():
            if n != 0:
                kids = depgraph.dependents_of(s, n)
                if len(kids) >= 2:
                    for s1, s2 in itertools.combinations(kids, 2):
                        yield f(s, s1), f(s, s2)

def compare_mi(one, two, num_samples=500):
    one = list(one)
    two = list(two)
    f = lambda x: entropy.mutual_information(Counter(x).items())
    p = permutation_test(f, one, two, num_samples)
    return one, two, p

def permutation_test(f, one, two, num_samples):
    one = list(one)
    two = list(two)
    diff = f(one) - f(two)
    N = len(one)
    together = one + two
    def gen():
        for i in range(num_samples):
            random.shuffle(together)
            one = together[:N]
            two = together[N:]
            yield f(one) - f(two)
        print("did %s permutations" % num_samples, file=sys.stderr)
    diffs = sorted(gen())
    i = bisect.bisect(diffs, diff)
    return min(i/num_samples, 1 - i/num_samples)

def subset_permutation_test(f, the_subset, the_set, num_samples):
    the_subset = list(the_subset)
    the_set = list(the_set)
    diff = f(the_subset) - f(the_set)
    N = len(the_subset)
    def gen():
        for i in range(num_samples):
            new_subset = random.sample(the_set, N)
            yield f(new_subset) - f(the_set)
        print("did %s permutations" % num_samples, file=sys.stderr)
    diffs = sorted(gen())
    i = bisect.bisect(diffs, diff)
    return min(i/num_samples, 1 - i/num_samples)

def mi_of_observations(xs):
    return entropy.mutual_information(Counter(xs).items())

def mi_factory(g):
    def mi(f, sentences):
        try:
            return mi_of_observations(g(f, sentences))
        except ValueError:
            return 0
    return mi

def skipgrams(xs, k):
    for gram in sliding(xs, k+2):
        yield gram[0], gram[-1]

def skip_dep_pairs(f, sentences, k):
    sentences = list(sentences)
    nondep_pairs = []
    dep_pairs = []
    for s in sentences:
        edges = set(s.edges())
        for n1, n2 in skipgrams(s.nodes()[1:], k):
            if (n1, n2) in edges or (n2, n1) in edges:
                dep_pairs.append((f(s, n1), f(s, n2)))
            else:
                nondep_pairs.append((f(s, n1), f(s, n2)))
    return dep_pairs, nondep_pairs

def skip_mi(f, sentences, k, num_samples=500):
    dep_pairs, nondep_pairs = skip_dep_pairs(f, sentences, k)
    dep_counts = Counter(dep_pairs)
    all_counts = dep_counts + Counter(nondep_pairs)
    def mi(xs):
        try:
            return entropy.mutual_information(xs)
        except ValueError:
            return 0
    dmi = mi(dep_counts.items())
    bmi = mi(all_counts.items())
    if not num_samples:
        pt = None
    else:
        pt = subset_permutation_test(
            mi,
            dep_counts.items(),
            all_counts.items(),
            num_samples
        )
    return dmi, bmi, pt

def skip_pmis(f, sentences, k, by_deptype=False):
    sentences = list(sentences)
    joint = Counter()
    marginal = Counter()
    for s in sentences:
        for n in s.nodes()[1:]:
            w = f(s, n)
            marginal[w] += 1
        for n1, n2 in skipgrams(s.nodes()[1:], k):
            w1 = f(s, n1)
            w2 = f(s, n2)
            joint[w1, w2] += 1
    log_Z_joint = log(sum(joint.values()))
    log_Z_marginal = log(sum(marginal.values()))
    pmi_Z = log_Z_joint - 2*log_Z_marginal
    def pmi(w1, w2):
        return (
            log(joint[w1, w2])
            - log(marginal[w1])
            - log(marginal[w2])
            - pmi_Z
        )
    if by_deptype:
        hdpmi = defaultdict(list)
    else:
        hdpmi = []
    baselinepmi = []
    for s in sentences:
        edges = set(s.edges())
        for n1, n2 in skipgrams(s.nodes()[1:], k):
            w1 = f(s, n1)
            w2 = f(s, n2)
            the_pmi = pmi(w1, w2)
            if (n1, n2) in edges or (n2, n1) in edges:
                if by_deptype:
                    try:
                        hdpmi[s.edge[n1][n2]['deptype']].append(the_pmi)
                    except KeyError:
                        hdpmi[s.edge[n2][n1]['deptype']].append(the_pmi)
                else:
                    hdpmi.append(the_pmi)
            elif by_deptype:
                hdpmi['nondep'].append(the_pmi)
            baselinepmi.append(the_pmi)
    return hdpmi, baselinepmi

def safemean(xs):
    try:
        return mean(xs)
    except ValueError:
        return 0

def skip_pmi(f, sentences, k, num_samples=500):
    hdpmi, baselinepmi = skip_pmis(f, sentences, k, by_deptype=False)
    if num_samples > 0:
        pt_result = subset_permutation_test(
            safemean,
            hdpmi,
            baselinepmi,
            num_samples
        )
    else:
        pt_result = None
    return (
        safemean(hdpmi),
        safemean(baselinepmi),
        len(hdpmi),
        len(baselinepmi),
        pt_result
    )

def hdpmi(f, sentences):
    joint = Counter()
    marginal = Counter()
    len_limit = 10
    for s in sentences:
        N = len(s) - 1
        if N > len_limit:
            continue
        for n in s.nodes():
            if n != 0:
                w = f(s, n)
                marginal[w] += 1
                for n2 in s.nodes():
                    if n2 != 0 and n2 != n:
                        w2 = f(s, n2)
                        joint[frozenset([w, w2])] += 1
    log_Z_joint = log(sum(joint.values()))
    log_Z_marginal = 2*log(sum(marginal.values()))
    def pmi(w1, w2):
        return (
            log(joint[frozenset([w1, w2])])
            - log(marginal[w1])
            - log(marginal[w2])
            - log_Z_joint
            + log_Z_marginal
        )
    hdpmi = 0
    hdn = 0
    baselinepmi = 0
    baselinen = 0
    for s in sentences:
        N = len(s) - 1
        if N > len_limit:
            continue
        edges = set(s.edges())
        for h, d in edges:
            if h != 0:
                w1 = f(s, h)
                w2 = f(s, d)
                hdpmi += pmi(w1, w2) 
                hdn += 1
        for n1 in s.nodes():
            for n2 in s.nodes():
                if n1 != 0 and n2 != 0 and n1 != n2:
                    if (n1, n2) not in edges and (n2, n1) not in edges:
                        w1 = f(s, n1)
                        w2 = f(s, n2)
                        baselinepmi += pmi(w1, w2)
                        baselinen += 1
    return hdpmi/hdn, baselinepmi/baselinen
                
def pairs(f, sentences):
    for s in sentences:
        for h, d in s.edges():
            if h != 0:
                yield f(s, h), f(s, d)

def gd(f, sentences):
    for s in sentences:
        for h, d in s.edges():
            if h != 0:
                g = depgraph.get_head_of(s, h)
                if g:
                    yield f(s, g), f(s, d)
                     
def hd(f, sentences):
    for s in sentences:
        for h, d in s.edges():
            if h != 0:
                yield f(s, h), f(s, d)
    
                     
                     
adjacents_mi = lambda f, s, i: mi_factory(lambda f, s: adjacents(f, s, i))(f, s)
ssmi = mi_factory(ss)
gdmi = mi_factory(gd)
hdmi = mi_factory(hd)

def hdmi_topologies_with_permutation_tests():
    d = {}
    c = cond.get_pos
    num_samples = 500
    def mi_pt(f, g, sentences):
        return permutation_test(
            mi_of_observations,
            list(f(c, sentences)),
            list(g(c, sentences)),
            num_samples
        )

    #for lang, corpus in [['en', corpora.ud_corpora['en']]]:
    for lang, corpus in corpora.ud_corpora.items():
        sentences = list(corpus.sentences(fix_content_head=False))
        d[lang] = {
            'hdmi': hdmi(c, sentences),
            'gdmi': gdmi(c, sentences),
            'ssmi': ssmi(c, sentences),
            'hd_gd_pt': mi_pt(hd, gd, sentences),
            'hd_ss_pt': mi_pt(hd, ss, sentences),
            'gd_ss_pt': mi_pt(gd, ss, sentences)
        }
    df = pd.DataFrame(d).T
    df['lang'] = df.index
    return df

def skip_mi_sweep():
    max_k = 15
    c = cond.get_pos
    def gen():
        for lang, corpus in corpora.ud_corpora.items():
            print(lang, file=sys.stderr)
            sentences = list(corpus.sentences(fix_content_head=False))
            for k in range(0, max_k):
                h, b, p = skip_mi(c, sentences, k, 500)
                hp, bp, num_dep, num_base, pp = skip_pmi(
                    c,
                    sentences,
                    k,
                    num_samples=500
                )
                yield {
                    'lang': lang,
                    'k': k,
                    'n_dep': num_dep,
                    'n_base': num_base,
                    'mi_dep': h,
                    'mi_base': b,
                    'mi_p': p,
                    'pmi_dep': hp,
                    'pmi_base': bp,
                    'pmi_p': pp,
                }
    df = pd.DataFrame(gen())
    return df

def skip_pmi_by_deptype():
    max_k = 10
    c = cond.get_pos
    def gen():
        #for lang, corpus in corpora.ud_corpora.items():
        for lang, corpus in [['en', corpora.ud_corpora['en']]]:
            sentences = list(corpus.sentences(fix_content_head=False))
            for k in range(0, max_k):
                h, _ = skip_pmi(c, sentences, k, by_deptype=True)
                for deptype, pmis in h.items():
                    yield lang, k, deptype, mean(pmis), count(pmis)
    df = pd.DataFrame(gen())
    return df

def hdmi_sweep():
    d = {}
    for lang, corpus in corpora.ud_corpora.items():
        sentences = list(corpus.sentences(fix_content_head=False))
        d[lang] = {
            'hd_mi': hdmi(cond.get_pos, sentences),
            'hd_n': count(hd(cond.nothing, sentences)),
            'gd_mi': gdmi(cond.get_pos, sentences),
            'gd_n': count(gd(cond.nothing, sentences)),
            'ss_mi': ssmi(cond.get_pos, sentences),
            'ss_n': count(ss(cond.nothing, sentences)),
        }
        for i in range(0, 25):
            d[lang]['a_%d_mi' % i] = adjacents_mi(cond.get_pos, sentences, i)
            d[lang]['a_%d_n' % i] = count(adjacents(cond.nothing, sentences, i))
    df = pd.DataFrame(d).T
    df['lang'] = df.index
    return df

        
