import operator
import random
import itertools
import functools
import copy
from collections import Counter

import networkx as nx
import pyrsistent as pyr

import rfutils
from rfutils.compat import *
import cliqs.depgraph as depgraph

def immutably(f):
    @functools.wraps(f)
    def wrapped(sentence, *a, **k):
        sentence = copy.deepcopy(sentence)
        return f(sentence, *a, **k)
    return wrapped


def sample_random_bits(n):
    """ Generate n random bits as a string of 1s and 0s. """
    assert n >= 0
    if n:
        return bin(random.randint(0, 2**n - 1))[2:].zfill(n)
    else:
        return ""

def enum_random_bits(n):
    assert n >= 0
    logp = -log(2**n)
    return zip(
        map("".join, itertools.product("01", repeat=n)),
        itertools.repeat(logp)
    )

def test_sample_random_bits():
    num_runs = 100
    max_size = 10000
    zero_or_one = {0, 1}
    lengths = [0, 1]
    lengths.extend(random.randint(0, max_size) for _ in range(num_runs))
    for length in lengths:
        bits = sample_random_bits(length)
        assert len(bits) == length
        assert all(int(bit) in zero_or_one for bit in bits)

def get_ancestor(s, n, k):
    result = n
    for _ in range(k):
        result = depgraph.get_head_of(s, result)
        if result is None:
            return None
    return result

def sample_two_arg(marking='hm',
                   head_pos=0,
                   head_self_info=0,
                   subj_self_info=0,
                   obj_self_info=0,
                   hso_mi=0,
                   hs_mi=1,
                   ho_mi=1,
                   hms_mi=1,
                   hmo_mi=1,
                   flip_order=True):
    
    H = sample_random_bits(head_self_info)
    hso = sample_random_bits(hso_mi)
    hs = sample_random_bits(hs_mi)
    ho = sample_random_bits(ho_mi)
    head_stem = H + hso + hs + ho

    S = sample_random_bits(subj_self_info)
    hms = sample_random_bits(hms_mi)
    S_stem = S + hms + hso + hs

    O = sample_random_bits(obj_self_info)
    hmo = sample_random_bits(hmo_mi)
    O_stem = O + hmo + hso + ho
    

    if marking == 'hm':
        head = head_stem + hms + hmo
        subj = S_stem
        obj = O_stem

    elif marking == 'dm':
        head = head_stem
        subj = S_stem + '0'
        obj = O_stem + '1'

    elif marking == 'fake_dm':
        head = head_stem
        flipper = sample_random_bits(1)
        subj = S_stem + flipper
        obj = O_stem + str(1-int(flipper))

    elif marking == 'fake_hm':
        marks = [hms, hmo]
        random.shuffle(marks)
        head = ''.join([head_stem] + marks)
        subj = S_stem
        obj = O_stem
    elif marking == 'passive':
        head = head_stem
        subj = S_stem
        obj = O_stem
    
    phrase = [subj, obj]
    if flip_order:
        random.shuffle(phrase)
    if marking == 'passive':
        if phrase[0] == subj:
            head += '0'
        else:
            head += '1'
        
    phrase.insert(0, head)
    return tuple(phrase)

def uncertain_order_lang(k1, k2, hm_pos=0, hm=True, dm=False, shuffled=True, suffix=True):
    one = sample_random_bits(k1)
    two = sample_random_bits(k2)
    if dm:
        if suffix:
            one += '0'
            two += '1'
        else:
            one = '0' + one
            two = '1' + two
    phrase = [one, two]
    if shuffled:
        random.shuffle(phrase)
    if hm:
        mark = one + two
        phrase.insert(hm_pos, mark)
    return tuple(phrase)

def which_one_lang(k, dm=False):
    one = sample_random_bits(k)
    two = sample_random_bits(k)
    if dm:
        one += '0'
        two += '1'
    if random.random() < .5:
        return (one+two, one)
    else:
        return (one+two, two)

def mi_lang(A, B, shared, redundancy, distance, sparse_first):
    A_bits = sample_random_bits(A)
    B_bits = sample_random_bits(B)
    a_bits = sample_random_bits(shared)
    b_bits = sample_random_bits(shared)
    filler = '0' * distance
    if sparse_first:
        return (
            A_bits + a_bits + b_bits * redundancy,
            filler,
            B_bits + b_bits + a_bits,
        )
    else:
        return (
            A_bits + a_bits + b_bits,
            filler,
            B_bits + b_bits + a_bits * redundancy
        )


def sample_markov_lang(length, mi=1, si=0, sparsity=1):
    nodes = range(length)
    tree = nx.DiGraph(zip(nodes, nodes[1:]))
    string = sample_string_for_tree(tree, [si, mi])
    sparse_string = tuple(x*sparsity for x in string)
    return sparse_string

# For embedding lang,
# first think about varying distance,
# with the probability of embedding held constnat.
# e.g., AAAxxxBBB vs. AAABBBxxx.
# I claim that in a language where p(AAABBBxxx) > p(AAAxxxBBB),
# the cost of AAAxxxBxB is less than the cost of AAAxxxBBB.
# What is the relationship between p_embed and (the relationship between p_distant and cost)?

def sample_embedding_length(length,
                            distance,
                            p_distant,
                            num_embedded=3,
                            embed_mi=1,
                            mi=1,
                            si=0,
                            sparsity=1):
    string = list(sample_markov_lang(length, mi=mi, si=si, sparsity=sparsity))
    outstanding = []
    for i in range(num_embedded):
        A = sample_random_bits(embed_mi)
        string[i] += A
        outstanding.append(A)
    if random.random() < p_distant:
        offset = num_embedded + distance
    else:
        offset = num_embedded
    for i in range(num_embedded):
        string[offset + i] += outstanding.pop()
    return tuple(string)

def sample_nodes_for_tree(tree, ks, sparsity=1):
    for node in tree.nodes():
        for i, k in enumerate(ks):
            tree.node[node]['own_%d' % i] = sample_random_bits(k) * sparsity
    for i, k in enumerate(ks[1:], 1):
        for node in tree.nodes():
            source_node = get_ancestor(tree, node, i)
            if source_node is None:
                tree.node[node]['mark_%d' % i] = "0" * k
            else:
                tree.node[node]['mark_%d' % i] = tree.node[source_node]['own_%d' % i]
    return tree

def test_sample_nodes_for_tree():
    t = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    assert (
        sample_nodes_for_tree(t, []).nodes(data=True)
        ==  [(0, {}), (1, {}), (2, {}), (3, {})]
    )

    one = sample_nodes_for_tree(t, [1])
    assert list(one.node[0].keys()) == ['own_0']
    assert list(one.node[1].keys()) == ['own_0']
    assert list(one.node[2].keys()) == ['own_0']
    assert list(one.node[3].keys()) == ['own_0']

    two = sample_nodes_for_tree(t, [0, 2])
    assert two.node[0]['own_1'] == two.node[1]['mark_1']
    assert two.node[1]['own_1'] == two.node[2]['mark_1']
    assert two.node[2]['own_1'] == two.node[3]['mark_1']

def sample_string_for_tree(tree, ks, sparsity=1):
    annotated = sample_nodes_for_tree(tree, ks, sparsity=sparsity)
    def gen():
        for node in sorted(tree.nodes()):
            yield "".join(v for k, v in sorted(tree.node[node].items()))
    return tuple(gen())

def test_sample_string_for_tree():
    t = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    nomi_samples = [
        immutably(sample_string_for_tree)(t, [2]) for _ in range(1000)
    ]

    mi_samples = [ # MI 1
        immutably(sample_string_for_tree)(t, [0, 1]) for _ in range(1000)
    ]

    # When self-information is high, samples become sparser, so more samples
    # are needed to get a distribution with the right MI.
    mi_samples2 = [ # MI 1
        immutably(sample_string_for_tree)(t, [2, 1]) for _ in range(10000)
    ]

    mi_samples3 = [ # MI 2
        immutably(sample_string_for_tree)(t, [0, 2]) for _ in range(1000)
    ]    

    assert len(set(mi_samples)) < len(set(nomi_samples))

    def is_close(x, y, tol):
        return abs(x - y) < tol

    import rfutils.entropy
    nomi_mi = rfutils.entropy.mutual_information(
        Counter(rfutils.flatmap(lambda s: rfutils.sliding(s, 2), nomi_samples))
    )
    assert is_close(nomi_mi, 0, 0.01)

    mi_mi = rfutils.entropy.mutual_information(
        Counter(rfutils.flatmap(lambda s: rfutils.sliding(s, 2), mi_samples))
    )
    assert is_close(mi_mi, 1, 0.01)

    mi2_mi = rfutils.entropy.mutual_information(
        Counter(rfutils.flatmap(lambda s: rfutils.sliding(s, 2), mi_samples2))
    )
    assert is_close(mi2_mi, 1, 0.01)

    mi3_mi = rfutils.entropy.mutual_information(
        Counter(rfutils.flatmap(lambda s: rfutils.sliding(s, 2), mi_samples3))
    )
    assert is_close(mi3_mi, 2, 0.1)

    # TODO test grandparent MI functionality

def _all_equivalence_classes(n, k):
    """ generate all possible mappings of n elements into k equivalence classes 
    This is the nice recursive implementation; for speed, use 
    all_equivalence_classes
    """
    # first is [0] * n, last is range(n)
    def g(prefix, num_symbols_so_far, n):
        if n == 0:
            yield prefix
        else:
            for symbol in range(num_symbols_so_far):
                for cont in g(prefix.append(symbol), num_symbols_so_far, n - 1):
                    yield cont
            if num_symbols_so_far < k:
                conts = g(prefix.append(num_symbols_so_far),
                          num_symbols_so_far + 1, n - 1)
                for cont in conts:
                    yield cont
    return g(pyr.v(), 0, n)

def num_equivalence_classes(n, k):
    from sympy.functions.combinatorial.numbers import stirling as S2
    return sum(S2(n, k_) for k_ in range(1, k+1))

def all_equivalence_classes(n, k):
    """ Generate all possible mappings of n elements into at most k equivalence 
    classes. There are $\sum_{k'=1}^{k} S_2(n, k')$ such mappings, where $S_2$ 
    is Stirling numbers of the second kind.
    https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
    You could also call these truncated Bell numbers; Bell numbers go up to k=n.
    """
    # Ugly 2x faster version of _all_equivalence_classes
    # As optimized as it's going to get unless I figure out a new algorithm
    assignments = [(1, pyr.v(0))]
    range_k = range(k) 
    for _ in range(1, n):
        new_assignments = []
        new_assignments_append = new_assignments.append 
        for next_symbol, assignment in assignments:
            assignment_append = assignment.append
            if next_symbol < k:
                for symbol in range(next_symbol):
                    new_assignments_append((next_symbol,
                                            assignment_append(symbol)))
                new_assignments_append((next_symbol+1,
                                        assignment_append(next_symbol)))
            else:
                for symbol in range_k:
                    new_assignments_append((next_symbol,
                                            assignment_append(symbol)))
        assignments = new_assignments
    return map(operator.itemgetter(1), assignments)

def test_all_equivalence_classes():
    classes = list(all_equivalence_classes(7, 4))
    # S_2(7, 4) + S_2(7, 3) + S_2(7, 2) + S_2(7, 1)    
    correct_num = 350 + 301 + 63 + 1 
    assert len(classes) == correct_num
    assert len(set(classes)) == correct_num # they are unique

        
if __name__ == '__main__':
    import nose
    nose.runmodule()
