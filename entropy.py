""" entropy

Module to calculate Shannon entropy and related measures, with features:
(1) Not dependent on numpy, so compatible with pypy;
(2) Focus on constant-space calculations in case input distribution is 
    represented as an iterator;
(3) Fast and accurate subject to the above constraints.

"""
from __future__ import division
from math import log
from collections import Counter, defaultdict

base = log(2)
def log2(x):
    return log(x, 2)

def entropy_of_tokens(tokens):
    return entropy(Counter(tokens))

def entropy(counts):
    """ entropy

    Fast calculation of Shannon entropy from an iterable of positive numbers. 
    Numbers are normalized to form a probability distribution, then entropy is
    computed.

    Generators are welcome.

    Params:
        counts: An iterable of positive numbers.

    Returns:
        Entropy of the counts, a positive number.

    """
    if isinstance(counts, dict):
        counts = counts.values()

    total = 0.0
    clogc = 0.0
    for c in counts:
        total += c
        try:
            clogc += c * log(c)
        except ValueError:
            pass
    try:
        return -(clogc/total - log(total)) / base
    except ValueError:
        return 0.0

def conditional_entropy(dict_of_counters):
    """ conditional entropy

    Give the conditional entropy of a conditional distribution X|Y,
    represented as:
        * A dictionary of dictionaries of counts {Y -> {X -> count}}, or
        * an iterable of joint counts (Y,X).

    """
    if isinstance(dict_of_counters, dict):
        return conditional_entropy_of_counters(dict_of_counters)
    else:
        counts = defaultdict(Counter)
        for x, y in pairs:
            counts[y][x] += 1
        conditional_counts = (counts_in_context.values()
                              for counts_in_context in counts.values())
        return conditional_entropy_of_counts(conditional_counts)    

def conditional_entropy_of_counts(iterable_of_iterables):
    """ conditional entropy of counts

    Conditional entropy of a conditional distribution X|Y 
    represented as an iterable of iterables of counts. 
    An index in the enclosing iterable corresponds to a value of Y;
    an index in an internal iterable corresponds to a value of X.

    Example:
    >> c = [[1, 1], [3, 1]] # A conditional distribution X|Y where c[0][0]
                            # is the counts of X=0 given Y=0, c[0][1] is 
                            # the counts of X=1 given Y=0, etc.
                            # The result is (1/3)*H(X|Y=0) + (2/3)*H(X|Y=1)
    >> conditional_entropy_of_counts(c)
    1.5010861115918823 

    """    
    entropy = 0.0
    grand_total = 0.0
    for counts in iterable_of_iterables:
        total = 0.0
        clogc = 0.0
        for c in counts:
            total += c
            try:
                clogc += c * log(c)
            except ValueError:
                pass
        grand_total += total
        try:
            entropy += total * -(clogc/total - log(total)) / base
        except ZeroDivisionError:
            pass
    try:
        return entropy / grand_total
    except ZeroDivisionError:
        return 0.0

def test_conditional_entropy_of_counts():
    def is_close(a, b):
        return abs(a - b) < 0.0000001
    c = [[1, 1], [3, 1]]
    result = conditional_entropy_of_counts(c)
    assert is_close(result, (1/3)*entropy([1, 1]) + (2/3)*entropy([3, 1]))    

class KLDomainError(Exception):
    pass

def kl(P, Q):
    """ Kullback-Leibler Divergence from P to Q.
    
    P and Q are dicts representing unnormalized discrete probability
    distributions.
    
    """
    result = 0.0
    Z_P = 0.0
    Z_Q = 0.0
    for i, c_i in P.items():
        try:
            result += c_i * log(c_i)
            Z_P += c_i
            cq_i = Q[i]
            try:
                result -= c_i * log(cq_i)
            except ValueError:
                error_str = "KL requires Q[i] == 0 -> P[i] == 0. "
                error_str += "Q[%s] == 0 but P[%s] == %s" % (i, i, c_i)
                raise KLDomainError(error_str)
            Z_Q += cq_i
        except ValueError:
            pass
    result += Z_P * (log(Z_Q) - log(Z_P))
    return (result / Z_P) / base
    
def mutual_information(counts):
    """ mutual information
    
    Takes iterable of tuples of form ((x_value, y_value), count)
    or counter whose keys are tuples (x_value, y_value).

    Stores marginal counts in memory.

    """
    if isinstance(counts, dict):
        counts = counts.items()

    total = 0
    c_x = Counter()
    c_y = Counter()
    clogc = 0

    for (x, y), c_xy in counts:
        total += c_xy
        c_x[x] += c_xy
        c_y[y] += c_xy
        try:
            clogc += c_xy * log(c_xy)
        except ValueError:
            pass

    return (log(total)
            + (clogc
               - sum(c*log(c) for c in c_x.values())
               - sum(c*log(c) for c in c_y.values()))
            / total) / base

def _generate_counts(lines):
    first_line = next(lines)
    first_line_elems = first_line.split()
    if any(x.isdigit() for x in first_line_elems):
        yield _get_count(first_line)
        for line in lines:
            yield _get_count(line)
    counts = Counter(lines)
    counts[line] += 1
    for count in counts.values():
        yield count
      
def _get_count(line):
    line = line.split()
    for x in line:
        if x.isdigit():
            return float(x)

if __name__ == "__main__":
    import sys
    lines = sys.stdin
    result = entropy(_generate_counts(lines))
    print(result)
