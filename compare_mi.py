import sys
import random
import bisect

import compute_mi

def read_file(lines, vocab):
    for line in lines:
        phrase, _ = line.split("\t")
        words = phrase.split()
        if all(word in vocab for word in words):
            yield line

def null_diffs(one, two, num_samples):
    N = len(one)
    together = one + two
    print("Permuting...", file=sys.stderr)
    for _ in range(num_samples):
        random.shuffle(together)
        one, two = together[:N], together[N:]
        yield compute_mi.mi(one, None) - compute_mi.mi(two, None)

def permutation_test(diff, one, two, num_samples):
    diffs = sorted(null_diffs(one, two, num_samples))
    N = len(diffs)
    position = bisect.bisect(diffs, diff)
    print(diffs)
    return min(position/N, 1-(position/N))

def main(filename1, filename2, vocab_filename, vocab_cutoff, num_samples):
    num_samples = int(num_samples)
    vocab_cutoff = int(vocab_cutoff)
    vocab = set()
    with open(vocab_filename) as infile:
        for i, line in zip(range(vocab_cutoff), infile):
            vocab.add(line.strip().split()[0])
    with open(filename1) as infile:
        one = list(read_file(infile, vocab))
    with open(filename2) as infile:
        two = list(read_file(infile, vocab))
    true1 = compute_mi.mi(one, None)
    true2 = compute_mi.mi(two, None)
    diff = true1 - true2
    print("I_1 = ", true1)
    print("I_2 = ", true2)
    print("D_I = ", diff)
    p = permutation_test(diff, one, two, num_samples)
    print("p = ", p)

if __name__ == '__main__':
    main(*sys.argv[1:])
