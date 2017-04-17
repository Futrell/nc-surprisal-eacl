import sys
from collections import Counter

import entropy

def mi_contributions(lines):
    for line in lines:
        try:
            phrase, count = line.strip().split("\t")
            count = int(count)
            if count:
                word1, word2 = phrase.split(" ")
                yield word1, word2, count
        except ValueError:
            pass

def mi(lines, vocab):
    d1 = Counter()
    d2 = Counter()
    djoint = Counter()
    for w1, w2, c in mi_contributions(lines):
        if (not vocab) or (w1 in vocab and w2 in vocab):
            d1[w1] += c
            d2[w2] += c
            djoint[w1, w2] += c
    return (
        entropy.entropy(d1.values()) 
        + entropy.entropy(d2.values()) 
        - entropy.entropy(djoint.values())
    )

def main(vocab_filename, vocab_cutoff):
    vocab_cutoff = int(vocab_cutoff)
    vocab = set()
    with open(vocab_filename) as infile:
        for i, line in zip(range(vocab_cutoff), infile):
            vocab.add(line.strip().split()[0])
    print(mi(sys.stdin, vocab))

if __name__ == '__main__':
    main(*sys.argv[1:])
