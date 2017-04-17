""" Calculate dependency pair MI from Syntactic N-grams """

import sys
import itertools
import operator

YEAR_RANGE = range(1960, 2000+1)

def sum_over_years(years):
    def gen():
        for part in years:
            year, count = part.split(",")
            if int(year) in YEAR_RANGE:
                yield int(count)
    return sum(gen())

def normalize(partcode):
    # 01 -> 01
    # 20 -> 01
    # 202 -> 011
    # 011 -> 011
    # 330 -> 011
    # 012 -> 012
    # 201 -> 012
    # 310 -> 012
    state = itertools.count()
    seen = {}
    for thing in partcode:
        if thing in seen:
            yield seen[thing]
        else:
            result = seen[thing] = next(state)
            yield result

def read_lines(lines, match_code, get_code):
    def read_line(line):
        _, phrase, _, *years = line.strip().split("\t")
        phrase = [part.split("/") for part in phrase.split()]
        phrase = sorted(phrase, key=operator.itemgetter(-1))
        partcode = tuple(normalize(sorted(int(part[-1]) for part in phrase)))
        if partcode != match_code:
            #print("Rejected %s" % phrase, file=sys.stderr)
            return None
        else:
            relevant_parts = [phrase[i][0].lower() for i in get_code]
            count = sum_over_years(years)
            if count:
                return relevant_parts, count
            else:
                return None
    return filter(None, map(read_line, lines))

def main(match_code, get_code):
    match_code = tuple(map(int, match_code))
    get_code = tuple(map(int, get_code))
    lines = read_lines(sys.stdin, match_code, get_code)
    for parts, count in lines:
        print(" ".join(parts), count, sep="\t")

if __name__ == '__main__':
    main(*sys.argv[1:])
    
