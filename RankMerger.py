import numpy as np


def merge(ranks1, ranks2):
    new_ranks = []
    for el1, el2 in zip(ranks1[0], ranks2[0]):
        new_el = []
        new_s = []
        l = 0
        m = 0
        i = 0
        while i < 10:
            if el1[l] not in new_el:
                new_el.append(el1[l])
                i += 1
            l += 1
            if el2[m] not in new_el:
                new_el.append(el2[m])
                i += 1
            m += 1
        new_ranks.append(new_el)

    # Scores are not correct
    return [new_ranks, ranks1[1]]




