""" Diagram generator

Author: Hugo U. R. Strand (2023) """


from copy import copy

from functools import reduce


def flatten_pairing_to_list(pairing):
    return reduce(lambda x, y: list(x) + list(y), pairing)


def pop_pair(vec, i, j):
    assert( i != j )
    indices = [i, j]
    p = tuple([ vec[idx] for idx in indices ])
    rest = copy(vec)
    for idx in sorted(indices, reverse=True):
        del rest[idx]
    return p, rest


def _recursive_pairing(parity, partial_pairing, unpaired):

    assert( len(unpaired) % 2 == 0 ) # even number of unpaired indices
    
    if len(unpaired) == 0:
        yield parity, partial_pairing

    for i in range(1, len(unpaired)):
        p, rest = pop_pair(unpaired, 0, i)
        new_partial_pairing = partial_pairing + [p]
        new_parity = -parity if i % 2 == 0 else parity
        yield from _recursive_pairing(new_parity, new_partial_pairing, rest)


def all_pairings(order):    
    unpaired = [ i for i in range(2*order) ]
    yield from _recursive_pairing(1, [], unpaired)


def all_connected_pairings(order, k=1):
    for parity, pairing in all_pairings(order):
        if is_connected(pairing, k=k):
            yield parity, pairing


def is_crossing(p1, p2):
    assert( p1[0] < p1[1] )
    assert( p2[0] < p2[1] )
    return \
        (p1[0] < p2[0] and p2[0] < p1[1] and p1[1] < p2[1]) or \
        (p2[0] < p1[0] and p1[0] < p2[1] and p2[1] < p1[1])


def is_k_connector(pair, k):
    k = k - 1
    return (pair[0] <= k and pair[1] > k) or (pair[1] <= k and pair[0] > k)


def split_pairing(pairing, k):
    rest, connectors = [], []
    for pair in pairing:
        if is_k_connector(pair, k):
            connectors.append(pair)
        else:
            rest.append(pair)
    return connectors, rest


def is_connected(pairing, k=1):

    connected = []
    stack, disconnected = split_pairing(pairing, k=k)
    
    #pairing = copy(pairing)
    #stack = [pairing.pop(0)]
    #disconnected = pairing    
    
    while len(stack) > 0:
        p_c = stack.pop(0)
        connected.append(p_c)
        for idx, p in reversed(list(enumerate(disconnected))):
            if is_crossing(p_c, p):
                disconnected.pop(idx)
                stack.append(p)

    return len(disconnected) == 0


if __name__ == '__main__':

    for order in [1, 2, 3, 4]:
        print('-'*72)
        print(f'order = {order}')
        print('-'*72)
        for par, pair in all_connected_pairings(order):
            print(f'{par:+d}, {pair}')


    def plot_pairing(pairing):

        n = 2*len(pairing)
        
    import numpy as np    
    import matplotlib.pyplot as plt

    from matplotlib.patches import Arc

    
    def plot_pairs(ax, pairs, ks=[]):

        colors = dict()
        for k in ks:
            c = ax.plot([], [])[0].get_color()
            colors[k] = c
        
        # arcs
        for p in pairs:
            r = p[1] - p[0]
            c = 0.5*np.sum(p)
            color = 'gray'
            for k in ks:
                if is_k_connector(p, k=k):
                    color = colors[k]
                    
            a = Arc((c, 0), r, r, theta1=0, theta2=180, fill=False, color=color)
            ax.add_patch(a)

        # backbone
        n = 2*len(pairs) - 1
        ax.plot([0, n], [0, 0], 'k', lw=2)

        # dashed vertical lines
        for k in ks:
            x = k - 0.5
            ax.plot([x], [0], '.r')

        ax.tick_params(bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.spines[:].set_visible(False)
        ax.set_ylim([-0.5, (n+1)//2 - 0.25])

    order = 1
    nks = { 1:1, 2:2, 3:7, 4:42 }

    nk = nks[order]
    nr = 2*order - 1
    subp = [nr, nk, 0]

    fig = plt.figure(figsize=(26 * nk/42, 4 * nr/7))

    for k in range(1, 2*order):
        
        parity_pairs = [ x for x in all_connected_pairings(order, k=k) ]
        n = len(parity_pairs)
        print(f'n = {n}')

        for par, pairs in parity_pairs:
            subp[-1] += 1
            ax = fig.add_subplot(*subp, aspect='equal')
            plot_pairs(ax, pairs, ks=[k])

        subp[-1] = nk * k
        
    plt.tight_layout()
    plt.savefig(f'figure_order_{order}.pdf')
    plt.show()
    
