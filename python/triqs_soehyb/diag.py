""" Diagram generator

Author: Hugo U. R. Strand (2023) """


from copy import copy


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


def all_connected_pairings(order):
    for parity, pairing in all_pairings(order):
        if is_connected(pairing):
            yield parity, pairing


def is_crossing(p1, p2):
    assert( p1[0] < p1[1] )
    assert( p2[0] < p2[1] )
    return \
        (p1[0] < p2[0] and p2[0] < p1[1] and p1[1] < p2[1]) or \
        (p2[0] < p1[0] and p1[0] < p2[1] and p2[1] < p1[1])

            
def is_connected(pairing):

    connected = []
    pairing = copy(pairing)
    stack = [pairing.pop(0)]
    disconnected = pairing
    
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

