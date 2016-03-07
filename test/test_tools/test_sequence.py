from itertools import count

from fluxion.tools import Sequence


def test_correctness():
    seq = Sequence(dict(
        a=[x * 0.5 for x in range(11)],
        b=[x for x in range(5)],
        c=[1.5, 2.5, 4]
        ))

    ts = count(0, 1.5)
    res = []
    for t in ts:
        res.append((t, seq.get_events_until(t)))
        if seq.empty():
            break

    res_ref = [
        (0, [(0.0, ['a', 'b'])]),
        (1, [(0.5, ['a']), (1.0, ['a', 'b'])]),
        (2, [(1.5, ['a', 'c']), (2.0, ['a', 'b'])]),
        (3, [(2.5, ['a', 'c']), (3.0, ['a', 'b'])]),
        (4, [(3.5, ['a']), (4.0, ['a', 'b', 'c'])]),
        (5, [(4.5, ['a']), (5.0, ['a'])])]

    assert res == res_ref


def test_profile():
    seq = Sequence(dict(
        a=[x * 1e-4 for x in range(101)],
        b=[x * 1e-3 for x in range(1001)],
        c=[x * 2.5e-4 for x in range(401)]
        ))

    ts = count(0, 1e-6)
    res = []
    for t in ts:
        events = seq.get_events_until(t)
        if seq.empty():
            break


if __name__ == '__main__':
    test_correctness()
    test_profile()

