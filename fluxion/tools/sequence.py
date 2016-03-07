"""
A simple implementation of a sorted sequence combinator.
"""

from collections import OrderedDict


def close(a, b, rtol=1e-5, atol=1e-8):
    """
    Determines if two values are close enough to be yielded together.
    """
    # Same formula as ``numpy.allclose()``, but extracted here for performance reasons
    # (``alclose()`` does various checks for arrays, we do not need it here).
    return abs(a - b) <= (atol + rtol * abs(b))


class Lookahead:

    def __init__(self, seq):
        self.iter = iter(seq)
        self.empty = False
        self.peek = None
        try:
            self.peek = next(self.iter)
        except StopIteration:
            self.empty = True

    def next(self):
        if self.empty:
            raise StopIteration()
        to_return = self.peek
        try:
            self.peek = next(self.iter)
        except StopIteration:
            self.peek = None
            self.empty = True
        return to_return


class Sequence:

    def __init__(self, seqs, close=close):
        # Keeping iterators in a sorted order to make output deterministic
        # (also, this will make returned events sorted by key)
        self.iters = OrderedDict((key, Lookahead(seqs[key])) for key in sorted(seqs))
        self.close = close

    def pop_events_until(self, max_val):

        close = self.close
        events = []
        to_delete = []
        for key, it in self.iters.items():

            while True:
                if it.empty:
                    to_delete.append(key)
                    break

                val = it.peek

                # handling this case separately to ensure values are snapped to max_val
                # when possible (this will help steppers to determine whether or not
                # to interpolate)
                if close(val, max_val):
                    val = max_val
                elif val > max_val:
                    break

                events.append((val, key))
                it.next()

        for key in to_delete:
            del self.iters[key]

        grouped_events = []

        # will be sorted by the first element in the tuple, that is, the value
        for val, key in sorted(events):
            if len(grouped_events) == 0:
                grouped_events.append((val, [key]))
            else:
                last_val = grouped_events[-1][0]
                if close(val, last_val):
                    grouped_events[-1][1].append(key)
                else:
                    grouped_events.append((val, [key]))

        return grouped_events

    def empty(self):
        return len(self.iters) == 0
