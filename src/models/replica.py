class Replica(object):
    """A representation of a replica (file)."""

    def __init__(self, name='', size=100):
        self._name = name
        self._size = size  # TODO: check for > 0?

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size
