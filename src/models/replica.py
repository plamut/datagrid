class Replica(object):
    """A representation of a replica (file)."""

    def __init__(self, name, size):
        """Initialize replica instance.

        :param str name: replica name
        :param int size: replica size in megabits
        """
        self._name = name

        if size <= 0:
            raise ValueError("Size must be positive.")
        self._size = size

    @property
    def name(self):
        """Replica's name."""
        return self._name

    @property
    def size(self):
        """Size of the replica in megabits."""
        return self._size
