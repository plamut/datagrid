from collections import namedtuple
from models.node import Node
from models.replica import Replica
from random import randint
from random import seed


Strategy = namedtuple('Strategy', [
    'EFS',  # Enhanced Fast Spread, used in the 2011 paper
    's2012',  # TODO
    's2013',  # TODO
])
Strategy = Strategy(*range(1, 4))


class _Clock(object):
    """Representation of a simulation clock."""

    def __init__(self):
        self._time = 0

    @property
    def time(self):
        return self._time

    def reset(self):
        self._time = 0

    # XXX: add step parameter? for advancing for more than one unit?
    def tick(self):
        self._time += 1


class Simulation(object):
    """TODO: docstring"""

    RND_SEED = 1  # XXX configurable?
    REPLICA_MIN_SIZE = 100  # megabits   XXX: make configurable?
    REPLICA_MAX_SIZE = 1000  # megabits   XXX: make configurable?
    TOTAL_REQUESTS = 100000  # simmulation steps   XXX: make configurable?

    def __init__(
        self, node_capacity=50000, strategy=Strategy.EFS,
        replica_count=1000
    ):
        self._node_capacity = node_capacity
        self._strategy = strategy  # TODO: convert to "enum"

        self._replica_count = replica_count
        self._replicas = []
        self._nodes = []

        self._clock = _Clock()

        # TODO: replica count, replica sizes

    def _add_node(self, *args, **kwargs):
        """TODO"""
        node = Node(*args, **kwargs)
        self._nodes.append(node)
        return node

    @property
    def time(self):
        return self._clock.time

    def init_grid(self):
        """TODO"""
        # XXX: why does the simulation say that the number of nodes is 50,
        # but in the picture there are only 14 (including the server)?
        self._clock.reset()

        self._replicas = []
        self._nodes = []

        # TODO: later remove (we have it now for deterministic behavior)
        seed(self.RND_SEED)

        # generate replicas
        for i in range(self._replica_count):
            replica = Replica(
                name='replica_' + str(i),  # TODO: zero padding
                size=randint(self.REPLICA_MIN_SIZE, self.REPLICA_MAX_SIZE)
            )
            self._replicas.append(replica)

        server = self._add_node(
            "Server Node",
            # server's capacity must be big enough to hold all replicas
            capacity=self._replica_count * self.REPLICA_MAX_SIZE,
            parent=None, sim=self
        )

        node_3 = self._add_node(
            "node_3", capacity=self._node_capacity, parent=server,
            sim=self)
        self._add_node(
            "node_1", capacity=self._node_capacity, parent=node_3,
            sim=self)
        self._add_node(
            "node_2", capacity=self._node_capacity, parent=node_3,
            sim=self)

        self._add_node(
            "node_5", capacity=self._node_capacity, parent=server,
            sim=self)
        node_4 = self._add_node(
            "node_4", capacity=self._node_capacity, parent=server,
            sim=self)
        self._add_node(
            "node_6", capacity=self._node_capacity, parent=node_4,
            sim=self)

        node_7 = self._add_node(
            "node_7", capacity=self._node_capacity, parent=server,
            sim=self)
        node_9 = self._add_node(
            "node_9", capacity=self._node_capacity, parent=node_7,
            sim=self)
        self._add_node(
            "node_12", capacity=self._node_capacity, parent=node_9,
            sim=self)

        node_8 = self._add_node(
            "node_8", capacity=self._node_capacity, parent=server,
            sim=self)
        self._add_node(
            "node_10", capacity=self._node_capacity, parent=node_8,
            sim=self)
        node_11 = self._add_node(
            "node_11", capacity=self._node_capacity, parent=node_8,
            sim=self)
        self._add_node(
            "node_13", capacity=self._node_capacity, parent=node_11,
            sim=self)

        # TODO: what to return? some Grid object? and Grid.get_server ...
        # or just store all replicas for later access?

    def run(self):
        """TODO:"""

        for i in range(self.TOTAL_REQUESTS):
            self._clock.tick()

            # random node requests a random replica
