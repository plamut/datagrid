from models.node import Node
from models.node import Strategy
from models.replica import Replica
from random import randint
from random import seed


class Simulation(object):
    """TODO: docstring"""

    RND_SEED = 1  # XXX configurable?
    REPLICA_MIN_SIZE = 100  # megabits   XXX: make configurable?
    REPLICA_MAX_SIZE = 1000  # megabits   XXX: make configurable?

    def __init__(
        self, node_capacity=50000, strategy=Strategy.EFS,
        replica_count=1000
    ):
        self._node_capacity = node_capacity
        self._strategy = strategy  # TODO: convert to "enum"

        self._replica_count = replica_count
        self._replicas = []
        self._nodes = []

        # TODO: replica count, replica sizes

    def _add_node(self, *args, **kwargs):
        """TODO"""
        node = Node(*args, **kwargs)
        self._nodes.append(node)
        return node

    def init_grid(self):
        """TODO"""
        # XXX: why does the simulation say that the number of nodes is 50,
        # but in the picture there are only 14 (including the server)?
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
            parent=None
        )

        node_3 = self._add_node(
            "node_3", capacity=self._node_capacity, parent=server)
        self._add_node("node_1", capacity=self._node_capacity, parent=node_3)
        self._add_node("node_2", capacity=self._node_capacity, parent=node_3)

        self._add_node("node_5", capacity=self._node_capacity, parent=server)
        node_4 = self._add_node(
            "node_4", capacity=self._node_capacity, parent=server)
        self._add_node("node_6", capacity=self._node_capacity, parent=node_4)

        node_7 = self._add_node(
            "node_7", capacity=self._node_capacity, parent=server)
        node_9 = self._add_node(
            "node_9", capacity=self._node_capacity, parent=node_7)
        self._add_node("node_12", capacity=self._node_capacity, parent=node_9)

        node_8 = self._add_node(
            "node_8", capacity=self._node_capacity, parent=server)
        self._add_node("node_10", capacity=self._node_capacity, parent=node_8)
        node_11 = self._add_node(
            "node_11", capacity=self._node_capacity, parent=node_8)
        self._add_node("node_13", capacity=self._node_capacity, parent=node_11)

        # TODO: what to return? some Grid object? and Grid.get_server ...
        # or just store all replicas for later access?