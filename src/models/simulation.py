from enum import IntEnum
from enum import unique
from models.replica import Replica


@unique
class Strategy(IntEnum):
    EFS = 1  # Enhanced Fast Spread, used in the 2012 paper
    s2012 = 2  # TODO
    s2013 = 3  # TODO


class Simulation(object):
    """TODO: docstring"""

    def __init__(node_capacity=50000, strategy=Strategy.EFS,
        replica_count=1000):
        self._node_capacity = node_capacity
        self._strategy = strategy  # TODO: convert to "enum"

        self._replica_count = replica_count

        # TODO: replica count, replica sizes

    def init_grid():
        """TODO"""
        # XXX: why does the simulation say that the number of nodes is 50,
        # but in the picture there are only 14 (including the server)?

        # generate replicas
        for i in range(self._replica_count):
            replica = Replica(
                name = 'replica_' + str(i),  # TODO: zero padding
                capacity = 100,  # TODO: random ... fixed seed optional
            )
            self._replicas.append(replica)

        # TODO: set larger than replica amount
        server = Node("Server Node", capacity=99999999, parent=None)

        node_3 = Node("node_3", capacity=self._node_capacity, parent=server)
        node_1 = Node("node_1", capacity=self._node_capacity, parent=node_3)
        node_2 = Node("node_2", capacity=self._node_capacity, parent=node_3)

        node_5 = Node("node_5", capacity=self._node_capacity, parent=server)
        node_4 = Node("node_4", capacity=self._node_capacity, parent=server)
        node_6 = Node("node_6", capacity=self._node_capacity, parent=node_4)

        node_7 = Node("node_7", capacity=self._node_capacity, parent=server)
        node_9 = Node("node_9", capacity=self._node_capacity, parent=node_7)
        node_12 = Node("node_12", capacity=self._node_capacity, parent=node_9)

        node_8 = Node("node_8", capacity=self._node_capacity, parent=server)
        node_10 = Node("node_10", capacity=self._node_capacity, parent=node_8)
        node_11 = Node("node_11", capacity=self._node_capacity, parent=node_8)
        node_13 = Node("node_13", capacity=self._node_capacity, parent=node_11)

        # TODO: what to return? some Grid object? and Grid.get_server ...
        # or just store all replicas for later access?
