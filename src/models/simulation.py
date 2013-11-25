from collections import namedtuple
from collections import OrderedDict
from models.node import Node
from models.replica import Replica
from random import randint
from random import seed

import itertools


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
    SERVER_NAME = 'server'

    def __init__(
        self, node_capacity=50000, strategy=Strategy.EFS,
        replica_count=1000, node_count=20, fste=10000,
        min_dist_km=1, max_dist_km=1000
    ):
        self._node_capacity = node_capacity
        self._strategy = strategy  # TODO: convert to "enum"

        self._node_count = node_count  # XXX: check > 0
        self._replica_count = replica_count
        self._fste = fste

        self._replicas = OrderedDict()
        self._nodes = OrderedDict()
        self._edges = OrderedDict()  # nodes' outbound edges

        self._clock = _Clock()

        # min and max distance between adjacent nodes
        self._min_dist_km = min_dist_km
        self._max_dist_km = max_dist_km
        # TODO: replica sizes

    def _new_node(self, *args, **kwargs):
        """TODO"""
        node = Node(*args, **kwargs)
        self._nodes[node.name] = node
        return node

    @property
    def time(self):
        return self._clock.time

    # later of zero padding
    # TODO: > 0
    # digits = int(math.log10(1000)) + 1
    # str = '0{}d'.format(digits)

    def _generate_nodes(self):
        """ TODO """
        self._nodes = OrderedDict()

        self._new_node(
            self.SERVER_NAME,
            # server's capacity must be big enough to hold all replicas
            capacity=self._replica_count * self.REPLICA_MAX_SIZE,
            replicas=self._replicas,
            sim=self,
        )

        for i in range(1, self._node_count):
            self._new_node(
                'node_{}'.format(i),  # XXX zero padding?
                capacity=self._node_capacity,
                sim=self
            )

    def _generate_edges(self):
        """TODO"""
        # generate edges - full graph with random edge distances
        self._edges = OrderedDict()

        for node_name in self._nodes:
            self._edges[node_name] = OrderedDict()

        for node_1, node_2 in itertools.combinations(self._nodes, 2):
            dist = randint(self._min_dist_km, self._max_dist_km)
            self._edges[node_1][node_2] = dist
            self._edges[node_2][node_1] = dist

    def _generate_replicas(self):
        """TODO:docstring"""
        self._replicas = OrderedDict()

        for i in range(self._replica_count):
            replica = Replica(
                name='replica_{}'.format(i),  # XXX: zero padding e.g. {03d}
                size=randint(self.REPLICA_MIN_SIZE, self.REPLICA_MAX_SIZE)
            )
            self._replicas[replica.name] = replica

    def initialize(self):
        """TODO"""
        # XXX: why does the simulation say that the number of nodes is 50,
        # but in the picture there are only 14 (including the server)?
        self._clock.reset()

        # TODO: make configurable (if None, then use random seed)
        seed(self.RND_SEED)

        # generate nodes
        self._generate_replicas()
        self._generate_nodes()
        self._generate_edges()

        # calculate shortest paths from server node to all other nodes
        # and update each the latter with a shortest path to server node

        # TODO

    def run(self):
        """TODO:"""

        for i in range(self.TOTAL_REQUESTS):
            self._clock.tick()

            # random node requests a random replica
