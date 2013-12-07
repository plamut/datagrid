from collections import namedtuple
from collections import OrderedDict
from models.event import ReplicaRequest
from models.event import ReplicaSendBack
from models.node import Node
from models.replica import Replica
from types import SimpleNamespace

import heapq
import itertools
import math
import random


def _digits(number):
    """Caclulate the number of digits in an integer.

    :param number: number in question
    :type number: int

    :returns: number of digits
    :rtype: int
    """
    if number == 0:
        return 1

    if number < 0:
        number = -number
    return int(math.log10(number)) + 1


Strategy = namedtuple('Strategy', [
    'EFS',  # Enhanced Fast Spread, used in the 2011 paper
    's2012',  # TODO
    's2013',  # TODO
])
Strategy = Strategy(*range(1, 4))
"""Possible strategies for the nodes to use when deciding, which
replica(s) (if any) to replace when locally storing a new replica."""


class _Clock(object):
    """Representation of a simulation clock."""

    def __init__(self):
        self._time = 0.0

    @property
    def time(self):
        """Current time."""
        return self._time

    def reset(self):
        """Reset current simulation time to zero (0)."""
        self._time = 0.0

    def tick(self, step=1.0):
        """Increase current simulation time by `step`.

        :param step: by how much to increase the time
            (optional, default: 1)
        :type step: int

        :returns: new simulation time
        :rtype: int
        """
        if step < 0:
            raise ValueError("Clock step must be non-negative.")
        # self._time += int(step)
        self._time += step
        return self._time


# XXX: provide a SimulationView for Node objects? A "readonly view" of the
# sim object for nodes to use

class Simulation(object):
    """Simulation runner."""

    SERVER_NAME = 'server'  # XXX: make configurable?

    def __init__(
        self, node_capacity=50000, strategy=Strategy.EFS,
        replica_count=1000, node_count=20, fsti=10000,
        min_dist_km=1, max_dist_km=1000,
        replica_min_size=100, replica_max_size=1000,
        rnd_seed=None, total_reqs=100000,
    ):
        """Initialize simulation parameters.

        :param node_count: number of nodes in the grid including the server
            node
            (optional, default: 20)
        :type node_count: int
        :param node_capacity: capacity of each non-server node in megabits
            (optional, default: 50000)
        :type node_capacity: int
        :param strategy: strategy to use for replica replacement
            (optional, deafult: Strategy.EFS)
        :type strategy: int
        :param min_dist_km:  min distance between two adjacent nodes in
            kilometers
            (optional, default: 1)
        :type min_dist_km: int
        :param max_dist_km:  max distance between two adjacent nodes in
            kilometers
            (optional, default: 1)
        :type max_dist_km: int
        :param replica_count: number of different replicas in simulation
            (optional, default: 1000)
        :type replica_count: int
        :param replica_min_size: min size of a single replica in megabits
            (optional, default: 100)
        :param replica_max_size: max size of a single replica in megabits
            (optional, default: 1000)
        :param total_reqs: number of replica requests to generate during the
            simulation
            (optional, default: 100000)
        :type total_reqs: int
        :param fsti: frequency specific time interval as defined in the paper
            (optional, default: 1000)
        :type fsti: int
        :param rnd_seed: seed for the random number generator
            (optional, default: current system time)
        :type rnd_seed: int
        """
        self._node_capacity = node_capacity
        self._strategy = strategy

        if node_count < 2:
            raise ValueError("Grid's node_count must be at least 2.")
        self._node_count = node_count

        if replica_count < 1:
            raise ValueError("Number of replicas must be positive.")
        self._replica_count = replica_count

        if fsti < 1:
            raise ValueError("FSTE must be positive.")
        self._fsti = fsti

        self._replicas = OrderedDict()
        self._nodes = OrderedDict()
        self._edges = OrderedDict()  # nodes' outbound edges

        self._clock = _Clock()

        if min_dist_km >= max_dist_km:
            raise ValueError("Min distance must be smaller than max distance")

        if min_dist_km <= 0:
            raise ValueError("Minimum distance must be positive.")
        self._min_dist_km = min_dist_km

        if max_dist_km <= 0:
            raise ValueError("Maximum distance must be positive.")
        self._max_dist_km = max_dist_km

        # XXX: don't hard-code?
        self._network_bw_mbps = 10e6  # network bandwidth (Mbits/s)
        self._pspeed_kmps = 6e3  # propagation speed (km/s)

        if replica_min_size >= replica_max_size:
            raise ValueError(
                "Min replica size must be smaller than max replica size.")

        if replica_min_size <= 0:
            raise ValueError("Minimum replica size must be positive.")
        self._replica_min_size = replica_min_size

        if replica_max_size <= 0:
            raise ValueError("Maximum replica size must be positive.")
        self._replica_max_size = replica_max_size

        self._rnd_seed = rnd_seed

        if total_reqs <= 0:
            raise ValueError("Total number of requests must me be positive.")
        self._total_reqs = total_reqs

        self._total_bw = 0.0  # total bandwidth used (in megabits)
        self._total_rt_s = 0.0  # total response time (in seconds)

        # two constants used in metrics calculations
        self._c1 = 0.001
        self._c2 = 0.001

    def _new_node(self, *args, **kwargs):
        """Create a new grid node and store it in the list of nodes.

        :returns: newly created node instance
        :rtype: :py:class:`~models.node.Node`
        """
        node = Node(*args, **kwargs)
        self._nodes[node.name] = node
        return node

    @property
    def now(self):
        """Current simulation time."""
        return self._clock.time

    @property
    def fsti(self):
        """Frequency specific time interval."""
        return self._fsti

    @property
    def nodes(self):
        """List of nodes in the grid."""
        return self._nodes

    @property
    def replicas(self):
        """List of all replicas."""
        return self._replicas

    def _generate_nodes(self):
        """Generate a new set of grid nodes."""
        self._nodes = OrderedDict()

        self._new_node(
            self.SERVER_NAME,
            # server's capacity must be big enough to hold all replicas
            capacity=self._replica_count * self._replica_max_size,
            sim=self,
            replicas=self._replicas,
        )

        name_tpl = 'node_{{0:0{}d}}'.format(_digits(self._node_count - 1))
        for i in range(1, self._node_count):
            self._new_node(
                name=name_tpl.format(i),
                capacity=self._node_capacity,
                sim=self,
            )

    def _generate_edges(self):
        """Generate edges with random distances between grid nodes
        (complete graph topology).

        Edge distances lie in the interval [`min_dist_km`, `max_dist_km`].
        """
        self._edges = OrderedDict()

        for node_name in self._nodes:
            self._edges[node_name] = OrderedDict()

        for node_1, node_2 in itertools.combinations(self._nodes, 2):
            dist = random.randint(self._min_dist_km, self._max_dist_km)
            self._edges[node_1][node_2] = dist
            self._edges[node_2][node_1] = dist

    def _generate_replicas(self):
        """Generate `replica_count` replicas with random sizes.

        Sizes lie in the interval [`replica_min_size`, `replica_max_size`].
        """
        name_tpl = 'replica_{{0:0{}d}}'.format(_digits(self._replica_count))

        self._replicas = OrderedDict()

        for i in range(1, self._replica_count + 1):
            replica = Replica(
                name=name_tpl.format(i),
                size=random.randint(
                    self._replica_min_size, self._replica_max_size)
            )
            self._replicas[replica.name] = replica

    def _dijkstra(self):
        """Find the shortest paths from server node to all other nodes.

        :returns: node information as calculated by Dijkstra algorithm
        :rtype: OrderedDict (key: node name, value: node info object)
            Node info object is a SimpleNamespace instance with the
            following attributes:
            * dist: total distance from node to server node
            * previous: previous node on the shortest path to server
            * visited: always True (result of Dijsktra algorithm)
        """
        node_info = OrderedDict()

        # initialize all distances to infinity, all nodes as not visited and
        # all previous nodes on shortest paths as non-existing
        for name in self._nodes:
            node_info[name] = SimpleNamespace(
                dist=float('inf'), visited=False, previous=None)

        q = set()  # a set of nodes still to examine
        source = self.SERVER_NAME

        node_info[source].dist = 0
        q.add(source)

        while q:
            # find: node in q with smallest distance (and has not been
            # visited) and remove it from the list
            # XXX: use heap for efficiency? (just if it runs too slow)
            u = min(q, key=lambda node: node_info[node].dist)
            q.remove(u)
            node_info[u].visited = True

            for v, dist_u_v in self._edges[u].items():  # all neighbors of u
                alt = node_info[u].dist + dist_u_v

                # if alternative path to v (through u) is smaller than the
                # currently known best path to v, update v
                if alt < node_info[v].dist and not node_info[v].visited:
                    node_info[v].dist = alt
                    node_info[v].previous = u
                    q.add(v)

        return node_info

    def notify_repl_req(self, node, parent, replica):
        """Notify simulation that `node` had to request `replica` from its
        `parent` (on the shortest path to server).

        :param node: node that issued a request for `replica`
        :type node: :py:class:`~models.node.Node`
        :param parent: `node`'s parent that received replica request
        :type parent: :py:class:`~models.node.Node`
        :param replica: replica requested
        :type replica: :py:class:`~models.replica.Replica`
        """
        # total BW consumpion
        self._total_bw += replica.size  # XXX: correct interpretation?

        # total response time
        dist_km = self._edges[node.name][parent.name]
        self._total_rt_s += replica.size / self._network_bw_mbps + \
            dist_km / self._pspeed_kmps

    def initialize(self):
        """Initialize (reset) a simulation."""
        self._clock.reset()
        random.seed(self._rnd_seed)

        self._total_bw = 0.0
        self._total_rt_s = 0.0

        # generate nodes
        self._generate_replicas()
        self._generate_nodes()
        self._generate_edges()

        # calculate shortest paths from server node to all the other nodes
        # and update each of the latter with a relevant path
        node_info = self._dijkstra()
        for name, info in node_info.items():
            self._nodes[name].set_parent(self._nodes.get(info.previous))

    def run(self):
        """Run simulation.

        NOTE: simulation must have already been initialized with initialize()
        """
        ef = _EventFactory(self)
        self._clock.reset()

        self.event_queue = []

        event = ef.get_random()

        heapq.heappush(self.event_queue, event)

        # while evenet queue not empty: process next event
        while len(self.event_queue) > 0:
            # print("[SIM @ {}] Events in queue:".format(self.now), len(self.event_queue))
            event = heapq.heappop(self.event_queue)

            msg = "[SIM @ {}] Next event at {}: {}".format(
                self.now, event.time, event)
            print(msg)

            # fast-forward time to event occurence
            self._clock.tick(step=event.time - self.now)
            msg = "[SIM @ {0}] Time changed to {0}".format(self.now)
            print(msg)

            # execute event
            self._exec_event(event)

        # for i in range(1, self._total_reqs + 1):
        #     #if i % 1000 == 0:
        #     print('\033[1m', "request {}/{} ({:.2f} %)".format(
        #             i, self._total_reqs, 100 * i / self._total_reqs),
        #         '\033[0m',
        #         sep='')

        #     event = ef.get_random()

        #     # fast-forward time to event occurence
        #     self._clock.tick(step=event.time - self.now)

        #     # execute an event (issue replica request)
        #     gen = event.node.request_replica(event.replica.name)
        #     replica = next(gen)
        #     print("[SIM] Got replica from {}: {}".format(
        #         event.node.name, replica.name))

        #     try:
        #         gen.send("NEW_TIME="+str(i))
        #     except StopIteration:
        #         print('')

        print("-" * 10, "END", "-" * 10)
        print("Total resp. time (s):", self._total_rt_s * self._c1)
        print("Total bandwidth:", self._total_bw * self._c2)

    def _exec_event(self, event):
        """TODO: docstring (executing an event)"""
        if type(event) == ReplicaRequest:

            print("[SIM @ {}] processing REPL_REQ event".format(self.now))
            g = event.target.request_replica(event.replica_name, event.source)

            new_event = next(g)
            # parent, replica_name, event_type = next(g)

            msg = "[SIM @ {}] {} returned event {})".format(
                self.now, new_event.source.name, new_event)
            print(msg)

            self._schedule_event(new_event)

        elif type(event) == ReplicaSendBack:
            print("[SIM @ {}] processing REPL_SEND_BACK event".format(self.now))

        else:
            raise ValueError("Unknown event", type(event))

    def _schedule_event(self, event):
        """TODO: schedules event based on its type and places it into event
        queue
        """
        if type(event) == ReplicaRequest:
            src_name = event.source.name
            target_name = event.target.name

            dist_km = self._edges[src_name][target_name]
            cl = dist_km / self._pspeed_kmps  # communication latency

            # it takes some time for the parent node to receive the request
            event.time += cl

        elif type(event) == ReplicaSendBack:
            print("[SIM @ {}] processing REPL_RECEIVED event".format(self.now))

        else:
            raise ValueError("Unknown event", type(event))

        heapq.heappush(self.event_queue, event)


class _EventFactory(object):
    """Simulation events factory."""

    def __init__(self, sim):
        self._sim = sim

    def get_random(self):
        """Create a new random event (some node requests a replica).

        :returns: new random instance of node request event
        :rtype: :py:class:`~models.simulation.NodeRequest`
        """
        # XXX: is there a better way to select a random element from a dict?
        # (without converting to list, which is slow) - BTW, does random.choice
        # do exactly that? converting to a list first?

        # XXX: exclude server node from this!
        node = random.sample(self._sim.nodes.keys(), 1)[0]
        node = self._sim.nodes[node]

        # XXX: in other scenarios, replica request distribution is not
        # uniform - change when needed
        replica = random.sample(self._sim.replicas.keys(), 1)[0]
        replica = self._sim.replicas[replica]

        # XXX: don't hardcode the limits, read them from the simulation
        # object (and the same for time?)
        time_from_now = random.randint(0, 99)
        event_time = time_from_now + self._sim.now

        # return NodeRequest(node, replica, self._sim.now + time_from_now)
        return ReplicaRequest(node, node, replica.name, event_time)


class NodeRequest(object):
    """An event when node requests a replica."""

    def __init__(self, node, replica, time):
        """Initialize newly created instance.

        :param node: node that requests a replica
        :type node: string
        :param replica: requested replica
        :type replica: string
        :param time: time when request is issued
        :type time: int
        """
        self.node = node
        self.replica = replica
        self.time = time

    def __str__(self):
        node_name = self.node.name if self.node else "None"
        return "NodeReqest ({} req. {} @ {})".format(
            node_name, self.replica.name, self.time)
