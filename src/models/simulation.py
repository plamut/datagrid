from collections import namedtuple
from collections import OrderedDict
from models.event import ReceiveReplica
from models.event import ReceiveReplicaRequest
from models.event import SendReplica
from models.event import SendReplicaRequest
from models.node import NodeEFS
from models.node import NodeLFU
from models.node import NodeLRU
from models.replica import Replica
from types import SimpleNamespace

import heapq
import itertools
import math
import random


def _digits(number):
    """Caclulate the number of digits in an integer.

    :param int number: number in question

    :returns: number of digits in `number`
    :rtype: int
    """
    if number == 0:
        return 1

    if number < 0:
        number = -number
    return int(math.log10(number)) + 1


Strategy = namedtuple('Strategy', [
    'EFS',  # Enhanced Fast Spread, used in the 2011 paper
    'LFU',  # Least Frequently Used
    'LRU',  # Least Recently Used
    's2012',  # TODO
    's2013',  # TODO
])
Strategy = Strategy(*range(1, 6))
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

        :param float step: by how much to increase the time
            (optional, default: 1)

        :returns: new simulation time
        :rtype: float
        """
        if step < 0.0:
            raise ValueError("Clock step must be non-negative.")
        self._time += step
        return self._time


class Simulation(object):
    """Simulation runner."""

    SERVER_NAME = 'server'  # XXX: make configurable?

    def __init__(
        self, node_capacity=50000, strategy=Strategy.EFS,
        replica_count=1000, replica_group_count=10, mwg_prob=0.1,
        node_count=20, fsti=10000,
        min_dist_km=1, max_dist_km=1000,
        replica_min_size=100, replica_max_size=1000,
        rnd_seed=None, total_reqs=100000,
        network_bw_mbps=10, pspeed_kmps=6000
    ):
        """Initialize simulation parameters.

        :param int node_count: number of nodes in the grid including the
            server node
            (optional, default: 20)
        :type node_count: int
        :param int node_capacity: capacity of each non-server node in megabits
            (optional, default: 50000)
        :param int strategy: strategy to use for replica replacement
            (optional, deafult: Strategy.EFS)
        :param int min_dist_km:  min distance between two adjacent nodes in
            kilometers
            (optional, default: 1)
        :param int max_dist_km:  max distance between two adjacent nodes in
            kilometers
            (optional, default: 1)
        :param int replica_count: number of different replicas in simulation
            (optional, default: 1000)
        :param int replica_group_count: number of different replica groups
            (optional, default: 1000)
        :param float mwg_prob: probability that node requests a replica from
            from its Most Wanted Group
            (optional, default: 0.1)
        :param int replica_min_size: min size of a single replica in megabits
            (optional, default: 100)
        :param int replica_max_size: max size of a single replica in megabits
            (optional, default: 1000)
        :param int total_reqs: number of replica requests to generate during
            the simulation
            (optional, default: 100000)
        :param int fsti: frequency specific time interval (as defined in paper)
            (optional, default: 1000)
        :param int network_bw_mbps:  network bandwidth (Mbit/s)
            (optional, default: 10)
        :param int pspeed_kmps:  propagation speed (km/s)
            (optional, default: 6000)
        :param int rnd_seed: seed for the random number generator
            (optional, default: current system time)
        """
        self._node_capacity = node_capacity
        self._strategy = strategy

        if node_count < 2:
            raise ValueError("Grid's node_count must be at least 2.")
        self._node_count = node_count

        if replica_count < 1:
            raise ValueError("Number of replicas must be positive.")
        self._replica_count = replica_count

        if replica_group_count < 1:
            raise ValueError("Number of replica groups must be positive.")
        self._replica_group_count = replica_group_count

        if mwg_prob < 0.0 or mwg_prob > 1.0:
            raise ValueError("MWG probability outside the interval [0, 1).")
        self._mwg_prob = mwg_prob

        if fsti <= 0.0:
            raise ValueError("FSTI must be positive.")
        self._fsti = fsti

        self._replicas = OrderedDict()
        self._replica_groups = dict()
        self._nodes = OrderedDict()
        self._nodes_mwg = OrderedDict()  # nodes' most wanted replica groups
        self._edges = OrderedDict()  # nodes' outbound edges

        self._clock = _Clock()

        if min_dist_km <= 0:
            raise ValueError("Minimum distance must be positive.")
        self._min_dist_km = min_dist_km

        if max_dist_km <= 0:
            raise ValueError("Maximum distance must be positive.")
        self._max_dist_km = max_dist_km

        if min_dist_km >= max_dist_km:
            raise ValueError("Min distance must be smaller than max distance")

        if network_bw_mbps <= 0:
            raise ValueError("Network bandwidth must be positive.")
        self._network_bw_mbps = network_bw_mbps

        if pspeed_kmps <= 0:
            raise ValueError("Propagation speed must be positive.")
        self._pspeed_kmps = pspeed_kmps

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

        self._event_queue = []

        # two constants used in metrics calculations
        self._c1 = 0.001
        self._c2 = 0.001

    def _new_node(self, *args, **kwargs):
        """Create a new grid node and store it in the list of nodes.

        :returns: newly created node instance
        :rtype: :py:class:`~models.node.Node`
        """
        if self._strategy == Strategy.EFS:
            node = NodeEFS(*args, **kwargs)
        elif self._strategy == Strategy.LFU:
            node = NodeLFU(*args, **kwargs)
        elif self._strategy == Strategy.LRU:
            node = NodeLRU(*args, **kwargs)
        else:
            raise NotImplementedError(
                "No Node implementation for the current strategy.")

        self._nodes[node.name] = node

        # XXX: move outside new_node, just as node stats are created elsewhere
        self._nodes_mwg[node.name] = random.randint(
            1, self._replica_group_count)
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

        # TODO: add tests for replica_groups!
        self._replica_groups = dict()
        for i in range(1, self._replica_group_count + 1):
            self._replica_groups[i] = []

        self._replicas = OrderedDict()

        for i in range(1, self._replica_count + 1):
            replica = Replica(
                name=name_tpl.format(i),
                size=random.randint(
                    self._replica_min_size, self._replica_max_size)
            )
            self._replicas[replica.name] = replica

            group_idx = (i - 1) % self._replica_group_count + 1
            self._replica_groups[group_idx].append(replica)

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
            # XXX: use heap for efficiency? (only if it runs too slow)
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

    def event_send_replica_request(self, *args, **kwargs):
        """Create and return a new SendReplicaRequest event instance.

        Parameters are the same as expected by the
        :py:class:`~models.event.SendReplicaRequest` class.

        :returns: new SendReplicaRequest instance
        :rtype: :py:class:`~models.event.SendReplicaRequest`
        """
        event = SendReplicaRequest(*args, **kwargs)
        return event

    def event_send_replica(self, *args, **kwargs):
        """Create and return a new SendReplica event instance.

        Parameters are the same as expected by the
        :py:class:`~models.event.SendReplica` class.

        :returns: new SendReplica instance
        :rtype: :py:class:`~models.event.SendReplica`
        """
        event = SendReplica(*args, **kwargs)
        return event

    def initialize(self):
        """Initialize (reset) simulation."""

        # XXX: accept settings here, too? (just like in init) ---> simplify
        # init and move all sim. parameter initializing logic to here
        self._clock.reset()
        random.seed(self._rnd_seed)

        self._total_bw = 0.0
        self._total_rt_s = 0.0

        # generate grid
        self._generate_replicas()
        self._generate_nodes()
        self._generate_edges()

        self._event_queue = []

        # calculate shortest paths from server node to all the other nodes
        # and update each of the latter with a relevant path
        node_info = self._dijkstra()
        for name, info in node_info.items():
            self._nodes[name].set_parent(self._nodes.get(info.previous))

    def run(self):
        """Run simulation.

        NOTE: simulation must have already been initialized with initialize()
        """
        self._clock.reset()

        print("[SIM    @ {:.8f}] SIMULATION STARTED".format(self.now))

        ef = _EventFactory(self)
        t, event = ef.get_random()
        self._schedule_event(event, t)

        total_reqs_gen = 1  # total replica request events generated so far

        # main event loop
        while len(self._event_queue) > 0:

            if total_reqs_gen < self._total_reqs:
                new_t, new_e = ef.get_random()
                self._schedule_event(new_e, new_t)
                total_reqs_gen += 1

                if total_reqs_gen % 1000 == 0:
                    print(
                        '\033[1;36m', "generated request {}/{}"
                        "({:.2f} %)".format(
                            total_reqs_gen, self._total_reqs,
                            100 * total_reqs_gen / self._total_reqs),
                        '\033[0m', sep='')

            t, event = self._pop_next_event()

            # fast-forward time to the next event occurence and then
            # process the event
            self._clock.tick(step=t - self.now)
            self._process_event(event)

        print("-" * 10, "END", "-" * 10)
        print("Total resp. time (s):", self._total_rt_s * self._c1)
        print("Total bandwidth:", self._total_bw * self._c2)

    def _pop_next_event(self):
        """Pop next event from the event queue and return it.

        :returns: time of next event and the event instance itself
        :rtype: tuple (float, :py:class:`~models.event._Event` instance)
        """
        return heapq.heappop(self._event_queue)

    def _schedule_event(self, event, event_time):
        """Add new event to event queue, scheduled at time `event_time`.

        :param event: event to add to the schedule
        :type event: subclass of :py:class:`~models.event._Event`
        :param float event_time: time at which `event` should occur
        """
        heapq.heappush(self._event_queue, (event_time, event))

    def _process_event(self, event):
        """Process a single event occuring in simulation.

        :param event: event to pocess
        :type event: subclass of :py:class:`~models.event._Event`
        """
        if type(event) == ReceiveReplicaRequest:
            g = event.target.request_replica(event.replica_name, event.source)
            returned_event = next(g)

            # node returns either SendReplicaRequest event (if it doesn't have
            # a requested replica) or SendReplica (if it has a replica and
            # wants to send it back to the requesting node)
            if type(returned_event) == SendReplicaRequest:

                # XXX: really need to copy or is it fine if some events share
                # a single generator list? probably better, because when a
                # generator is axhausted (and removed), it is no longer needed
                # *anywhere*
                new_gens = event._generators.copy()
                new_gens.append(g)
                returned_event._generators = new_gens
                self._schedule_event(returned_event, self.now)

            elif type(returned_event) == SendReplica:
                # pass generators from preceding ReceiveReplicaRequest event
                returned_event._generators = event._generators.copy()
                self._schedule_event(returned_event, self.now)
            else:
                raise TypeError(
                    "Node returned unexpected event", returned_event)

        elif type(event) == SendReplicaRequest:
            # some node sends a replica request, schedule ReceiveReplicaRequest
            # event for that node's parent

            # it takes some time for the parent node to receive the request,
            # so calculate the latency and schedule the event accordingly
            dist_km = self._edges[event.source.name][event.target.name]
            cl = dist_km / self._pspeed_kmps  # communication latency
            event_time = self.now + cl

            new_event = ReceiveReplicaRequest(
                event.source, event.target, event.replica_name)
            new_event._generators = event._generators.copy()  # pass generators

            self._schedule_event(new_event, event_time)

            self._total_rt_s += cl  # update simulation stats

        elif type(event) == SendReplica:
            # some node sends a replica, schedule replica receive event
            # for the receiver (after a delay, of course)

            # calculate the time needed for the replica to reach target
            # XXX: add CL here, too?
            delay_s = event.replica.size / self._network_bw_mbps
            event_time = self.now + delay_s

            new_event = ReceiveReplica(
                event.source, event.target, event.replica)
            new_event._generators = event._generators.copy()  # pass generators

            self._schedule_event(new_event, event_time)

            # simulation update stats
            self._total_rt_s += delay_s
            self._total_bw += event.replica.size  # XXX: correct interpretat.?

        elif type(event) == ReceiveReplica:
            # some node's request for a replica has completed (replica
            # received), we need to notify that node about it so that it
            # continues its work from where it stopped
            if event._generators:
                g = event._generators.pop()
                new_event = g.send(event.replica)  # we get a SendReplica event

                # if target is not None, node did not request the replica
                # by itself, thus we need to send the replica to another node
                # down the chain
                if new_event.target is not None:
                    new_event._generators = event._generators.copy()
                    self._schedule_event(new_event, self.now)
        else:
            raise TypeError("Unknown event", type(event))


class _EventFactory(object):
    """Simulation events factory."""

    def __init__(self, sim):
        self._sim = sim
        self._node_names = list(self._sim.nodes.keys())[1:]  # omit server node

        # XXX: add tests for _replica_gorups
        self._replica_groups = list(self._sim._replica_groups.keys())

    def get_random(self):
        """Create new random event (some node receives a request for replica).

        :returns: new random instance of receive replica request
        :rtype: :py:class:`~models.simulation.ReceiveReplicaRequest`
        """
        receiver = random.choice(self._node_names)
        receiver = self._sim.nodes[receiver]

        # TODO: add tests for replica_groups! MWG etc.
        # and test that server node is expluded from the list etc.

        mwg = self._sim._nodes_mwg[receiver.name]

        # first choose a group - either MWG or any other non-MWG group,
        # then choose a random replica from this randomly chosen group
        if random.random() < self._sim._mwg_prob:
            group = mwg
        else:
            # XXX: more efficient than constructing a new list with MWG
            # omitted, but there might be some more elegant ways
            while True:
                group = random.choice(self._replica_groups)
                if group != mwg:
                    break

        # XXX: not uniform distribution if groups do not contain the same
        # number of replicas!
        replica = random.choice(self._sim._replica_groups[group])

        # XXX: don't hardcode the limits, read them from the simulation
        # object (and the same for time?)
        time_from_now = random.randint(0, 99)
        event_time = time_from_now + self._sim.now

        # return NodeRequest(node, replica, self._sim.now + time_from_now)
        return event_time, ReceiveReplicaRequest(None, receiver, replica.name)
