from collections import deque
from collections import OrderedDict
from copy import deepcopy


class _ReplicaStats(object):
    """A helper class for the Node class, representing stats of a single
    replica.
    """

    def __init__(self, nor=0, fsti=0, lrt=0):
        """Initialize replica stat values.

        :param nor: number of requests of the replica
            (optional, default: 0)
        :type nor: int
        :param fsti: width of the FSTI interval
            (optional, default: 0)
        :type fsti: int
        :param lrt: the time of the last request of the replica
            (optional, default: 0)
        :type lrt: int
        """
        if fsti < 0:
            raise ValueError("FSTI must be a non-negative number.")
        self._fsti = fsti

        if nor < 0:
            raise ValueError("NOR must be a non-negative number.")
        self._nor = nor

        if lrt < 0:
            raise ValueError("LRT must be a non-negative number.")
        self._lrt = lrt

        self._req_hist = deque()  # request history (we store request times)

        @property
        def lrt(self):
            """Time when replica was last requested."""
            return self._lrt

        @property
        def nor(self):
            """Number of requests of the replica."""
            return self._nor

        def nor_fsti(self, time):
            """Return the number of requests of the replica in FSTI interval.

            NOTE: as a side effect, the requests older than (`time` - FSTE)
            are removed from the internal requests history so make sure you
            indeed provide current time, or else subsequent calls to nor_fsti
            might return invalid results.

            :param time: current time
            :type time: int

            :returns: number of requests in the last FSTI interval
            :rtype: int
            """
            while (
                len(self._req_hist) > 0 and
                (time - self._req_hist[0]) > self._fsti
            ):
                self._req_history.popleft()

            return len(self._req_history)

        def new_request_made(self, time):
            """Update stats with a new request that has arrived at `time`.

            NOTE: time of the request is appended to the internal requests
            history, so make sure you always call this method in order (i.e.
            for older requests first).

            :param time: time when the request was made
            :type time: int
            """
            self._requests_history.append(time)
            self._lrt = time
            self._nor += 1


class Node(object):
    """A representation of a node (either client or server) in the grid."""

    def __init__(self, name, capacity, sim, replicas=None):
        """Initialize node instance.

        :param name: name of the node
        :type name: string
        :param capacity: node capacity in megabits
        :type capacity: int
        :param sim: simulation which the node is a part of
        :type sim: :py:class:`~models.simulation.Simulation`
        :param replicas: initial replicas the node contans copies of
        :type replicas: list of :py:class:`~models.replica.Replica` instances
            (optional, default: None)
        """
        # XXX: instead of "full" simulation object pass an adapter with
        # a limited interface? (it doesn't make sense for the Node to call,
        # e.g, sim.init_grid())
        self._sim = sim

        self._name = name

        if capacity <= 0:
            raise ValueError("Capacity must be a positive number.")
        self._capacity = capacity
        self._free_capacity = capacity

        self._client_nodes = OrderedDict()
        self._nsp_list = []  # node shortest path list to the server

        self._replicas = OrderedDict()  # replicas sorted by their value ASC
        self._replica_stats = OrderedDict()
        if replicas is not None:
            for repl in replicas.values():
                # TODO: set to true? replicas must always be sorted by RV
                # better: coppy all replicas and sort at the end
                self._copy_replica(repl, run_sort=False)

    @property
    def name(self):
        """Name of the node."""
        return self._name

    @property
    def capacity(self):
        """Node's total capacity in megabits."""
        return self._capacity

    @property
    def capacity_free(self):
        """Node's current available capacity in megabits."""
        return self._capacity_free

    def update_nsp_path(self, nsp_list):
        """Update the list of node names on the shortest path to server node.

        :param nsp_list: shortest path from node to server node, including
            the node itself.
        :type nsp_list: list of strings
        """
        self._nsp_list = nsp_list

    def _GV(self, replicas):
        """Calculate value of a group of replicas.

        :param replicas: list representing a replica group
        :type replicas: list of :py:class:`~models.replica.Replica` instances

        :returns: value of the replica group
        :rtype: float
        """
        if not replicas:
            return 0.0  # empty gorup has a vaue of zero

        s_nor = 0  # sum of NORs
        s_size = 0  # sum of replica sizes
        s_nor_fsti = 0  # sum of replicas' nor_fsti values
        s_lrt = 0  # sum of replicas' LRT times

        for r in replicas:
            stats = self._replica_stats[r.name]
            s_size += r.size
            s_nor += stats.nor
            s_nor_fsti += stats.nor_fsti
            s_lrt += stats[r.name].lrt

        fsti = self._sim.fsti
        ct = self._sim.now  # current simulation time

        gv = s_nor / s_size + s_nor_fsti / fsti + \
            1 / (ct - s_lrt / len(replicas))

        return gv

    def _RV(self, replica):
        """Calculate value of a replica.

        :param replica: replica to calclulate the value of
        :type replica: :py:class:`~models.replica.Replica`

        :returns: value of the `replica`
        :rtype: float
        """
        fsti = self._sim.fsti
        ct = self._sim.now  # current simulation time

        stats = self._replica_stats.get(replica.name)
        if stats is None:
            stats = _ReplicaStats()

        rv = stats.nor / replica.size + stats.nor_fsti / fsti + \
            1 / (ct - stats.lrt)

        return rv

    def _store_if_valuable(self, replica):
        """Store a local copy of the given replica if valuable enough.

        If the current free capacity is big enough, a local copy of `replica`
        is stored. Otherwise its replica value is calculated and if it is
        greater than the value of some group of replicas, that group of
        replicas is deleted to make enough free space for the local copy of
        `replica`.

        :param replica: replica to consider storing locally
        :type replica: :py:class:`~models.replica.Replica`
        """
        # XXX: perhaps rename copy_replica (or retain the name for easier
        # comparison with the pseudocode in the paper)

        # TODO: notify simulation machinery about this? sim.increase_count
        # actually wrap everything into a sim object ... e.g. sim.clock.time
        # (or sim.now) - simulation should provide an API to the simulated
        # entities
        # UPDATE: after refactoring, _store_if_valuable will be probably
        # called by Simulation itself which will then know how to calculate
        # grid load stats (or maybe not ...)

        if self.capacity_free >= replica.size:
            self._copy_replica(replica)
            self._replica_stats[replica.name].new_request_made()
        else:
            # not enough space to copy replica, might replace some
            # of the existing replicas
            sos = 0  # sum of sizes
            marked_replicas = []  # replicas visited and marked for deletion
            for x, rep in enumerate(self._replicas):
                if sos + self.capacity_free < replica.size:
                    sos += rep.size
                    marked_replicas.append(rep)
                else:
                    break

            # x: index of the first replica *excluded* from the group of
            # replicas that would make enough free space for req. replica
            # in case this group is deleted. - XXX: needed? we have
            # a list of "makred" replicas for that
            gv = self._GV(marked_replicas)

            # TODO: what stats to use for a replica, which does not yet exist
            # on the node? Stick with NOR=0 etc.?
            rv = self._RV(replica)

            if gv < rv:
                # delete all replicas needed to free enough space
                for mr in marked_replicas:
                    self.delete_replica(mr.name)
                self._copy_replica(replica)
                self._replica_stats[replica.name].new_request_made()

    def request_replica(self, replica_name):
        """Request a replica from the node.

        If a local copy of replica is currently available, it is immediately
        returned, otherwise the node requests a replica from its parent (on
        the shortest path to server) and then return it.

        If replica has been obtained from the parent, the node also decides
        whether to keep a local copy of it or not (a copy is kept if it is
        determined more important than a group of existing local replicas).

        :param replica_name: name of the replica to request
        :type replica_name: string

        :returns: requested replica
        :rtype: :py:class:`~models.replica.Replica`
        """
        replica = self._replicas.get(replica_name)
        if replica is not None:
            # "UseReplica()" - update its stats
            self._replica_stats[replica_name].new_request_made(
                self._sim.now)
        else:
            # replica not available locally, request it from parent
            replica = self._nsp_list[1].request_replica(replica_name)

            # XXX get rid of nsp list and only have info about parent?
            # would make sense now, nsp list not needed anymore

            # XXX: notify simulation machinery that replica has been obtained
            # from parent? So that it can calculate the bandwidth consumed
            # and response time

            # NOTE: stats are automatically updated when a replica is stored
            # in the _store_if_valuable() method
            self._store_if_valuable(replica)

        return replica

    def _copy_replica(self, replica, run_sort=True):
        """Store a local copy of the given replica.

        :param replica: replica to copy
        :type replica: :py:class:`~models.replica.Replica`
        :param run_sort: whether or not to sort the internal replica list by
            replica value after storing a copy of the new replica
            (optional, default: True)
        :type run_sort: bool
        """
        if replica.size > self._free_capacity:
            raise ValueError(
                "Cannot store a copy of replica, not enough free capacity.")

        self._replicas[replica.name] = deepcopy(replica)
        self._free_capacity -= replica.size

        # initialize with default stats - the latter need to be updated
        # separately (in case this is needed)
        self._replica_stats[replica.name] = _ReplicaStats(fsti=self._sim.fsti)

        # XXX: this OK? if default stats, then subsequent sorting by replica
        # value (RV) might be wrong

        if run_sort:
            # re-create dictionary ordered by replica value (lowest first)
            self._replicas = OrderedDict(
                sorted(self._replicas.items(), key=lambda x: self._RV(x[1]))
            )

    def delete_replica(self, replica_name):
        """Remove a local copy of a replica. If replica does not exist
        on the node, an error is raised.

        :param replica_name: name of the replica to delete
        :type replica_name: string
        """
        replica = self._replicas.pop(replica_name)
        self._replica_stats.pop(replica_name)
        self._free_capacity += replica.size

# Node: store NOR (# of requests) of each replica which resides on it
# For a given node, the NOR of a replica is increased by one each
# time that replica is requested by that node


# TODO: Node2011 (strategy = EFS)
#       Node2012 (strategy = ...)
#       Node2013 (strategy = ...)
