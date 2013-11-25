from collections import OrderedDict
from copy import deepcopy


class _ReplicaStats(object):
    """A helper class for the Node class."""

    def __init__(self, nor=0, nor_fsti=0, lrt=0):
        self.nor = nor
        self.nor_fsti = nor_fsti
        self.lrt = lrt


class Node(object):
    """A representation of a node (either client or server) in the grid."""

    def __init__(
        self, name='', capacity=50000, parent=None, replicas=None, sim=None
    ):
        if sim is None:
            raise ValueError("Simulation object instance expected, not None.")
        # XXX: instead of "full" simulation object pass an adapter with
        # a limited interface? (it doesn't make sense for the Node to call,
        # e.g, sim.init_grid())
        self._sim = sim

        self._name = name

        if capacity <= 0:
            raise ValueError("Capacity must be a positive number.")
        self._capacity = capacity
        self._free_capacity = capacity

        if parent is self:
            raise ValueError("Node cannot be its own parent.")
        self._parent = parent  # XXX: check all the way up the hierarchy?
        self._client_nodes = OrderedDict()
        self._nsp_list = []

        self._replicas = OrderedDict()  # replicas sorted by their value ASC
        self._replica_stats = OrderedDict()
        if replicas is not None:
            for repl in replicas.values():
                self._copy_replica(repl, run_sort=False)

    @property
    def name(self):
        return self._name

    @property
    def capacity(self):
        return self._capacity

    @property
    def capacity_free(self):
        return self._capacity_free

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def is_server(self):
        return self._parent is None

    def add_client_node(self, node):
        # XXX: implement as a collection? e.g. self[] = node?
        # overrdie __setitem__ and __getitem__ and __delitem__
        self._client_nodes[node.name] = node
        node.parent = self

    def get_replica(self, replica_name):
        """Return replica with the given name or None if replica does not
        exist on the node.
        """
        return self._replicas.get(replica_name)

    def path_to_server(self, rebuild=False):
        """TODO: return a list of nodes on the shortest path to the server
        including the node itself
        If it does not yet exist or rebuild is forced, it is reconstructed
        """
        if rebuild or not self._nsp_list:
            self._nsp_list = [self]
            node = self.parent
            while node is not None:
                self._nsp_list.append(node)
                node = node.parent

        return self._nsp_list

    def _GV(self, replicas):
        """Calculate group value of the given list of replicas."""

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

        fsti = 1234   # TODO: part of simulation object! self._sim.fsti
        ct = self._sim.time  # current simulation time

        gv = s_nor / s_size + s_nor_fsti / fsti + \
            1 / (ct - s_lrt / len(replicas))

        return gv

    def _RV(self, replica):
        """Calculate value of the given replica."""
        fsti = 1234   # TODO: part of simulation object! self._sim.fsti
        ct = self._sim.time  # current simulation time

        stats = self._replica_stats[replica.name]

        rv = stats.nor / replica.size + stats.nor_fsti / fsti + \
            1 / (ct - stats.lrt)

        return rv

    def store_if_valuable(self, replica):
        """Store a local copy of the given replica if valuable enough.

        If the current free capacity is big enough, a local copy of `replica`
        is stored. Otherwise its replica value is calculated and if it is
        greater than the value of some group of replicas, that group of
        replicas is deleted to make enough free space for the local copy of
        `replica`.
        """
        # XXX: perhaps rename copy_replica (or retain the name for easier
        # comparison with the pseudocode in the paper)

        # TODO: notify simulation machinery about this? sim.increase_count
        # actually wrap everything into a sim object ... e.g. sim.clock.time
        # (or sim.time) - simulation should provide an API to the simulated
        # entities

        if self.capacity_free >= replica.size:
            self._copy_replica(replica)
            return  # nothing more to do

        # else: not enough space to copy replica, might replace some
        # of the existing replicas

        sos = 0  # sum of sizes
        marked_replicas = []  # visited and marked for deletion
        for x, rep in enumerate(self._replicas):
            if sos + self.capacity_free < replica.size:
                sos += rep.size
                marked_replicas.append(rep)
            else:
                break

        # x: index of the first replica *excluded* from the group of
        # replicas that would make enough free space for req. replica
        # in case this group is deleted. - TODO: needed? we have
        # a list of "makred" replicas for that
        gv = self._GV(marked_replicas)
        rv = self._RV(replica)

        if gv < rv:
            # delete all replicas needed to free enough space
            for mr in marked_replicas:
                self.delete_replica(mr.name)
            self._copy_replica(replica)

    def request_replica(self, replica_name):
        """TODO: trigger a request for a particular replica"""
        replica = self.get_replica(replica_name)
        if replica is not None:
            # ... use RR ... ---> XXX: refactor to a method?
            rep_stats = self._replica_stats[replica.name]
            rep_stats.nor += 1
            rep_stats.lrt = self._sim.time
            # TODO: NOR_FSTI update
            return

        # else: replica does not exist on the node ..

        # nodes on the shortest path from here to server
        nsp_list = self.path_to_server()

        # go up the hierarchy to the server
        for i, node in enumerate(nsp_list[1:]):
            replica = node.get_replica(replica_name)  # requested replica
            if replica is not None:  # RR exists on NSPList(i)
                # from node where replica is found all the way
                # back down to self
                for cn_node in nsp_list[i - 1::-1]:  # "checked node"
                    cn_node.store_if_valuable(replica)

                break  # TODO: not in a paper but should be here
                        # no point in searching the replica further up the
                        # hierarchy once it has been found?

        # TODO: useReplica now that it has been copied here ... to update stats

    def _copy_replica(self, replica, run_sort=True):
        """Store a local copy of the given replica.

        TODO: run_sort description and usage (when init, don't use)
        """
        # XXX: raise error if not enough space available
        if replica.size > self._free_capacity:
            raise ValueError(
                "Cannot store a copy of replica, not enough free capacity.")

        self._replicas[replica.name] = deepcopy(replica)
        self._free_capacity -= replica.size

        # initialize with default stats - the latter need to be updated
        # separately (in case this is needed)
        self._replica_stats[replica.name] = _ReplicaStats()

        if run_sort:
            # re-create dictionary ordered by replica value (lowest first)
            self._replicas = OrderedDict(
                sorted(self._replicas.items(), key=lambda x: self._RV(x[1]))
            )

    def delete_replica(self, replica_name):
        """TODO"""
        replica = self._replicas.pop(replica_name)
        self._replica_stats.pop(replica_name)
        self._free_capacity += replica.size

# Node: store NOR (# of requests) of each replica which resides on it
# For a given node, the NOR of a replica is increased by one each
# time that replica is requested by that node


# TODO: Node2011 (strategy = EFS)
#       Node2012 (strategy = ...)
#       Node2013 (strategy = ...)
