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
        self, name='', capacity=50000, parent=None, replicas=None, clock=None
    ):
        if clock is None:
            raise ValueError("Clock instance expected, not None.")
        self._clock = clock

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

        self._replicas = OrderedDict()
        self._replica_stats = OrderedDict()
        if replicas is not None:
            for rep in replicas:
                self.copy_replica(rep)

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
        # TODO: implement as a collection? e.g. self[] = node?
        # overrdie __setitem__ and __getitem__ and __delitem__
        self._client_nodes[node.name] = node
        node.parent = self

    def replica_idx(self, replica_name):
        """Find an index of replica_name in self._replicas list.
           Return -1 if replica does not exist.
        """
        for i in range(len(self._replicas)):
            if self._replicas[i].name == replica_name:
                return i
        return -1

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

    def request_replica(self, replica_name):
        """TODO: trigger a request for particular replica"""
        r_idx = self.replica_idx(replica_name)
        if r_idx >= 0:
            self._replica_stats[r_idx].nor += 1
            self.lrt = self._clock.time()
            # TODO: USE_REPLICA ... increase NOR count etc.
            return

        # nodes on the shortest path from here to server
        nsp_list = self.path_to_server()

        # go up the hierarchy to the server
        for i, node in enumerate(nsp_list[1:]):
            req_replica = node.get_replica(replica_name)  # requested replica
            if req_replica is None:  # RR does not exist on NSPList(i)
                continue

            # from node where replica is found all the way back down to self
            for cn_node in nsp_list[i - 1::-1]:  # "checked node"
                if cn_node.capacity_free >= req_replica.size:
                    cn_node.copy_replica(req_replica)
                    continue

                # else: not enough space to copy replica, might replace some
                # of the existing replicas
                # XXX: _replicas should be sorted based on RV!
                # XXX: accessing "private" attribute
                sos = 0  # sum of sizes
                marked_replicas = []  # visited and marked for deletion
                for x, rep in enumerate(cn_node._replicas):
                    if sos + cn_node.capacity_free < req_replica.size:
                        sos += rep.size
                        marked_replicas.append(rep)
                    else:
                        break

                GV = "TODO"
                RV_rr = "TODO 2"

                if GV < RV_rr:
                    # delete all replicas needed to free enough space
                    for mr in marked_replicas:
                        cn_node.delete_replica(mr.name)
                    cn_node.copy_replica(req_replica)
            break  # TODO: not in a paper but should be here
                    # no point in searching the replica further up the
                    # hierarchy once it has been found?

    def copy_replica(self, replica):
        """Store a local copy of the given replica."""
        # XXX: raise error if not enough space available
        if replica.size > self._free_capacity:
            raise ValueError(
                "Cannot store a copy of replica, not enough free capacity.")

        self._replicas[replica.name] = deepcopy(replica)
        self._free_capacity -= replica.size

        # XXX: NOR - is it 1?
        self._replica_stats[replica.name] = \
            _ReplicaStats(nor=0, nor_fsti=0, lrt=self._clock.time())

        # XXX: now use decorate-sort-undecorate? for sorting by replica value?

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
