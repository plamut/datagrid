from collections import namedtuple
from collections import OrderedDict
from models.replica import Replica

# TODO: move to __init__.py? shouldn't be in node.py
# perhaps move to simulation.py?
Strategy = namedtuple('Strategy', [
    'EFS',  # Enhanced Fast Spread, used in the 2011 paper
    's2012',  # TODO
    's2013',  # TODO
])
Strategy = Strategy(*range(1, 4))


class _ReplicaStats(object):
    """A helper class for the Node class."""

    def __init__(self, nor=0, nor_fsti=0, lrt=0):
        self.nor = nor
        self.nor_fsti = nor_fsti
        self.lrt = lrt


class Node(object):
    """A representation of a node (either client or server) in the grid."""

    def __init__(self, name='', capacity=50000, parent=None, replicas=None):
        self._name = name
        self._capacity = capacity  # XXX: check for > 0?
        self._free_capacity = capacity
        self._parent = parent  # XXX: check for is not self
        self._client_nodes = OrderedDict()

        # XXX: check for total replica capacity?
        # TODO: update self._free capacity! (use copy replica)
        self._replicas = replicas if replicas is not None else []

        # we keep some stats for each replica
        self._replica_stats = [_ReplicaStats() for r in self._replicas]

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
        # TODO:
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

    def request_replica(self, replica_name):
        """TODO: trigger a request for particular replica"""
        # TODO: implement the algorithm from the paper here ...

        sos = 0

        r_idx = self.replica_idx(replica_name)
        if r_idx >= 0:
            self._replica_stats[r_idx].nor += 1
            self.lrt = 42  # TODO: current simulation time
              # TODO: USE_REPLICA ... increase NOR count etc.
            return

        # build a list of node parents (XXX: don't compute every time)
        nsplist = [self]  # nodes on the shortest path from here to server
        node = self.parent
        while node is not None:
            nsplist.append(node)
            node = node.parent

        # go up the hierarchy (parent to server)
        for i in range(1, len(nsplist)):
            node = nsplist[i]
            r_idx = node.replica_idx(replica_name)
            if r_idx < 0:  # RR does not exist on NSPList(i)
                continue

            # RR exists on NSPList(i)
            replica = self._replicas[r_idx]

            # from node where replica is found all the way back down to self
            for j in range(i - 1, -1, -1):
                cn_node = nsplist[j]  # "checked node"
                if cn_node.capacity_free >= replica.size:
                    cn_node.copy_replica(replica)
                    continue

                # not enough space to copy replica, perhaps need to replace
                # some of the existing replicas
                # XXX: _replicas should be sorted based on RV!
                 # XXX: accessing "private" attribute
                for x, rep in enumerate(cn_node._replicas):
                    if sos + cn_node.capacity_free < replica.size:
                        sos += rep.size
                    else:
                        break

                GV = "TODO"
                RV_rr = "TODO 2"

                if GV < RV_rr:
                    # delete all replicas needed to free enough space
                    for y in range(0, x):
                        cn_node.del_replica_at(y)
                    cn_node.copy_replica(replica)
            break  # TODO: not in a paper but should be here
                    # no point in searching the replica further up the
                    # hierarchy once it has been found?

        # if replica_name exists

    def copy_replica(self, replica):
        """Store a local copy of the given replica."""
        # XXX: raise error if not enough space available
        if replica.size > self._free_capacity:
            raise ValueError(
                "Cannot store a copy of replica, not enough free capacity.")

        self._replicas.append(Replica(replica.name, replica.size))

        # XXX: NOR - is it 1? Is the time current simulation time?
        # IDEA: have a global object issuing simulation time ... and
        # have the method curr_time (in integers ... steps)
        self._replica_stats.append(_ReplicaStats(0, 0, 0))

        # XXX: now use decorate-sort-undecorate?

    def del_replica_at(self, idx):
        """TODO"""
        self._replicas.pop(idx)
        self._replica_stats.pop(idx)

# Node: store NOR (# of requests) of each replica which resides on it
# For a given node, the NOR of a replica is increased by one each
# time that replica is requested by that node


# TODO: Node2011 (strategy = EFS)
#       Node2012 (strategy = ...)
#       Node2013 (strategy = ...)
