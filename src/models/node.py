from collections import namedtuple
from collections import OrderedDict


Strategy = namedtuple('Strategy', [
    'EFS',  # Enhanced Fast Spread, used in the 2011 paper
    's2012',  # TODO
    's2013',  # TODO
])
Strategy = Strategy(*range(1, 4))


class Node(object):
    """TODO: docstring"""

    def __init__(self, name='', capacity=50000, parent=None):
        self._name = name
        self._capacity = capacity  # XXX: check for > 0?
        self._parent = parent  # XXX: check for is not self
        self._client_nodes = OrderedDict()

    @property
    def name(self):
        return self._name

    @property
    def capacity(self):
        return self._capacity

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

    # add_replica ... and algorithm (if size ...)


# TODO: Node2011 (strategy = EFS)
#       Node2012 (strategy = ...)
#       Node2013 (strategy = ...)
