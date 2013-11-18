from collections import OrderedDict


class Node(object):
    """TODO: docstring"""

    def __init__(self, name='', capacity=50000, parent=None):
        self._name = name
        self._capacity = capacity  # XXX: check for > 0?
        self._parent = parent
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
