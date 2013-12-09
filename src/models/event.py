"""Definitions of events tha can occur in simulation."""


class _Event(object):
    """Base class for all simulation events."""
    event_id = 0

    def __init__(self):
        self._generators = []  # a stack of callbacks (generators)
        _Event.event_id += 1
        self.event_id = _Event.event_id

    def __lt__(self, other):
        return self.event_id < other.event_id


class ReceiveReplicaRequest(_Event):
    """An event when node receives a requests for a replica."""

    def __init__(self, source, target, replica_name):
        """Initialize new instance.

        :param source: node that requests a replica
        :type source: :py:class:`~models.node.Node`
        :param target: node that receives replica request
        :type target: :py:class:`~models.node.Node`
        :param str replica_name: requested replica's name
        """
        super().__init__()
        self.source = source
        self.target = target
        self.replica_name = replica_name

    def __str__(self):
        source_name = self.source.name if self.source else "None"
        return "<ReceiveReplicaRequest event> ({} --> {}, {})".format(
            source_name, self.target.name, self.replica_name)


class SendReplicaRequest(_Event):
    """An event when node sends a requests for a replica."""

    def __init__(self, source, target, replica_name):
        """Initialize new instance.

        :param source: node that issues a request
        :type source: :py:class:`~models.node.Node`
        :param target: target node from which `replica` is requested
        :type target: :py:class:`~models.node.Node`
        :param str replica_name: requested replica's name
        """
        super().__init__()
        self.source = source
        self.target = target
        self.replica_name = replica_name

    def __str__(self):
        return "<SendReplicaRequest event> ({} --> {}, {})".format(
            self.source.name, self.target.name, self.replica_name)


class SendReplica(_Event):
    """An event when node sends back a replica to the requester."""

    def __init__(self, source, target, replica):
        """Initialize new instance.

        :param source: node that sends back requested `replica`
        :type source: :py:class:`~models.node.Node`
        :param target: node that should receive requested `replica`
        :type target: :py:class:`~models.node.Node`
        :param replica: replica that was requested
        :type replica: :py:class:`~models.replica.Replica`
        """
        super().__init__()
        self.source = source
        self.target = target
        self.replica = replica

    def __str__(self):
        target_name = self.target.name if self.target else "None"
        return "<SendReplica event> ({} <-- {}, {})".format(
            target_name, self.source.name, self.replica.name)


class ReceiveReplica(_Event):
    """An event when node sends back a replica to the requester."""

    def __init__(self, source, target, replica):
        """Initialize new instance.

        :param source: node that sent requested `replica`
        :type source: :py:class:`~models.node.Node`
        :param target: node that receives requested `replica`
        :type target: :py:class:`~models.node.Node`
        :param replica: replica that was requested
        :type replica: :py:class:`~models.replica.Replica`
        """
        super().__init__()
        self.source = source
        self.target = target
        self.replica = replica

    def __str__(self):
        return "<ReceiveReplica event> ({} <-- {}, {})".format(
            self.target.name, self.source.name, self.replica.name)
