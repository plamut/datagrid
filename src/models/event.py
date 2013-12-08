"""Definitions of events tha can occur in simulation."""


class _Event(object):
    """Base class for all simulation events."""


class ReceiveReplicaRequest(_Event):
    """An event when node receives a requests for a replica."""

    def __init__(self, source, target, replica_name, time):
        """Initialize new instance.

        :param source: node that requests a replica
        :type source: :py:class:`~models.node.Node`
        :param target: node that receives replica request
        :type target: :py:class:`~models.node.Node`
        :param str replica_name: requested replica's name
        :param int time: time when request was received
        """
        self.source = source
        self.target = target
        self.replica_name = replica_name
        self.time = time

    def __str__(self):
        source_name = self.source.name if self.source else "None"
        return "<ReceiveReplicaRequest event @ {}> ({} --> {}, {})".format(
            self.time, source_name, self.target.name, self.replica_name)


class SendReplicaRequest(_Event):
    """An event when node sends a requests for a replica."""

    def __init__(self, source, target, replica_name, time):
        """Initialize new instance.

        :param source: node that issues a request
        :type source: :py:class:`~models.node.Node`
        :param target: target node from which `replica` is requested
        :type target: :py:class:`~models.node.Node`
        :param str replica_name: requested replica's name
        :param int time: time when request was received
        """
        self.source = source
        self.target = target
        self.replica_name = replica_name
        self.time = time

    def __str__(self):
        return "<SendReplicaRequest event @ {}> ({} --> {}, {})".format(
            self.time, self.source.name, self.target.name, self.replica_name)


class SendReplica(_Event):
    """An event when node sends back a replica to the requester."""

    def __init__(self, source, target, replica, time):
        """Initialize new instance.

        :param source: node that sends back requested `replica`
        :type source: :py:class:`~models.node.Node`
        :param target: node that should receive requested `replica`
        :type target: :py:class:`~models.node.Node`
        :param replica: replica that was requested
        :type replica: :py:class:`~models.replica.Replica`
        :param int time: time of occurence (sending back a replica)
        """
        self.source = source
        self.target = target
        self.replica = replica
        self.time = time

    def __str__(self):
        target_name = self.target.name if self.target else "None"
        return "<SendReplica event @ {}> ({} <-- {}, {})".format(
            self.time, target_name, self.source.name, self.replica.name)


class ReceiveReplica(_Event):
    """An event when node sends back a replica to the requester."""

    def __init__(self, source, target, replica, time):
        """Initialize new instance.

        :param source: node that sent requested `replica`
        :type source: :py:class:`~models.node.Node`
        :param target: node that receives requested `replica`
        :type target: :py:class:`~models.node.Node`
        :param replica: replica that was requested
        :type replica: :py:class:`~models.replica.Replica`
        :param int time: time of occurence (sending back a replica)
        """
        self.source = source
        self.target = target
        self.replica = replica
        self.time = time

    def __str__(self):
        return "<ReceiveReplica event @ {}> ({} <-- {}, {})".format(
            self.time, self.target.name, self.source.name, self.replica.name)
