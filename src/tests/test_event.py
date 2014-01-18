"""Tests for :py:mod:`models.event` module."""

from unittest.mock import Mock

import unittest


class TestEvent(unittest.TestCase):
    """Tests for :py:class:`~models.event._Event` base class."""

    def _get_target_class(self):
        from models.event import _Event
        return _Event

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_init(self):
        """Test than new instances are correctly initialized."""
        event = self._make_instance()

        self.assertEqual(event._generators, [])


class TestReceiveReplicaRequest(unittest.TestCase):
    """Tests for :py:class:`~models.event.ReceiveReplicaRequest` class."""

    def _get_target_class(self):
        from models.event import ReceiveReplicaRequest
        return ReceiveReplicaRequest

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_is_subclass_of_event(self):
        """Test that `ReceiveReplicaRequest` is a subclass of _Event."""
        from models.event import _Event
        self.assertTrue(issubclass(self._get_target_class(), _Event))

    def test_init(self):
        """Test than new instances are correctly initialized."""
        source_node = Mock()
        target_node = Mock()
        event = self._make_instance(source_node, target_node, 'replica_XYZ')

        self.assertIs(event.source, source_node)
        self.assertIs(event.target, target_node)
        self.assertEqual(event.replica_name, 'replica_XYZ')

    def test_str(self):
        """Test string reprezentation of an instance."""
        source_node = Mock()
        source_node.name = 'node_S'

        target_node = Mock()
        target_node.name = 'node_T'

        event = self._make_instance(source_node, target_node, 'replica_XYZ')

        self.assertEqual(
            str(event),
            "<ReceiveReplicaRequest event> (node_S --> node_T, replica_XYZ)"
        )

    def test_str_source_is_none(self):
        """Test string reprezentation of an instance when source node is None.
        """
        target_node = Mock()
        target_node.name = 'node_T'

        event = self._make_instance(None, target_node, 'replica_XYZ')

        self.assertEqual(
            str(event),
            "<ReceiveReplicaRequest event> (None --> node_T, replica_XYZ)"
        )


class TestSendReplicaRequest(unittest.TestCase):
    """Tests for :py:class:`~models.event.SendReplicaRequest` class."""

    def _get_target_class(self):
        from models.event import SendReplicaRequest
        return SendReplicaRequest

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_is_subclass_of_event(self):
        """Test that `SendReplicaRequest` is a subclass of _Event."""
        from models.event import _Event
        self.assertTrue(issubclass(self._get_target_class(), _Event))

    def test_init(self):
        """Test than new instances are correctly initialized."""
        source_node = Mock()
        target_node = Mock()
        event = self._make_instance(source_node, target_node, 'replica_XYZ')

        self.assertIs(event.source, source_node)
        self.assertIs(event.target, target_node)
        self.assertEqual(event.replica_name, 'replica_XYZ')

    def test_str(self):
        """Test string reprezentation of an instance."""
        source_node = Mock()
        source_node.name = 'node_S'

        target_node = Mock()
        target_node.name = 'node_T'

        event = self._make_instance(source_node, target_node, 'replica_XYZ')

        self.assertEqual(
            str(event),
            "<SendReplicaRequest event> (node_S --> node_T, replica_XYZ)"
        )


class TestSendReplica(unittest.TestCase):
    """Tests for :py:class:`~models.event.SendReplica` class."""

    def _get_target_class(self):
        from models.event import SendReplica
        return SendReplica

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_is_subclass_of_event(self):
        """Test that `SendReplica` is a subclass of _Event."""
        from models.event import _Event
        self.assertTrue(issubclass(self._get_target_class(), _Event))

    def test_init(self):
        """Test than new instances are correctly initialized."""
        source_node = Mock()
        target_node = Mock()
        replica = Mock()
        event = self._make_instance(source_node, target_node, replica)

        self.assertIs(event.source, source_node)
        self.assertIs(event.target, target_node)
        self.assertIs(event.replica, replica)

    def test_str(self):
        """Test string reprezentation of an instance."""
        source_node = Mock()
        source_node.name = 'node_S'

        target_node = Mock()
        target_node.name = 'node_T'

        replica = Mock()
        replica.name = 'replica_XYZ'

        event = self._make_instance(source_node, target_node, replica)

        self.assertEqual(
            str(event),
            "<SendReplica event> (node_T <-- node_S, replica_XYZ)"
        )

    def test_str_target_is_none(self):
        """Test string reprezentation of an instance when target node is None.
        """
        source_node = Mock()
        source_node.name = 'node_S'

        replica = Mock()
        replica.name = 'replica_XYZ'

        event = self._make_instance(source_node, None, replica)

        self.assertEqual(
            str(event),
            "<SendReplica event> (None <-- node_S, replica_XYZ)"
        )


class TestReceiveReplica(unittest.TestCase):
    """Tests for :py:class:`~models.event.SendReplica` class."""

    def _get_target_class(self):
        from models.event import ReceiveReplica
        return ReceiveReplica

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_is_subclass_of_event(self):
        """Test that `ReceiveReplica` is a subclass of _Event."""
        from models.event import _Event
        self.assertTrue(issubclass(self._get_target_class(), _Event))

    def test_init(self):
        """Test than new instances are correctly initialized."""
        source_node = Mock()
        target_node = Mock()
        replica = Mock()
        event = self._make_instance(source_node, target_node, replica)

        self.assertIs(event.source, source_node)
        self.assertIs(event.target, target_node)
        self.assertIs(event.replica, replica)

    def test_str(self):
        """Test string reprezentation of an instance."""
        source_node = Mock()
        source_node.name = 'node_S'

        target_node = Mock()
        target_node.name = 'node_T'

        replica = Mock()
        replica.name = 'replica_XYZ'

        event = self._make_instance(source_node, target_node, replica)

        self.assertEqual(
            str(event),
            "<ReceiveReplica event> (node_T <-- node_S, replica_XYZ)"
        )
