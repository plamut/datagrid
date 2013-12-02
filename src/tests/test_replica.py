import unittest

class TestReplica(unittest.TestCase):
    """Tests for Replica model."""

    def _getTargetClass(self):
        from models.replica import Replica
        return Replica

    def _makeInstance(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_init(self):
        """Test than new instances are correctly initialized."""
        replica = self._makeInstance('replica_1', 1234)
        self.assertEqual(replica.name, 'replica_1')
        self.assertEqual(replica.size, 1234)

    def test_init_checks_size_to_be_positive(self):
        """Test that new replica instance cannot be initialized with a
        non-positive size.
        """
        self.assertRaises(ValueError, self._makeInstance, 'replica_1', 0)
        self.assertRaises(ValueError, self._makeInstance, 'replica_1', -1234)

    def test_name_readonly(self):
        """Test that instance's `name` property is read-only."""
        replica = self._makeInstance('replica_1', 1234)
        try:
            replica.name = 'New Name'
        except AttributeError:
            pass
        else:
            self.fail("Name attribute is not read-only.")

    def test_size_readonly(self):
        """Test that instance's `size` property is read-only."""
        replica = self._makeInstance('replica_1', 1234)
        try:
            replica.size = 5678
        except AttributeError:
            pass
        else:
            self.fail("Size attribute is not read-only.")
