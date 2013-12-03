import unittest


class TestReplica(unittest.TestCase):
    """Tests for :py:class:`~models.replica.Replica` model."""

    def _get_target_class(self):
        from models.replica import Replica
        return Replica

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_init(self):
        """Test than new Replica instances are correctly initialized."""
        replica = self._make_instance('replica_1', 1234)
        self.assertEqual(replica.name, 'replica_1')
        self.assertEqual(replica.size, 1234)

    def test_init_checks_size_to_be_positive(self):
        """Test that new Replica instances cannot be initialized with a
        non-positive size.
        """
        self.assertRaises(ValueError, self._make_instance, 'replica_1', 0)
        self.assertRaises(ValueError, self._make_instance, 'replica_1', -1234)

    def test_name_readonly(self):
        """Test that instance's `name` property is read-only."""
        replica = self._make_instance('replica_1', 1234)
        try:
            replica.name = 'New Name'
        except AttributeError:
            pass
        else:
            self.fail("Name attribute is not read-only.")

    def test_size_readonly(self):
        """Test that instance's `size` property is read-only."""
        replica = self._make_instance('replica_1', 1234)
        try:
            replica.size = 5678
        except AttributeError:
            pass
        else:
            self.fail("Size attribute is not read-only.")
