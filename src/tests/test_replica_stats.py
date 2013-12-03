import unittest


class TestReplicaStats(unittest.TestCase):
    """Tests for :py:class:`~models.node._ReplicaStats` helper class."""

    def _getTargetClass(self):
        from models.node import _ReplicaStats
        return _ReplicaStats

    def _makeInstance(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_init(self):
        """Test than new _ReplicaStats instances are correctly initialized."""
        stats = self._makeInstance(nor=5, fsti=2, lrt=100)
        self.assertEqual(stats.nor, 5)
        self.assertEqual(stats.lrt, 100)
        self.assertEqual(stats.fsti, 2)

    def test_init_checks_nor_to_be_non_negative(self):
        """Test that new Replica instances cannot be initialized with a
        negative NOR.
        """
        try:
            self._makeInstance(nor=0, fsti=10, lrt=10)
        except ValueError:
            self.fail('Init should not fail if NOR is zero (0)')

        self.assertRaises(ValueError, self._makeInstance,
                          nor=-1, fsti=10, lrt=10)

    def test_init_checks_fsti_to_be_non_negative(self):
        """Test that new Replica instances cannot be initialized with a
        negative FSTI.
        """
        try:
            self._makeInstance(nor=10, fsti=0, lrt=10)
        except ValueError:
            self.fail('Init should not fail if FSTI is zero (0)')

        self.assertRaises(ValueError, self._makeInstance,
                          nor=10, fsti=-1, lrt=10)

    def test_init_checks_lrt_to_be_non_negative(self):
        """Test that new Replica instances cannot be initialized with a
        negative LRT.
        """
        try:
            self._makeInstance(nor=10, fsti=10, lrt=0)
        except ValueError:
            self.fail('Init should not fail if LRT is zero (0)')

        self.assertRaises(ValueError, self._makeInstance,
                          nor=10, fsti=10, lrt=-1)

    def test_lrt_readonly(self):
        """Test that instance's `lrt` property is read-only."""
        stats = self._makeInstance(nor=10, fsti=100, lrt=5)
        try:
            stats.lrt = 25
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'lrt' is not read-only.")

    def test_nor_readonly(self):
        """Test that instance's `nor` property is read-only."""
        stats = self._makeInstance(nor=10, fsti=100, lrt=5)
        try:
            stats.nor = 15
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'nor' is not read-only.")

    def test_nor_fsti(self):
        """Test nor_fsti() method."""
        stats = self._makeInstance(nor=0, fsti=10, lrt=0)
        stats.new_request_made(time=2)
        stats.new_request_made(time=4)
        stats.new_request_made(time=8)

        self.assertEqual(stats.nor_fsti(10), 3)

        # should *include* times (T - FSTI)
        self.assertEqual(stats.nor_fsti(12), 3)

        self.assertEqual(stats.nor_fsti(13), 2)
        self.assertEqual(stats.nor_fsti(15), 1)
        self.assertEqual(stats.nor_fsti(18), 1)
        self.assertEqual(stats.nor_fsti(19), 0)

    def test_new_request_made(self):
        """Test new_request_made() method."""
        stats = self._makeInstance(nor=0, fsti=10, lrt=0)
        stats.new_request_made(time=7)
        stats.new_request_made(time=16)

        self.assertEqual(stats.lrt, 16)
        self.assertEqual(stats.nor, 2)
        self.assertEqual(stats.nor_fsti(20), 1)
