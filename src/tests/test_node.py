"""Tests for :py:mod:`models.node` module."""

from collections import OrderedDict
from unittest.mock import Mock

import unittest


class TestReplicaStats(unittest.TestCase):
    """Tests for :py:class:`~models.node._ReplicaStats` helper class."""

    def _get_target_class(self):
        from models.node import _ReplicaStats
        return _ReplicaStats

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_init(self):
        """Test than new _ReplicaStats instances are correctly initialized."""
        stats = self._make_instance(nor=5, fsti=2, lrt=100)
        self.assertEqual(stats.nor, 5)
        self.assertEqual(stats.lrt, 100)
        self.assertEqual(stats.fsti, 2)

    def test_init_checks_nor_to_be_non_negative(self):
        """Test that new Replica instances cannot be initialized with a
        negative NOR.
        """
        try:
            self._make_instance(nor=0, fsti=10, lrt=10)
        except ValueError:
            self.fail('Init should not fail if NOR is zero (0)')

        self.assertRaises(ValueError, self._make_instance,
                          nor=-1, fsti=10, lrt=10)

    def test_init_checks_fsti_to_be_non_negative(self):
        """Test that new Replica instances cannot be initialized with a
        negative FSTI.
        """
        try:
            self._make_instance(nor=10, fsti=0, lrt=10)
        except ValueError:
            self.fail('Init should not fail if FSTI is zero (0)')

        self.assertRaises(ValueError, self._make_instance,
                          nor=10, fsti=-1, lrt=10)

    def test_init_checks_lrt_to_be_non_negative(self):
        """Test that new Replica instances cannot be initialized with a
        negative LRT.
        """
        try:
            self._make_instance(nor=10, fsti=10, lrt=0)
        except ValueError:
            self.fail('Init should not fail if LRT is zero (0)')

        self.assertRaises(ValueError, self._make_instance,
                          nor=10, fsti=10, lrt=-1)

    def test_lrt_readonly(self):
        """Test that instance's `lrt` property is read-only."""
        stats = self._make_instance(nor=10, fsti=100, lrt=5)
        try:
            stats.lrt = 25
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'lrt' is not read-only.")

    def test_nor_readonly(self):
        """Test that instance's `nor` property is read-only."""
        stats = self._make_instance(nor=10, fsti=100, lrt=5)
        try:
            stats.nor = 15
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'nor' is not read-only.")

    def test_nor_fsti(self):
        """Test nor_fsti() method."""
        stats = self._make_instance(nor=0, fsti=10, lrt=0)
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
        stats = self._make_instance(nor=0, fsti=10, lrt=0)
        stats.new_request_made(time=7)
        stats.new_request_made(time=16)

        self.assertEqual(stats.lrt, 16)
        self.assertEqual(stats.nor, 2)
        self.assertEqual(stats.nor_fsti(20), 1)


class TestNode(unittest.TestCase):
    """Tests for :py:class:`~models.node.Node` class."""

    def _get_target_class(self):
        from models.node import Node
        return Node

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def _make_sim(self, *args, **kwargs):
        """Make an instance of Simulation."""
        from models.simulation import Simulation
        return Simulation(*args, **kwargs)

    def _make_replica(self, *args, **kwargs):
        """Make an instance of Replica."""
        from models.replica import Replica
        return Replica(*args, **kwargs)

    def test_init(self):
        """Test than new instances are correctly initialized."""
        sim = self._make_sim()
        replicas = OrderedDict(
            replica_1=self._make_replica('replica_1', 3000),
            replica_2=self._make_replica('replica_2', 7500),
            replica_3=self._make_replica('replica_3', 12000),
        )
        node = self._make_instance('node_1', 30000, sim, replicas=replicas)

        self.assertEqual(node.name, 'node_1')
        self.assertEqual(node.capacity, 30000)
        self.assertEqual(node.free_capacity, 7500)

        self.assertIsNone(node._parent)
        self.assertIs(node._sim, sim)
        self.assertEqual(node._replicas.keys(), replicas.keys())

    def test_init_checks_capacity_to_be_positive(self):
        """Test than new instances are correctly initialized."""
        sim = self._make_sim()

        with self.assertRaises(ValueError):
            self._make_instance('node_1', -1000, sim)

        with self.assertRaises(ValueError):
            self._make_instance('node_1', 0, sim)

    def test_name_readonly(self):
        """Test that instance's `name` property is read-only."""
        sim = self._make_sim()
        node = self._make_instance('node_1', 1000, sim)

        try:
            node.name = 'new_name'
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'name' is not read-only.")

    def test_capacity_readonly(self):
        """Test that instance's `capacity` property is read-only."""
        sim = self._make_sim()
        node = self._make_instance('node_1', 1000, sim)

        try:
            node.capacity = 5000
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'capacity' is not read-only.")

    def test_free_capacity_readonly(self):
        """Test that instance's `free_capacity` property is read-only."""
        sim = self._make_sim()
        node = self._make_instance('node_1', 1000, sim)

        try:
            node.free_capacity = 200
        except AttributeError:
            pass
        else:
            self.fail("Attribute 'free_capacity' is not read-only.")

    def test_set_parent(self):
        """Test that set_parent() correctly sets node's parent."""
        sim = self._make_sim()
        node = self._make_instance('node_1', 1000, sim)
        parent_node = self._make_instance('parent', 10000, sim)

        self.assertIsNone(node._parent)
        node.set_parent(parent_node)
        self.assertIs(node._parent, parent_node)

    # TODO: GV_

    def test_replica_value(self):
        """Test that the value of a replica is calculated correctly.

        Replica value is calculated by the following formula (from the paper):

        NOR / replica.size + NOR_FSTI / FSTI + 1 / (now - repl_last_req_time)
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        repl_stats = Mock(nor=0, lrt=0)
        repl_stats.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = repl_stats

        self.assertAlmostEqual(node._RV(replica), 0.25)

        repl_stats.nor = 20
        self.assertAlmostEqual(node._RV(replica), 0.35)

        replica._size = 50
        self.assertAlmostEqual(node._RV(replica), 0.65)

        repl_stats.nor_fsti.return_value = 15
        self.assertAlmostEqual(node._RV(replica), 2.15)

        repl_stats.lrt = 3
        self.assertAlmostEqual(node._RV(replica), 2.90)

        sim.now = 8
        self.assertAlmostEqual(node._RV(replica), 2.10)

    # TODO: _store_if_valuable

    # TODO: request_replica

    # TODO: _copy_replica

    # TODO: delete_replica
