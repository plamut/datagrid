"""Tests for :py:mod:`models.node` module."""

from collections import OrderedDict
from copy import deepcopy
from unittest.mock import Mock
from unittest.mock import patch

import inspect
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
        """Test that new instances are correctly initialized."""
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
        """Test that init checks that `capacity` is a positive number."""
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

    def test_gv_raises_not_implemented(self):
        """Test that _GV() method raises NotImplementedError."""
        node = self._make_instance('node_1', 1000, Mock())
        with self.assertRaises(NotImplementedError):
            node._GV([])

    def test_rv_raises_not_implemented(self):
        """Test that _RV() raises NotImplementedError."""
        node = self._make_instance('node_1', 1000, Mock())
        with self.assertRaises(NotImplementedError):
            node._RV(Mock())

    def test_reorder_replicas(self):
        """Test that _reorder_replicas orders replicas by their replica value.
        """
        sim = Mock()

        replicas = [
            self._make_replica('replica_1', size=270),
            self._make_replica('replica_2', size=800),
            self._make_replica('replica_3', size=500),
            self._make_replica('replica_4', size=750),
            self._make_replica('replica_5', size=110),
        ]

        node = self._make_instance('node_1', 2000, sim)
        node._replicas = OrderedDict((r.name, r) for r in replicas)
        node._replica_stats = OrderedDict((r.name, Mock()) for r in replicas)

        # make smaller replicas less valuable
        node._RV = Mock(side_effect=lambda r: r.size)
        node._reorder_replicas()

        self.assertEqual(
            list(node._replicas.keys()),
            ['replica_5', 'replica_1', 'replica_3', 'replica_4', 'replica_2']
        )

    def test_store_if_valuable_enough_free_space(self):
        """Test that _store_if_valuable method stores a new replica when there
        is enough free space.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=300),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 1000, sim, replicas)

        # make sure replica is considered less valuable than the group, so that
        # if it gets stored, it is because of enough free space on the node
        node._RV = Mock(return_value=7)
        node._GV = Mock(return_value=10)

        new_replica = self._make_replica('new_replica', size=500)

        node._store_if_valuable(new_replica)
        self.assertEqual(len(node._replicas), 3)
        self.assertIn('new_replica', node._replicas)

    def test_store_if_valuable_limited_space_replica_more_important(self):
        """Test that _store_if_valuable method stores a new replica when there
        is not enough free space, but this replica is valued high enough.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=300),
            self._make_replica('replica_3', size=400),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 1000, sim, replicas)

        node._RV = Mock(return_value=15)
        node._GV = Mock(return_value=8)

        new_replica = self._make_replica('new_replica', size=501)
        node._store_if_valuable(new_replica)
        self.assertEqual(len(node._replicas), 2)
        self.assertIn('new_replica', node._replicas)
        self.assertNotIn('replica_1', node._replicas)
        self.assertNotIn('replica_2', node._replicas)

    def test_store_if_valuable_limited_space_replica_less_important(self):
        """Test that _store_if_valuable method does not store a new replica
        when there is *not* enough free space and this replica is not valued
        high enough.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=300),
            self._make_replica('replica_3', size=400),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 1000, sim, replicas)

        node._RV = Mock(return_value=2)
        node._GV = Mock(return_value=14)

        new_replica = self._make_replica('new_replica', size=501)
        node._store_if_valuable(new_replica)
        self.assertEqual(len(node._replicas), 3)
        self.assertNotIn('new_replica', node._replicas)

    def test_request_replica_returns_generator(self):
        """Test that `request_replica` method returns a generator."""
        node = self._make_instance('node_1', 50000, Mock(name='sim_obj'))
        g = node.request_replica('replica_XYZ', Mock(name='requesting_node'))
        self.assertTrue(inspect.isgenerator(g))

    def test_request_replica_replica_present(self):
        """Test request_replica method when node has a copy of the requested
        replica.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4
        event_send_repl = Mock(name='SendReplica')
        sim.event_send_replica.return_value = event_send_repl

        replica_1 = self._make_replica('replica_1', size=100)
        replica_2 = self._make_replica('replica_2', size=350)
        replicas = OrderedDict([
            (replica_1.name, replica_1),
            (replica_2.name, replica_2),
        ])

        node = self._make_instance('node_1', 5000, sim, replicas)
        node._replica_stats['replica_1']._nor = 10
        requester = self._make_instance('req_node', 5000, sim)

        g = node.request_replica('replica_1', requester)
        event = next(g)

        # check that correct event was yielded (send replica)
        self.assertIs(event, event_send_repl)
        self.assertEqual(
            sim.event_send_replica.call_args,
            ((node, requester, replica_1), {})
        )

        # check that requested replica's stats have been updated as well
        stats = node._replica_stats['replica_1']
        self.assertEqual(stats.nor, 11)
        self.assertEqual(stats.nor_fsti(sim.now), 1)
        self.assertEqual(stats.lrt, sim.now)

        with self.assertRaises(StopIteration):
            next(g)  # no more events yielded

    @patch('models.node.Node._RV', lambda self, replica: id(replica))
    def test_request_replica_replica_not_present(self):
        """Test request_replica method when node does not have the requested
        replica.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        event_send_repl_req = Mock(name='SendReplicaRequest')
        sim.event_send_replica_request.return_value = event_send_repl_req

        event_send_repl = Mock(name='SendReplica')
        sim.event_send_replica.return_value = event_send_repl

        replica_1 = self._make_replica('replica_1', size=100)
        replicas = OrderedDict(replica_1=replica_1)

        parent = self._make_instance('parent', 5000, sim, replicas)

        node = self._make_instance('node_1', 5000, sim)
        node.set_parent(parent)
        node._store_if_valuable = Mock(wraps=node._store_if_valuable)

        requester = self._make_instance('req_node', 5000, sim)

        g = node.request_replica('replica_1', requester)
        event = next(g)

        # check that correct event was yielded (send replica reques)
        self.assertIs(event, event_send_repl_req)
        self.assertEqual(
            sim.event_send_replica_request.call_args,
            ((node, parent, 'replica_1'), {})
        )

        # simulate response (replica receive event after some delay) and see
        # what we get
        sim.now += 10
        event = g.send(replica_1)

        self.assertEqual(
            node._store_if_valuable.call_args,
            ((replica_1,), {})
        )

        self.assertIs(event, event_send_repl)
        self.assertEqual(
            sim.event_send_replica.call_args,
            ((node, requester, replica_1), {})
        )

        # check that requested replica's stats have been updated as well
        stats = node._replica_stats['replica_1']
        self.assertEqual(stats.nor, 1)
        self.assertEqual(stats.nor_fsti(sim.now), 1)
        self.assertEqual(stats.lrt, 4)  # the time replica was requested

        with self.assertRaises(StopIteration):
            next(g)  # no more events yielded

    def test_request_replica_replica_received_during_waiting(self):
        """Test request_replica method when replica is received (because of an
        earlier request) while the node is still waiting for this same replica.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        event_send_repl_req = Mock(name='SendReplicaRequest')
        sim.event_send_replica_request.return_value = event_send_repl_req

        event_send_repl = Mock(name='SendReplica')
        sim.event_send_replica.return_value = event_send_repl

        replica_1 = self._make_replica('replica_1', size=100)
        replicas = OrderedDict(replica_1=replica_1)

        parent = self._make_instance('parent', 5000, sim, replicas)

        node = self._make_instance('node_1', 5000, sim)
        node.set_parent(parent)
        node._store_if_valuable = Mock(wraps=node._store_if_valuable)

        requester = self._make_instance('req_node', 5000, sim)

        g = node.request_replica('replica_1', requester)
        event = next(g)

        # simulate early arrival of a replica
        node._copy_replica(replica_1)
        sim.now += 10
        stats_1 = node._replica_stats[replica_1.name]
        stats_1.new_request_made = Mock(wraps=stats_1.new_request_made)

        event = g.send(replica_1)

        self.assertFalse(node._store_if_valuable.called)
        self.assertEqual(stats_1.new_request_made.call_count, 1)
        self.assertEqual(
            stats_1.new_request_made.call_args,
            ((4,), {})  # the last time replica was requested
        )

        self.assertIs(event, event_send_repl)
        self.assertEqual(
            sim.event_send_replica.call_args,
            ((node, requester, replica_1), {})
        )

        with self.assertRaises(StopIteration):
            next(g)  # no more events yielded

    @patch('models.node.Node._RV', lambda self, replica: id(replica))
    def test_copy_replica_already_exists(self):
        """Test that _copy_replica raises an error if replica already exists.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replica = self._make_replica('replica', size=120)
        node = self._make_instance(
            'node_1', 500, sim, OrderedDict(replica=replica))

        replica_clone = deepcopy(replica)

        with self.assertRaises(ValueError):
            node._copy_replica(replica_clone)

    def test_copy_replica_not_enough_space(self):
        """Test that _copy_replica raises ValueError if there is not enough
        free space.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=100),
            self._make_replica('replica_2', size=350),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 500, sim, replicas)

        new_replica = self._make_replica('new_replica', size=51)
        with self.assertRaises(ValueError):
            node._copy_replica(new_replica)

    def test_copy_replica_enough_space(self):
        """Test that _copy_replica correctly stores a replica when there is
        enough free space.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=100),
            self._make_replica('replica_2', size=800),
            self._make_replica('replica_3', size=500),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 2000, sim, replicas)

        # make smaller replicas more valuable
        node._RV = Mock(side_effect=lambda r: 1000 - r.size)

        new_replica = self._make_replica('new_replica', size=400)
        node._copy_replica(new_replica)

        self.assertEqual(len(node._replicas), 4)
        self.assertIn('new_replica', node._replicas)
        self.assertIn('new_replica', node._replica_stats)
        self.assertEqual(node.free_capacity, 200)
        self.assertEqual(
            list(node._replicas.keys()),
            ['replica_1', 'replica_2', 'replica_3', 'new_replica']
        )

    def test_delete_replica_replica_exists(self):
        """Test that _delete_replica correctly deletes an existing replica.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=300),
            self._make_replica('replica_3', size=400),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 1000, sim, replicas)

        node._delete_replica('replica_3')
        self.assertEqual(len(node._replicas), 2)
        self.assertNotIn('replica_3', node._replicas)
        self.assertNotIn('replica_3', node._replica_stats)
        self.assertEqual(node.free_capacity, 500)

    def test_delete_replica_non_existent(self):
        """Test that _delete_replica raises ValueError for non-existent
        replicas.
        """
        sim = Mock()
        sim.fsti = 10
        sim.now = 4

        replicas = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=300),
            self._make_replica('replica_3', size=400),
        ]
        replicas = OrderedDict((r.name, r) for r in replicas)
        node = self._make_instance('node_1', 1000, sim, replicas)

        with self.assertRaises(ValueError):
            node._delete_replica('non-existent')


class TestNodeEFS(unittest.TestCase):
    """Tests for :py:class:`~models.node.NodeEFS` class."""

    # XXX: refactor some common utility methods into some base testcase class?
    def _get_target_class(self):
        from models.node import NodeEFS
        return NodeEFS

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

    def test_is_node_subclass(self):
        """Test if target class is a subclass of Node."""
        from models.node import Node
        self.assertTrue(issubclass(self._get_target_class(), Node))

    def test_group_value(self):
        """Test that the value of a replica group is calculated correctly.

        Group value is calculated by the following formula (from the paper):

            sum(NOR) / sum(sizes) + sum(NOR_FSTI) / FSTI +
            1 / (now - avg(last_req_time))
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4

        node = self._make_instance('node_1', 1000, sim)

        replica_group = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=400),
            self._make_replica('replica_3', size=900),
        ]

        stats_1 = Mock(nor=0, lrt=0)
        stats_1.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = stats_1

        stats_2 = Mock(nor=0, lrt=0)
        stats_2.nor_fsti.return_value = 0
        node._replica_stats['replica_2'] = stats_2

        stats_3 = Mock(nor=0, lrt=0)
        stats_3.nor_fsti.return_value = 0
        node._replica_stats['replica_3'] = stats_3

        self.assertAlmostEqual(node._GV(replica_group), 0.25)

        stats_1.nor = 15
        self.assertAlmostEqual(node._GV(replica_group), 0.26)
        stats_2.nor = 45
        self.assertAlmostEqual(node._GV(replica_group), 0.29)

        replica_group[2]._size = 2400
        self.assertAlmostEqual(node._GV(replica_group), 0.27)

        stats_1.nor_fsti.return_value = 5
        self.assertAlmostEqual(node._GV(replica_group), 0.77)
        stats_2.nor_fsti.return_value = 20
        self.assertAlmostEqual(node._GV(replica_group), 2.77)

        stats_1.lrt = 1
        stats_2.lrt = 2
        stats_3.lrt = 3
        self.assertAlmostEqual(node._GV(replica_group), 3.02)

        sim.now = 7
        self.assertAlmostEqual(node._GV(replica_group), 2.72)

    def test_group_value_when_empty(self):
        """Test that the calculated value of an empty replica group is zero."""
        sim = Mock()
        node = self._make_instance('node_1', 1000, sim)
        self.assertAlmostEqual(node._GV([]), 0.0)

    def test_group_value_zero_denominator(self):
        """Test that the value of a replica group is calculated correctly in
        cases of a zero denominator.

        This can happen if the current simulation time is exactly equal to
        replica's last requested time (e.g. when sorting replicas by
        importance) and this replica is the only one in replica group.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4.0

        node = self._make_instance('node_1', 1000, sim)

        replica_group = [self._make_replica('replica_1', size=200)]

        stats_1 = Mock(nor=0, lrt=4.0)
        stats_1.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = stats_1

        try:
            result = node._GV(replica_group)
        except ZeroDivisionError:
            self.fail("Incorrect handling of a zero denominator.")
        else:
            self.assertEqual(result, float('inf'))

    def test_replica_value(self):
        """Test that the value of a replica is calculated correctly.

        Replica value is calculated by the following formula (from the paper):

            NOR / replica.size + NOR_FSTI / FSTI +
            1 / (now - repl_last_req_time)
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

    def test_replica_value_zero_denominator(self):
        """Test that _RV correctly handles cases with a denominator of zero.

        This can happen if the current simulation time is exactly equal to
        replica's last requested time (e.g. when sorting replicas by
        importance).
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        repl_stats = Mock(nor=0, lrt=4)
        repl_stats.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = repl_stats

        try:
            result = node._RV(replica)
        except ZeroDivisionError:
            self.fail("Incorrect handling of a zero denominator.")
        else:
            self.assertEqual(result, float('inf'))


class TestNodeLRU(unittest.TestCase):
    """Tests for :py:class:`~models.node.NodeLRU` class."""

    # XXX: refactor some common utility methods into some base testcase class?
    def _get_target_class(self):
        from models.node import NodeLRU
        return NodeLRU

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

    def test_is_node_subclass(self):
        """Test if target class is a subclass of Node."""
        from models.node import Node
        self.assertTrue(issubclass(self._get_target_class(), Node))

    def test_group_value(self):
        """Test that the value of a replica group is calculated correctly.

        Group value is always negative, so that it is less important than any
        new replica.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4

        node = self._make_instance('node_1', 1000, sim)

        replica_group = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=400),
        ]

        stats_1 = Mock(nor=0, lrt=0)
        stats_1.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = stats_1

        stats_2 = Mock(nor=0, lrt=0)
        stats_2.nor_fsti.return_value = 0
        node._replica_stats['replica_2'] = stats_2

        self.assertAlmostEqual(node._GV(replica_group), -1.0)

    def test_replica_value_no_stats(self):
        """Test that the value of a replica with no stats is calculated
        correctly.

        Replica does not have any stats yet in cases when it has just been
        retrieved from a parent node and is a candidate for storing a copy of
        it on the node that requested it.

        When replica does not yet have any stats, its value is the current
        simulation time.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4.77

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        self.assertEqual(node._RV(replica), 4.77)

    def test_replica_value_stats_exist(self):
        """Test that the value of a replica with existing stats is calculated
        correctly.

        Replica value is just a time when it was last requested.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4.77

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        repl_stats = Mock(nor=0, lrt=2.57)
        repl_stats.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = repl_stats

        self.assertEqual(node._RV(replica), 2.57)


class TestNodeLFU(unittest.TestCase):
    """Tests for :py:class:`~models.node.NodeLFU` class."""

    # XXX: refactor some common utility methods into some base testcase class?
    def _get_target_class(self):
        from models.node import NodeLFU
        return NodeLFU

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

    def test_is_node_subclass(self):
        """Test if target class is a subclass of Node."""
        from models.node import Node
        self.assertTrue(issubclass(self._get_target_class(), Node))

    def test_group_value(self):
        """Test that the value of a replica group is calculated correctly.

        Group value is always negative, so that it is less important than any
        new replica.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4

        node = self._make_instance('node_1', 1000, sim)

        replica_group = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=400),
        ]

        stats_1 = Mock(nor=0, lrt=0)
        stats_1.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = stats_1

        stats_2 = Mock(nor=0, lrt=0)
        stats_2.nor_fsti.return_value = 0
        node._replica_stats['replica_2'] = stats_2

        self.assertAlmostEqual(node._GV(replica_group), -1.0)

    def test_replica_value_no_stats(self):
        """Test that the value of a replica with no stats is calculated
        correctly.

        Replica does not have any stats yet in cases when it has just been
        retrieved from a parent node and is a candidate for storing a copy of
        it on the node that requested it.

        When replica does not yet have any stats, its value is the default
        NOR value (0).
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4.77

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        self.assertEqual(node._RV(replica), 0.0)

    def test_replica_value_stats_exist(self):
        """Test that the value of a replica with existing stats is calculated
        correctly.

        Replica does not have any stats yet in cases when it has just been
        retrieved from a parent node and is a candidate for storing a copy of
        it on the node that requested it.

        Replica value is just the number of time it has been requested.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4.77

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        repl_stats = Mock(nor=6, lrt=2.57)
        repl_stats.nor_fsti.return_value = 0
        node._replica_stats['replica_1'] = repl_stats

        self.assertEqual(node._RV(replica), 6)


class TestNodeMFS(unittest.TestCase):
    """Tests for :py:class:`~models.node.NodeMFS` class."""

    # XXX: refactor some common utility methods into some base testcase class?
    def _get_target_class(self):
        from models.node import NodeMFS
        return NodeMFS

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

    def test_is_node_subclass(self):
        """Test if target class is a subclass of Node."""
        from models.node import Node
        self.assertTrue(issubclass(self._get_target_class(), Node))

    def test_reorder_replicas(self):
        """Test that _reorder_replicas correctly sorts the replica list."""
        sim = Mock(spec=self._make_sim())
        node = self._make_instance('node_1', 5000, sim)

        replica_list = [
            self._make_replica('replica_1', size=500),
            self._make_replica('replica_2', size=400),
            self._make_replica('replica_3', size=200),
            self._make_replica('replica_4', size=300),
            self._make_replica('replica_5', size=600),
        ]

        for i, replica in enumerate(replica_list):
            node._replicas['replica_{}'.format(i + 1)] = replica

        stats_list = [
            Mock(nor=7),
            Mock(nor=9),
            Mock(nor=7),
            Mock(nor=2),
            Mock(nor=7),
        ]

        for i, stats in enumerate(stats_list):
            node._replica_stats['replica_{}'.format(i + 1)] = stats

        node._reorder_replicas()

        sorted_keys = list(node._replicas.keys())
        self.assertEqual(
            sorted_keys,
            ['replica_4', 'replica_5', 'replica_1', 'replica_3', 'replica_2']
        )

    def test_group_value(self):
        """Test that the value of a replica group is calculated correctly.

        Group value is simply the sum of numbers of requests for each replica
        in the group.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 1.50
        sim.now = 4

        node = self._make_instance('node_1', 1000, sim)

        replica_group = [
            self._make_replica('replica_1', size=200),
            self._make_replica('replica_2', size=400),
            self._make_replica('replica_3', size=900),
        ]

        stats_1 = Mock(nor=6, lrt=3.50)
        stats_1.nor_fsti.return_value = 5
        node._replica_stats['replica_1'] = stats_1

        stats_2 = Mock(nor=0, lrt=0.04)
        stats_2.nor_fsti.return_value = 0
        node._replica_stats['replica_2'] = stats_2

        stats_3 = Mock(nor=11, lrt=2.40)
        stats_3.nor_fsti.return_value = 4
        node._replica_stats['replica_3'] = stats_3

        self.assertAlmostEqual(node._GV(replica_group), 17)

    def test_group_value_when_empty(self):
        """Test that the calculated value of an empty replica group is zero."""
        sim = Mock()
        node = self._make_instance('node_1', 1000, sim)
        self.assertAlmostEqual(node._GV([]), 0.0)

    def test_replica_value_no_stats(self):
        """Test that the value of a replica with no stats is calculated
        correctly.

        Replica does not have any stats yet in cases when it has just been
        retrieved from a parent node and is a candidate for storing a copy of
        it on the node that requested it.

        When replica does not yet have any stats, its value is zero (because
        its NOR is zero as well).
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 10
        sim.now = 4.77

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)

        self.assertEqual(node._RV(replica), 0.0)

    def test_replica_value_stats_exist(self):
        """Test that the value of a replica with existing stats is calculated
        correctly.

        Replica does not have any stats yet in cases when it has just been
        retrieved from a parent node and is a candidate for storing a copy of
        it on the node that requested it.

        Replica value is just the number of time it has been requested.
        """
        sim = Mock(spec=self._make_sim())
        sim.fsti = 1.50
        sim.now = 4.77

        replica = self._make_replica('replica_1', size=200)
        node = self._make_instance('node_1', 1000, sim)
        node._free_capacity = 50

        repl_stats = Mock(nor=10, lrt=2.57)
        repl_stats.nor_fsti.return_value = 4
        node._replica_stats['replica_1'] = repl_stats

        self.assertEqual(node._RV(replica), 7.50)
