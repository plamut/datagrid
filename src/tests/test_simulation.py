"""Tests for :py:mod:`models.simulation` module."""

from collections import OrderedDict
from unittest.mock import Mock

import unittest


class TestDigits(unittest.TestCase):
    """Tests for :py:func:`models.simulation._digits` helper function."""

    def _FUT(self, *args, **kwargs):
        """Invoke function under test."""
        from models.simulation import _digits
        return _digits(*args, **kwargs)

    def test_typical_values(self):
        """Test behavior for "typical" input values."""
        self.assertEqual(self._FUT(5), 1)
        self.assertEqual(self._FUT(97), 2)
        self.assertEqual(self._FUT(60049), 5)
        self.assertEqual(self._FUT(1204560), 7)

    def test_negative_values(self):
        """Test behavior for negative input values."""
        self.assertEqual(self._FUT(-5), 1)
        self.assertEqual(self._FUT(-97), 2)
        self.assertEqual(self._FUT(-60049), 5)
        self.assertEqual(self._FUT(-1204560), 7)

    def test_zero(self):
        """Test behavior for zero (0)."""
        self.assertEqual(self._FUT(0), 1)


class TestStrategy(unittest.TestCase):
    """Tests for :py:obj:`models.simulation.Strategy`."""

    def _get_OUT(self):
        """Return object under test."""
        from models.simulation import Strategy
        return Strategy

    def test_items(self):
        """Test that `Strategy` contains all the right items and nothing else.
        """
        strategy = self._get_OUT()
        self.assertEqual(len(strategy), 3)

        self.assertIn('EFS', strategy._fields)
        self.assertIn('s2012', strategy._fields)
        self.assertIn('s2013', strategy._fields)

        self.assertEqual(strategy.EFS, 1)
        self.assertEqual(strategy.s2012, 2)
        self.assertEqual(strategy.s2013, 3)


class TestClock(unittest.TestCase):
    """Tests for :py:class:`models.simulation._Clock`."""

    def _get_target_class(self):
        from models.simulation import _Clock
        return _Clock

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_init(self):
        """Test than new _Clock instances are correctly initialized."""
        clock = self._make_instance()
        self.assertEqual(clock.time, 0.0)

    def test_time_readonly(self):
        """Test that instance's `time` property is read-only."""
        clock = self._make_instance()
        try:
            clock.time = 123.45
        except AttributeError:
            pass
        else:
            self.fail("Time attribute is not read-only.")

    def test_reset(self):
        """Test that `reset` method resets time to zero (0)."""
        clock = self._make_instance()
        clock._time = 50.0  # cheating for easier testing

        self.assertEqual(clock.time, 50.0)
        clock.reset()
        self.assertEqual(clock.time, 0.0)

    def test_tick_default_step(self):
        """Test that `tick` method increases time by 1 unit by default."""
        clock = self._make_instance()
        self.assertEqual(clock.time, 0.0)

        clock.tick()
        self.assertEqual(clock.time, 1.0)
        clock.tick()
        self.assertEqual(clock.time, 2.0)
        clock.tick()
        self.assertEqual(clock.time, 3.0)

    def test_tick_non_default_step(self):
        """Test that `tick` method correctly increases time by given step."""
        clock = self._make_instance()
        self.assertEqual(clock.time, 0.0)

        clock.tick(step=4)
        self.assertEqual(clock.time, 4.0)
        clock.tick(step=0)
        self.assertEqual(clock.time, 4.0)

    def test_tick_checks_step_to_be_positive(self):
        """Test that `tick` method rejects negative steps."""
        clock = self._make_instance()
        with self.assertRaises(ValueError):
            clock.tick(step=-1.0)


class TestSimulation(unittest.TestCase):
    """Tests for :py:class:`models.simulation.Simulation`."""

    def _get_target_class(self):
        from models.simulation import Simulation
        return Simulation

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def _get_settings(self):
        """Get example simulation configuration settings.

        :returns: An example of simulation configuration settings.
        :rtype: dict
        """
        from models.simulation import Strategy
        return dict(
            node_capacity=12000,
            strategy=Strategy.EFS,
            node_count=14,
            replica_count=100,
            fsti=500,  # frequency specific time interval
            min_dist_km=5,  # min distance between two adjacent nodes
            max_dist_km=800,  # max distance between two adjacent nodes
            network_bw_mbps=20,
            pspeed_kmps=7e3,
            replica_min_size=70,  # megabits
            replica_max_size=700,  # megabits
            rnd_seed=42,
            total_reqs=10000,
        )

    def test_name_for_server_node(self):
        """Test that predefined name of the server node is correct."""
        name = self._get_target_class().SERVER_NAME
        self.assertEqual(name, 'server')

    def test_init(self):
        """Test that new instances are correctly initialized."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        self.assertEqual(sim._node_capacity, settings['node_capacity'])
        self.assertEqual(sim._strategy, settings['strategy'])
        self.assertEqual(sim._node_count, settings['node_count'])
        self.assertEqual(sim._replica_count, settings['replica_count'])
        self.assertEqual(sim._fsti, settings['fsti'])
        self.assertEqual(sim._min_dist_km, settings['min_dist_km'])
        self.assertEqual(sim._max_dist_km, settings['max_dist_km'])
        self.assertEqual(sim._network_bw_mbps, settings['network_bw_mbps'])
        self.assertEqual(sim._pspeed_kmps, settings['pspeed_kmps'])
        self.assertEqual(sim._replica_min_size, settings['replica_min_size'])
        self.assertEqual(sim._replica_max_size, settings['replica_max_size'])
        self.assertEqual(sim._rnd_seed, settings['rnd_seed'])
        self.assertEqual(sim._total_reqs, settings['total_reqs'])

        self.assertEqual(sim._replicas, OrderedDict())
        self.assertEqual(sim._nodes, OrderedDict())
        self.assertEqual(sim._edges, OrderedDict())

        from models.simulation import _Clock
        self.assertTrue(isinstance(sim._clock, _Clock))

        self.assertEqual(sim._total_bw, 0.0)
        self.assertEqual(sim._total_rt_s, 0.0)
        self.assertEqual(sim._c1, 0.001)
        self.assertEqual(sim._c2, 0.001)

    def test_init_checks_node_count_big_enough(self):
        """Test that init checks that `node_count` is at least two (2)."""
        settings = self._get_settings()
        settings['node_count'] = 1

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

        settings['node_count'] = 2
        try:
            self._make_instance(**settings)
        except:
            self.fail("Number of nodes (2) should have been accepted.")

    def test_init_rejects_non_positive_replica_count(self):
        """Test that init rejects non-positive values for `replica_count`."""
        settings = self._get_settings()
        settings['replica_count'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_fsti(self):
        """Test that init rejects non-positive values for `fsti`."""
        settings = self._get_settings()
        settings['fsti'] = 0.0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_min_dist_km(self):
        """Test that init rejects non-positive values for `min_dist_km`."""
        settings = self._get_settings()
        settings['min_dist_km'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_max_dist_km(self):
        """Test that init rejects non-positive values for `max_dist_km`."""
        settings = self._get_settings()
        settings['max_dist_km'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_checks_max_dist_is_greater_than_min_dist(self):
        """Test that init checks that `max_dist_km` is greater than
        `min_dist_km`.
        """
        settings = self._get_settings()
        settings['min_dist_km'] = settings['max_dist_km']

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_network_bw_mbps(self):
        """Test that init rejects non-positive values for `network_bw_mbps`."""
        settings = self._get_settings()
        settings['network_bw_mbps'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_replica_min_size(self):
        """Test that init rejects non-positive values for `replica_min_size`.
        """
        settings = self._get_settings()
        settings['replica_min_size'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_replica_max_size(self):
        """Test that init rejects non-positive values for `replica_max_size`.
        """
        settings = self._get_settings()
        settings['replica_max_size'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_checks_replica_max_size_is_greater_than_min_size(self):
        """Test that init checks that `replica_max_size` is greater than
        `replica_min_size`.
        """
        settings = self._get_settings()
        settings['replica_min_size'] = settings['replica_max_size']

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_non_positive_total_reqs(self):
        """Test that init rejects non-positive values for `total_reqs`.
        """
        settings = self._get_settings()
        settings['total_reqs'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_new_node(self):
        """Test that _new_node indeed creates and returns a new Node instance.
        """
        from models.node import Node

        settings = self._get_settings()
        sim = self._make_instance(**settings)

        new_node = sim._new_node('node_1', 15000, sim)
        self.assertTrue(isinstance(new_node, Node))
        self.assertEqual(sim._nodes.get('node_1'), new_node)

    def test_now(self):
        """Test that `now` returns internal clock's current time."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        sim._clock._time = 8.7602
        self.assertEqual(sim.now, 8.7602)

    def test_now_readonly(self):
        """Test that `now`property is read-only."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        try:
            sim.now = 10.501
        except AttributeError:
            pass
        else:
            self.fail("Now attribute is not read-only.")

    def test_nodes(self):
        """Test that `nodes` property returns simulation's list of nodes."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        fake_nodes = Mock()
        sim._nodes = fake_nodes
        self.assertIs(sim.nodes, fake_nodes)

    def test_replicas(self):
        """Test that `replicas` property returns simulation's list of replicas.
        """
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        fake_replicas = Mock()
        sim._replicas = fake_replicas
        self.assertIs(sim.replicas, fake_replicas)

    def test_generate_nodes(self):
        """Test that _generate_nodes creates a new set of nodes."""
        settings = self._get_settings()
        settings['node_count'] = 5
        sim = self._make_instance(**settings)

        sim._generate_nodes()

        self.assertEqual(len(sim.nodes), 5)
        for name in ['server', 'node_1', 'node_2', 'node_3', 'node_4']:
            self.assertIn(name, sim.nodes)
            self.assertIs(sim.nodes[name]._sim, sim)

            if name == 'server':
                expected_capacity = \
                    settings['replica_count'] * settings['replica_max_size']
            else:
                expected_capacity = settings['node_capacity']
            self.assertEqual(sim.nodes[name].capacity, expected_capacity)

    def test_generate_edges(self):
        """Test that _generate_edges creates edges with random lengths between
        all node pairs.
        """
        # TODO
