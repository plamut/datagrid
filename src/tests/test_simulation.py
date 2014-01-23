"""Tests for :py:mod:`models.simulation` module."""

from collections import OrderedDict
from inspect import isgenerator
from unittest.mock import Mock
from unittest.mock import patch

import heapq
import itertools
import random
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
        self.assertEqual(len(strategy), 4)

        self.assertIn('EFS', strategy._fields)
        self.assertIn('LFU', strategy._fields)
        self.assertIn('LRU', strategy._fields)
        self.assertIn('MFS', strategy._fields)

        self.assertEqual(strategy.EFS, 1)
        self.assertEqual(strategy.LFU, 2)
        self.assertEqual(strategy.LRU, 3)
        self.assertEqual(strategy.MFS, 4)


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
            replica_group_count=10,
            mwg_prob=0.3,
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
        self.assertEqual(
            sim._replica_group_count, settings['replica_group_count'])
        self.assertEqual(sim._mwg_prob, settings['mwg_prob'])
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
        self.assertEqual(sim._replica_groups, dict())
        self.assertEqual(sim._nodes, OrderedDict())
        self.assertEqual(sim._nodes_mwg, OrderedDict())
        self.assertEqual(sim._edges, OrderedDict())

        from models.simulation import _Clock
        self.assertTrue(isinstance(sim._clock, _Clock))

        self.assertEqual(sim._total_bw, 0.0)
        self.assertEqual(sim._total_rt_s, 0.0)
        self.assertEqual(sim._event_queue, [])
        self.assertEqual(sim._event_index, {})
        self.assertEqual(sim._node_transfers, {})

        self.assertEqual(next(sim._autoinc), 1)
        self.assertEqual(next(sim._autoinc), 2)

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

    def test_init_rejects_non_positive_replica_group_count(self):
        """Test that init rejects non-positive values for
        `replica_group_count`.
        """
        settings = self._get_settings()
        settings['replica_group_count'] = 0

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_negative_mwg_prob_values(self):
        """Test that init rejects invalid values for `mwg_prob`."""
        settings = self._get_settings()
        settings['mwg_prob'] = -0.0000001

        with self.assertRaises(ValueError):
            self._make_instance(**settings)

    def test_init_rejects_too_big_mwg_prob_values(self):
        """Test that init rejects values greater than 1.0 for `mwg_prob`."""
        settings = self._get_settings()
        settings['mwg_prob'] = 1.0000001

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

    def test_new_node_unknown_strategy(self):
        """Test that _new_node raises an error if node strategy is unknown.
        """
        settings = self._get_settings()
        settings['strategy'] = -1  # a non-existing strategy
        sim = self._make_instance(**settings)

        with self.assertRaises(NotImplementedError):
            sim._new_node('node_1', 15000, sim)

    @patch('random.randint', Mock(return_value=6))
    def test_new_node_EFS(self):
        """Test that _new_node currectly creates a new NodeEFS instance.
        """
        from models.node import NodeEFS
        from models.simulation import Strategy

        settings = self._get_settings()
        settings['strategy'] = Strategy.EFS
        sim = self._make_instance(**settings)

        new_node = sim._new_node('node_1', 15000, sim)
        self.assertTrue(isinstance(new_node, NodeEFS))
        self.assertIs(sim._nodes.get('node_1'), new_node)
        self.assertIn('node_1', sim._nodes_mwg)
        self.assertEqual(sim._nodes_mwg['node_1'], 6)
        self.assertTrue(
            random.randint.called_with(1, settings['replica_group_count']))

    @patch('random.randint', Mock(return_value=6))
    def test_new_node_LFU(self):
        """Test that _new_node currectly creates a new NodeLFU instance.
        """
        from models.node import NodeLFU
        from models.simulation import Strategy

        settings = self._get_settings()
        settings['strategy'] = Strategy.LFU
        sim = self._make_instance(**settings)

        new_node = sim._new_node('node_1', 15000, sim)
        self.assertTrue(isinstance(new_node, NodeLFU))
        self.assertIs(sim._nodes.get('node_1'), new_node)
        self.assertIn('node_1', sim._nodes_mwg)
        self.assertEqual(sim._nodes_mwg['node_1'], 6)
        self.assertTrue(
            random.randint.called_with(1, settings['replica_group_count']))

    @patch('random.randint', Mock(return_value=6))
    def test_new_node_LRU(self):
        """Test that _new_node currectly creates a new NodeLRU instance.
        """
        from models.node import NodeLRU
        from models.simulation import Strategy

        settings = self._get_settings()
        settings['strategy'] = Strategy.LRU
        sim = self._make_instance(**settings)

        new_node = sim._new_node('node_1', 15000, sim)
        self.assertTrue(isinstance(new_node, NodeLRU))
        self.assertIs(sim._nodes.get('node_1'), new_node)
        self.assertIn('node_1', sim._nodes_mwg)
        self.assertEqual(sim._nodes_mwg['node_1'], 6)
        self.assertTrue(
            random.randint.called_with(1, settings['replica_group_count']))

    @patch('random.randint', Mock(return_value=6))
    def test_new_node_MFS(self):
        """Test that _new_node currectly creates a new NodeMFS instance.
        """
        from models.node import NodeMFS
        from models.simulation import Strategy

        settings = self._get_settings()
        settings['strategy'] = Strategy.MFS
        sim = self._make_instance(**settings)

        new_node = sim._new_node('node_1', 15000, sim)
        self.assertTrue(isinstance(new_node, NodeMFS))
        self.assertIs(sim._nodes.get('node_1'), new_node)
        self.assertIn('node_1', sim._nodes_mwg)
        self.assertEqual(sim._nodes_mwg['node_1'], 6)
        self.assertTrue(
            random.randint.called_with(1, settings['replica_group_count']))

    def test_now(self):
        """Test that `now` returns internal clock's current time."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        sim._clock._time = 8.7602
        self.assertEqual(sim.now, 8.7602)

    def test_now_readonly(self):
        """Test that `now` property is read-only."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        try:
            sim.now = 10.501
        except AttributeError:
            pass
        else:
            self.fail("Now attribute is not read-only.")

    def test_fsti(self):
        """Test that `fsti` returns correct value."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        sim._fsti = 423
        self.assertEqual(sim.fsti, 423)

    def test_fsti_readonly(self):
        """Test that `fsti` property is read-only."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        try:
            sim.fsti = 99
        except AttributeError:
            pass
        else:
            self.fail("FSTI attribute is not read-only.")

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
        from models.node import NodeEFS

        settings = self._get_settings()
        settings['node_count'] = 20

        # shorten [min_dist, max_dist] interval to detect out-of-range random
        # edge distances with greater probability
        settings['min_dist_km'] = 5
        settings['max_dist_km'] = 20
        sim = self._make_instance(**settings)

        for i in range(1, 21):
            name = 'node_' + str(i)
            sim._nodes[name] = NodeEFS(name, 2000, sim)

        sim._generate_edges()

        for node_1, node_2 in itertools.combinations(sim.nodes, 2):
            self.assertIn(node_1, sim._edges)
            self.assertEqual(len(sim._edges[node_1]), 19)
            self.assertIn(node_2, sim._edges[node_1])

            self.assertIn(node_2, sim._edges)
            self.assertEqual(len(sim._edges[node_2]), 19)
            self.assertIn(node_1, sim._edges[node_2])

            # check distance symmetry
            dist = sim._edges[node_1][node_2]
            self.assertEqual(dist, sim._edges[node_2][node_1])

            # check distance lies withing the specified limits
            self.assertGreaterEqual(dist, settings['min_dist_km'])
            self.assertLessEqual(dist, settings['max_dist_km'])

    def test_generate_replicas(self):
        """Test that _generate_replicas creates replicas as specified by the
        simulation parameters.
        """
        settings = self._get_settings()
        settings['replica_count'] = 100
        settings['replica_group_count'] = 5

        # shorten [min_size, max_size] interval to detect out-of-range random
        # replica sizes distances with greater probability
        settings['replica_min_size'] = 10
        settings['replica_max_size'] = 20
        sim = self._make_instance(**settings)

        sim._generate_replicas()

        self.assertEqual(len(sim.replicas), 100)
        for i in range(1, 101):
            name = 'replica_{:03d}'.format(i)
            self.assertIn(name, sim.replicas)

            replica = sim.replicas[name]
            self.assertEqual(replica.name, name)
            self.assertGreaterEqual(replica.size, settings['replica_min_size'])
            self.assertLessEqual(replica.size, settings['replica_max_size'])

        # check that all replica groups have been initialized and contain the
        # same number of replicas
        for i in range(1, settings['replica_group_count'] + 1):
            self.assertIn(i, sim._replica_groups)
            self.assertEqual(len(sim._replica_groups[i]), 20)

    def test_dijkstra(self):
        """Test that _dijkstra finds the shortest paths between the server
        node and all other nodes.
        """
        settings = self._get_settings()
        settings['node_count'] = 6
        settings['min_dist_km'] = 1
        settings['max_dist_km'] = 20
        sim = self._make_instance(**settings)

        sim._generate_nodes()
        sim._generate_edges()

        # NOTE: example was taken from Wikipedia (Dijsktra's algorithm) with
        # node 1 renamed to 'server' and all other nodes' names reduced by 1:
        # node_<X> --> node_<X-1>
        distances = dict(
            server=dict(node_1=7, node_2=9, node_5=14),
            node_1=dict(server=7, node_2=10, node_3=15),
            node_2=dict(server=9, node_1=10, node_3=11, node_5=2),
            node_3=dict(node_1=15, node_2=11, node_4=6),
            node_4=dict(node_3=6, node_5=9),
            node_5=dict(server=14, node_2=2, node_4=9),
        )

        for node_1, node_2 in itertools.combinations(sim.nodes, 2):
            dist = distances[node_1].get(node_2, float('inf'))
            sim._edges[node_1][node_2] = dist
            sim._edges[node_2][node_1] = dist

        node_info = sim._dijkstra()

        self.assertEqual(node_info['node_3'].dist, 20)
        self.assertEqual(node_info['node_3'].previous, 'node_2')

        self.assertEqual(node_info['node_4'].dist, 20)
        self.assertEqual(node_info['node_4'].previous, 'node_5')

        self.assertEqual(node_info['node_5'].dist, 11)
        self.assertEqual(node_info['node_5'].previous, 'node_2')

        self.assertEqual(node_info['node_2'].dist, 9)
        self.assertEqual(node_info['node_2'].previous, 'server')

        self.assertEqual(node_info['node_1'].dist, 7)
        self.assertEqual(node_info['node_1'].previous, 'server')

        self.assertEqual(node_info['server'].dist, 0)
        self.assertIs(node_info['server'].previous, None)

    def test_event_send_replica_request(self):
        """Test that event_send_replica_request creates a new
        SendReplicaRequest event instance.
        """
        from models.event import SendReplicaRequest

        settings = self._get_settings()
        sim = self._make_instance(**settings)

        source = Mock()
        target = Mock()
        event = sim.event_send_replica_request(source, target, 'replica_XY')

        self.assertTrue(isinstance(event, SendReplicaRequest))
        self.assertIs(event.source, source)
        self.assertIs(event.target, target)
        self.assertEqual(event.replica_name, 'replica_XY')

    def test_event_send_replica(self):
        """Test that event_send_replica_request creates a new SendReplica
        event instance.
        """
        from models.event import SendReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)

        source = Mock()
        target = Mock()
        replica = Mock()
        event = sim.event_send_replica(source, target, replica)

        self.assertTrue(isinstance(event, SendReplica))
        self.assertIs(event.source, source)
        self.assertIs(event.target, target)
        self.assertIs(event.replica, replica)

    def test_initialize(self):
        """Test that initialize resets simulation state."""
        from models.node import NodeEFS

        settings = self._get_settings()
        settings['node_count'] = 3
        settings['replica_count'] = 2
        sim = self._make_instance(**settings)

        sim._clock._time = 12.506
        wrapped_rndseed = Mock(wrap=random.seed)
        sim._total_bw = 27.338
        sim._total_rt_s = 4.7682

        event_mock = Mock()
        event_entry = (13.60, event_mock)
        heapq.heappush(sim._event_queue, event_entry)
        heapq.heappush(sim._event_queue, (18.72, Mock()))

        sim._event_index[event_mock] = event_entry

        sim._node_transfers = {'server': [Mock(), Mock()]}
        next(sim._autoinc)  # advance internal auto-increment counter

        # manually construct the grid
        sim._generate_replicas = Mock()
        sim._generate_nodes = Mock()
        sim._generate_edges = Mock()

        sim._replicas = OrderedDict(
            replica_1=Mock(),
            replica_2=Mock(),
        )
        sim._nodes = OrderedDict(
            server=NodeEFS('server', 1000, sim),
            node_1=NodeEFS('node_1', 1000, sim),
            node_2=NodeEFS('node_2', 1000, sim),
        )
        sim._edges = OrderedDict(
            server=OrderedDict(node_1=10, node_2=20),
            node_1=OrderedDict(server=10, node_2=5),
            node_2=OrderedDict(server=20, node_1=5),
        )

        sim.initialize()

        self.assertEqual(sim._clock.time, 0.0)
        self.assertTrue(wrapped_rndseed.called_with(sim._rnd_seed))
        self.assertEqual(sim._total_bw, 0.0)
        self.assertEqual(sim._total_rt_s, 0.0)
        self.assertEqual(sim._event_queue, [])
        self.assertEqual(sim._event_index, {})

        self.assertEqual(sim._node_transfers, {})
        self.assertEqual(next(sim._autoinc), 1)
        self.assertEqual(next(sim._autoinc), 2)

        self.assertIs(sim._nodes['server']._parent, None)
        self.assertEqual(sim._nodes['node_1']._parent.name, 'server')
        self.assertEqual(sim._nodes['node_2']._parent.name, 'node_1')

    # TODO: test run

    def test_pop_next_event(self):
        """Test that _pop_next_event pops the next event from the event queue.
        """
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event_1 = Mock()
        event_2 = Mock()
        event_3 = Mock()
        event_4 = Mock()
        entry_1 = [10, 1, event_1]
        entry_2 = [20, 2, event_2]
        entry_3 = [30, 3, event_3]
        entry_4 = [40, 4, event_4]

        sim._event_queue.append(entry_4)
        sim._event_queue.append(entry_2)
        sim._event_queue.append(entry_1)
        sim._event_queue.append(entry_3)
        heapq.heapify(sim._event_queue)

        sim._event_index = {
            event_1: entry_1,
            event_2: entry_2,
            event_3: entry_3,
            event_4: entry_4,
        }

        time, event = sim._pop_next_event()

        self.assertEqual(len(sim._event_queue), 3)
        self.assertEqual(time, 10)
        self.assertIs(event, event_1)

    def test_pop_next_event_skip_canceled(self):
        """Test that _pop_next_event skips events marked as canceled."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event_1 = Mock()
        event_2 = Mock()
        event_3 = Mock()
        entry_1 = [10, 1, sim._CANCELED]
        entry_2 = [20, 2, event_2]
        entry_3 = [30, 3, event_3]

        sim._event_queue.append(entry_1)
        sim._event_queue.append(entry_2)
        sim._event_queue.append(entry_3)
        heapq.heapify(sim._event_queue)

        sim._event_index = {
            event_1: entry_1,
            event_2: entry_2,
            event_3: entry_3,
        }

        time, event = sim._pop_next_event()

        self.assertEqual(len(sim._event_queue), 1)
        self.assertEqual(time, 20)
        self.assertIs(event, event_2)

    def test_pop_next_event_empty_queue(self):
        """Test that _pop_next_event raises an error if event queue is empty.
        """
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        with self.assertRaises(KeyError):
            sim._pop_next_event()

    def test_cancel_event(self):
        """Test that existing event is correctly canceled."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event_1 = Mock()
        event_2 = Mock()
        entry_1 = [10, event_1]
        entry_2 = [20, event_2]

        sim._event_queue.append(entry_1)
        sim._event_queue.append(entry_2)
        heapq.heapify(sim._event_queue)

        sim._event_index = {
            event_1: entry_1,
            event_2: entry_2,
        }

        sim._cancel_event(event_1)

        self.assertEqual(len(sim._event_queue), 2)  # entry_1 is still there!
        self.assertIs(sim._event_queue[0][0], 10)   # time unchanged
        self.assertIs(sim._event_queue[0][1], sim._CANCELED)

    def test_cancel_event_nonexistent(self):
        """Test that an error is raised if canceling a nonexistent event."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event_1 = Mock()
        event_2 = Mock()
        entry_1 = [10, event_1]
        entry_2 = [20, event_2]

        sim._event_queue.append(entry_1)
        sim._event_queue.append(entry_2)
        heapq.heapify(sim._event_queue)

        sim._event_index = {
            event_1: entry_1,
            event_2: entry_2,
        }

        with self.assertRaises(KeyError):
            sim._cancel_event(Mock(name='nonexistent event'))

    def test_schedule_event(self):
        """Test that _schedule_event adds an event to the event queue."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event_1 = Mock()
        event_2 = Mock()
        event_4 = Mock()
        entry_1 = [10, 1, event_1]
        entry_2 = [20, 2, event_2]
        entry_4 = [40, 3, event_4]

        sim._event_queue.append(entry_1)
        sim._event_queue.append(entry_2)
        sim._event_queue.append(entry_4)
        heapq.heapify(sim._event_queue)
        sim._autoinc = itertools.count(start=4)

        sim._event_index = {
            event_1: entry_1,
            event_2: entry_2,
            event_4: entry_4,
        }

        event_3 = Mock()
        sim._schedule_event(event_3, 30)

        self.assertEqual(len(sim._event_queue), 4)
        self.assertIn([30, 4, event_3], sim._event_queue)

    def test_schedule_event_in_the_past(self):
        """Test that _schedule_event rejects events which would occur in past
        time.
        """
        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 8.00

        with self.assertRaises(ValueError):
            sim._schedule_event(Mock(), 7.999)

    def test_schedule_event_reschedule_existing(self):
        """Test that _schedule_event reschedules an event if the latter already
        exists.
        """
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event_1 = Mock()
        event_2 = Mock()
        event_3 = Mock()
        entry_1 = [10, 1, event_1]
        entry_2 = [20, 2, event_2]
        entry_3 = [30, 3, event_3]

        sim._event_queue.append(entry_1)
        sim._event_queue.append(entry_2)
        sim._event_queue.append(entry_3)
        heapq.heapify(sim._event_queue)
        sim._autoinc = itertools.count(start=4)

        sim._event_index = {
            event_1: entry_1,
            event_2: entry_2,
            event_3: entry_3,
        }

        sim._schedule_event(event_2, 42)

        # original event entry is marked as canceled and a new rescheduled
        # event entry is added to event queue
        self.assertEqual(len(sim._event_queue), 4)
        self.assertIn([20, 2, sim._CANCELED], sim._event_queue)
        self.assertIn([42, 4, event_2], sim._event_queue)

    def test_process_event_type_unknown(self):
        """Test that _process_event raises an error on unknown event types."""
        settings = self._get_settings()
        sim = self._make_instance(**settings)

        with self.assertRaises(TypeError):
            sim._process_event(object())

    def test_process_event_type_receive_replica_request(self):
        """Test that _process_event invokes correct handler for receive
        replica request events.
        """
        from models.event import ReceiveReplicaRequest

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        fake_handler = Mock()
        sim._process_receive_replica_request = fake_handler

        event = ReceiveReplicaRequest(Mock(), Mock(), Mock())
        sim._process_event(event)

        self.assertEqual(fake_handler.call_count, 1)
        try:
            fake_handler.assert_called_with(event)
        except AssertionError:
            self.fail("Event handler called with incorrect parameter.")

    def test_process_event_type_send_replica_request(self):
        """Test that _process_event invokes correct handler for send replica
        request events.
        """
        from models.event import SendReplicaRequest

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        fake_handler = Mock()
        sim._process_send_replica_request = fake_handler

        event = SendReplicaRequest(Mock(), Mock(), Mock())
        sim._process_event(event)

        self.assertEqual(fake_handler.call_count, 1)
        try:
            fake_handler.assert_called_with(event)
        except AssertionError:
            self.fail("Event handler called with incorrect parameter.")

    def test_process_event_type_send_replica(self):
        """Test that _process_event invokes correct handler for send replica
        events.
        """
        from models.event import SendReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        fake_handler = Mock()
        sim._process_send_replica = fake_handler

        event = SendReplica(Mock(), Mock(), Mock())
        sim._process_event(event)

        self.assertEqual(fake_handler.call_count, 1)
        try:
            fake_handler.assert_called_with(event)
        except AssertionError:
            self.fail("Event handler called with incorrect parameter.")

    def test_process_event_type_receive_replica(self):
        """Test that _process_event invokes correct handler for receive
        replica events.
        """
        from models.event import ReceiveReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        fake_handler = Mock()
        sim._process_receive_replica = fake_handler

        event = ReceiveReplica(Mock(), Mock(), Mock())
        sim._process_event(event)

        self.assertEqual(fake_handler.call_count, 1)
        try:
            fake_handler.assert_called_with(event)
        except AssertionError:
            self.fail("Event handler called with incorrect parameter.")

    def test_process_receive_replica_request_replica_not_found(self):
        """Test that _process_receive_replica_request correctly processes
        ReceiveReplicaRequest events when target node does not have the
        requested replica.
        """
        from models.node import NodeEFS
        from models.event import ReceiveReplicaRequest
        from models.event import SendReplicaRequest

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 2.55

        target = NodeEFS('target', 10000, sim)
        target._parent = Mock()
        target._parent.name = 'server'

        source = Mock()
        source._parent = target
        event = ReceiveReplicaRequest(source, target, 'replica_X')
        event._generators = [Mock(), Mock()]

        sim._process_receive_replica_request(event)

        # ReceiveReplicaRequest should have resulted in a SendReplicaRequest
        # (target node requests replica from its parent) with no time delay
        self.assertEqual(len(sim._event_queue), 1)
        next_event_time, entry_id, next_event = sim._event_queue[0]
        self.assertEqual(next_event_time, 2.55)
        self.assertTrue(isinstance(next_event, SendReplicaRequest))
        self.assertEqual(next_event.source.name, 'target')
        self.assertEqual(next_event.target.name, 'server')
        self.assertEqual(next_event.replica_name, 'replica_X')

        # a new generator should have been added to the end of the list
        self.assertEqual(len(next_event._generators), 3)
        self.assertEqual(next_event._generators[:2], event._generators)
        self.assertTrue(isgenerator(next_event._generators[2]))

    def test_process_receive_replica_request_replica_found(self):
        """Test that _process_receive_replica_request correctly processes
        ReceiveReplicaRequest events when target node has a copy of the
        requested replica.
        """
        from models.node import NodeEFS
        from models.event import ReceiveReplicaRequest
        from models.event import SendReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 2.55

        replica = Mock()
        replica.name = 'replica_X'

        target = NodeEFS('target', 10000, sim)
        target._replicas[replica.name] = replica
        target._replica_stats[replica.name] = Mock()

        source = NodeEFS('source', 10000, sim)
        source._parent = target
        event = ReceiveReplicaRequest(source, target, 'replica_X')
        event._generators = [Mock(), Mock()]

        sim._process_receive_replica_request(event)

        # ReceiveReplicaRequest should have resulted in a SendReplica with no
        # delay (target node had the replica and sent it back)
        self.assertEqual(len(sim._event_queue), 1)
        next_event_time, entry_id, next_event = sim._event_queue[0]
        self.assertEqual(next_event_time, 2.55)
        self.assertTrue(isinstance(next_event, SendReplica))
        self.assertEqual(next_event.source.name, 'target')
        self.assertEqual(next_event.target.name, 'source')
        self.assertIs(next_event.replica, replica)
        self.assertEqual(next_event._generators, event._generators)

    @patch('builtins.next', lambda g: object())
    def test_process_receive_replica_request_unknown_event_yielded(self):
        """Test that _process_receive_replica_request raises an error when
        processing ReceiveReplicaRequest resulting in an event of an unknown
        type.

        For this to happen we patch the next() function so that it returns a
        plain object instance instead of what the Node.request_replica()
        generator would have returned.
        """
        from models.event import ReceiveReplicaRequest

        settings = self._get_settings()
        sim = self._make_instance(**settings)

        event = ReceiveReplicaRequest(Mock(), Mock(), 'replica_X')

        with self.assertRaises(TypeError):
            sim._process_receive_replica_request(event)

    def test_process_send_replica_request(self):
        """Test that _process_send_replica_request correctly processes
        SendReplicaRequest events.
        """
        from models.node import NodeEFS
        from models.event import ReceiveReplicaRequest
        from models.event import SendReplicaRequest

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 4.22
        sim._total_rt_s = 0.72

        # NOTE: signal propagation speed (as defined in settings) is 7000 km/s
        sim._edges['source'] = OrderedDict(target=700)

        target = Mock()
        target.name = 'target'
        source = NodeEFS('source', 10000, sim)
        source._parent = target
        event = SendReplicaRequest(source, target, 'replica_X')
        event._generators = [Mock(), Mock()]

        sim._process_send_replica_request(event)

        # SendReplicaRequest should have resulted in a ReceiveReplicaRequest
        # event on the target node after some propagation speed latency
        self.assertEqual(len(sim._event_queue), 1)
        next_event_time, entry_id, next_event = sim._event_queue[0]
        self.assertAlmostEqual(next_event_time, 4.32)
        self.assertTrue(isinstance(next_event, ReceiveReplicaRequest))
        self.assertEqual(next_event.source.name, 'source')
        self.assertEqual(next_event.target.name, 'target')
        self.assertEqual(next_event.replica_name, 'replica_X')
        self.assertEqual(next_event._generators, event._generators)

        # total response time statistics should have to be updates as well
        self.assertAlmostEqual(sim._total_rt_s, 0.82)

    def test_process_send_replica_no_other_transfers(self):
        """Test that _process_send_replica correctly processes SendReplica
        events when there are no other concurrent replica transfers.
        """
        from models.event import ReceiveReplica
        from models.event import SendReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 22.57
        sim._total_rt_s = 7.08
        sim._total_bw = 110.25

        source = Mock()
        source.name = 'server'
        target = Mock()
        replica = Mock(size=80)  # NOTE: network bandwidth is 20 Mb/s
        event = SendReplica(source, target, replica)
        event._generators = [Mock(), Mock()]

        sim._process_send_replica(event)

        # SendReplica should have resulted in a ReceiveReplica event on the
        # target node after the time needed to send replica over the network
        self.assertEqual(len(sim._event_queue), 1)
        next_event_time, entry_id, next_event = sim._event_queue[0]
        self.assertAlmostEqual(next_event_time, 26.57)
        self.assertTrue(isinstance(next_event, ReceiveReplica))
        self.assertIs(next_event.source, source)
        self.assertIs(next_event.target, target)
        self.assertIs(next_event.replica, replica)
        self.assertEqual(next_event._generators, event._generators)

        # check that node's transfers have been updated
        transfers = sim._node_transfers[source.name]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0][0], next_event_time)
        self.assertIs(transfers[0][2], next_event)

        # simulation statistics should have been updated as well
        self.assertAlmostEqual(sim._total_rt_s, 11.08)
        self.assertAlmostEqual(sim._total_bw, 190.25)

    def test_process_send_replica_other_longer_transfers(self):
        """Test that _process_send_replica correctly processes SendReplica
        events when there are other replica transfers on the node, which will
        last longer than the new transfer.
        """
        from models.event import ReceiveReplica
        from models.event import SendReplica

        # Scenario: node is currently sending replicas R1, R2, R3 which are
        # supposed to arrive at now +6, +14 and +17 respectively.
        # Sizes of not-yet-transfered replica parts are 20, 60 and 90 Mbit.
        # The current total BW consumption stat is currently at 170 Mbit,
        # while the total delay stat is 37 seconds.
        #
        # Node then starts sending replica R4 (80 Mbit) which delays
        # the arrivals of all replicas and affects the stats.

        settings = self._get_settings()
        settings['network_bw_mbps'] = 10
        sim = self._make_instance(**settings)
        sim._clock._time = 1000.00
        sim._total_rt_s = 37.00
        sim._total_bw = 170.00

        ev_receive_r1 = Mock(name='receive_R1')
        ev_receive_r2 = Mock(name='receive_R2')
        ev_receive_r3 = Mock(name='receive_R3')

        transfers = sim._node_transfers['server']
        transfers.append([1006.0, 1, ev_receive_r1])
        transfers.append([1014.0, 2, ev_receive_r2])
        transfers.append([1017.0, 3, ev_receive_r3])
        heapq.heapify(transfers)
        sim._autoinc = itertools.count(start=4)

        entry_1 = [1006.0, ev_receive_r1]
        entry_2 = [1014.0, ev_receive_r2]
        entry_3 = [1017.0, ev_receive_r3]
        heapq.heappush(sim._event_queue, entry_1)
        heapq.heappush(sim._event_queue, entry_2)
        heapq.heappush(sim._event_queue, entry_3)
        sim._event_index = {
            ev_receive_r1: entry_1,
            ev_receive_r2: entry_2,
            ev_receive_r3: entry_3,
        }

        source = Mock()
        source.name = 'server'
        target = Mock()
        replica_4 = Mock(name='replica_4', size=80)
        event = SendReplica(source, target, replica_4)
        event._generators = [Mock(), Mock()]

        sim._process_send_replica(event)

        # SendReplica should have resulted in a ReceiveReplica event on the
        # target node and with current active transfers delayed a bit due to
        # bandwidth sharing
        for key in sim._event_index.keys():
            if isinstance(key, ReceiveReplica):
                ev_receive_r4 = key
                break
        else:
            self.fail("ReceiveReplica event not present in event index.")

        # 7 == 3 canceled events + 3 rescheduled events + 1 new event
        self.assertEqual(len(sim._event_queue), 7)

        entry = sim._event_index[ev_receive_r4]
        self.assertAlmostEqual(entry[0], 1024.00)  # event time
        self.assertTrue(isinstance(ev_receive_r4, ReceiveReplica))
        self.assertIs(ev_receive_r4.source, source)
        self.assertIs(ev_receive_r4.target, target)
        self.assertIs(ev_receive_r4.replica, replica_4)
        self.assertEqual(ev_receive_r4._generators, event._generators)

        # now check if other ReceiveReplica events have been correctly delayed
        self.assertAlmostEqual(sim._event_index[ev_receive_r1][0], 1008.00)
        self.assertAlmostEqual(sim._event_index[ev_receive_r2][0], 1020.00)
        self.assertAlmostEqual(sim._event_index[ev_receive_r3][0], 1025.00)

        # check that node's transfers have been updated
        transfers = sim._node_transfers['server']
        self.assertEqual(len(transfers), 4)

        for entry in transfers:
            if entry[-1] is ev_receive_r4:
                break
        else:
            self.fail("New replica transfer not found in transfer list.")

        self.assertAlmostEqual(entry[0], 1024.00)

        # simulation statistics should have been updated as well
        self.assertAlmostEqual(sim._total_rt_s, 77.00)
        self.assertAlmostEqual(sim._total_bw, 250.00)

    def test_process_send_replica_all_other_transfers_longer(self):
        """Test that _process_send_replica correctly processes SendReplica
        events when the transfer of new replica would finish before other
        currently active transfers.

        The main purpose of this test is to cover some cases when the current
        node's transer queue is not correctly updated (some existing transfers
        disappear).
        """
        from models.event import ReceiveReplica
        from models.event import SendReplica

        # Scenario: node is currently sending replica R1 and R2 which are
        # supposed to arrive at now +50 and +80 respectively.
        # Sizes of not-yet-transfered replica parts are 250 and 450 Mbit.
        # The current total BW consumption stat is currently at 80 Mbit,
        # while the total delay stat is 12 seconds.
        #
        # Node then starts sending replica R3 (100 Mbit) which delays
        # the arrivals of all replicas and affects the stats.

        settings = self._get_settings()
        settings['network_bw_mbps'] = 10
        sim = self._make_instance(**settings)
        sim._clock._time = 1000.0
        sim._total_rt_s = 12.00
        sim._total_bw = 80.00

        ev_receive_r1 = Mock(name='receive_R1')
        ev_receive_r2 = Mock(name='receive_R2')

        transfers = sim._node_transfers['server']
        transfers.append([1050.0, 1, ev_receive_r1])
        transfers.append([1070.0, 2, ev_receive_r2])
        heapq.heapify(transfers)
        sim._autoinc = itertools.count(start=3)

        entry_1 = [1050.0, ev_receive_r1]
        entry_2 = [1070.0, ev_receive_r2]
        heapq.heappush(sim._event_queue, entry_1)
        heapq.heappush(sim._event_queue, entry_2)
        sim._event_index = {
            ev_receive_r1: entry_1,
            ev_receive_r2: entry_2,
        }

        source = Mock()
        source.name = 'server'
        target = Mock()
        replica_3 = Mock(name='replica_3', size=100)
        event = SendReplica(source, target, replica_3)
        event._generators = [Mock(), Mock()]

        sim._process_send_replica(event)

        # SendReplica should have resulted in a ReceiveReplica event on the
        # target node and with current active transfers delayed a bit due to
        # bandwidth sharing
        for key in sim._event_index.keys():
            if isinstance(key, ReceiveReplica):
                ev_receive_r3 = key
                break
        else:
            self.fail("ReceiveReplica event not present in event index.")

        # 5 == 2 canceled events + 2 rescheduled events + 1 new event
        self.assertEqual(len(sim._event_queue), 5)

        return

        entry = sim._event_index[ev_receive_r3]
        self.assertAlmostEqual(entry[0], 1030.00)  # event time
        self.assertTrue(isinstance(ev_receive_r3, ReceiveReplica))
        self.assertIs(ev_receive_r3.source, source)
        self.assertIs(ev_receive_r3.target, target)
        self.assertIs(ev_receive_r3.replica, replica_3)
        self.assertEqual(ev_receive_r3._generators, event._generators)

        # now check if other ReceiveReplica events have been correctly delayed
        self.assertAlmostEqual(sim._event_index[ev_receive_r1][0], 1060.00)
        self.assertAlmostEqual(sim._event_index[ev_receive_r2][0], 1080.00)

        # check that node's transfers have been updated
        transfers = sim._node_transfers['server']
        self.assertEqual(len(transfers), 3)

        for entry in transfers:
            if entry[-1] is ev_receive_r3:
                break
        else:
            self.fail("New replica transfer not found in transfer list.")

        self.assertAlmostEqual(entry[0], 1030.00)

        # simulation statistics should have been updated as well
        self.assertAlmostEqual(sim._total_rt_s, 62.00)  # added 2*10 + 30
        self.assertAlmostEqual(sim._total_bw, 180.00)

    def test_process_receive_replica_with_subtarget(self):
        """Test that _process_receive_replica correctly processes
        ReceiveReplica events when receiving node (target) has another
        sub-target to send replica to.
        """
        from models.event import ReceiveReplica
        from models.event import SendReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 8.11

        source = Mock()
        source.name = 'source_node'
        target = Mock()
        target_child = Mock()
        replica = Mock()
        event = ReceiveReplica(source, target, replica)

        # set some node transfers
        sim._node_transfers[source.name] = [
            [8.11, 100, event],
            [12.01, 101, Mock()],
            [17.20, 102, Mock()],
        ]

        def g():
            yield
            yield SendReplica(target, target_child, replica)
        gen = g()
        next(gen)

        another_gen = Mock()
        event._generators = [another_gen, gen]

        sim._process_receive_replica(event)

        # ReceiveReplica should have resulted in a SendReplica event (target
        # node sends replica to its own requester)
        self.assertEqual(len(sim._event_queue), 1)
        next_event_time, entry_id, next_event = sim._event_queue[0]
        self.assertEqual(next_event_time, 8.11)  # immediately (no delay)
        self.assertTrue(isinstance(next_event, SendReplica))
        self.assertIs(next_event.source, target)
        self.assertIs(next_event.target, target_child)
        self.assertIs(next_event.replica, replica)
        self.assertEqual(next_event._generators, [another_gen])

        # check node's replica transfer list
        transfers = sim._node_transfers[source.name]
        self.assertEqual(len(transfers), 2)
        self.assertNotIn([8.11, 100, event], transfers)

    def test_process_receive_replica_without_subtarget(self):
        """Test that _process_receive_replica correctly processes
        ReceiveReplica events when receiving node (target) does not have any
        sub-target to send replica to.
        """
        from models.event import ReceiveReplica
        from models.event import SendReplica

        settings = self._get_settings()
        sim = self._make_instance(**settings)
        sim._clock._time = 8.11

        source = Mock()
        target = Mock()
        replica = Mock()
        event = ReceiveReplica(source, target, replica)

        # set some node transfers
        sim._node_transfers[source.name] = [
            [8.11, 100, event],
            [12.01, 101, Mock()],
            [17.20, 102, Mock()],
        ]

        def g():
            yield
            yield SendReplica(target, None, replica)
        gen = g()
        next(gen)

        event._generators = [gen]

        sim._process_receive_replica(event)

        # after processing ReceiveReplica event, no more events should have
        # resulted from that (because there is no sub-target node to send
        # replica to)
        self.assertEqual(len(sim._event_queue), 0)

        # check node's replica transfer list
        transfers = sim._node_transfers[source.name]
        self.assertEqual(len(transfers), 2)
        self.assertNotIn([8.11, 100, event], transfers)


class TestEventFactory(unittest.TestCase):
    """Tests for :py:class:`models.simulation._EventFactory`."""

    def _get_target_class(self):
        from models.simulation import _EventFactory
        return _EventFactory

    def _make_instance(self, *args, **kw):
        return self._get_target_class()(*args, **kw)

    def test_init(self):
        """Test that new instances are correctly initialized."""

        node_keys = ['server', 'node_1', 'node_2']
        group_keys = [1, 2, 3]

        simulation = Mock()
        simulation.nodes.keys.return_value = node_keys
        simulation._replica_groups.keys.return_value = group_keys

        instance = self._make_instance(simulation)

        self.assertIs(instance._sim, simulation)
        self.assertEqual(instance._node_names, node_keys[1:])  # server omitted
        self.assertEqual(instance._replica_groups, group_keys)

    @patch('random.random')
    @patch('random.randint')
    @patch('random.choice')
    def test_get_random(self, choice, randint, random):
        """Test that get_random behaves as expected."""
        from models.event import ReceiveReplicaRequest

        nodes = OrderedDict(
            server=Mock(name='server'),
            node_1=Mock(name='node_1'),
            node_2=Mock(name='node_2'),
            node_3=Mock(name='node_3'),
            node_4=Mock(name='node_4'),
        )
        for key, item in nodes.items():
            item.name = key

        replica = Mock()
        replica.name = 'replica_1'

        replica_groups = {
            1: Mock(),
            2: Mock(),
        }

        nodes_mwg = dict(
            server=1,  # XXX: irrelevant?
            node_1=1,
            node_2=2,
            node_3=2,
            node_4=1,
        )

        simulation = Mock(name='simulation')
        simulation.nodes = nodes
        simulation._replica_groups = replica_groups
        simulation._nodes_mwg = nodes_mwg
        simulation._mwg_prob = 0.500
        simulation.now = 2.7

        choice.side_effect = ['node_3', 2, 1, replica]
        randint.return_value = 61
        random.return_value = 0.501

        event_factory = self._make_instance(simulation)
        ret_val = event_factory.get_random()

        self.assertTrue(len(ret_val), 2)
        self.assertEqual(ret_val[0], 61.0)  # event time

        event = ret_val[1]
        self.assertTrue(isinstance(event, ReceiveReplicaRequest))
        self.assertIs(event.source, None)
        self.assertIs(event.target, nodes['node_3'])
        self.assertEqual(event.replica_name, 'replica_1')

        # TODO: test for MWG probability? select mwg, select non-mwg
        # ... refactor setup into some helper method
