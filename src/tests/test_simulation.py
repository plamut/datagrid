"""Tests for `simulation` module."""

import unittest


# XXX: perhaps combine tests into a test suite (i.e. for testing the
# simulation module)
# might not be that necessary if autodiscover works just fine ...

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
        self.assertEqual(clock.time, 0)

    def test_time_readonly(self):
        """Test that instance's `time` property is read-only."""
        clock = self._make_instance()
        try:
            clock.time = 123
        except AttributeError:
            pass
        else:
            self.fail("Time attribute is not read-only.")

    def test_reset(self):
        """Test that `reset` method resets time to zero (0)."""
        clock = self._make_instance()
        clock._time = 50  # cheating for easier testing

        self.assertEqual(clock.time, 50)
        clock.reset()
        self.assertEqual(clock.time, 0)

    def test_tick_default_step(self):
        """Test that `tick` method increases time by 1 unit by default."""
        clock = self._make_instance()
        self.assertEqual(clock.time, 0)

        clock.tick()
        self.assertEqual(clock.time, 1)
        clock.tick()
        self.assertEqual(clock.time, 2)
        clock.tick()
        self.assertEqual(clock.time, 3)

    def test_tick_non_default_step(self):
        """Test that `tick` method correctly increases time by given step."""
        clock = self._make_instance()
        self.assertEqual(clock.time, 0)

        clock.tick(step=4)
        self.assertEqual(clock.time, 4)
        clock.tick(step=0)
        self.assertEqual(clock.time, 4)

    def test_tick_checks_step_to_be_positive(self):
        """Test that `tick` method rejects negative steps."""
        clock = self._make_instance()
        try:
            clock.tick(step=-1)
        except ValueError:
            pass
        except:
            self.fail("Invalid exception type raised, ValueError expected.")
        else:
            self.fail("Tick (incorrectly) accepts negative steps.")
