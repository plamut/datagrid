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


# TODO: _Clock
