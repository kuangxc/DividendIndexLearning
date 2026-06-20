#!/usr/bin/env python3
"""Tests for chart axis formatting."""

import importlib.util
import unittest
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "update_data.py"


def load_update_data_module():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    spec = importlib.util.spec_from_file_location("update_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDateAxisFormatting(unittest.TestCase):
    def setUp(self):
        self.update_data = load_update_data_module()

    def test_short_range_labels_include_days(self):
        dates = [
            datetime(2026, 5, 22),
            datetime(2026, 5, 29),
            datetime(2026, 6, 5),
            datetime(2026, 6, 12),
            datetime(2026, 6, 18),
        ]

        fig, ax = plt.subplots(figsize=(14, 6))
        try:
            ax.plot(dates, [1, 2, 3, 4, 5])
            self.update_data.format_date_axis(ax, dates)
            fig.canvas.draw()

            labels = [tick.get_text() for tick in ax.get_xticklabels() if tick.get_text()]
            self.assertEqual(labels, list(dict.fromkeys(labels)))
            self.assertTrue(any(label.startswith("2026-05-") for label in labels))
            self.assertTrue(any(label.startswith("2026-06-") for label in labels))
        finally:
            plt.close(fig)

    def test_full_history_range_is_preserved(self):
        full_history_dates = [
            datetime(2020, 1, 1),
            datetime(2026, 6, 18),
        ]
        recent_metric_dates = [
            datetime(2026, 5, 22),
            datetime(2026, 6, 18),
        ]

        fig, ax = plt.subplots(figsize=(14, 6))
        try:
            ax.plot(recent_metric_dates, [4.3, 5.1])
            self.update_data.format_date_axis(ax, full_history_dates)
            fig.canvas.draw()

            min_xlim, max_xlim = ax.get_xlim()
            min_date = mdates.num2date(min_xlim).replace(tzinfo=None)
            max_date = mdates.num2date(max_xlim).replace(tzinfo=None)
            labels = [tick.get_text() for tick in ax.get_xticklabels() if tick.get_text()]

            self.assertLessEqual(min_date.date(), datetime(2020, 1, 1).date())
            self.assertGreaterEqual(max_date.date(), datetime(2026, 6, 18).date())
            self.assertIn("2020-01", labels)
            self.assertIn("2026-01", labels)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
