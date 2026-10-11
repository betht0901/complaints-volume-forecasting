"""Standard-library tests; run with python -m unittest discover -s tests -v."""
import unittest

import numpy as np
import pandas as pd

from forecast import HORIZON, make_features, ridge_forecast, score, seasonal_naive


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2023-01-01', periods=400, freq='D')
        self.y = pd.Series(30 + self.dates.dayofweek.to_numpy() * 2, index=self.dates, dtype=float)
        self.future = pd.date_range(self.dates[-1] + pd.Timedelta(days=1), periods=HORIZON)

    def test_features_depend_only_on_dates(self):
        features = make_features(self.future, self.dates[0])
        self.assertFalse(features.isna().any().any())
        self.assertEqual(len(features), HORIZON)
        self.assertNotIn('centered_7d_mean', features.columns)

    def test_forecasts_are_finite_and_horizon_length(self):
        for estimator in (seasonal_naive, ridge_forecast):
            with self.subTest(estimator=estimator.__name__):
                predictions = estimator(self.y, self.future)
                self.assertEqual(len(predictions), HORIZON)
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())

    def test_baseline_handles_missing_recent_days(self):
        data = self.y.copy()
        data.iloc[-7:] = np.nan
        predictions = seasonal_naive(data, self.future)
        self.assertTrue(np.isfinite(predictions).all())

    def test_score_ignores_missing_truth(self):
        metrics = score(np.array([10., np.nan, 20.]), np.array([12., 99., 17.]))
        self.assertEqual(metrics['actual_days'], 2)
        self.assertEqual(metrics['MAE'], 2.5)


if __name__ == '__main__':
    unittest.main()
