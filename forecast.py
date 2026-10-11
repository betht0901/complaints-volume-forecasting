"""Reproducible 90-day daily complaint forecasting assessment.

Run: python forecast.py --input data/assessment.xlsx --output outputs
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZON = 90
SEED = 42


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name='daily records')
    raw['date'] = pd.to_datetime(raw['date'], errors='raise')
    if raw.date.isna().any() or raw.date.duplicated().any():
        raise ValueError('Dates must be valid and unique.')
    if (raw.complaints.dropna() < 0).any():
        raise ValueError('Complaint counts must be non-negative.')
    raw = raw.sort_values('date').set_index('date')
    # Keep NaN on days with no observation: never treat an absent day as zero complaints.
    observed = raw[['complaints']].asfreq('D')
    return observed


def make_features(dates: pd.DatetimeIndex, anchor: pd.Timestamp) -> pd.DataFrame:
    """Strictly future-known date features; excludes centred rolling and operational data."""
    age = np.asarray((dates - anchor).days, dtype=float)
    d = pd.DataFrame(index=dates)
    d['trend_years'] = age / 365.25
    d['trend_sq'] = (age / 365.25) ** 2
    for day in range(1, 7):
        d[f'weekday_{day}'] = (dates.dayofweek == day).astype(float)
    for period, harmonics, label in [(365.25, 2, 'year'), (30.4375, 1, 'month')]:
        for n in range(1, harmonics + 1):
            d[f'{label}_sin_{n}'] = np.sin(2 * np.pi * n * age / period)
            d[f'{label}_cos_{n}'] = np.cos(2 * np.pi * n * age / period)
    return d


def seasonal_naive(history: pd.Series, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Repeat weekday-specific mean from the most recent 28 observed-calendar days."""
    recent = history.tail(28)
    global_mean = float(history.dropna().tail(28).mean())
    means = recent.groupby(recent.index.dayofweek).mean()
    return np.array([means.get(d.dayofweek, np.nan) for d in future_dates], dtype=float) if means.notna().all() else np.array([float(means.get(d.dayofweek)) if pd.notna(means.get(d.dayofweek)) else global_mean for d in future_dates])


def ridge_forecast(history: pd.Series, dates: pd.DatetimeIndex, alpha: float = 250.0) -> np.ndarray:
    anchor = history.index.min()
    features = make_features(history.index, anchor)
    known = history.notna()
    train_x = features.loc[known]
    train_y = history.loc[known].to_numpy(float)
    # Regularise the seasonal and trend coefficients. The intercept is unpenalised.
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y)
    return np.maximum(0, model.predict(make_features(dates, anchor)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = np.isfinite(y_true)
    y, p = y_true[valid], y_pred[valid]
    return {'MAE': round(float(mean_absolute_error(y, p)), 3),
            'RMSE': round(float(np.sqrt(mean_squared_error(y, p))), 3),
            'WAPE_pct': round(float(100 * np.abs(y - p).sum() / np.abs(y).sum()), 2),
            'actual_days': int(len(y))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/assessment.xlsx'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.input)
    target = data.complaints
    # Expanding-window evaluation: 3 non-overlapping 90-day blocks, strictly earlier training.
    windows = []
    for remaining in [3, 2, 1]:
        split = len(target) - HORIZON * remaining
        history = target.iloc[:split]
        future = target.index[split:split + HORIZON]
        truth = target.loc[future].to_numpy(float)
        for name, predict in [('weekday_28d', seasonal_naive), ('calendar_ridge', ridge_forecast)]:
            preds = predict(history, future)
            result = {'model': name, 'train_end': str(history.index[-1].date()),
                      'test_start': str(future[0].date()), 'test_end': str(future[-1].date()),
                      **score(truth, preds)}
            windows.append(result)
    validation = pd.DataFrame(windows)
    validation.to_csv(args.output / 'backtest_metrics.csv', index=False)
    leaderboard = validation.groupby('model', as_index=False)[['MAE', 'RMSE', 'WAPE_pct']].mean()
    leaderboard = leaderboard.sort_values(['MAE', 'RMSE'])
    leaderboard.to_csv(args.output / 'model_comparison.csv', index=False)
    selected = str(leaderboard.iloc[0]['model'])
    dates = pd.date_range(target.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq='D')
    forecast = seasonal_naive(target, dates) if selected == 'weekday_28d' else ridge_forecast(target, dates)
    output = pd.DataFrame({'date': dates, 'predicted_complaints': np.maximum(0, np.rint(forecast)).astype(int)})
    output.to_csv(args.output / 'forecast_90_days.csv', index=False, date_format='%Y-%m-%d')
    missing_in_sheet = int(target.isna().sum())
    report = {'first_observation_date': str(target.index.min().date()),
              'last_observation_date': str(target.index.max().date()),
              'calendar_days': int(len(target)), 'missing_target_calendar_days': missing_in_sheet,
              'selected_model': selected,
              'forecast_start': str(dates[0].date()), 'forecast_end': str(dates[-1].date()),
              'expected_90_day_total': int(output.predicted_complaints.sum()),
              'validation': windows, 'comparison_mean': leaderboard.to_dict(orient='records')}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(12, 5))
    recent = target.tail(180)
    ax.plot(recent.index, recent.to_numpy(), color='grey', alpha=.65, label='Observed complaints')
    ax.plot(dates, forecast, color='tab:blue', label=f'90-day forecast ({selected})')
    ax.set(title='Daily complaint volumes — 90-day forecast', ylabel='Complaints', xlabel='Date')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'forecast_chart.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'selected_model': selected, 'comparison': leaderboard.to_dict(orient='records'),
                      'forecast_start': report['forecast_start'], 'forecast_end': report['forecast_end'],
                      '90_day_volume': report['expected_90_day_total']}, indent=2))


if __name__ == '__main__':
    main()
