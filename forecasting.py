"""
FitSmart forecasting module.
Menggunakan Statsmodels ARIMA untuk memprediksi berat badan.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError as exc:
    raise ImportError("statsmodels is required. Install it with `pip install statsmodels`.") from exc

def forecast_weight(
    df: pd.DataFrame, date_col: str = "tanggal", weight_col: str = "berat badan",
    periods: int = 30, freq: str = "D", min_observations: int = 10, absolute_min_weight: float = 35.0
) -> pd.DataFrame:
    
    data = df[[date_col, weight_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, weight_col]).sort_values(by=date_col)
    data = data.drop_duplicates(subset=[date_col], keep="last").set_index(date_col)
    data.index = pd.DatetimeIndex(data.index)
    
    if data.index.duplicated().any():
        data = data[~data.index.duplicated(keep="last")]

    data = data.asfreq(freq)
    data[weight_col] = data[weight_col].astype(float).interpolate(method="time").ffill().bfill()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*frequency.*")
        try:
            model = ARIMA(data[weight_col], order=(1, 1, 1))
            fitted = model.fit()
        except Exception:
            model = ARIMA(data[weight_col], order=(0, 1, 1))
            fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=periods)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=periods, freq=data.index.freq or freq)

    conf_int = forecast_result.conf_int(alpha=0.05)
    
    forecast_df = pd.DataFrame({
        "forecast": forecast_result.predicted_mean,
        "lower_bound": conf_int.iloc[:, 0],
        "upper_bound": conf_int.iloc[:, 1],
    }, index=forecast_index)
    
    forecast_df.index.name = date_col
    forecast_df['forecast'] = forecast_df['forecast'].clip(lower=absolute_min_weight)
    forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=absolute_min_weight)
    forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(lower=absolute_min_weight)
    forecast_df['lower_bound'] = np.minimum(forecast_df['lower_bound'], forecast_df['forecast'])

    return forecast_df