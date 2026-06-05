"""
Data Loader Module untuk FitSmart Application
Menangani proses loading, processing, imputation, dan pembuatan mock data.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Optional

def load_nutrition_data(file_path: str) -> pd.DataFrame:
    try:
        nutrition_df = pd.read_csv(file_path)
        return nutrition_df
    except FileNotFoundError:
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV: {e}")

def create_mock_user_tracking(days: int = 60, seed: Optional[int] = None, nutrition_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
        
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - timedelta(days=days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    n_rows = len(dates)
    
    weight_trend = np.linspace(0, -2.5, n_rows) 
    weight = 75 + weight_trend + np.random.normal(0, 0.3, n_rows)
    weight[np.random.choice(n_rows, size=int(0.15 * n_rows), replace=False)] = np.nan
    
    calories_burned = np.random.uniform(300, 800, n_rows)
    calories_burned[np.random.choice(n_rows, size=int(0.10 * n_rows), replace=False)] = np.nan
    
    sleep_hours = np.random.uniform(5, 9, n_rows)
    sleep_hours[np.random.choice(n_rows, size=int(0.12 * n_rows), replace=False)] = np.nan
    
    workout_duration_mins = np.random.choice(
        [0, 30, 45, 60, 90, 120], size=n_rows, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
    )

    if nutrition_df is not None and 'calories' in nutrition_df.columns:
        intake_pool = nutrition_df['calories'].dropna().values
        calories_intake = np.random.choice(intake_pool, size=n_rows) * 4.5 + np.random.normal(0, 50, n_rows)
    else:
        calories_intake = np.random.normal(2000, 200, n_rows)

    df = pd.DataFrame({
        'date': dates,
        'weight': weight,
        'calories_intake': calories_intake,
        'calories_burned': calories_burned,
        'sleep_hours': sleep_hours,
        'workout_duration_mins': workout_duration_mins
    })
    
    df.set_index('date', inplace=True)
    return df

def impute_user_tracking_data(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()
    df_imputed['weight'] = df_imputed['weight'].ffill().bfill()
    df_imputed['calories_burned'] = df_imputed['calories_burned'].fillna(df_imputed['calories_burned'].median())
    df_imputed['sleep_hours'] = df_imputed['sleep_hours'].fillna(df_imputed['sleep_hours'].median())
    return df_imputed

def load_and_process_data(nutrition_path: str, days: int = 60, apply_imputation: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nutrition_df = load_nutrition_data(nutrition_path)
    user_tracking_df = create_mock_user_tracking(days=days, seed=42, nutrition_df=nutrition_df)
    
    if apply_imputation:
        user_tracking_df = impute_user_tracking_data(user_tracking_df)
        
    return nutrition_df, user_tracking_df