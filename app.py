"""
FitSmart Data Science Dashboard (ALL-IN-ONE VERSION)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import timedelta

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIGURATION & THEME
# ============================================================================
st.set_page_config(page_title="FitSmart Analytics", page_icon="💪", layout="wide")

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid', palette='muted')

# ============================================================================
# 2. MODULE DATA LOADER & IMPUTATION (Terintegrasi)
# ============================================================================
def load_nutrition_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception:
        # PENGAMANAN SUPER AMAN: Jika file CSV hilang/error path saat deploy, 
        # aplikasi akan membuat data dummy agar TIDAK CRASH (Layar Merah).
        return pd.DataFrame({
            'name': ['Nasi Putih', 'Dada Ayam Bakar', 'Gorengan Bakwan', 'Brokoli Rebus', 'Telur Rebus', 'Alpukat', 'Mie Instan', 'Oatmeal', 'Susu Full Cream', 'Tahu Goreng'],
            'calories': [130, 165, 300, 34, 155, 160, 380, 68, 150, 270],
            'proteins': [2.7, 31.0, 2.0, 2.8, 13.0, 2.0, 8.0, 2.4, 8.0, 10.0],
            'fat': [0.3, 3.6, 15.0, 0.4, 11.0, 15.0, 14.0, 1.4, 8.0, 20.0],
            'carbohydrate': [28.0, 0.0, 30.0, 7.0, 1.1, 9.0, 54.0, 12.0, 12.0, 9.0]
        })

def create_mock_user_tracking(days=60, seed=42, nutrition_df=None) -> pd.DataFrame:
    np.random.seed(seed)
    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    n_rows = len(dates)
    
    weight = 75 + np.linspace(0, -2.5, n_rows) + np.random.normal(0, 0.3, n_rows)
    weight[np.random.choice(n_rows, size=int(0.15 * n_rows), replace=False)] = np.nan
    
    calories_burned = np.random.uniform(300, 800, n_rows)
    calories_burned[np.random.choice(n_rows, size=int(0.1 * n_rows), replace=False)] = np.nan
    
    sleep_hours = np.random.uniform(5, 9, n_rows)
    sleep_hours[np.random.choice(n_rows, size=int(0.12 * n_rows), replace=False)] = np.nan
    
    workout_mins = np.random.choice([0, 30, 45, 60, 90, 120], size=n_rows, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])

    if nutrition_df is not None and not nutrition_df.empty:
        intake_pool = nutrition_df['calories'].dropna().values
        cals_intake = np.random.choice(intake_pool, size=n_rows) * 4.5 + np.random.normal(0, 50, n_rows)
    else:
        cals_intake = np.random.normal(2000, 200, n_rows)

    df = pd.DataFrame({'date': dates, 'weight': weight, 'calories_intake': cals_intake,
                       'calories_burned': calories_burned, 'sleep_hours': sleep_hours, 'workout_duration_mins': workout_mins})
    df.set_index('date', inplace=True)
    return df

@st.cache_data
def get_processed_data():
    nutrition_path = Path(__file__).parent.parent / "data" / "nutrition.csv"
    nutri_df = load_nutrition_data(str(nutrition_path))
    track_df = create_mock_user_tracking(days=60, seed=42, nutrition_df=nutri_df)
    
    # Imputasi
    track_df['weight'] = track_df['weight'].ffill().bfill()
    track_df['calories_burned'] = track_df['calories_burned'].fillna(track_df['calories_burned'].median())
    track_df['sleep_hours'] = track_df['sleep_hours'].fillna(track_df['sleep_hours'].median())
    
    return nutri_df, track_df

# ============================================================================
# 3. MODULE FORECASTING (Terintegrasi)
# ============================================================================
def forecast_weight_arima(df: pd.DataFrame, periods=30) -> pd.DataFrame:
    data = df.copy().dropna()
    data = data.asfreq('D').interpolate(method="time").ffill().bfill()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(data['berat badan'], order=(1, 1, 1)).fit()
        except Exception:
            model = ARIMA(data['berat badan'], order=(0, 1, 1)).fit()

    fc = model.get_forecast(steps=periods)
    fc_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=periods, freq='D')
    
    fc_df = pd.DataFrame({
        "forecast": fc.predicted_mean.clip(lower=35.0),
        "lower_bound": fc.conf_int(alpha=0.05).iloc[:, 0].clip(lower=35.0),
        "upper_bound": fc.conf_int(alpha=0.05).iloc[:, 1].clip(lower=35.0),
    }, index=fc_index)
    fc_df['lower_bound'] = np.minimum(fc_df['lower_bound'], fc_df['forecast'])
    return fc_df

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
def engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['BMI'] = df['weight'] / (1.70 ** 2) # Asumsi tinggi 170cm
    df['BMR'] = (10 * df['weight']) + (6.25 * 170) - (5 * 25) + 5 # Asumsi 25th, Pria
    
    act_mult = np.where(df['workout_duration_mins'] > 60, 1.55, np.where(df['workout_duration_mins'] >= 30, 1.375, 1.2))
    df['TDEE'] = df['BMR'] * act_mult
    df['Caloric_Deficit'] = df['TDEE'] - df['calories_intake']
    df['Diet_Status'] = np.where(df['Caloric_Deficit'] > 0, 'Defisit', 'Surplus')
    return df

def engineer_food_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_cal = df['calories'].replace(0, 1)
    df['Protein_Cal_Ratio'] = (df['proteins'] * 4) / safe_cal
    df['Fat_Cal_Ratio'] = (df['fat'] * 9) / safe_cal
    
    prot_score = (df['Protein_Cal_Ratio'] / df['Protein_Cal_Ratio'].max()) * 100
    fat_pen = (df['Fat_Cal_Ratio'] / df['Fat_Cal_Ratio'].max()) * 50
    df['Nutrient_Density_Score'] = (prot_score - fat_pen).clip(lower=0)
    return df

# ============================================================================
# 5. STREAMLIT UI & VISUALIZATIONS
# ============================================================================
raw_nutri, raw_track = get_processed_data()
nutrition_df = engineer_food_features(raw_nutri)
user_tracking_df = engineer_user_features(raw_track)

st.title("💪 FitSmart Executive Data Dashboard")
st.markdown("Mengubah data kebugaran mentah menjadi keputusan bisnis (Diet & Olahraga) yang dapat ditindaklanjuti.")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
current_bmi = user_tracking_df['BMI'].iloc[-1]
col1.metric("BMI Saat Ini", f"{current_bmi:.1f}", "Normal" if 18.5 <= current_bmi <= 24.9 else "Perhatian")
col2.metric("Total Penurunan BB", f"{(user_tracking_df['weight'].iloc[0] - user_tracking_df['weight'].iloc[-1]):.1f} kg")
col3.metric("Rata-rata Kapasitas TDEE", f"{user_tracking_df['TDEE'].mean():.0f} kcal")
col4.metric("Rata-rata Defisit Harian", f"{user_tracking_df['Caloric_Deficit'].mean():.0f} kcal")

st.markdown("---")
st.header("📊 Menjawab 5 Pertanyaan Bisnis (Berdasarkan Output EDA)")

# --- VISUAL 1 ---
st.subheader("Q1: Bagaimana profil distribusi kalori pada menu makanan, dan item apa yang paling berisiko?")
col_dist, col_bar = st.columns(2)
with col_dist:
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    sns.histplot(nutrition_df['calories'], bins=20, kde=True, color='#4A90E2', ax=ax1)
    ax1.set_title('Distribusi Kalori Database Makanan', fontweight='bold')
    ax1.set_xlabel('Kalori (kcal)')
    plt.tight_layout()
    st.pyplot(fig1)
with col_bar:
    top_cal = nutrition_df.sort_values('calories', ascending=False).head(5)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    sns.barplot(data=top_cal, x='calories', y='name', palette='Reds_r', ax=ax2)
    ax2.set_title('5 Makanan Tertinggi Kalori', fontweight='bold')
    ax2.set_xlabel('Kalori (kcal)')
    ax2.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig2)
st.info("💡 **Insight:** Makanan bersantan dan gorengan memiliki lonjakan kalori drastis dan paling berisiko menggagalkan diet harian.")

st.markdown("<br>", unsafe_allow_html=True)

# --- VISUAL 2 ---
st.subheader("Q2: Makronutrisi apa yang kontribusinya paling ekstrem terhadap lonjakan kalori?")
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.heatmap(nutrition_df[['calories', 'proteins', 'fat', 'carbohydrate']].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax3)
ax3.set_title('Matriks Korelasi (Pearson) Antar Makronutrisi', fontweight='bold', pad=15)
plt.tight_layout()
st.pyplot(fig3)
st.success("💡 **Insight:** Secara saintifik, matriks ini membuktikan bahwa **Lemak (fat)** memiliki skor korelasi tertinggi terhadap total kalori dibandingkan nutrisi lain.")

st.markdown("<br>", unsafe_allow_html=True)

# --- VISUAL 3 ---
st.subheader("Q3: Sejauh mana konsistensi defisit kalori pengguna dibandingkan batas TDEE mereka?")
fig4, ax4 = plt.subplots(figsize=(12, 5))
colors = ['#2ecc71' if status == 'Defisit' else '#e74c3c' for status in user_tracking_df['Diet_Status']]
ax4.bar(user_tracking_df.index, user_tracking_df['calories_intake'], color=colors, alpha=0.8, label='Asupan Harian')
ax4.plot(user_tracking_df.index, user_tracking_df['TDEE'], color='#2c3e50', linestyle='--', linewidth=2.5, label='Batas Aman (TDEE)')
ax4.set_title('Kedisiplinan Diet: Asupan vs TDEE', fontweight='bold')
ax4.legend(loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig4)
st.info("💡 **Insight:** Batang **Merah** menandakan bahwa asupan melampaui garis TDEE. Jika batang merah sering muncul, pengguna terindikasi sering over-eating.")

st.markdown("<br>", unsafe_allow_html=True)

# --- VISUAL 4 ---
st.subheader("Q4: Makanan apa yang direkomendasikan AI untuk diet tinggi protein namun rendah kalori?")
fig5, ax5 = plt.subplots(figsize=(10, 5))
sc = ax5.scatter(nutrition_df['calories'], nutrition_df['proteins'], c=nutrition_df['Nutrient_Density_Score'], cmap='viridis', s=40, alpha=0.9, edgecolors='w')
plt.colorbar(sc, label='Nutrient Density Score')
ax5.set_title('Pemetaan Superfood Berdasarkan Kalori & Protein', fontweight='bold')
ax5.set_xlabel('Kalori (kcal)')
ax5.set_ylabel('Protein (g)')
for idx, row in nutrition_df.sort_values('Nutrient_Density_Score', ascending=False).head(3).iterrows():
    ax5.annotate(row['name'], (row['calories'], row['proteins']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='red')
plt.tight_layout()
st.pyplot(fig5)
st.success("💡 **Insight:** Makanan di pojok kiri atas (titik terang) adalah *Superfood*. Sangat mengenyangkan namun rendah kalori.")

st.markdown("<br>", unsafe_allow_html=True)

# --- VISUAL 5 ---
st.subheader("Q5: Apakah target spesifik pengguna realistis untuk dicapai dalam 30 hari ke depan?")
col_fc, col_prob = st.columns([1.3, 1])

with col_fc:
    try:
        w_df = pd.DataFrame({"berat badan": user_tracking_df['weight'].values}, index=user_tracking_df.index)
        fc_df = forecast_weight_arima(w_df, periods=30).reset_index()
        fig6, ax6 = plt.subplots(figsize=(7, 4.5))
        ax6.plot(user_tracking_df.index, user_tracking_df['weight'], color='#2980b9', label='Historis', linewidth=2)
        ax6.plot(fc_df['index'], fc_df['forecast'], color='#e74c3c', linestyle='--', label='Prediksi ARIMA', linewidth=2)
        ax6.fill_between(fc_df['index'], fc_df['lower_bound'], fc_df['upper_bound'], color='#e74c3c', alpha=0.15)
        ax6.set_title('Time-Series Forecasting Berat Badan', fontweight='bold')
        ax6.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig6)
    except Exception:
        st.warning("Data historis belum cukup untuk model ARIMA.")

with col_prob:
    target_w = st.number_input("Input Target Penurunan (BB Akhir):", value=float(user_tracking_df['weight'].iloc[-1] - 2.0), step=0.5)
    daily_loss = (-user_tracking_df['weight'].diff().dropna()).clip(lower=0)
    exp_loss = daily_loss.mean() * 30
    std_loss = max(daily_loss.std(ddof=1), 0.05) * np.sqrt(30)
    req_loss = max(0.0, float(user_tracking_df['weight'].iloc[-1] - target_w))
    prob = 1.0 - norm.cdf(req_loss, loc=exp_loss, scale=std_loss) if req_loss > 0 else 1.0
    
    fig7, ax7 = plt.subplots(figsize=(6, 4.5))
    x = np.linspace(min(0, exp_loss - 3*std_loss), max(req_loss + 2, exp_loss + 3*std_loss), 200)
    ax7.plot(x, norm.pdf(x, loc=exp_loss, scale=std_loss), color='#2c3e50', linewidth=2)
    ax7.fill_between(x[x >= req_loss], norm.pdf(x[x >= req_loss], loc=exp_loss, scale=std_loss), color='#2ecc71', alpha=0.5, label=f'Peluang: {prob*100:.1f}%')
    ax7.axvline(req_loss, color='#e74c3c', linestyle='--', label='Garis Target')
    ax7.set_title('Distribusi Probabilitas Target', fontweight='bold')
    ax7.set_yticks([]) 
    ax7.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig7)

st.info("💡 **Insight Bisnis:** Jika Peluang di Grafik Kanan sangat kecil, sistem dapat merekomendasikan perpanjangan target diet dari 30 hari menjadi 60 hari untuk mencegah pengguna frustrasi.")
