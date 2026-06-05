"""
FitSmart Data Science Dashboard

Aplikasi Streamlit untuk menampilkan statistik dan analisis data FitSmart.
Berfokus pada 5 Pertanyaan Bisnis Utama dengan visualisasi Matplotlib & Seaborn
(Replikasi dari FitSmart_EDA.ipynb) dan insight lengkap.

Author: FitSmart Team
Date: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from data_loader import load_and_process_data
from forecasting import forecast_weight

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIGURATION & THEME (SESUAI EDA NOTEBOOK)
# ============================================================================

st.set_page_config(
    page_title="FitSmart Analytics",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi Matplotlib & Seaborn persis seperti di FitSmart_EDA.ipynb
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid', palette='muted')

# ============================================================================
# 2. DATA PROCESSING & FEATURE ENGINEERING
# ============================================================================

@st.cache_data
def load_data():
    nutrition_path = Path(__file__).parent.parent / "data" / "nutrition.csv"
    try:
        nutrition_df, user_tracking_df = load_and_process_data(
            nutrition_path=str(nutrition_path), days=60, apply_imputation=True
        )
    except Exception:
        st.error("❌ Gagal memuat data. Periksa file CSV Anda.")
        return pd.DataFrame(), pd.DataFrame()
    return nutrition_df, user_tracking_df

def engineer_user_features(df: pd.DataFrame, height_cm=170, age=25, gender='M') -> pd.DataFrame:
    df = df.copy()
    height_m = height_cm / 100
    df['BMI'] = df['weight'] / (height_m ** 2)
    
    # Mifflin-St Jeor Equation
    if gender.upper() == 'M':
        df['BMR'] = (10 * df['weight']) + (6.25 * height_cm) - (5 * age) + 5
    else:
        df['BMR'] = (10 * df['weight']) + (6.25 * height_cm) - (5 * age) - 161
        
    activity_multiplier = np.where(df['workout_duration_mins'] > 60, 1.55,
                                  np.where(df['workout_duration_mins'] >= 30, 1.375, 1.2))
    df['TDEE'] = df['BMR'] * activity_multiplier
    
    if 'calories_intake' not in df.columns:
        np.random.seed(42)
        df['calories_intake'] = np.random.normal(2100, 250, len(df))
        
    df['Caloric_Deficit'] = df['TDEE'] - df['calories_intake']
    df['Diet_Status'] = np.where(df['Caloric_Deficit'] > 0, 'Defisit', 'Surplus')
    return df

def engineer_food_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = ['calories', 'proteins', 'fat', 'carbohydrate']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    safe_cal = df['calories'].replace(0, 1)
    df['Protein_Cal_Ratio'] = (df['proteins'] * 4) / safe_cal
    df['Fat_Cal_Ratio'] = (df['fat'] * 9) / safe_cal
    
    prot_score = (df['Protein_Cal_Ratio'] / df['Protein_Cal_Ratio'].max()) * 100
    fat_pen = (df['Fat_Cal_Ratio'] / df['Fat_Cal_Ratio'].max()) * 50
    df['Nutrient_Density_Score'] = (prot_score - fat_pen).clip(lower=0)
    return df

# ============================================================================
# 3. HEADER & KPI DASHBOARD
# ============================================================================

raw_nutrition_df, raw_user_tracking_df = load_data()
nutrition_df = engineer_food_features(raw_nutrition_df)
user_tracking_df = engineer_user_features(raw_user_tracking_df)

st.title("💪 FitSmart Executive Data Dashboard")
st.markdown("Mengubah data kebugaran mentah menjadi keputusan bisnis (Diet & Olahraga) yang dapat ditindaklanjuti.")
st.markdown("---")

if not user_tracking_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_bmi = user_tracking_df['BMI'].iloc[-1]
        st.metric("BMI Saat Ini", f"{current_bmi:.1f}", "Normal" if 18.5 <= current_bmi <= 24.9 else "Perhatian")
    with col2:
        st.metric("Total Penurunan BB", f"{(user_tracking_df['weight'].iloc[0] - user_tracking_df['weight'].iloc[-1]):.1f} kg")
    with col3:
        st.metric("Rata-rata Kapasitas TDEE", f"{user_tracking_df['TDEE'].mean():.0f} kcal")
    with col4:
        st.metric("Rata-rata Defisit Harian", f"{user_tracking_df['Caloric_Deficit'].mean():.0f} kcal")

st.markdown("---")

# ============================================================================
# 4. BUSINESS QUESTIONS & VISUALIZATIONS (MATPLOTLIB/SEABORN)
# ============================================================================
st.header("📊 Menjawab 5 Pertanyaan Bisnis (Berdasarkan Output EDA)")

# ----------------------------------------------------------------------------
# Pertanyaan 1: Distribusi Kalori & Risiko
# ----------------------------------------------------------------------------
st.subheader("Q1: Bagaimana profil distribusi kalori pada menu makanan, dan item apa yang paling berisiko?")
if not nutrition_df.empty:
    col_dist, col_bar = st.columns(2)
    
    with col_dist:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(nutrition_df['calories'], bins=20, kde=True, color='#4A90E2', ax=ax1)
        ax1.set_title('Distribusi Kalori Database Makanan', fontweight='bold')
        ax1.set_xlabel('Kalori (kcal)')
        ax1.set_ylabel('Frekuensi')
        plt.tight_layout()
        st.pyplot(fig1)
        
    with col_bar:
        top_cal = nutrition_df.dropna(subset=['calories']).sort_values('calories', ascending=False).head(5)
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=top_cal, x='calories', y='name', palette='Reds_r', ax=ax2)
        ax2.set_title('5 Makanan Tertinggi Kalori (Risiko Surplus)', fontweight='bold')
        ax2.set_xlabel('Kalori (kcal)')
        ax2.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig2)
        
    st.info("💡 **Insight Bisnis:** Kurva distribusi menunjukkan mayoritas makanan FitSmart berada di rentang rendah-menengah (aman). Namun, grafik di kanan menunjukkan bahwa **gorengan dan makanan bersantan** memiliki lonjakan kalori drastis. Pengguna harus dibatasi (diberi peringatan) saat mengonsumsi *Top 5* menu ini agar defisit kalori tidak rusak dalam satu kali makan.")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Pertanyaan 2: Korelasi Makronutrisi
# ----------------------------------------------------------------------------
st.subheader("Q2: Makronutrisi apa yang kontribusinya paling ekstrem terhadap lonjakan kalori?")
if not nutrition_df.empty:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    corr_cols = ['calories', 'proteins', 'fat', 'carbohydrate']
    corr_matrix = nutrition_df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax3, linewidths=0.5)
    ax3.set_title('Matriks Korelasi (Pearson) Antar Makronutrisi', fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.success("💡 **Insight Bisnis:** Secara saintifik, matriks ini membuktikan bahwa **Lemak (fat)** memiliki skor korelasi tertinggi terhadap total kalori. Artinya, untuk program penurunan berat badan (*weight loss*), memangkas kadar minyak/lemak akan memberikan dampak reduksi kalori yang jauh lebih besar dan cepat dibandingkan memangkas karbohidrat.")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Pertanyaan 3: Evaluasi Defisit Kalori (TDEE)
# ----------------------------------------------------------------------------
st.subheader("Q3: Sejauh mana konsistensi defisit kalori pengguna dibandingkan batas TDEE mereka?")
if not user_tracking_df.empty:
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    
    colors = ['#2ecc71' if status == 'Defisit' else '#e74c3c' for status in user_tracking_df['Diet_Status']]
    
    ax4.bar(user_tracking_df.index, user_tracking_df['calories_intake'], color=colors, alpha=0.8, label='Asupan Harian')
    ax4.plot(user_tracking_df.index, user_tracking_df['TDEE'], color='#2c3e50', linestyle='--', linewidth=2.5, label='Batas Aman (TDEE)')
    
    ax4.set_title('Kedisiplinan Diet: Asupan vs Total Daily Energy Expenditure (TDEE)', fontweight='bold')
    ax4.set_ylabel('Kalori (kcal)')
    ax4.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)

    st.info("💡 **Insight Bisnis:** Batang **Merah** menandakan " + "pelanggaran" + " di mana asupan melampaui garis TDEE (garis putus-putus). Jika batang merah sering muncul, artinya pengguna sering *cheat meal*. Sistem aplikasi harus mengirimkan notifikasi intervensi pada hari-hari rawan ini (misalnya setiap akhir pekan).")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Pertanyaan 4: Sistem Rekomendasi (Density Score)
# ----------------------------------------------------------------------------
st.subheader("Q4: Makanan apa yang direkomendasikan AI untuk diet tinggi protein namun rendah kalori?")
if not nutrition_df.empty:
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    
    sc = ax5.scatter(nutrition_df['calories'], nutrition_df['proteins'], 
                     c=nutrition_df['Nutrient_Density_Score'], cmap='viridis', 
                     s=40, alpha=0.9, edgecolors='w')
    
    plt.colorbar(sc, label='Nutrient Density Score (Lebih terang = Lebih baik)')
    ax5.set_title('Pemetaan Superfood Berdasarkan Kalori & Protein', fontweight='bold')
    ax5.set_xlabel('Total Kalori (kcal)')
    ax5.set_ylabel('Protein (g)')
    
    # Labeling top 3 food
    top_3 = nutrition_df.sort_values('Nutrient_Density_Score', ascending=False).head(3)
    for idx, row in top_3.iterrows():
        ax5.annotate(row['name'], (row['calories'], row['proteins']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='red')

    plt.tight_layout()
    st.pyplot(fig5)
    
    st.success("💡 **Insight Bisnis:** Melalui *Feature Engineering* Density Score, kita berhasil mengidentifikasi makanan di pojok kiri atas (titik terang/kuning) sebagai *Superfood*. Makanan berlabel merah ini wajib dimasukkan ke dalam daftar *Meal Plan* otomatis karena sangat mengenyangkan (tinggi protein) namun rendah kalori.")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Pertanyaan 5: Prediksi & Probabilitas A/B Testing
# ----------------------------------------------------------------------------
st.subheader("Q5: Apakah target spesifik pengguna realistis untuk dicapai dalam 30 hari ke depan?")

col_fc, col_prob = st.columns([1.3, 1])

with col_fc:
    try:
        weight_df = pd.DataFrame({"tanggal": user_tracking_df.index, "berat badan": user_tracking_df['weight'].values})
        fc_df = forecast_weight(weight_df, periods=30).reset_index()
        
        fig6, ax6 = plt.subplots(figsize=(7, 4.5))
        ax6.plot(weight_df['tanggal'], weight_df['berat badan'], color='#2980b9', label='Historis', linewidth=2)
        ax6.plot(fc_df['tanggal'], fc_df['forecast'], color='#e74c3c', linestyle='--', label='Prediksi ARIMA', linewidth=2)
        ax6.fill_between(fc_df['tanggal'], fc_df['lower_bound'], fc_df['upper_bound'], color='#e74c3c', alpha=0.15, label='95% Confidence Interval')
        
        ax6.set_title('Time-Series Forecasting Berat Badan (ARIMA)', fontweight='bold')
        ax6.set_ylabel('Berat Badan (kg)')
        ax6.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig6)
    except Exception:
        st.warning("Data historis belum cukup untuk model ARIMA.")

with col_prob:
    current_w = user_tracking_df['weight'].iloc[-1]
    target_w = st.number_input("Input Target Penurunan (BB Akhir):", value=float(current_w - 2.0), step=0.5)
    
    weight_diff = user_tracking_df['weight'].diff().dropna()
    daily_loss = (-weight_diff).clip(lower=0)
    
    expected_loss = daily_loss.mean() * 30
    std_total_loss = max(daily_loss.std(ddof=1), 0.05) * np.sqrt(30)
    required_loss = max(0.0, float(current_w - target_w))
    
    prob = 1.0 - norm.cdf(required_loss, loc=expected_loss, scale=std_total_loss) if required_loss > 0 else 1.0
    
    fig7, ax7 = plt.subplots(figsize=(6, 4.5))
    x_min = min(0, expected_loss - 3 * std_total_loss)
    x_max = max(required_loss + 2, expected_loss + 3 * std_total_loss)
    x = np.linspace(x_min, x_max, 200)
    y = norm.pdf(x, loc=expected_loss, scale=std_total_loss)
    
    ax7.plot(x, y, color='#2c3e50', linewidth=2)
    success_x = x[x >= required_loss]
    success_y = y[x >= required_loss]
    if len(success_x) > 0:
        ax7.fill_between(success_x, success_y, color='#2ecc71', alpha=0.5, label=f'Peluang: {prob*100:.1f}%')
        
    ax7.axvline(required_loss, color='#e74c3c', linestyle='--', linewidth=2, label='Garis Target')
    ax7.set_title('Distribusi Probabilitas Target', fontweight='bold')
    ax7.set_xlabel('Estimasi Penurunan (kg)')
    ax7.set_yticks([]) 
    ax7.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig7)

st.info("💡 **Insight Bisnis:** Model prediksi ARIMA memproyeksikan lintasan berat badan pengguna ke depan (Grafik Kiri). Sementara Grafik Kanan (*Bell Curve*) mengkuantifikasi apakah ambisi pengguna realistis. Jika Peluang di bawah 50%, sistem dapat mencegah kegagalan (*churn*) dengan merekomendasikan target yang lebih kecil atau perpanjangan durasi dari 30 hari menjadi 60 hari.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 13px;'>FitSmart Data Science Dashboard | Backend: Matplotlib, Seaborn, Scipy, Statsmodels</div>", unsafe_allow_html=True)