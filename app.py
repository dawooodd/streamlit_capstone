"""
FitSmart Data Science Dashboard
Aplikasi Streamlit Terintegrasi untuk Analisis Kebugaran, Nutrisi, dan Machine Learning.
Dirancang khusus untuk Laporan Sidang Skripsi.

Author: Senior Data Scientist (Gemini AI)
Project: FitSmart
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import datetime
import warnings

# Mengabaikan warning untuk tampilan yang lebih bersih
warnings.filterwarnings('ignore')

# Coba import statsmodels untuk ARIMA, gunakan try-except agar aplikasi tidak crash jika library belum terinstall
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    st.error("Library statsmodels belum terinstall. Silakan jalankan: pip install statsmodels")

# ============================================================================
# 1. PAGE CONFIGURATION & THEME
# ============================================================================
st.set_page_config(
    page_title="FitSmart Analytics Engine",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi Grafik Standar Akademik
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_theme(style='whitegrid', palette='muted')

# ============================================================================
# 2. DATA GENERATION & MOCKING (AGAR KODE BISA LANGSUNG JALAN)
# ============================================================================
@st.cache_data
def load_food_data():
    """Memuat 24 dataset makanan dari hasil ekstraksi Computer Vision FitSmart"""
    data = [
        {'label': 'Ayam Goreng', 'n_images': 400, 'calories': 302.0, 'protein': 18.2, 'fat': 25.0, 'carbs': 0.0},
        {'label': 'Ikan Goreng', 'n_images': 125, 'calories': 380.0, 'protein': 33.6, 'fat': 27.5, 'carbs': 0.0},
        {'label': 'Mie Goreng', 'n_images': 119, 'calories': 476.0, 'protein': 0.1, 'fat': 21.1, 'carbs': 71.3},
        {'label': 'alpukat', 'n_images': 60, 'calories': 85.0, 'protein': 0.9, 'fat': 6.5, 'carbs': 7.7},
        {'label': 'apel', 'n_images': 40, 'calories': 58.0, 'protein': 0.3, 'fat': 0.4, 'carbs': 14.9},
        {'label': 'bakso', 'n_images': 315, 'calories': 76.0, 'protein': 4.1, 'fat': 2.5, 'carbs': 9.2},
        {'label': 'bayam', 'n_images': 97, 'calories': 30.0, 'protein': 1.3, 'fat': 0.7, 'carbs': 5.8},
        {'label': 'gado_gado', 'n_images': 263, 'calories': 137.0, 'protein': 6.1, 'fat': 3.2, 'carbs': 21.0},
        {'label': 'gudeg', 'n_images': 213, 'calories': 53.0, 'protein': 1.6, 'fat': 1.6, 'carbs': 8.8},
        {'label': 'jeruk', 'n_images': 40, 'calories': 48.0, 'protein': 0.6, 'fat': 0.2, 'carbs': 12.4},
        {'label': 'kentang', 'n_images': 77, 'calories': 83.0, 'protein': 2.0, 'fat': 0.1, 'carbs': 19.1},
        {'label': 'mangga', 'n_images': 39, 'calories': 63.0, 'protein': 2.4, 'fat': 0.4, 'carbs': 12.4},
        {'label': 'nasi_goreng', 'n_images': 366, 'calories': 180.0, 'protein': 3.0, 'fat': 0.3, 'carbs': 39.8},
        {'label': 'pempek', 'n_images': 286, 'calories': 162.0, 'protein': 4.9, 'fat': 3.0, 'carbs': 27.8},
        {'label': 'pisang', 'n_images': 40, 'calories': 43.0, 'protein': 2.6, 'fat': 0.0, 'carbs': 11.6},
        {'label': 'rawon', 'n_images': 235, 'calories': 60.0, 'protein': 5.4, 'fat': 2.5, 'carbs': 4.0},
        {'label': 'rendang', 'n_images': 237, 'calories': 193.0, 'protein': 22.6, 'fat': 7.9, 'carbs': 7.8},
        {'label': 'sate', 'n_images': 355, 'calories': 283.0, 'protein': 12.1, 'fat': 16.8, 'carbs': 20.9},
        {'label': 'semangka', 'n_images': 84, 'calories': 28.0, 'protein': 0.5, 'fat': 0.2, 'carbs': 6.9},
        {'label': 'soto', 'n_images': 358, 'calories': 42.0, 'protein': 3.9, 'fat': 1.7, 'carbs': 2.8},
        {'label': 'tomat', 'n_images': 92, 'calories': 98.0, 'protein': 2.0, 'fat': 0.4, 'carbs': 24.5},
        {'label': 'ubi', 'n_images': 69, 'calories': 110.0, 'protein': 1.4, 'fat': 1.1, 'carbs': 23.4},
        {'label': 'wortel', 'n_images': 82, 'calories': 15.0, 'protein': 0.6, 'fat': 0.1, 'carbs': 3.1}
    ]
    return pd.DataFrame(data)

@st.cache_data
def load_user_tracking_data():
    """Membuat data historis pengguna selama 60 hari terakhir"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.date.today(), periods=60)
    # Simulasi tren penurunan berat badan dari 80kg ke 75kg
    weight_trend = np.linspace(80, 75, 60) + np.random.normal(0, 0.4, 60)
    calories_intake = np.random.normal(1900, 250, 60)
    workout_mins = np.random.choice([0, 30, 45, 60, 90], size=60, p=[0.3, 0.2, 0.3, 0.15, 0.05])
    
    df = pd.DataFrame({
        'tanggal': dates,
        'berat_badan': weight_trend,
        'kalori_masuk': calories_intake,
        'durasi_olahraga_menit': workout_mins
    })
    df.set_index('tanggal', inplace=True)
    return df

# ============================================================================
# 3. ADVANCED FEATURE ENGINEERING FUNCTIONS
# ============================================================================
def engineer_user_features(df, height_cm, age, gender):
    """Menghitung metrik saintifik (BMI, BMR, TDEE, Defisit)"""
    df = df.copy()
    height_m = height_cm / 100
    df['BMI'] = df['berat_badan'] / (height_m ** 2)
    
    # Mifflin-St Jeor Equation untuk Basal Metabolic Rate (BMR)
    if gender == 'Pria':
        df['BMR'] = (10 * df['berat_badan']) + (6.25 * height_cm) - (5 * age) + 5
    else:
        df['BMR'] = (10 * df['berat_badan']) + (6.25 * height_cm) - (5 * age) - 161
        
    # Kalkulasi Total Daily Energy Expenditure (TDEE) berdasarkan olahraga
    activity_multiplier = np.where(df['durasi_olahraga_menit'] > 60, 1.55,
                          np.where(df['durasi_olahraga_menit'] >= 30, 1.375, 1.2))
    df['TDEE'] = df['BMR'] * activity_multiplier
    
    df['Defisit_Kalori'] = df['TDEE'] - df['kalori_masuk']
    df['Status_Diet'] = np.where(df['Defisit_Kalori'] > 0, 'Defisit (Aman)', 'Surplus (Pelanggaran)')
    return df

def engineer_food_features(df):
    """Menghitung Nutrient Density Score untuk sistem rekomendasi"""
    df = df.copy()
    safe_cal = df['calories'].replace(0, 1) # Menghindari division by zero
    df['Protein_Ratio'] = (df['protein'] * 4) / safe_cal
    df['Fat_Ratio'] = (df['fat'] * 9) / safe_cal
    
    # Score = (Protein tinggi baik, Fat tinggi dikurangi)
    prot_score = (df['Protein_Ratio'] / df['Protein_Ratio'].max()) * 100
    fat_pen = (df['Fat_Ratio'] / df['Fat_Ratio'].max()) * 30
    df['Nutrient_Density_Score'] = (prot_score - fat_pen).clip(lower=0)
    return df

# ============================================================================
# 4. SIDEBAR & USER PROFILE CONFIGURATION
# ============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2936/2936886.png", width=80)
st.sidebar.title("FitSmart Profile")
st.sidebar.markdown("Konfigurasi Bio-Metrik Pengguna")

user_gender = st.sidebar.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
user_age = st.sidebar.number_input("Umur (Tahun)", min_value=15, max_value=80, value=25)
user_height = st.sidebar.number_input("Tinggi Badan (cm)", min_value=140, max_value=210, value=170)
target_weight = st.sidebar.number_input("Target Berat Badan (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)

# Load and process data
raw_food_df = load_food_data()
food_df = engineer_food_features(raw_food_df)

raw_user_df = load_user_tracking_data()
user_df = engineer_user_features(raw_user_df, user_height, user_age, user_gender)

# ============================================================================
# 5. MAIN DASHBOARD UI & KPIs
# ============================================================================
st.title("🍏 FitSmart Executive Data Dashboard")
st.markdown("""
Dashboard ini adalah implementasi *end-to-end Machine Learning* dan *Data Science Pipeline*. 
Mentransformasi data mentah komputer visi dan historis *tracking* menjadi *actionable insights* (Metodologi Skripsi Bab 4).
""")

# Menampilkan KPI Header
current_weight = user_df['berat_badan'].iloc[-1]
start_weight = user_df['berat_badan'].iloc[0]
current_bmi = user_df['BMI'].iloc[-1]
avg_deficit = user_df['Defisit_Kalori'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Berat Badan Saat Ini", f"{current_weight:.1f} kg", f"{current_weight - start_weight:.1f} kg (Total)")
col2.metric("BMI (Body Mass Index)", f"{current_bmi:.1f}", "Normal" if 18.5 <= current_bmi <= 24.9 else "Perhatian", delta_color="off")
col3.metric("Rata-rata TDEE Harian", f"{user_df['TDEE'].mean():.0f} kcal")
col4.metric("Rata-rata Defisit Harian", f"{avg_deficit:.0f} kcal", "Sesuai Target" if avg_deficit > 0 else "Gagal Diet")

st.markdown("---")

# ============================================================================
# 6. DATA VISUALIZATION TABS (SESUAI ANALISIS BAB 4)
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Computer Vision Data", 
    "🔥 Diet Consistency Tracker", 
    "🧠 AI Food Recommender", 
    "📈 ARIMA Forecasting & Probabilitas"
])

# ----------------------------------------------------------------------------
# TAB 1: Exploratory Data Analysis (Seperti yang dibahas di prompt sebelumnya)
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Analisis Distribusi Kalori & Deteksi Risiko Outlier")
    st.markdown("Identifikasi *High-Risk High-Calorie Items* dan analisis *Class Imbalance* pada dataset Computer Vision.")
    
    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Distribusi Kalori (Mendeteksi Outlier Atas)
    df_plot = food_df.sort_values('calories')
    colors = ['#E24B4A' if c > food_df['calories'].quantile(0.75) 
              else '#1D9E75' if c < food_df['calories'].quantile(0.25) 
              else '#378ADD' for c in df_plot['calories']]
    
    bars = axes[0].barh(df_plot['label'], df_plot['calories'], color=colors, edgecolor='white')
    axes[0].axvline(food_df['calories'].mean(), color='#BA7517', linestyle='--', linewidth=2, label=f'Mean={food_df["calories"].mean():.0f} kcal')
    axes[0].set_xlabel('Total Kalori (kcal)', fontweight='bold')
    axes[0].set_title('Kepadatan Kalori per Label Makanan', fontweight='bold', pad=15)
    axes[0].legend()
    
    # Plot 2: Class Imbalance (Jumlah Gambar)
    df_imgs = food_df.sort_values('n_images')
    mean_img = food_df['n_images'].mean()
    img_colors = ['#E24B4A' if v < mean_img * 0.5 else '#1D9E75' for v in df_imgs['n_images']]
    
    axes[1].barh(df_imgs['label'], df_imgs['n_images'], color=img_colors, edgecolor='white')
    axes[1].axvline(mean_img, color='#185FA5', linestyle='--', linewidth=2, label=f'Rata-rata={mean_img:.0f} Gbr')
    axes[1].set_xlabel('Jumlah Gambar Latih', fontweight='bold')
    axes[1].set_title('Analisis Keseimbangan Kelas (Class Imbalance)', fontweight='bold', pad=15)
    axes[1].legend()
    
    plt.tight_layout()
    st.pyplot(fig1)
    
    st.info("""
    **Insight Akademik (Bab 4):**
    1. **Deteksi Outlier:** 'Mie Goreng' (476 kcal) dan 'Ikan Goreng' (380 kcal) merupakan *outlier* batas atas. Sistem Rule-based FitSmart akan menandai kelas merah ini sebagai *Warning Items* untuk mencegah surplus kalori.
    2. **Penanganan Imbalance:** Terjadi ketidakseimbangan kelas (*imbalance*) yang tajam, contoh: 'Ayam Goreng' (400) vs 'Mangga' (39). Model CNN Computer Vision akan rentan mengalami *overfitting*. Hal ini diatasi melalui proses *Data Augmentation* dan pembobotan *Class Weights* saat training.
    """)

# ----------------------------------------------------------------------------
# TAB 2: Diet Consistency (TDEE vs Intake)
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Evaluasi Kedisiplinan Diet (Intake vs TDEE)")
    st.markdown("Memantau anomali (*Cheat Days*) di mana kalori masuk melebihi batas batas ambang metabolisme.")
    
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    colors_diet = ['#2ecc71' if status == 'Defisit (Aman)' else '#e74c3c' for status in user_df['Status_Diet']]
    
    ax2.bar(user_df.index, user_df['kalori_masuk'], color=colors_diet, alpha=0.8, label='Asupan Kalori Harian')
    ax2.plot(user_df.index, user_df['TDEE'], color='#2c3e50', linestyle='--', linewidth=2.5, label='Batas TDEE (BMR + Olahraga)')
    
    ax2.set_ylabel('Energi (kcal)')
    ax2.set_title('Monitoring Defisit Kalori Time-Series', fontweight='bold')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.success("**Implikasi Sistem:** Batang berwarna **merah** mendeteksi hari di mana terjadi *Surplus Kalori* (pelanggaran diet). FitSmart mendeteksi pola ini dan dapat memicu fitur *Push Notification* intervensi secara otomatis.")

# ----------------------------------------------------------------------------
# TAB 3: AI Superfood Recommender
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Sistem Rekomendasi Berbasis Nutrient Density Score")
    st.markdown("Pemetaan makanan yang tinggi protein namun rendah kalori (Superfoods) menggunakan perhitungan matriks komposit.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sc = ax3.scatter(food_df['calories'], food_df['protein'], 
                     c=food_df['Nutrient_Density_Score'], cmap='viridis', 
                     s=100, alpha=0.9, edgecolors='black')
    
    plt.colorbar(sc, label='Nutrient Density Score (Kuning = Direkomendasikan)')
    ax3.set_title('Pemetaan Superfood: Protein vs Kalori', fontweight='bold')
    ax3.set_xlabel('Total Kalori (kcal)')
    ax3.set_ylabel('Total Protein (gram)')
    
    # Anotasi 5 makanan terbaik
    top_5 = food_df.sort_values('Nutrient_Density_Score', ascending=False).head(5)
    for idx, row in top_5.iterrows():
        ax3.annotate(row['label'], (row['calories'], row['protein']), 
                     xytext=(5, 5), textcoords='offset points', 
                     fontsize=10, fontweight='bold', color='red')
                     
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.info("**Algoritma Rekomendasi:** Semakin tinggi dan ke kiri titik berada (zona terang), makanan tersebut diklasifikasikan sebagai sangat mengenyangkan dengan kalori minim. Item berlabel teks merah diprioritaskan oleh algoritma FitSmart ke dalam modul *Automated Meal Plan* pengguna.")

# ----------------------------------------------------------------------------
# TAB 4: Time-Series Forecasting & Probability
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Machine Learning: ARIMA Forecasting & Scipy Probability")
    st.markdown("Memprediksi lintasan berat badan 30 hari ke depan dan mengevaluasi peluang keberhasilan target.")
    
    col_fc, col_prob = st.columns([1.2, 1])
    
    with col_fc:
        try:
            # Training Model ARIMA secara Real-time
            ts_data = user_df['berat_badan'].copy()
            model = ARIMA(ts_data, order=(1, 1, 1)) # Simple ARIMA model
            fitted = model.fit()
            
            # Forecasting 30 days
            forecast_steps = 30
            forecast = fitted.get_forecast(steps=forecast_steps)
            fc_mean = forecast.predicted_mean
            fc_conf = forecast.conf_int(alpha=0.05) # 95% CI
            
            fc_index = pd.date_range(start=ts_data.index[-1] + datetime.timedelta(days=1), periods=forecast_steps)
            
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.plot(ts_data.index, ts_data, label='Historis Aktual', color='#2980b9', linewidth=2)
            ax4.plot(fc_index, fc_mean, label='Prediksi ARIMA (30 Hari)', color='#e74c3c', linestyle='--', linewidth=2)
            ax4.fill_between(fc_index, fc_conf.iloc[:, 0], fc_conf.iloc[:, 1], color='#e74c3c', alpha=0.2, label='95% Confidence Interval')
            
            ax4.axhline(target_weight, color='green', linestyle=':', linewidth=2, label='Target Berat Badan')
            ax4.set_title('Time-Series Forecasting (Model ARIMA)', fontweight='bold')
            ax4.set_ylabel('Berat Badan (kg)')
            ax4.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig4)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membangun model ARIMA: {e}")

    with col_prob:
        # Kalkulasi Probabilitas dengan Distribusi Normal
        weight_diff = user_df['berat_badan'].diff().dropna()
        daily_loss = (-weight_diff).clip(lower=0) # Ambil nilai penurunan saja
        
        expected_loss = daily_loss.mean() * 30
        std_loss = daily_loss.std() * np.sqrt(30)
        required_loss = current_weight - target_weight
        
        # Hitung peluang (Area under curve dari required_loss hingga tak hingga)
        if required_loss <= 0:
            prob_success = 1.0 # Target sudah tercapai
        else:
            prob_success = 1.0 - norm.cdf(required_loss, loc=expected_loss, scale=std_loss)
            
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        x_axis = np.linspace(expected_loss - 3*std_loss, expected_loss + 3*std_loss, 100)
        y_axis = norm.pdf(x_axis, expected_loss, std_loss)
        
        ax5.plot(x_axis, y_axis, color='#2c3e50', linewidth=2)
        
        # Arsiran area sukses
        x_success = x_axis[x_axis >= required_loss]
        y_success = y_axis[x_axis >= required_loss]
        if len(x_success) > 0:
            ax5.fill_between(x_success, y_success, color='#2ecc71', alpha=0.5, label=f'Peluang Sukses: {prob_success*100:.1f}%')
            
        ax5.axvline(required_loss, color='red', linestyle='--', linewidth=2, label=f'Syarat Penurunan: {required_loss:.1f} kg')
        ax5.set_title('Uji Probabilitas Target (Bell Curve)', fontweight='bold')
        ax5.set_xlabel('Estimasi Total Penurunan (kg)')
        ax5.set_yticks([])
        ax5.legend()
        plt.tight_layout()
        st.pyplot(fig5)

    st.success(f"**Insight Evaluasi Model:** Berdasarkan laju penurunan berat badan historis, Model memproyeksikan pengguna akan mencapai berat **{fc_mean.iloc[-1]:.1f} kg** dalam 30 hari ke depan. Uji statistik mengkuantifikasi peluang keberhasilan target tersebut sebesar **{prob_success*100:.1f}%**.")

# Footer Akademik
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray; font-size: 13px;'>FitSmart Skripsi Dashboard | Dibangun menggunakan Python, Streamlit, Statsmodels & Seaborn</div>", unsafe_allow_html=True)
