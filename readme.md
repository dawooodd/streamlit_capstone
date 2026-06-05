# 📊 FitSmart Data Science Dashboard

Folder ini berisi kode sumber untuk **Aplikasi Dashboard Interaktif Data Science** dari proyek FitSmart. Dashboard ini dibangun menggunakan **Streamlit** dan berfungsi sebagai representasi visual dari tahap *Exploratory Data Analysis* (EDA), *Feature Engineering*, dan *Machine Learning Modeling* (Time-Series Forecasting).

Dashboard ini dirancang dengan prinsip **Data Storytelling**, di mana setiap visualisasi secara langsung menjawab pertanyaan bisnis spesifik lengkap dengan *insight* yang dapat ditindaklanjuti (*actionable insights*).

---

## ✨ Fitur Utama

1. **Advanced Feature Engineering Terintegrasi:** Melakukan komputasi *real-time* untuk menentukan metrik saintifik pengguna seperti BMI, BMR (Mifflin-St Jeor), TDEE, Defisit/Surplus Kalori, serta *Nutrient Density Score* untuk database makanan.
2. **Menjawab 5 Pertanyaan Bisnis:**
   * Analisis Distribusi Kalori & Deteksi Risiko Menu Makanan.
   * Matriks Korelasi (Pearson) antar Makronutrisi.
   * Evaluasi Kedisiplinan Diet (Asupan Harian vs Batas TDEE).
   * Pemetaan Sistem Rekomendasi Makanan Cerdas (Superfood AI).
   * Evaluasi Metrik A/B Testing menggunakan Model Probabilitas (*Bell Curve*).
3. **Time-Series Forecasting:** Memproyeksikan lintasan berat badan pengguna untuk 30 hari ke depan menggunakan algoritma **ARIMA** yang dilengkapi dengan *Confidence Interval* 95% dan *safety guardrails* matematis.
4. **Visualisasi Profesional:** Menggunakan *backend* **Matplotlib** dan **Seaborn** untuk mereplikasi keakuratan dan kebersihan visualisasi sesuai standar *Jupyter Notebook* industri.

---

## � Akses Dashboard Streamlit

1. Masuk ke folder `streamlit/`:

   ```bash
   cd streamlit
   ```

2. Jalankan Streamlit:

   ```bash
   streamlit run app.py
   ```

3. Buka URL berikut di browser:

   [http://localhost:8501](http://localhost:8501)

> Jika port default sudah digunakan, Streamlit akan menampilkan alamat alternatif di terminal.

---

## �📂 Struktur Folder

```text
streamlit/
├── app.py                # Script utama aplikasi Streamlit (UI & Data Storytelling)
├── data_loader.py        # Modul backend untuk memuat CSV, membuat mock data, & imputasi (ETL)
├── forecasting.py        # Modul pemodelan Time-Series menggunakan Statsmodels (ARIMA)
└── README.md             # Dokumentasi panduan instalasi dan deployment
