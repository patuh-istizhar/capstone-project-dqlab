# Capstone Project: Pengembangan Model Klasifikasi Prediktif

## Deskripsi Proyek

Proyek ini merupakan Capstone Project yang dikembangkan sebagai bagian dari program DQLab. Fokus utamanya adalah pembangunan model klasifikasi prediktif komprehensif dari awal hingga akhir. Proyek ini mendemonstrasikan penerapan machine learning dan alur kerja data science standar industri.

## Fitur Utama

- **Tiga Model Klasifikasi:** Implementasi dan perbandingan kinerja model Random Forest, Regresi Logistik, dan XGBoost untuk tugas klasifikasi.
- **Multi-Dataset Analysis:** Penerapan model pada tiga dataset berbeda dari Scikit-learn:
  - **Breast Cancer Dataset** (klasifikasi biner)
  - **Iris Dataset** (klasifikasi multi-kelas)
  - **Wine Dataset** (klasifikasi multi-kelas)
- **Metodologi CRISP-DM:** Seluruh siklus proyek data science diikuti secara sistematis berdasarkan metodologi **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, mulai dari pemahaman bisnis hingga deployment.
- **Hyperparameter Tuning:** Pemanfaatan **GridSearch** untuk optimasi hyperparameter guna meningkatkan performa dan akurasi model.
- **Interaktif UI dengan Streamlit:** Model yang telah dilatih di-deploy ke dalam aplikasi web interaktif yang intuitif menggunakan **Streamlit**, memungkinkan pengguna untuk berinteraksi dengan model secara langsung.

## Teknologi yang Digunakan

- **Python**
- **Pandas** (untuk manipulasi dan analisis data)
- **Scikit-learn** (untuk model Regresi Logistik dan Random Forest, serta dataset)
- **XGBoost** (untuk model XGBoost)
- **Joblib** (untuk menyimpan dan memuat model)
- **Streamlit** (untuk pengembangan aplikasi web interaktif)
- **Jupyter Notebook** (untuk eksplorasi data dan pengembangan model)

## Struktur Repositori

```
capstone-project-dqlab/
├── notebooks/
│   └── breast_cancer_classification_crisp_dm.ipynb
│   └── iris_classification_crisp_dm.ipynb
│   └── wine_classification_crisp_dm.ipynb
├── saved_models/
│   ├── breast_cancer_classification/
│   │   ├── logistic_regression_model.joblib
│   │   ├── model_config.txt
│   │   └── scaler.joblib
│   ├── iris_classification/
│   │   ├── model_config.txt
│   │   ├── random_forest_model.joblib
│   │   └── scaler.joblib
│   └── wine_classification/
│       ├── logistic_regression_model.joblib
│       ├── model_config.txt
│       └── scaler.joblib
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

## Cara Menjalankan Aplikasi Streamlit

Aplikasi Streamlit dari proyek ini dapat diakses secara online melalui tautan berikut:

[Akses Aplikasi Streamlit di Sini](https://capstone-project-dqlab-f8idk5xxjrvvcx5nuksdqb.streamlit.app/)

Untuk menjalankan aplikasi ini secara lokal menggunakan **uv**:

1. **Pastikan Anda sudah menginstal uv:**
   Jika belum, Anda bisa menginstalnya dengan `pip install uv` atau mengikuti instruksi di dokumentasi uv.
2. **Clone repositori ini:**

   ```bash
   git clone https://github.com/patuh-istizhar/capstone-project-dqlab.git
   cd capstone-project-dqlab
   ```

3. **Instal semua dependensi menggunakan uv:**
   Karena `pyproject.toml` dan `uv.lock` sudah ada di repositori Anda, uv akan secara otomatis menggunakan file-file ini untuk menginstal dependensi.

   ```bash
   uv sync
   ```

   (Ini akan membuat virtual environment dan menginstal dependensi sesuai `uv.lock`.)

4. Jalankan aplikasi Streamlit:

   ```bash
   uv run streamlit run main.py
   ```

## Kontribusi

Proyek ini dikerjakan secara mandiri sebagai bagian dari Capstone Project. Namun, saran atau masukan untuk pengembangan lebih lanjut selalu diterima.
