# Laporan Proyek Machine Learning: Prediksi Peluang Diterima Mahasiswa Pascasarjana - Ananta Boemi Adji

## **Domain Project**
Penerimaan mahasiswa pascasarjana merupakan salah satu proses penting bagi institusi pendidikan tinggi. Universitas harus mempertimbangkan berbagai faktor akademik dan non-akademik dalam mengevaluasi kandidat. Di sisi lain, calon mahasiswa sering kali ingin mengetahui seberapa besar peluang mereka untuk diterima di program impian berdasarkan latar belakang akademik yang dimiliki [1].

Dengan adanya kemajuan dalam teknologi data science dan machine learning, kini memungkinkan untuk membuat model prediktif yang dapat memperkirakan peluang diterima mahasiswa berdasarkan data historis. Model ini dapat memberikan nilai tambah baik untuk institusi pendidikan dalam meningkatkan efisiensi proses seleksi maupun untuk calon mahasiswa dalam pengambilan keputusan.

Alasan kepentingan dari pembuatan model ini adalah karena:
1. Meningkatkan efisiensi proses seleksi secara lebih efisien dengan memanfaatkan data historis dan algoritma pembelajaran mesin. Hal ini memungkinkan proses seleksi yang lebih cepat dan objektif.
2. Mengidentifikasi faktor-faktor kunci penerimaan seperti nilai akademik, skor tes kemampuan bahasa, dan kualitas surat rekomendasi.
3. Mengurangi tingkat kegagalan studi dengan mengetahui faktor mana yang memiliki pengaruh besar terhadap penerimaan mereka.

Penerapan analitik prediktif sangat relevan dalam membantu institusi pendidikan melakukan proses seleksi yang lebih efisien, adil, dan transparan, sekaligus memberikan insight personalisasi kepada calon mahasiswa.

Dataset saya ambil dari sumber Kaggle yang berjudul Graduate Admission 2 dengan sumber: Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019

## **Business Understanding**
Peningkatan jumlah pendaftar program pascasarjana menyebabkan tantangan tersendiri bagi universitas dalam menyaring kandidat yang sesuai. Evaluasi manual terhadap ribuan aplikasi tidak hanya memakan waktu, tetapi juga rentan terhadap bias manusia.

Bagi calon mahasiswa, tidak mengetahui seberapa besar peluang mereka untuk diterima dapat menyebabkan ketidakpastian dalam proses perencanaan studi, termasuk pemilihan universitas, persiapan tes, dan strategi peningkatan profil.

Dengan menggunakan pendekatan prediktif berbasis data historis penerimaan, institusi dapat melakukan **filter awal** kandidat secara efisien, sementara calon mahasiswa mendapatkan **estimasi objektif** terhadap peluang mereka berdasarkan data numerik dan faktor akademik yang terukur.

**Problem Statements**

1. Bagaimana memodelkan hubungan antara profil akademik calon mahasiswa (seperti GRE, TOEFL, CGPA, dan faktor lainnya) dengan peluang diterima di program pascasarjana?
2. Fitur-fitur apa saja yang paling berpengaruh dalam menentukan peluang diterima?
3. Algoritma machine learning regresi mana yang paling akurat dan dapat diandalkan dalam memprediksi `Chance of Admit`?

**Goals**
- Membangun model machine learning berbasis regresi untuk memprediksi peluang diterima calon mahasiswa.
- Menganalisis faktor-faktor yang paling memengaruhi hasil prediksi.
- Membandingkan beberapa model regresi untuk memilih model dengan performa terbaik berdasarkan metrik evaluasi seperti MSE, RMSE, MAE, dan R² Score.
- Menyimpan model terbaik agar dapat digunakan kembali untuk prediksi di masa mendatang.
- Mengembangkan model *supervised regression* untuk memprediksi **Chance of Admit**.
- Melakukan analisis visual dan korelasi antar fitur.

**Solution Statements**
Untuk mencapai tujuan di atas, berikut pendekatan solusi yang dilakukan:
1. **Data Collection**  
   Menggunakan dataset *Graduate Admission Prediction* dari Kaggle yang berisi data historis penerimaan mahasiswa pascasarjana.

2. **Data Exploration & Preprocessing**  
   - Mengeksplorasi distribusi dan hubungan antar variabel.
   - Melakukan pembersihan data dan normalisasi fitur numerik.
   - Melakukan *Exploratory Data Analysis* untuk memahami pola data dan hubungan antar fitur.
   - Melakukan preprocessing seperti normalisasi dan penghapusan fitur yang tidak relevan.

3. **Model Building & Evaluation**  
   - Melatih beberapa model regresi: Linear Regression, Random Forest, dan Gradient Boosting.
   - Mengevaluasi performa masing-masing model menggunakan metrik regresi.

## Data Understanding & Exploratory Data Analysis (EDA)

**Informasi Dataset**
Dataset saya ambil dari sumber Kaggle yang berjudul Graduate Admission 2:

Sumber: Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019

Link: https://www.kaggle.com/datasets/mohansacharya/graduate-admissions

### Struktur Dataset

Dataset terdiri dari 500 data pelamar dan 9 fitur (termasuk target variabel). Berikut deskripsi masing-masing fitur:

| Nama Kolom          | Deskripsi                                              | Tipe Data | Rentang / Contoh |
|---------------------|--------------------------------------------------------|-----------|------------------|
| Serial No.          | Nomor urut data                                        | int       | 1, 2, 3, ...     |
| GRE Score           | Skor GRE (Graduate Record Examination)                 | int       | 290 - 340        |
| TOEFL Score         | Skor TOEFL (Test of English as a Foreign Language)     | int       | 92 - 120         |
| University Rating   | Peringkat universitas (1–5)                            | int       | 1 - 5            |
| SOP                 | Kekuatan *Statement of Purpose* (1–5)                  | float     | 1.0 - 5.0        |
| LOR                 | Kekuatan surat rekomendasi (1–5)                       | float     | 1.0 - 5.0        |
| CGPA                | IPK dalam skala 10                                     | float     | 6.8 - 9.92       |
| Research            | Pengalaman riset (0 = tidak, 1 = ya)                   | int       | 0, 1             |
| Chance of Admit     | Peluang diterima program master (target variabel)      | float     | 0.34 - 0.97      |

### Ukuran dan Tipe Data

- Dimensi dataset: **500 baris** × **9 kolom**
- Tipe data:
  - Integer: `GRE_Score`, `TOEFL_Score`, `University_Rating`, `Research`, `Serial_No.`
  - Float: `SOP`, `LOR`, `CGPA`, `Chance_of_Admit`

### Kondisi Data
| Pemeriksaan             | Hasil                     |
|-------------------------|---------------------------|
| Missing value           | Tidak ada                 |
| Duplikasi data          | Tidak ada                 |
| Tipe data sesuai        | Ya                        |

Dataset bersih dan siap digunakan untuk proses eksplorasi dan pemodelan, tanpa perlu tahap pembersihan yang kompleks.

### Statistik Deskriptif

Berikut ringkasan statistik dari fitur numerik:
| Fitur             | Mean     | Std Dev | Min   | 25%   | 50%   | 75%   | Max   |
|------------------|----------|---------|-------|-------|-------|-------|-------|
| GRE Score        | 316.47   | 11.30   | 290   | 308   | 317   | 325   | 340   |
| TOEFL Score      | 107.19   | 6.08    | 92    | 103   | 107   | 112   | 120   |
| University Rating| 3.11     | 1.14    | 1     | 2     | 3     | 4     | 5     |
| SOP              | 3.37     | 0.99    | 1.0   | 2.5   | 3.5   | 4.0   | 5.0   |
| LOR              | 3.48     | 0.93    | 1.0   | 3.0   | 3.5   | 4.0   | 5.0   |
| CGPA             | 8.58     | 0.60    | 6.8   | 8.13  | 8.56  | 9.04  | 9.92  |
| Research         | 0.56     | 0.50    | 0     | 0     | 1     | 1     | 1     |
| Chance of Admit  | 0.72     | 0.14    | 0.34  | 0.63  | 0.72  | 0.82  | 0.97  |

**Insight:**
- Sebagian besar nilai berada pada rentang yang wajar dan tidak ekstrem.
- Rata-rata Chance of Admit sebesar 0.72 menunjukkan banyak kandidat memiliki peluang cukup tinggi untuk diterima.
- Fitur Research bersifat biner, dengan distribusi hampir seimbang antara 0 dan 1.

### Korelasi Antar Fitur

Matriks korelasi menunjukkan kekuatan hubungan linier antar fitur:
| Fitur             | Korelasi dengan Chance of Admit |
|------------------|----------------------------------|
| CGPA             | **0.88**                         |
| GRE Score        | **0.81**                         |
| TOEFL Score      | **0.79**                         |
| SOP              | 0.68                             |
| LOR              | 0.66                             |
| University Rating| 0.71                             |
| Research         | 0.55                             |
| Serial No.       | ~0.00 *(tidak relevan)*          |

**Insight:**
- Fitur CGPA, GRE Score, dan TOEFL Score memiliki korelasi tertinggi terhadap peluang diterima, menjadikannya prediktor utama.
- Fitur Serial No. tidak memiliki kontribusi terhadap target dan akan dibuang sebelum proses modeling.

### Distribusi Fitur

- **GRE Score & TOEFL Score:** Distribusi normal (bell-shaped), cocok untuk regresi linier.
- **CGPA:** Distribusi juga normal, tanpa outlier signifikan.
- **SOP & LOR:** Distribusi agak merata dengan beberapa puncak karena skala diskrit.
- **University Rating:** Diskrit dan terpusat di nilai tertentu (3 dan 4).
- **Research:** Variabel biner (0 dan 1).
- **Chance of Admit:** Skew ke kiri (mayoritas memiliki peluang tinggi).

### Deteksi Outlier (Boxplot)

- **Outlier Terdeteksi:**  
  - LOR: 1 data di bawah minimum normal.  
  - Chance of Admit: 1 data outlier di bawah 0.4, namun masih valid.

- **Tidak Perlu Penghapusan Outlier:**
    - Outlier tergolong ringan dan tidak berdampak signifikan terhadap model regresi.
    - Semua data tetap digunakan karena masih mencerminkan kondisi riil dan valid secara kontekstual.

### Kesimpulan EDA
- Dataset lengkap, tidak memiliki missing value atau duplikasi.
- Fitur `Serial No.` tidak informatif dan akan dihapus.
- Fitur terpenting secara korelasi terhadap peluang diterima: **CGPA**, **GRE Score**, dan **TOEFL Score**.
- Distribusi fitur sesuai untuk digunakan dalam model regresi.
- Tidak diperlukan penanganan khusus terhadap outlier.

## Preprocessing / Data Preparation
Beberapa teknik *data preparation* yang saya lakukan untuk mempersiapkan data agar optimal untuk proses pemodelan:

### 1. Penghapusan Fitur Tidak Relevan
Fitur `Serial No.` dihapus dari dataset karena tidak memberikan kontribusi informasi terhadap prediksi. Fitur ini bersifat hanya sebagai penomoran baris dan tidak memiliki korelasi dengan target (`Chance of Admit`).

### 2. Feature Scaling (Standardization)
Karena setiap fitur memiliki skala nilai yang berbeda (misalnya `GRE Score` dalam skala 260–340, sedangkan `CGPA` dalam skala 0–10), dilakukan proses standardisasi menggunakan **StandardScaler** pada semua fitur yang digunakan. Ini bertujuan untuk memastikan semua fitur berada dalam skala yang setara (rata-rata 0 dan standar deviasi 1), sehingga model tidak bias terhadap fitur tertentu dengan rentang nilai yang lebih besar. sebelum masuk ke tahap selanjutnya yaitu splitting.

### 3. Pemisahan Fitur dan Target
Data dibagi menjadi dua bagian utama:
    - Fitur (X): Semua kolom yang merepresentasikan karakteristik kandidat, seperti `GRE Score`, `TOEFL Score`, `University Rating`, `SOP`, `LOR`, `CGPA`, dan `Research`.
    - Target (y): Kolom `Chance of Admit` yang menjadi fokus prediksi.

### 4. Pembagian Data Latih dan Uji
Dataset dibagi menjadi:
    - **80% Data Latih (Training Set)**: Digunakan untuk melatih model.
    - **20% Data Uji (Testing Set)**: Digunakan untuk menguji performa model terhadap data yang belum pernah dilihat sebelumnya.

Pembagian dilakukan secara acak namun terkontrol dengan menetapkan `random_state` agar hasil pembagian konsisten di setiap eksekusi.

Data preparation penting agar:
- Menghilangkan fitur yang tidak relevan untuk mencegah noise dan overfitting.
- Menyeragamkan skala fitur agar model tidak bias terhadap fitur berskala besar.
- Meningkatkan stabilitas dan performa model, terutama algoritma regresi dan distance-based.
- Memastikan evaluasi model adil dan tidak bias, dengan membagi data secara proporsional.
- Meningkatkan akurasi prediksi, karena model dilatih dari data yang bersih dan representatif.

### Jumlah Data dan Kolom
Dataset yang digunakan memiliki jumlah 500 baris dan juga 8 kolom

## Modeling
Pada tahap ini, dilakukan pembangunan dan evaluasi terhadap tiga algoritma regresi: Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor. Tujuannya adalah menentukan model terbaik berdasarkan performa prediksi terhadap target Chance of Admit.

### Model 1: Linear Regression
**Cara Kerja**
Linear Regression memodelkan hubungan linier antara fitur dan target. Model mencari garis terbaik (hiperplane dalam dimensi tinggi) yang meminimalkan jumlah kuadrat selisih antara nilai aktual dan nilai prediksi.

**Parameter**
- Model ini tidak memiliki banyak parameter utama; bekerja dengan estimasi koefisien linier untuk setiap fitur.
- Model yang digunakan: LinearRegression() dari sklearn.linear_model.

**Kelebihan**
- Cepat dan efisien untuk dataset kecil hingga menengah.
- Hasilnya mudah diinterpretasikan.
- Cocok sebagai baseline model.

**Kekurangan**
- Tidak mampu menangkap hubungan non-linear antar fitur.
- Rentan terhadap multikolinearitas dan outlier ekstrem.

### Model 2: Random Forest Regressor

**Cara Kerja**
Random Forest adalah model **ensemble** berbasis pohon keputusan. Ia membangun banyak decision tree pada subset data dan fitur, lalu menggabungkan hasil prediksi dari semua pohon (rata-rata untuk regresi).

**Parameter Penting**
- `n_estimators`: jumlah pohon (default: 100).
- `max_depth`: kedalaman maksimum pohon (jika tidak ditentukan, tumbuh hingga sempurna).
- `random_state=42`: untuk memastikan hasil reprodusibel.

**Kelebihan**
- Dapat menangani hubungan non-linear.
- Tahan terhadap overfitting.
- Dapat mengukur pentingnya fitur (feature importance).

**Kekurangan**
- Lebih lambat dibanding Linear Regression.
- Interpretasi model lebih sulit karena merupakan "black-box".

### Model 3: Gradient Boosting Regressor

**Cara Kerja**
Gradient Boosting bekerja dengan membangun model secara **berurutan**. Setiap model baru memperbaiki kesalahan dari model sebelumnya menggunakan teknik **gradien penurunan kesalahan**. Model akhir merupakan kombinasi dari semua model lemah (biasanya decision tree kecil).

**Parameter Penting**
- `n_estimators`: jumlah boosting stage (default: 100).
- `learning_rate`: seberapa besar kontribusi setiap model baru terhadap total prediksi.
- `max_depth`: kedalaman maksimal setiap pohon.
- `random_state=42`: untuk konsistensi hasil.

**Kelebihan**
- Sangat akurat untuk regresi kompleks.
- Menangani outlier lebih baik dari regresi linier.
- Dapat mengatasi interaksi antar fitur.

**Kekurangan**
- Lebih lambat saat training dibanding model lain.
- Perlu tuning parameter yang lebih teliti.
- Risiko overfitting jika jumlah estimator terlalu banyak.

## Evaluation performa semua Model
| Model                       | MSE          | RMSE         | MAE          | R² Score     |
| --------------------------- | ------------ | ------------ | ------------ | ------------ |
| **Linear Regression**       | 0.0037       | 0.0609       | 0.0427       | 0.8188       |
| Random Forest Regressor     | 0.0043       | 0.0654       | 0.0437       | 0.7908       |
| Gradient Boosting Regressor | 0.0044       | 0.0667       | 0.0462       | 0.7826       |

**Kesimpulan Sementara:**
Model **Linear Regression** memberikan hasil terbaik secara keseluruhan, dengan nilai R² tertinggi dan error terendah. Meskipun model ini sederhana, performanya sangat baik dalam memodelkan hubungan linier antara fitur dan peluang diterima (`Chance of Admit`).

## Cross-Validation dan Hyperparameter Tuning

### Cross-Validation
Untuk memastikan kestabilan dan konsistensi performa model, dilakukan **5-Fold Cross-Validation** pada ketiga model utama:

-  Linear Regression
-  Random Forest Regressor
-  Gradient Boosting Regressor

**Hasil Cross-Validation:**
| Model             | R² Mean | Std Dev |
| ----------------- | ------- | ------- |
| Linear Regression | 0.8108  | 0.0756  |
| Random Forest     | 0.7743  | 0.0789  |
| Gradient Boosting | 0.7825  | 0.0880  |

**Insight:**
Linear Regression menunjukkan performa paling stabil dan terbaik dengan nilai rata-rata R² tertinggi (0.8108), mendukung kesimpulan bahwa hubungan antara fitur dan target bersifat linier.

## Hyperparameter Tuning pada Random Forest
Untuk meningkatkan performa model berbasis pohon, dilakukan **Grid Search** pada Random Forest dengan parameter berikut:

| Parameter           | Nilai Dicoba |
| ------------------- | ------------ |
| n\_estimators       | 50, 100      |
| max\_depth          | None, 5, 10  |
| min\_samples\_split | 2, 5         |

**Hasil terbaik dari Grid Search:**
| Parameter           | Nilai Terbaik |
| ------------------- | ------------- |
| max\_depth          | 5             |
| min\_samples\_split | 5             |
| n\_estimators       | 50            |

Setelah melakukan tuning pada **Random Forest** hasil terbaik terdapat pada `max_depth=5`, `min_samples_split=5`, dan `n_estimators=50` dengan R² Score: **0.7863**

**Insight:**
Tuning parameter Random Forest menghasilkan sedikit peningkatan performa, namun masih di bawah Linear Regression. Ini menegaskan bahwa model linier lebih sesuai untuk dataset ini.

## Evaluasi Project dan Kesimpulan Akhir
Proyek ini bertujuan membangun model prediksi peluang diterima kuliah (Chance of Admit) berdasarkan berbagai fitur akademik dan non-akademik, antara lain GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, dan pengalaman riset. Tahapan utama yang dilakukan meliputi eksplorasi data (EDA), persiapan data (preprocessing dan standarisasi), pembangunan dan evaluasi beberapa model regresi, serta penerapan cross-validation dan hyperparameter tuning untuk meningkatkan performa model.

Model yang telah dibuat maka dievaluasi menggunakan beberapa metrik berikut ini:
- **Mean Absolute Error (MAE)**  
  \[
  MAE = $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|\$ 
  \]
  Mengukur seberapa jauh prediksi dari nilai aktual secara rata-rata dalam satuan absolut.
- **Mean Squared Error (MSE)**
  \[
  MSE = $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
  \]  
  Memberi penalti lebih besar pada kesalahan besar.
- **R-squared (R² Score)**  
  \[
  R^2 = $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
  \]  
- R² Score (koefisien determinasi)

**Hasil perbandingan output dari setiap hasil model**
| Model                       | MSE    | RMSE   | MAE    | R² Score   | R² (CV Mean) | Std Dev (CV) | Catatan                                                    |
| --------------------------- | ------ | ------ | ------ | ---------- | ------------ | ------------ | -----------------------------------------------------------|
| **Linear Regression**       | 0.0037 | 0.0609 | 0.0427 | 0.8188     | **0.8108**   | 0.0756       | Performa & konsistensi terbaik                             |
| Random Forest Regressor     | 0.0043 | 0.0654 | 0.0437 | 0.7908     | 0.7743       | 0.0789       | Performa cukup baik, belum dituning (Cross-Validation)     |
| Gradient Boosting Regressor | 0.0044 | 0.0667 | 0.0462 | 0.7826     | 0.7825       | 0.0880       | Stabil namun performa masih di bawah LR (Cross-Validation) |
| **Random Forest (Tuned)**   | –      | –      | –      | **0.7863** | –            | –            | Setelah tuning, tetap di bawah LR                          |

Berdasarkan hasil akhir dari proses modeling, cross validation dan juga hyperparameter tuning dapat dilihat bahwa model **Linear Regression** bisa dipilih sebagai model akhir karena:
* Memiliki akurasi tertinggi dengan R² Score terbaik dan nilai error terendah.
* Performa yang konsisten berdasarkan hasil cross-validation.
* Sifatnya yang sederhana dan mudah diinterpretasikan untuk kebutuhan bisnis dan pengambilan keputusan.

**Hubungan dengan Business Understanding**
Evaluasi model terhadap problem statement dengan memberikan prediksi peluang diterima kuliah secara akurat dan konsisten
- **Mendukung Seleksi Mahasiswa:** Model prediksi membantu universitas melakukan seleksi awal yang lebih objektif dan efisien berdasarkan profil kandidat.
- **Informasi untuk Calon Mahasiswa:** Membantu calon mahasiswa menilai peluang diterima dan merencanakan persiapan akademik dengan lebih baik.
- **Efisiensi Operasional:** Mengoptimalkan proses seleksi dan mengurangi beban kerja manual serta bias subjektif.
- **Pengembangan Sistem Rekomendasi:** Bisa dijadikan dasar sistem rekomendasi dan konsultasi akademik yang memberikan saran personalisasi.
- **Peningkatan Mutu Pendidikan:** Memberikan insight untuk evaluasi dan pengembangan program peningkatan kemampuan calon mahasiswa serta kebijakan penerimaan.

**Potensi Perbaikan dan Pengembangan**
Untuk pengembangan proyek di masa depan, beberapa rekomendasi berikut dapat dipertimbangkan:
- Menggunakan dataset yang lebih besar dan lebih beragam guna meningkatkan kemampuan generalisasi model.
- Menambahkan fitur baru yang relevan, seperti latar belakang pendidikan, aktivitas ekstrakurikuler, atau data psikometrik.
- Mengeksplorasi model yang lebih kompleks dan modern seperti XGBoost, LightGBM, atau neural networks.
- Melakukan feature engineering dan seleksi fitur secara lebih mendalam untuk meningkatkan kualitas data input.

**Rekomendasi Implementasi Model**
Berdasarkan hasil proyek, model Linear Regression direkomendasikan untuk implementasi karena memberikan performa terbaik dengan akurasi tinggi dan konsistensi yang stabil, serta kemudahan interpretasi hasil prediksi. Untuk pengembangan lebih lanjut, disarankan menggabungkan model ini dalam sistem pendukung keputusan penerimaan mahasiswa, melengkapi dengan fitur tambahan dan pipeline otomatisasi agar proses prediksi dapat berjalan efisien dan mudah diintegrasikan

## Referensi
1. Ismail, A., & Elnagar, A. (2021). *Graduate Admission Prediction Using Machine Learning*. ResearchGate.
   [https://www.researchgate.net/publication/348433004\_Graduate\_Admission\_Prediction\_Using\_Machine\_Learning](https://www.researchgate.net/publication/348433004_Graduate_Admission_Prediction_Using_Machine_Learning)

2. Mohan S Acharya, Asfia Armaan, Aneeta S Antony. (2019). *A Comparison of Regression Models for Prediction of Graduate Admissions*. IEEE International Conference on Computational Intelligence in Data Science. Kaggle:
   [https://www.kaggle.com/mohansacharya/graduate-admissions](https://www.kaggle.com/mohansacharya/graduate-admissions)

3. Scikit-learn documentation. *Supervised learning*.
   [https://scikit-learn.org/stable/supervised\_learning.html](https://scikit-learn.org/stable/supervised_learning.html)