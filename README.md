# SISTEM PERINGATAN DINI RESIKO DROP OUT MAHASISWA
##### Kelompok 9:
##### - Neli Agustin (G1A022048)
##### - Rizki Ramadani Dalimunthe (G1A022054)
##### - Yuda Reyvandra Herman (G1A022072)


#### Link Streamlit [Early Warning Drop Out Mahasiswa](https://team9-data-mining.streamlit.app/)

## Gambaran Umum Proyek
Proyek ini mengembangkan sistem peringatan dini untuk mendeteksi mahasiswa berisiko tinggi drop out (DO). Dengan mengenali mereka lebih awal, universitas dapat segera memberikan dukungan dan intervensi yang tepat. Tujuannya meningkatkan keberhasilan studi mahasiswa dan menekan angka DO. Sistem ini dibangun menggunakan data akademik dan non-akademik mahasiswa.

## Metode Data Mining: Random Forest
Dipilihnya Random Forest Classifier sebagai metode utama didasari beberapa keunggulan:
- Sangat Akurat: Metode ini dikenal memberikan hasil prediksi yang akurat, bahkan untuk data yang kompleks.
- Anti Overfitting: Model efektif dalam mengurangi risiko overfitting, memastikan generalisasi yang baik pada data baru.
- Fleksibel: Cocok untuk berbagai jenis data, baik angka maupun kategori.
- Identifikasi Faktor Penting: Model ini bisa menunjukkan fitur mana yang paling berpengaruh dalam memprediksi risiko DO, memberikan wawasan berharga untuk intervensi.

## Dataset
Dataset yang digunakan adalah dataset dummy yang dibuat secara sintetis, meniru karakteristik data mahasiswa sungguhan. Dataset ini mencakup 5.000 entri mahasiswa dengan berbagai informasi relevan yang diyakini memengaruhi risiko DO.

## Fitur-fitur (Kolom) dalam Dataset:
Setiap baris dalam dataset merepresentasikan satu mahasiswa, dengan kolom-kolom berikut sebagai fiturnya:
- NIM: Nomor Induk Mahasiswa. Ini adalah pengidentifikasi unik untuk setiap mahasiswa.
- IPK_Semester_1 s/d IPK_Semester_6: Indeks Prestasi Kumulatif (IPK) mahasiswa untuk setiap semester, dari semester 1 hingga semester 6. Ini mensimulasikan kinerja akademik mahasiswa dari waktu ke waktu.
- Kehadiran_Per_Mata_Kuliah: Rata-rata persentase kehadiran mahasiswa di seluruh mata kuliah. Fitur ini penting untuk mengukur tingkat komitmen dan partisipasi di kelas.
- Riwayat_Pengambilan_Ulang: Jumlah mata kuliah yang pernah diulang oleh mahasiswa. Pengulangan mata kuliah seringkali menjadi indikator kesulitan akademik.
- Aktivitas_Sistem_Pembelajaran_Daring: Skor yang mencerminkan tingkat aktivitas atau keterlibatan mahasiswa dalam sistem pembelajaran daring (misalnya, LMS). Ini bisa meliputi frekuensi login, partisipasi forum, atau penyelesaian tugas daring.
- Status_Pekerjaan: Status pekerjaan mahasiswa, yang bisa berupa "Bekerja" atau "Tidak Bekerja". Ini adalah faktor non-akademik yang dapat memengaruhi fokus studi.
- Beban_Kerja_JamPerMinggu: Jumlah jam kerja per minggu jika mahasiswa berstatus "Bekerja". Beban kerja yang tinggi dapat memengaruhi waktu dan energi untuk belajar.
- Status_Risiko_DO: Ini adalah variabel target kita, yang mengindikasikan apakah mahasiswa tersebut memiliki risiko tinggi untuk drop out ("Risiko Tinggi") atau tidak ("Aman"). Penentuan status ini didasarkan pada logika skor yang menggabungkan beberapa fitur lainnya.

Dataset ini dirancang untuk memberikan variasi yang cukup realistis agar dapat digunakan untuk pelatihan model data mining dalam memprediksi risiko DO.

## Metodologi Proyek: CRISP-DM
Proyek ini mengikuti kerangka kerja CRISP-DM (Cross-Industry Standard Process for Data Mining), sebuah pendekatan standar industri yang terbukti efektif untuk proyek data mining. CRISP-DM terdiri dari enam fase utama:
1. Pemahaman Bisnis (Business Understanding):
- Identifikasi Tujuan: Tahap awal ini fokus pada pemahaman mendalam tentang mengapa proyek ini penting. Tujuannya jelas: membangun sistem peringatan dini untuk mahasiswa berisiko DO agar universitas bisa memberikan intervensi tepat waktu, yang pada akhirnya meningkatkan keberhasilan studi dan mengurangi angka DO.
- Penentuan Kriteria Sukses: Ditentukan metrik keberhasilan yang jelas, baik dari sisi teknis (akurasi model, recall, precision, F1-Score, AUC) maupun sisi bisnis (penurunan angka DO yang terukur setelah implementasi sistem).
- Perencanaan Skenario Penggunaan: Sistem akan secara periodik menganalisis data mahasiswa. Hasil prediksinya akan disalurkan ke departemen kemahasiswaan atau penasihat akademik, dan rangkuman tren akan disajikan melalui dashboard interaktif kepada rektorat.

2. Pemahaman Data (Data Understanding):
- Pengumpulan Data Awal: Data dikumpulkan dari berbagai sumber internal yang relevan, seperti Sistem Informasi Akademik (untuk IPK, riwayat pengulangan), Sistem Absensi (untuk kehadiran), dan Sistem Pembelajaran Daring (untuk aktivitas online).
- Deskripsi Data Detail: Setiap kolom data dianalisis untuk memahami tipe data (numerik/kategorikal), rentang nilai, dan maknanya (misalnya, NIM sebagai pengidentifikasi unik, IPK sebagai indikator kinerja akademik).
- Eksplorasi Data Awal (EDA): Analisis statistik deskriptif dan visualisasi (histogram, count plot, grafik korelasi) dilakukan untuk memahami distribusi variabel, mengidentifikasi pola, potensi outliers, dan memeriksa keseimbangan kelas pada variabel target (Status_Risiko_DO).

3. Persiapan Data (Data Preparation):
- Pemilihan Data: Fitur-fitur yang paling relevan dari dataset dipilih untuk pemodelan. Kolom NIM dihilangkan sebelum pelatihan model karena berfungsi sebagai pengidentifikasi, bukan fitur prediktif.
- Pembersihan Data:
Penanganan Duplikat: Baris data duplikat, jika ditemukan, dihapus berdasarkan NIM untuk memastikan setiap entri mahasiswa unik.
Penanganan Missing Values: Dilakukan pengecekan missing values pada setiap kolom. Dalam dataset dummy ini, tidak ada missing values yang ditemukan, sehingga tidak diperlukan operasi pengisian. Namun, jika ada, missing values pada fitur numerik akan diisi dengan median, dan pada fitur kategorikal dengan modus.
- Transformasi Data:
Encoding Kategorikal: Variabel target Status_Risiko_DO diubah menjadi representasi numerik (0 untuk 'Aman', 1 untuk 'Risiko Tinggi') menggunakan LabelEncoder. Fitur Status_Pekerjaan juga di-encode menjadi angka (0 untuk 'Tidak Bekerja', 1 untuk 'Bekerja') menggunakan LabelEncoder. Kolom asli yang sudah di-encode dihapus.
Feature Scaling: Semua fitur numerik distandarisasi menggunakan StandardScaler. Proses ini mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1, memastikan fitur dengan rentang nilai yang berbeda tidak mendominasi perhitungan model.
- Pembagian dan Penanganan Data Tidak Seimbang: Dataset dibagi menjadi set pelatihan dan set pengujian. SMOTE (Synthetic Minority Over-sampling Technique) diterapkan pada data pelatihan yang sudah di-scale untuk menyeimbangkan jumlah sampel antara kelas 'Aman' dan 'Risiko Tinggi'. Ini sangat penting agar model tidak bias terhadap kelas mayoritas dan mampu mendeteksi risiko dengan lebih baik.

4. Pemodelan (Modeling):
- Pemilihan Algoritma: Menggunakan RandomForestClassifier.
- Pelatihan Model: Model dilatih pada data pelatihan yang sudah distandarisasi dan diseimbangkan dengan SMOTE.
- Hyperparameter Tuning: GridSearchCV digunakan untuk mencari kombinasi hyperparameter terbaik bagi Random Forest. Ini melibatkan pengujian berbagai nilai untuk parameter seperti n_estimators (jumlah pohon), max_depth (kedalaman maksimum pohon), min_samples_split (jumlah minimum sampel untuk membagi node), min_samples_leaf (jumlah minimum sampel di daun), dan max_features (jumlah fitur yang dipertimbangkan untuk setiap split). GridSearchCV melakukan cross-validation untuk menemukan kombinasi yang optimal.

5. Evaluasi (Evaluation):
1.) Metrik Kinerja Komprehensif: Model dievaluasi menggunakan berbagai metrik pada data pengujian yang belum diseimbangkan untuk memberikan gambaran kinerja yang realistis:
   - Confusion Matrix: Tabel yang menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN). Ini krusial untuk memahami jenis kesalahan model.
   - Recall (Sensitivitas): TP/(TP+FN). Metrik paling penting dalam konteks ini, menunjukkan seberapa banyak mahasiswa berisiko yang berhasil dideteksi.
   - Precision (Presisi): TP/(TP+FP). Mengukur seberapa akurat prediksi 'Risiko Tinggi'.
   - F1-Score: Rata-rata harmonik precision dan recall, sangat berguna untuk data tidak seimbang.
   - Balanced Accuracy: Rata-rata recall dari setiap kelas, memberikan akurasi yang lebih adil.
   - ROC AUC Score: Mengukur kemampuan model dalam membedakan kelas positif dan negatif.=
   - Matthews Correlation Coefficient (MCC): Metrik tunggal yang robust untuk data tidak seimbang.

   2.) Feature Importance: Visualisasi feature importance dari model Random Forest dianalisis untuk mengidentifikasi fitur-fitur yang paling berpengaruh dalam prediksi risiko DO, memberikan wawasan berharga bagi universitas.

   3.) Cross-Validation Score: Melakukan cross-validation pada model terbaik menggunakan data pelatihan yang seimbang untuk mengonfirmasi konsistensi dan robustness kinerjanya.

6. Penyebaran (Deployment):
- Penyimpanan Model: Model Random Forest terbaik dan objek scaler disimpan ke file .pkl agar dapat dimuat dan digunakan kembali di lingkungan produksi.
- Integrasi Sistem: Merencanakan pembangunan API ringan yang memungkinkan sistem informasi akademik universitas mengirim data mahasiswa dan menerima prediksi risiko secara otomatis.
- Automasi Alur Kerja: Mengembangkan alur kerja otomatis untuk mengumpulkan, memproses, dan memasukkan data terbaru ke dalam model prediksi secara berkala.
- Sistem Peringatan Dini: Mengimplementasikan notifikasi otomatis (misalnya, email) kepada penasihat akademik atau departemen kemahasiswaan saat mahasiswa teridentifikasi berisiko tinggi.
- Visualisasi dan Dashboard: Membuat dashboard interaktif (menggunakan tools seperti Power BI atau Streamlit) untuk rektorat dan manajemen, menampilkan tren risiko DO, faktor pendorong utama, dan distribusi risiko.

##  Preprocessing Data (Persiapan Data)
Tahap persiapan data sangat krusial untuk memastikan data bersih dan siap pakai untuk pemodelan. Ini adalah serangkaian langkah sistematis yang dilakukan sebelum data digunakan untuk melatih model:
1. Penanganan Duplikat:
- Tujuan: Memastikan setiap entri mahasiswa dalam dataset adalah unik.
- Implementasi: Baris data duplikat, jika ditemukan, dihapus berdasarkan NIM (Nomor Induk Mahasiswa) sebagai identifikasi tunggal mahasiswa.
- 
2. Penanganan Missing Values:
- Tujuan: Mengisi atau mengatasi nilai-nilai yang hilang dalam dataset agar tidak mengganggu proses pemodelan.
- Implementasi: Pengecekan missing values dilakukan pada setiap kolom. Dalam dataset dummy ini, tidak ada missing values yang terdeteksi, sehingga operasi pengisian tidak diperlukan. Namun, dalam skenario data nyata, missing values pada fitur numerik umumnya diisi dengan median (untuk mengurangi dampak outlier), sementara pada fitur kategorikal diisi dengan modus (nilai yang paling sering muncul).

3. Encoding Kategorikal:
- Tujuan: Mengubah variabel kategorikal (teks) menjadi format numerik yang dapat dipahami oleh algoritma machine learning.
- Implementasi:
Variabel target Status_Risiko_DO ("Aman", "Risiko Tinggi") diubah menjadi representasi numerik: 0 untuk "Aman" dan 1 untuk "Risiko Tinggi" menggunakan LabelEncoder.
Fitur Status_Pekerjaan ("Bekerja", "Tidak Bekerja") juga di-encode menjadi angka: 0 untuk "Tidak Bekerja" dan 1 untuk "Bekerja" menggunakan LabelEncoder.
Kolom asli Status_Risiko_DO dan Status_Pekerjaan kemudian dihapus dari dataset setelah proses encoding.

4. Feature Scaling:
- Tujuan: Menstandardisasi rentang nilai fitur-fitur numerik. Ini penting agar fitur dengan rentang nilai besar tidak secara tidak proporsional mendominasi perhitungan model.
- Implementasi: Semua fitur numerik dalam dataset distandarisasi menggunakan StandardScaler. Proses ini mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1.

5. Penanganan Data Tidak Seimbang (Imbalanced Data):
- Tujuan: Mengatasi masalah ketidakseimbangan kelas pada variabel target, di mana satu kelas memiliki jumlah sampel jauh lebih sedikit dibandingkan kelas lainnya (misalnya, lebih banyak mahasiswa "Aman" daripada "Risiko Tinggi"). Ketidakseimbangan ini dapat membuat model bias dan kurang efektif dalam memprediksi kelas minoritas.
- Implementasi: SMOTE (Synthetic Minority Over-sampling Technique) diterapkan pada data pelatihan yang sudah di-scale. SMOTE bekerja dengan membuat sampel sintetis dari kelas minoritas, sehingga menyeimbangkan jumlah sampel antara kelas 'Aman' dan 'Risiko Tinggi'. Hal ini membantu model belajar lebih baik dari kelas minoritas dan meningkatkan kemampuannya mendeteksi risiko dengan lebih akurat.

## Hasil dan Evaluasi Model
Model Random Forest yang terlatih menunjukkan kinerja yang kuat:
Akurasi (Test Set): 0.99
Classification Report:

                 precision    recall  f1-score   support
          0       1.00      0.99      0.99       943
          1       0.89      0.95      0.92        57
          
    accuracy                           0.99      1000
    macro avg       0.94      0.97      0.95      1000
    weighted avg       0.99      0.99      0.99      1000

Confusion Matrix:

       [[936   7]  # True Negative: 936 (Aman diprediksi Aman)
       [  3  54]] # False Negative: 3 (Risiko diprediksi Aman); True Positive: 54 (Risiko diprediksi Risiko)

Balanced Accuracy: 0.9700
Matthews Correlation Coefficient (MCC): 0.9105
ROC AUC Score: Sekitar 0.9958 (nilai dapat sedikit bervariasi karena randomness dalam pembuatan data).

Hasil ini menunjukkan bahwa model memiliki kemampuan yang sangat baik dalam mendeteksi mahasiswa berisiko tinggi (Recall 0.95 untuk kelas 1) sambil tetap menjaga precision yang tinggi.

## Rencana Pengembangan Sistem ke Depan
Untuk menjadikan sistem ini lebih fungsional dan berdampak, rencana pengembangan meliputi:
1. Integrasi dengan Sistem Akademik: Membangun API untuk memungkinkan sistem informasi universitas mengirim data mahasiswa dan menerima prediksi risiko secara otomatis.
2. Dashboard Visual Interaktif: Membuat dashboard (menggunakan Power BI, Tableau, Dash, atau Streamlit) untuk rektorat dan penasihat. Dashboard ini akan menampilkan tren risiko DO, faktor-faktor pendorong utama (berdasarkan feature importance), dan distribusi risiko berdasarkan kriteria tertentu.
3. Peningkatan Berkelanjutan: Menerapkan feedback loop dengan mengumpulkan data hasil intervensi. Model akan dilatih ulang secara berkala dengan data terbaru untuk mempertahankan akurasi dan beradaptasi dengan perubahan tren.
4. Explanaible AI (XAI): Menerapkan teknik seperti SHAP atau LIME untuk memberikan penjelasan yang lebih detail mengapa seorang mahasiswa diprediksi berisiko, membantu penasihat memberikan bimbingan yang lebih personal.
