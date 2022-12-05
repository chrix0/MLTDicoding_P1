# Laporan Proyek Machine Learning - Chris Tianto Pratama
## Domain Proyek

Kesehatan merupakan salah satu indikator penting bagi tercapainya kehidupan yang sejahtera bagi suatu masyarakat. Kondisi tubuh yang sehat mendukung manusia untuk melakukan aktivitas sehari-hari dengan baik. Namun, tidak ada yang bisa menduga kapan tubuh akan terserang penyakit [2], [3].  

Biaya pengobatan yang tinggi menyebabkan tidak semua anggota  masyarakat  mampu  memperoleh  pelayanan  kesehatan  yang  layak. Golongan miskin, terutama, tidak  memiliki kemampuan untuk membayar biaya kesehatan sehingga mereka mengalami musibah ganda manakala sakit, sehingga mereka tidak bisa segera memulihkan kesehatan dan tidak bisa memperoleh penghasilan [4]. Oleh karena itu, diperlukan perlindungan untuk meringankan beban mengenai biaya  yang  ditanggung  seseorang saat terkena penyakit, yaitu asuransi kesehatan [3].

Asuransi kesehatan merupakan jaminan mengatasi risiko atas pembiayaan dan perawatan kesehatan. Kepemilikan asuransi kesehatan (pemerintah atau swasta) merupakan jaminan kesehatan kepada masyarakat agar dapat menjangkau fasilitas pelayanan kesehatan. Dengan adanya asuransi kesehatan, masyarakat dapat mengurangi biaya pengobatan dengan membayar uang asuransi yang biasa disebut dengan iuran atau premi dengan syarat dan ketentuan yang berlaku. Banyaknya masyarakat yang belum memiliki asuransi kesehatan akan dapat menyebabkan peningkatan pembiayaan kesehatan [2].

Penentuan biaya premi asuransi kesehatan bersifat tidak mudah karena banyaknya variabel-variabel yang harus diperhatikan. Beberapa variabel di antaranya adalah umur pemegang polis asuransi, jenis kelamin, BMI (*Body Mass Index*), jumlah anak yang ditanggung oleh asuransi kesehatan, dan status merokok. Variabel-variabel tersebut memiliki pengaruh dalam menentukan premi asuransi. Namun, setiap variabel yang digunakan memiliki pengaruh yang berbeda terhadap hasil penentuan [1].

Masalah ini harus diselesaikan karena analisis premi asuransi kesehatan merupakan bagian dari permasalahan dalam *big data* untuk bidang terkait, sehingga diperlukan pendekatan *Computational Intelligence* (CI) untuk menangani jumlah data yang sangat banyak.  Salah satu solusi dengan pendekatan CI adalah dengan mengembangkan fitur perangkat lunak yang dapat memprediksi premi asuransi kesehatan setiap pemegang polis berdasarkan variabel-variabel yang ditentukan dengan memanfaatkan *machine learning* [1]. Untuk mengembangkan model *machine learning* yang dapat memprediksi premi asuransi, pengembangan harus melalui beberapa tahap, yaitu menganalisis variabel-variabel yang digunakan, menyiapkan data, mengembangkan model, dan mengevaluasi model. Dalam tahap evaluasi model, pengembang akan melakukan perbandingan terhadap lima algoritma, yaitu K-Nearest Neighbor (KNN), Random Forest (RF), AdaBoost, *Gradient Boosting*, dan LGBM *Regression* untuk menenentukan model dengan algoritma terbaik untuk kasus ini.

## Business Understanding

### Problem Statements

- Fitur / variabel apa yang paling berpengaruh terhadap premi asuransi kesehatan?
- Berapa biaya / premi asuransi kesehatan dengan karakteristik pemegang polis asuransi kesehatan tertentu?
- Algoritma apa yang dapat memprediksi premi asuransi kesehatan dengan akurasi terbaik?

### Goals

- Mengetahui fitur / variabel yang memiliki kolerasi dengan premi asuransi kesehatan.
- Membuat model machine learning yang dapat memprediksi premi asuransi kesehatan.
- Mengetahui algoritma yang menghasilkan prediksi dengan akurasi tinggi dalam memprediksi premi asuransi kesehatan.

### Solution statements

- Untuk dapat mengetahui variabel yang paling berpengaruh terhadap premi asuransi kesehatan, Univariate dan Multivariate Analysis dapat dilakukan untuk memahami variabel-variabel serta hubungan atau kolerasinya dengan variabel lain.
- Untuk dapat memprediksi biaya / premi asuransi kesehatan, model yang digunakan harus menggunakan algoritma yang sesuai dan dapat digunakan untuk melakukan prediksi premi asuransi kesehatan.
- Untuk mengetahui algoritma terbaik untuk memprediksi premi asuransi kesehatan, dilakukan perbandingan terhadap lima algoritma machine learning yang telah dioptimasi dengan hyperparemeter tuning (Grid Search), yaitu K-Nearest Neighbor (KNN), Random Forest (RF), AdaBoost, Gradient Boosting, dan LBGM Regression dengan metrik Mean Squared Error (MSE) dan Mean Absolute Error (MAE).

## Data Understanding

Berikut informasi dari dataset yang digunakan:

Tabel 1. Informasi dataset
|   |   |
|---|---|
|__Nama dataset__| Medical Cost Personal Datasets |
|__Deskripsi dataset__| Insurance Forecast by using Linear Regression |
|__Jumlah sampel__| 1338 |
|__Jumlah variabel__| 7 |

Dataset ini berisi data pemegang polis asuransi kesehatan yang berasal dari Amerika Serikat dengan karakteristik berbeda beserta premi asuransi yang dibayar.

Link download dataset: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance?datasetId=13720).


### Exploratory Data Analysis

#### Deksripsi Variabel

Berikut variabel / fitur yang terdapat pada Medical Cost Personal Dataset.

Tabel 2. Deskripsi dan tipe data variabel
| Nama fitur | Deskripsi | Tipe data |
|---|---|---|
|age| Umur pemegang polis asuransi kesehatan. |int64|
|sex| Jenis kelamin pemegang polis asuransi kesehatan. |object|
|bmi| Indeks massa tubuh pemegang polis asuransi kesehatan. |float64|
|children| Jumlah anak yang ditanggung oleh asuransi kesehatan. |int64|
|smoker| Menunjukkan apakah pemegang polis asuransi kesehatan merupakan perokok. |object|
|region| Daerah perumahan pemegang polis asuransi kesehatan di Amerika Serikat (northeast, southeast, southwest, northwest). |object|
|charges| Iuran atau premi asuransi kesehatan yang harus dibayar oleh pemegang polis asuransi. |float64 |

Tabel 2 menunjukkan bahwa:
- Terdapat 2 fitur numerik dengan tipe data int64, yaitu age dan children.
- Terdapat 2 fitur numerik dengan tipe data float64, yaitu bmi dan charges. Kolom charges akan digunakan sebagai target prediksi.
- Terdapat 3 fitur kategori dengan tipe data object, yaitu sex, smoker, dan region.


Beirkut statistik deskriptif setiap fitur numerik.

Tabel 3. Statistik deskriptif fitur numerik
| | age | bmi | children | charges |
|---|---|---|---|---|
|__count__| 1338.000000	| 1338.000000 |	1338.000000	| 1338.000000 |
|__mean__| 39.207025 | 30.663397 | 1.094918	| 13270.422265 |
|__std__| 14.049960	| 6.098187	| 1.205493	| 12110.011237 |
|__min__| 18.000000	| 15.960000 | 0.000000 | 1121.873900 |
|__25%__| 27.000000	| 26.296250	| 0.000000 | 4740.287150 |
|__50%__| 39.000000 | 30.400000 | 1.000000 | 9382.033000 |
|__75%__| 39.000000 | 30.400000 | 1.000000 | 9382.033000 |
|__max__| 64.000000	| 53.130000 | 5.000000 | 63770.428010 |


#### Memeriksa data dengan missing value
Dalam tahap ini dilakukan pemeriksaan data dengan missing value pada dataset. Dataset "Medical Cost Personal Datasets" tidak memiliki data dengan *missing value*, sehingga tidak dilakukan penghapusan data.

#### Pendeteksian data outlier

Pengidentifikasian outlier dapat dilakukan dengan memanfaatkan *box plot*. *Box plot* merupakan ringkasan distribusi sampel yang disajikan secara grafis yang bisa menggambarkan bentuk distribusi data (skewness), ukuran tendensi sentral dan ukuran penyebaran (keragaman) data pengamatan [5].

###### Box plot variabel age

![Box plot variabel age](https://i.ibb.co/DKNptgK/boxage.png "Box plot variabel age")

Gambar 1. Box plot variabel age

Box plot pada Gambar 1 menunjukkan bahwa tidak ada outliers pada fitur age.

###### Box plot variabel bmi

![Box plot variabel bmi](https://i.ibb.co/yPcJqbG/boxbmi.png "Box plot variabel bmi")

Gambar 2. Box plot variabel bmi

Box plot pada Gambar 2 menunjukkan bahwa terdapat beberapa outliers pada variabel bmi.

###### Box plot variabel children

![Box plot variabel children](https://i.ibb.co/fdtvrHy/boxchildren.png "Box plot variabel children")

Gambar 3. Box plot variabel children

Box plot pada Gambar 3 menunjukkan bahwa tidak ada outliers pada variabel children.

###### Box plot variabel charges

![Box plot variabel charges](https://i.ibb.co/SXdzrMQ/boxcharges.png "Box plot variabel charges")

Gambar 4. Box plot variabel charges

Box plot pada Gambar 4 menunjukkan bahwa terdapat banyak outliers pada variabel charges.

###### Penanganan outliers

Secara matematis, pengidentifikasian data outliers dapat dilakukan dengan metode *Interquartile Range* (IQR). Dalam kasus ini, IQR dimanfaatkan untuk menentukan nilai batas atas dan batas bawah setiap fitur untuk menyaring data-data outliers dalam dataset yang digunakan [5]. Berikut persamaan yang digunakan:

![Formula IQR](https://i.ibb.co/tDkMQfc/IQRform.png)

Data yang memiliki satu atau lebih fitur yang berada di luar nilai batas awal dan batas akhir akan dihapus. Setelah penghapusan data outlier, jumlah data berkurang dari 1338 data menjadi 1193 data. Hal ini berarti bahwa terdapat 145 data outlier dalam dataset ini.

#### Univariate Analysis

*Univariate analysis* merupakan analisis dari setiap atau masing-masing variabel penelitian yang memberikan informasi atau ringkasan data yang dikumpulkan sehingga diperoleh informasi yang jelas. Informasi ini dapat disajikan dalam bentuk tabel, grafik atau ukuran statistik lainnya [6].

##### Analisis fitur kategori

###### Fitur sex (jenis kelamin)

![Grafik jumlah data berdasarkan jenis kelamin](https://i.ibb.co/1GJnpbK/jkstat.png "Grafik jumlah data berdasarkan jenis kelamin")

Gambar 5. Grafik jumlah data berdasarkan jenis kelamin

Fitur sex terdiri atas dua kategori, yaitu male dan female. Grafik pada Gambar 5 menunjukkan bahwa jumlah data dengan jenis kelamin perempuan lebih banyak daripada laki-laki.

###### Fitur smoker

![Grafik jumlah data berdasarkan status merokok](https://i.ibb.co/Jtxfgbq/smokerstat.png "Grafik jumlah data berdasarkan status merokok")

Gambar 6. Grafik jumlah data berdasarkan status merokok

Fitur smoker terdiri atas dua kategori, yaitu yes dan no. Grafik pada Gambar 6 menunjukkan bahwa jumlah data pemegang polis asuransi yang tidak merokok jauh lebih banyak daripada yang merokok.

###### Fitur region

![Grafik jumlah data berdasarkan lokasi perumahan](https://i.ibb.co/TH1Lnbj/regionstat.png "Grafik jumlah data berdasarkan lokasi perumahan")

Gambar 7. Grafik jumlah data berdasarkan lokasi perumahan

Fitur region terdiri atas empat kategori, yaitu northwest, southeast, northeast, dan southwest. Keempat kategori teresbut memiliki jumlah data dengan persentase kurang lebih 25%. Grafik pada Gambar 7 menunjukkan bahwa region northwest memiliki jumlah data tertinggi, sedangkan region southwest memiliki jumlah data terendah.


##### Analisis fitur numerik

![Grafik fitur numerik](https://i.ibb.co/4MdLYk3/univariate-num.png "Grafik fitur numerik")

Gambar 8. Grafik fitur numerik

Berdasarkan grafik pada Gambar 8, dapat disimpulkan bahwa:
- Distribusi nilai fitur charges bersifat miring ke kanan (right-skewed).
- Sebagian besar pemegang polis asuransi memiliki premi dari rentang 0 sampai 15000.
- Sebagian besar pemegang polis asuransi memiliki bmi sekitar 30.
- Jumlah data pemegang polis asuransi yang berumur 20 tahun merupakan yang terbanyak.
- Sebagian besar pemegang polis asuransi belum meiliki anak.

#### Multivariate Analysis

Multivariate Analysis merupakan analisis yang menggunakan lebih dari dua variabel untuk menguji pengaruh atau hubungan antara dua atau lebih variabel bebas dengan satu variabel terikat [6].

##### Fitur charges dengan fitur kategori

###### Fitur charges dengan fitur sex

![Grafik harga premi rata-rata untuk jenis kelamin perempuan dan laki-laki](https://i.ibb.co/4KKT3bT/chargejk.png "Grafik harga premi rata-rata untuk jenis kelamin perempuan dan laki-laki")

Gambar 9. Grafik harga premi rata-rata untuk jenis kelamin perempuan dan laki-laki

Grafik pada Gambar 9 menunjukkan bahwa nilai rata-rata fitur charges untuk jenis kelamin male dan female memiliki perbedaan yang sangat sedikit. Premi untuk jenis kelamin perempuan sedikit lebih tinggi daripada premi untuk jenis kelamin laki-laki. Rentang nilai rata-rata untuk kedua kategori tersebut berada antara 9500 hingga 10000. Dapat disimpulkan bahwa fitur sex memiliki pengaruh yang sangat kecil terhadap fitur charges.

###### Fitur charges dengan fitur smoker

![Grafik harga premi rata-rata untuk pemegang polis asuransi yang merokok dan tidak merokok](https://i.ibb.co/sKxLmGN/chargesmoker.png "Grafik harga premi rata-rata untuk pemegang polis asuransi yang merokok dan tidak merokok")

Gambar 10. Grafik harga premi rata-rata untuk pemegang polis asuransi yang merokok dan tidak merokok

Grafik pada Gambar 10 menunjukkan bahwa nilai rata-rata fitur charges untuk pemegang polis asuransi yang merokok dan tidak merokok memiliki perbedaan yang sangat besar. Pemegang polis asuransi yang merokok memiliki premi yang lebih besar daripada yang tidak merokok. Dapat disimpulkan bahwa fitur smoker memiliki pengaruh yang sangat besar terhadap fitur charges.

###### Fitur charges dengan fitur region

![Grafik harga premi rata-rata untuk setiap lokasi perumahan pemegang polis asuransi kesehatan di Amerika Serikat](https://i.ibb.co/fGKv1dn/chargeregion.png "Grafik harga premi rata-rata untuk setiap lokasi perumahan pemegang polis asuransi kesehatan di Amerika Serikat")

Gambar 11. Grafik harga premi rata-rata untuk setiap lokasi perumahan pemegang polis asuransi kesehatan di Amerika Serikat

Grafik pada Gambar 11 menunjukkan bahwa nilai rata-rata fitur charges untuk setiap region perbedaan yang jelas. Pemegang polis asuransi yang tinggal di wilayah northeast memiliki nilai premi tertinggi, dan wilayah southwest dengan nilai premi terendah. Dapat disimpulkan bahwa fitur region memiliki pengaruh terhadap fitur charges.

##### Fitur charges dengan fitur numerik lainnya

![Grafik hubungan fitur charges dengan fitur-fitur numerik](https://i.ibb.co/khCMZsy/multinum.png "Grafik hubungan fitur charges dengan fitur-fitur numerik")

Gambar 12. Grafik hubungan fitur charges dengan fitur-fitur numerik

Grafik pada Gambar 12 menunjukkan bahwa:
- Biaya asuransi meningkat seiring dengan umur pemegang polis asuransi.
- Tidak terlihat hubungan atau kolerasi antara fitur *children* dan bmi dengan fitur *charges*.

*Correlation matrix* dapat digunakan untuk memudahkan pengembang dalam melihat kolerasi antar fitur numerik. Correlation  Matrix  bertujuan  untuk  menunjukkan  hubungan  antar  suatu  variabel  dengan  variabel lainnya. Tingkat hubungan kolerasi antar variabel ditentukan dengan simbol tertentu dan nilai tertentu, baik positif, negatif ataupun tanpa korelasi [7]. Berikut *correlation matrix* yang menilai kolerasi antar fitur numerik:

![Correlation matrix](https://i.ibb.co/zsBMspB/coreelation-matrix.png "Correlation matrix")

Gambar 13. Correlation matrix fitur numerik

Correlation matrix pada Gambar 13 menunjukkan bahwa terdapat kolerasi yang jelas antara fitur age dan fitur charges. Fitur bmi dan children memiliki kolerasi yang sangat rendah terhadap fitur charges (mendekati 0), sehingga kedua fitur tersebut tidak akan digunakan dalam memprediksi premi asuransi. 

## Data Preparation

Dalam kasus ini, data preparation terdiri atas 3 tahap, yaitu melakukan *encoding* terhadap fitur kategorikal, membagi dataset menjadi data training dan data testing (*Train Test Split*), dan melakukan standarisasi terhadap fitur age data training dan data testing.

### Encoding fitur kategori

Pada tahapan ini dilakukan proses *encoding* dengan teknik *One-hot Encoding* terhadap fitur-fitur kategori. *One-hot Encoding* merupakan proses pengubahan nilai-nilai pada fitur kategori menjadi format yang dapat diterima oleh model machine learning (berupa numerik). Dalam teknik ini akan dilakukan penambahan variabel atau fitur *dummy* terhadap dataframe untuk setiap nilai unik dalam fitur kategori. Angka nol dan satu kemudian dimasukkan ke dalam variabel dummy tersebut untuk menunjukkan kategori yang digunakan. Tahapan ini diperlukan untuk meningkatkan akurasi dari model machine learning yang akan dibuat nantinya [8].

Salah satu cara untuk mengimplementasi teknik *One-hot Encoding* adalah dengan memanfaatkan method `get_dummies()` dari [library pandas](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), yang berfungsi untuk mengubah variabel kategori menjadi kumpulan variabel *dummy* atau variabel indikator. Kolom-kolom yang dihasilkan oleh method ini lalu digabungkan di dalam *dataframe* dengan method `concat()`, yang juga berasal dari [library pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html).

Setelah *One-hot Encoding* dilakukan, terdapat penambahan fitur-fitur dummy seperti yang ditampilkan pada Tabel 4.

Tabel 4. Dataframe setelah mengimplementasi *One-hot Encoding*
| | age | charges | Sex_female | Sex_male | Smoker_no | Smoker_yes | Region_northeast | Region_northwest | Region_southeast | Region_southwest |
|---|---|---|---|---|---|---|---|---|---|---|
|0| 19 | 16884.92400 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
|1| 18 | 1725.55230 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
|2| 28 | 4449.46200 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
|3| 33 | 21984.47061 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
|4| 32 | 3866.85520 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
|...| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Pembagian dataset

Pada tahap ini dilakukan pembagian dataset menjadi dua bagian, yaitu data training dan data testing. Data training merupakan data-data yang akan digunakan untuk melatih model machine learning. Dalam kasus ini, data training dibagi lagi menjadi dua bagian, yaitu data training yang tidak memiliki fitur target (`x_train`) dan data training yang hanya memiliki fitur target (`y_train`). Data testing juga dibagi menjadi dua bagian, yaitu data testing yang tidak memiliki fitur target (`x_test`) dan data testing yang hanya memiliki fitur target (`y_test`). Salah satu cara untuk membagi dataset menjadi keempat bagian tersebut adalah dengan memanfaatkan method `train_test_split()` yang berasal dari [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Tahapan ini dilakukan untuk menyiapkan data-data untuk melakukan evaluasi terhadap model, sehingga pengembang dapat mengetahui akurasi dari prediksi yang dihasilkan oleh model tersebut.

Dalam kasus ini, 80% dataset akan digunakan sebagai data training, dan 20% dataset akan digunakan sebagai data testing. Setelah dilakukan pembagian dataset:
- `x_train` dan `x_test` memiliki 954 data
- `y_train` dan `y_test` memiliki 239 data 

### Standarisasi dengan StandardScaler

Pada tahap ini dilakukan standarisasi terhadap fitur numerik pada `x_train` dan `x_test`. Standarisasi merupakan proses transformasi nilai dari fitur dalam dataset agar nilai-nilai dalam fitur numerik berada pada skala yang relatif sama atau mendekati distribusi normal. Tahapan ini diperlukan agar algoritma yang digunakan dalam model memiliki performa lebih baik dan konvergen lebih cepat [9].

Dalam kasus ini, standarisasi dilakukan dengan method `StandardScaler()` yang berasal dari [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). StandardScaler berfungsi untuk mengubah nilai-nilai pada fitur numerik, sehingga nilai rata-rata fitur tersebut menjadi 0, dan standar deviasi menjadi 1 [9].

Terdapat satu fitur numerik pada x_train dan x_test, yaitu fitur age. Setelah melakukan standarisasi, fitur age memiliki mean dengan nilai 0 dan standar deviasi dengan nilai 1.

## Model Development

Pada tahap ini dilakukan pengembangan model *machine learning* dengan menggunakan lima algoritma, yaitu K-Nearest Neighbor (KNN), Random Forest (RF), AdaBoost, Gradient Boosting, dan LBGM Regression. Dari kelima algoritma tersebut akan dipilih salah satu dengan performa terbaik.

### Pembahasan algoritma

Berikut pembahasan mengenai algoritma yang digunakan.

#### K-Nearest Neighbor (KNN)

Algoritma K-Nearest Neighbor (KNN) merupakan algoritma yang bekerja dengan mengambil sejumlah K data terdekat (tetangganya) sebagai acuan untuk menentukan atau memprediksi suatu data. Algoritma KNN dapat digunakan untuk kasus-kasus klasifikasi dan regresi. Algoritma ini mengklasifikasikan atau memprediksikan data berdasarkan similarity atau kemiripan atau kedekatannya terhadap data lainnya [10].

Kelebihan algoritma KNN:
- KNN bersifat sangat nonlinear, karena KNN merupakan algoritma pembelajaran yang bersifat non-parametrik, yang berarti algoritma ini tidak mengasumsikan apa-apa mengenai distribusi *instance* dalam data maupun dokumen
- KNN mudah dipahami dan diimplementasikan

Kekurangan algoritma KNN:
- Perlu memberikan parameter K (jumlah tetangga terdekat).
- Tidak menangani *missing value* secara implisit.
- Sensitif terhadap data *outlier*.
- Rentan terhadap variabel yang tidak informatif.
- Rentan terhadap dimensionalitas (jumlah variabel) yang tinggi, karena semakin banyak dimensi, ruang yang bisa ditempati *instance* semakin besar, sehingga semakin besar pula kemungkinan bahwa *neighbor* terdekat dari suatu *instance* sebenarnya memiliki jarak yang sangat jauh.
- Rentan terhadap perbedaan rentang variabel.
- Memiliki nilai komputasi yang tinggi.

Dalam kasus ini, pengembang menggunakan method `KNeighborsRegressor()` yang tersedia dalam [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html). Berikut parameter yang akan digunakan dalam algoritma ini.

- `n_neighbors`: jumlah neighbor yang dibutuhkan setiap sampel data.
- `algorithm`: algoritma yang digunakan untuk mencari neighbor terdekat. Terdapat 3 algoritma yang dapat digunakan:
    - `ball_tree`
    - `kd_tree`
    - `brute`
- `weights`: pembobotan setiap neighbor yang terhubung dengan sampel data. Terdapat dua metode yang dapat digunakan:
    -   `uniform`: Bobot semua poin neighbor yang terhubung bernilai sama
    -   `distance`: Bobot setiap poin neighbor disesuaikan dengan jarak antar poin.

#### Random Forest

Algoritma Random Forest merupakan algoritma yang digunakan dalam kasus klasifikasi dan regresi dengan data dalam jumlah yang besar. Random forest terdiri dari kombinasi dari masing – masing pohon (tree) dari model *Decision Tree*, dan kemudian dikombinasikan ke dalam satu model. Penentuan hasil dilakukan dengan mengambil prediksi terbaik di antara semua model *Decision Tree* yang ada [11].

Kelebihan algoritma Random Forest:
- Menghasilkan error yang lebih rendah.
- Dapat menangani data training dengan jumlah banyak secara efisien.
- Metode yang efektif untuk mengestimasi hilangnya data.
- Dapat memperkiraan variabel-variabel penting yang mempengaruhi hasil.
- Menyediakan metode eksperimental untuk mendeteksi interaksi variabel.

Kekurangan algoritma Random Forest:
- Waktu pemrosesan yang lama, karena menggunakan data yang banyak dan membangun model tree yang banyak pula untuk membentuk random trees karena menggunakan single processor.
- Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data.
- Ketika digunakan untuk kasus regresi, algoritma ini sulit memprediksi di luar kisaran dalam data *testing*, hal ini disebabkan karena model dengan algoritma ini terlalu mudah beradaptasi dengan kumpulan data pengganggu (*noisy data*).

Dalam kasus ini, pengembang menggunakan method `RandomForestRegressor()` yang tersedia dalam [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Berikut parameter yang akan digunakan dalam algoritma ini:
- `n_estimators`: Menunjukkan jumlah model *Decision Tree* yang akan digunakan
- `max_depth`: Menunjukkan kedalaman maksimum Decision Tree.
- `random_state`: Menunjukkan keacakan dalam *bootstrapping* sampel yang digunakan saat membangun *Decision Tree*
- `min_samples_leaf`: Fraksi bobot minimum dari jumlah total bobot (dari semua sampel input) yang diperlukan untuk berada di simpul daun.

#### Adaboost

Algoritma AdaBoost (*Adaptive Boosting*) merupakan algoritma ensemble yang memanfaatkan *bagging* dan *boosting* untuk mengembangkan peningkatan akurasi prediktor. Sama seperti Random Forest, algoritma ini juga menggunakan beberapa pohon keputusan untuk memperoleh data prediksi [12].

Kelebihan algoritma AdaBoost:
- Mudah digunakan, karena tidak memerlukan banyak pengaturan parameter
- Kemungkinan kecil untuk mengalami overfitting karena parameter-parameter yang digunakan belum sepenuhnya teroptimasi dan learning rate yang cenderung rendah.

Kekurangan algoritma AdaBoost:
- Adaboost menggunakan teknik *learning boosting* secara progresif, sehingga algoritma ini sensitif terhadap data *outlier* atau *noise*.

Dalam kasus ini, pengembang menggunakan method `AdaBoostRegressor()` yang tersedia dalam [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html). Berikut parameter yang akan digunakan dalam algoritma ini:
- `learning_rate`: Bobot yang diterapkan pada setiap *regressor* pada setiap iterasi *boosting*.
- `n_estimators`: Jumlah maksimum estimator di mana *boosting* dihentikan.
- `random_state`: Mengatur *random seed* dalam setiap iterasi *boosting*.
- `loss`: Metode yang digunakan saat memperbarui bobot setelah setiap iterasi *boosting*. Terdapat beberapa metode loss yang dapat digunakan, yaitu `linear`, `square`, dan `exponetial`.

#### Gradient Boosting

*Gradient boosting* merupakan algoritma klasifikasi *machine learning* yang juga menggunakan algoritma *ensemble* dengan decision tree untuk memprediksi nilai. Gradient boosting dimulai dengan menghasilkan pohon klasifikasi awal, kemudian menyesuaikan pohon baru melalui minimalisasi kerugian dengan fungsi *loss* [13].

Kelebihan algoritma Gradient Boosting:
- Sering memberikan akurasi prediksi yang tidak dapat dipalsukan.
- Banyak fleksibilitas. Algoritma ini dapat mengoptimalkan fungsi loss yang berbeda dan menyediakan beberapa opsi penyetelan *hyperparameter* yang membuat fungsi tersebut sangat fleksibel.
- Tidak diperlukan pra-pemrosesan data. Algoritma ini sering kali berfungsi dengan baik dengan nilai kategorik dan numerik apa adanya.
- Penanganan *missing value* tidak perlu dilakukan.

Kelebihan algoritma Gradient Boosting:
- Model Gradient Boosting akan terus berusaha untuk meminimalkan semua kesalahan. Hal ini dapat menyebabkan model ini terlalu fokus pada outlier dan bisa mengalami *overfitting*.
- Komputasi yang terlalu kompleks. Algoritma ini sering membutuhkan banyak Decision Tree (>1000) yang dapat menghabiskan waktu dan memori.
- Fleksibilitas yang terlalu tinggi menghasilkan banyaknya parameter yang saling berinteraksi dan sangat mempengaruhi perilaku pendekatan (jumlah iterasi, kedalaman pohon, parameter regularisasi, dll.). Oleh karena itu, algoritma ini membutuhkan pencarian *hyperparameter* dengan skala yang besar selama penyetelan.
- Sifatnya kurang interpretatif.

Dalam kasus ini, pengembang menggunakan method `GradientBoostingRegressor()` yang tersedia dalam [library sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.htmll). Berikut parameter yang akan digunakan dalam algoritma ini:
- `learning_rate`: Dalam algoritma ini, *learning rate* berfungsi untuk mengecilkan kontribusi setiap *Decision Tree*.
- `n_estimators`: Jumlah tahapan *boosting* yang harus dilakukan.
- `max_depth`: Kedalaman maksimum estimator regresi individu.
- `random_state`: Mengatur *random seed* dalam setiap iterasi *boosting*.

#### LightGBM (LGBM)

Light Gradient-Boosting Machine (LightGBM atau LGBM) merupakan algoritma yang mengimplementasikan algoritma Gradient Boosting konvensional dengan penambahan dua teknik baru, yaitu Gradient Based One Side Sampling (GOSS) dan Exclusive Feature Bundling (EFB). Teknik-teknik ini dirancang untuk secara signifikan meningkatkan efisiensi dan skalabilitas algoritma Gradient Boosting [14].

LGBM berkinerja baik dalam kompetisi machine learning karena penanganannya yang kuat dari berbagai jenis data, hubungan, distribusi, dan keragaman hyperparameter. LightGBM digunakan dalam kasus regresi, klasifikasi (biner dan multiclass), dan pemeringkatan.

Kelebihan algoritma LGBM:
- Kecepatan pelatihan yang lebih cepat dan efisiensi yang lebih tinggi
- Penggunaan memori yang lebih rendah
- Akurasi yang lebih baik daripada algoritma boosting lainnya
- Kompatibilitas dengan dataset dengan ukuran yang besar.

Kekurangan algoritma LGBM:
- LGBM membagi *Decision Tree* berdasarkan bagian daunnya, yang dapat menyebabkan overfitting karena menghasilkan banyak model *Decision Tree* yang kompleks.
- Sensitif terhadap overfitting.

Dalam kasus ini, pengembang menggunakan method `LGBMRegressor()` yang tersedia dalam [library lightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html). Berikut parameter yang akan digunakan dalam algoritma ini:
- `boosting_type`: Metode boosting yang digunakan. Terdapat empat metode yang dapat digunakan, yaitu: `gdbt`, `dart`, `goss`, dan `rf`.
- `learning_rate`: *Learning rate* dalam proses *boosting*.
- `n_estimators`: Jumlah *Decision tree* yang akan dilakukan *fit*.
- `random_state`: Mengatur *random seed* dalam setiap iterasi *boosting*. 

### Hyperparameter Tuning

*Hyperparameter Tuning* bertujuan untuk melakukan memberikan *improvement* terhadap model yang akan dilatih dengan mengatur parameter-parameter dalam metode algoritma yang digunakan. Dalam kasus ini, pencarian *hyperparameter* yang sesuai untuk setiap algoritma dilakukan dengan Grid Search. Metrik pengukuran performa dari *Grid Search* adalah *Mean Cross Validation* (CV). Setiap kemungkinan kombinasi nilai parameter akan diukur dengan metrik CV. Kombinasi parameter dengan Skor CV terbaik akan digunakan dalam pelatihan model.

Setelah dilakukan Hyperparameter Tuning dengan metode [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) dari library sklearn, kombinasi nilai paramater dengan skor CV terbaik untuk setiap algoritma akan diberikan.

Tabel 5. Hasil Hyperparameter Tuning dengan Grid Search
|index|model|best\_score|best\_params|
|---|---|---|---|
|0|KNN|0\.4221451625752903|\{'algorithm': 'brute', 'n\_neighbors': 9, 'weights': 'uniform'\}|
|1|RandomForest|0\.5980567985427531|\{'max\_depth': 4, 'min\_samples\_leaf': 16, 'n\_estimators': 25, 'random\_state': 33\}|
|2|AdaBoost|0\.5907888310169224|\{'learning\_rate': 0\.01, 'loss': 'exponential', 'n\_estimators': 5, 'random\_state': 22\}|
|3|GradientBoosting|0\.5976592080666286|\{'learning\_rate': 0\.05, 'max\_depth': 3, 'n\_estimators': 75, 'random\_state': 11\}|
|4|LGBM|0\.5845705479785525|\{'boosting\_type': 'goss', 'learning\_rate': 0\.05, 'n\_estimators': 50, 'random\_state': 44\}|

Tabel 5  menunjukkan bahwa algoritma Random Forest merupakan algoritma terbaik untuk kasus ini karena memiliki skor CV tertinggi, yaitu dengan skor 0.598. Skor algoritma Random Forest memiliki perbedaan yang sangat kecil dengan skor algoritma Gradient Boosting, yaitu sekitar 0.0004. Parameter yang tertera pada tabel di atas akan diimplementasikan dalam pengembangan model.

## Evaluation

Pada tahap *evaluation*, metrik-metrik evaluasi yang digunakan adalah *Mean Squared Error* (MSE) dan *Mean Absolute Error* (MAE).

### Pembahasan Metrik Evaluasi

Berikut pembahasan mengenai metrik yang digunakan dalam tahap evaluasi.

#### Mean Squared Error

*Mean Squared Error* (MSE) adalah rata-rata kesalahan kuadrat antara nilai aktual dan nilai prediksi. MSE secara umum digunakan untuk memeriksa estimasi kesalahan pada nilai prediksi. Nilai MSE yang rendah atau mendekati nol menunjukkan bahwa hasil prediksi sesuai dengan data aktual dan bisa dimanfaatkan untuk memprediksi di periode mendatang [15].

Berikut formula perhitungan MSE:

![Formula MSE](https://i.ibb.co/XCYVK7B/mseform.png)

Keterangan:

- n = jumlah data
- Yi = nilai aktual
- Ŷi = nilai yang diprediksi

Formula di atas menunjukkan bahwa perhitungan MSE dilakukan dengan melakukan pengkuadratan pada pengurangan nilai aktual dengan nilai prediksi setiap data, kemudian dijumlahkan secara keseluruhan dan membaginya dengan jumlah data yang ada.

#### Mean Absolute Error
*Mean Absolute Error* (MAE) adalah rata-rata selisih mutlak nilai aktual dengan nilai prediksi. MAE juga digunakan untuk memeriksa estimasi kesalahan pada nilai prediksi secara umum. Nilai MAE yang rendah atau mendekati nol menunjukkan bahwa hasil prediksi sesuai dengan data aktual dan bisa dimanfaatkan untuk memprediksi di periode mendatang [16]. 

Berikut formula perhitungan MAE:

![Formula MAE](https://i.ibb.co/b6jMDsS/maeform.png)

Keterangan:

- n = jumlah data
- Yi = nilai aktual
- Ŷi = nilai yang diprediksi

Formula di atas menunjukkan bahwa perhitungan MAE dilakukan dengan melakukan pengurangan nilai aktual dengan nilai prediksi setiap data, dimana hasil pengurangan tersebut selalu merupakan bilangan positif, yang kemudian dijumlahkan secara keseluruhan dan membaginya dengan jumlah data yang ada.

### Evaluasi model
Setiap model dengan algoritma berbeda akan dievaluasi untuk menentukan model yang dapat menghasilkan nilai prediksi dengan nilai eror terendah. Berikut hasil evaluasi dengan metrik MSE:

Tabel 6. Hasil evaluasi dengan metrik MSE
|index|MSE Train|MSE Test|
|---|---|---|
|KNN|19017\.73449906815|23360\.01040835056|
|RandomForest|20224\.663778011884|20830\.255662282183|
|AdaBoost|20841\.928071098522|22319\.002061981377|
|GradientBoosting|19686\.41797426854|21444\.284104005863|
|LGBM|18600\.154909200846|22517\.850226414972|

![Hasil evaluasi - MSE](https://i.ibb.co/LryfQVF/mse-eva.png "Hasil evaluasi - MSE")

Gambar 14. Hasil evaluasi dengan metrik MSE dalam bentuk grafik

Berikut hasil evaluasi dengan metrik MAE:

Tabel 7. Hasil evaluasi dengan metrik MAE
|index|MAE Train|MAE Test|
|---|---|---|
|KNN|2544\.1310571410436|2545\.6182073621576|
|RandomForest|2605\.8146088027365|2447\.9915850260563|
|AdaBoost|2660\.7876673805863|2583\.5350039293453|
|GradientBoosting|2563\.584107595491|2452\.6459536895345|
|LGBM|2520\.1222823306584|2457\.8821017350106|

![Hasil evaluasi - MAE](https://i.ibb.co/P5jPHrK/mae-eva.png "Hasil evaluasi - MAE")

Gambar 15. Hasil evaluasi dengan metrik MAE dalam bentuk grafik

Tabel 6, Tabel 7, Gambar 14, dan Gambar 15 menunjukkan bahwa model dengan algoritma Random Forest memiliki nilai MSE dan MAE terendah dalam pengujian prediksi data testing. Hal ini menunjukkan bahwa algoritma Random Forest akan memberikan hasil prediksi yang paling akurat jika dibandingkan dengan algoritma Gradient Boosting, LGBM, KNN, dan AdaBoost.

Berikut contoh hasil prediksi nilai premi salah satu data dengan kelima model tersebut:

Tabel 8. Contoh prediksi
|index|y\_true \(nilai aktual\)|prediksi\_KNN|prediksi\_RandomForest|prediksi\_AdaBoost|prediksi\_GradientBoosting|prediksi\_LGBM|
|---|---|---|---|---|---|---|
|1332|11411\.685|10104\.4|11277\.9|11199\.2|10379\.3|10822\.1|

Tabel 8 menunjukkan bahwa hasil prediksi model dengan algoritma *Random Forest* memberikan hasil yang paling mendekati nilai aktual. Oleh karena itu, algoritma terbaik untuk memprediksi premi asuransi kesehatan berdasarkan karakteristik pemegang polis asuransi adalah algoritma ***Random Forest***.

## Kesimpulan

- Fitur-fitur yang berpengaruh terhadap premi asuransi kesehatan (*charges*) adalah umur (*age*), status merokok (*smoker*), wilayah perumahan (*region*), dan jenis kelamin (*sex*).
- Model machine learning yang dapat memprediksi premi asuransi kesehatan adalah model dengan algoritma regresi, seperti K-Nearest Neighbor (KNN), Random Forest (RF), AdaBoost, Gradient Boosting, dan LGBM Regression. Dengan algoritma tersebut, model dapat memprediksi premi asuransi kesehatan berdasarkan karakteristik pemegang polis.
- Di antara kelima algoritma tersebut, algoritma yang dapat memprediksi premi asuransi kesehatan dengan akurasi terbaik adalah algoritma Random Forest. Namun, nilai MSE dan MAE yang dihasilkan dengan algoritma tersebut masih tergolong tinggi, sehingga perlu dilakukan improvisasi lebih lanjut, seperti menambahkan parameter lain yang berpengaruh terhadap performa model saat melakukan *hyperparameter tuning* atau mencoba algoritma regresi lain.


## Referensi

[1] S. S. Mladenovic et al., “[Identification of the important variables for prediction of individual medical costs billed by health insurance](https://sci-hub.se/https://doi.org/10.1016/j.techsoc.2020.101307),” Technology in Society, vol. 62, p. 101307, Aug. 2020, doi: 10.1016/j.techsoc.2020.101307.

[2] P. E. Arimbawa, “[HUBUNGAN KEPEMILIKAN ASURANSI KESEHATAN DENGAN PENGGUNAAN OBAT RASIONAL (POR) PADA PASIEN SWAMEDIKASI](https://e-journal.unmas.ac.id/index.php/Medicamento/article/view/866/782) | Jurnal Ilmiah Medicamento,” Jan. 2022, Accessed: Nov. 8, 2022. [Online]. Available: https://e-journal.unmas.ac.id/index.php/Medicamento/article/view/866

[3] S. N. Islami, L. Noviyanti, and F. Indrayatna, “[Prediksi Eror Cadangan klaim berdasarkan Metode Chain Ladder Pendekatan Bootstrap pada Produk Asuransi Kesehatan](http://biastatistics.statistics.unpad.ac.id/index.php/biastatistics/article/view/140/151),” E-Journal BIAStatistics | Departemen Statistika FMIPA Universitas Padjadjaran, vol. 16, no. 2, Art. no. 2, Sep. 2022.

[4] B. Setiyono, “[PERLUNYA REVITALISASI KEBIJAKAN JAMINAN KESEHATAN DI INDONESIA](https://e-journal.unmas.ac.id/index.php/Medicamento/article/view/866/782),” Politika: Jurnal Ilmu Politik, vol. 9, no. 2, pp. 38–60, Oct. 2018.

[5] N. Rokhman and H. N. D. Saputri, “[Peningkatan Efisiensi Penugasan Guru di Provinsi Daerah Istimewa Yogyakarta Melalui Penghapusan Outlier](http://jurnal.atmaluhur.ac.id/index.php/knsi2018/article/view/368/293),” Konferensi Nasional Sistem Informasi (KNSI) 2018, no. 0, Art. no. 0, Mar. 2018, Accessed: Nov. 9, 2022. [Online]. Available: http://jurnal.atmaluhur.ac.id/index.php/knsi2018/article/view/368

[6] S. Riyanto and A. A. Hatmawan, [Metode Riset Penelitian Kuantitatif Penelitian Di Bidang Manajemen, Teknik, Pendidikan Dan Eksperimen](https://books.google.co.id/books?id=W2vXDwAAQBAJ&printsec=copyright&hl=id#v=onepage&q&f=false). Deepublish, 2020.

[7] S. Kurniawan and N. D. Nahdi, “[Penggunaan Metode QFD Menerjemahkan Suara Konsumen Untuk Pengembangan Lip Product Lavine Beaute](https://journal.binus.ac.id/index.php/BECOSS/article/view/6532),” Business Economic, Communication, and Social Sciences Journal (BECOSS), vol. 2, no. 3, Art. no. 3, 2020, doi: 10.21512/becossjournal.v2i3.6532.

[8] L. Yu, R. Zhou, R. Chen, and K. K. Lai, “[Missing Data Preprocessing in Credit Classification: One-Hot Encoding or Imputation?](https://www.tandfonline.com/doi/abs/10.1080/1540496X.2020.1825935),” Emerging Markets Finance and Trade, vol. 58, no. 2, pp. 472–482, Jan. 2022, doi: 10.1080/1540496X.2020.1825935.

[9] A. Agus, J. A. Qadhli, and H. Yeni, “[Analysis of the Effect of Data Scaling on the Performance of the Machine Learning Algorithm for Plant Identification](https://jurnal.iaii.or.id/index.php/RESTI/article/view/1517/210)," Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi),  Apr. 2020, Accessed: Nov. 21, 2022. [Online]. Available: http://www.jurnal.iaii.or.id/index.php/RESTI/article/view/1517

[10] D. Syahid, J. Jumadi, and D. Nursantika, “[Sistem Klasifikasi Jenis Tanaman Hias Daun Philodendron
Menggunakan Metode K-Nearest Neighboor (KNN) Berdasarkan Nilai Hue, Saturation, Value (HSV)](http://join.if.uinsgd.ac.id/index.php/join/article/view/v1i15/15),” Jurnal Online Informatika, vol. 1, no. 1, Art. no. 1, Jun. 2016, doi: 10.15575/join.v1i1.6.

[11] D. Irawan, E. B. Perkasa, Y. Yurindra, D. Wahyuningsih, and E. Helmud, “[Perbandingan Klassifikasi SMS Berbasis Support Vector Machine, Naive Bayes Classifier, Random Forest dan Bagging Classifier](http://jurnal.atmaluhur.ac.id/index.php/sisfokom/article/view/1302),” Jurnal Sisfokom (Sistem Informasi dan Komputer), vol. 10, no. 3, Art. no. 3, Dec. 2021, doi: 10.32736/sisfokom.v10i3.1302.

[12] S. I. Gultom, “[Implementasi Data Mining Menentukan Pola Hidup Sehat Bagi Pengguna KB Menggunakan Algoritma Adaboost (Studi Kasus :Dinas Serdang Bedagai)](https://ejurnal.stmik-budidarma.ac.id/index.php/inti/article/view/2405/1833),” Informasi dan Teknologi Ilmiah (INTI), vol. 7, no. 3, Art. no. 3, Jun. 2020.

[13] A. Natekin and A. Knoll, “[Gradient boosting machines, a tutorial](https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full),” Frontiers in Neurorobotics, vol. 7, 2013, Accessed: Nov. 21, 2022. [Online]. Available: https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021

[14] A. Shehadeh, O. Alshboul, R. E. Al Mamlook, and O. Hamedat, “[Machine learning models for predicting the residual value of heavy construction equipment: An evaluation of modified decision tree, LightGBM, and XGBoost regression](https://www.sciencedirect.com/science/article/abs/pii/S0926580521002788),” Automation in Construction, vol. 129, p. 103827, Sep. 2021, doi: 10.1016/j.autcon.2021.103827.

[15] D. I. Y. Situmorang, “[ANALISA PREDIKSI PENYEWAAN ALAT TRANSPORTASI MENGGUNAKAN METODE SINGLE EXPONENTIAL SMOOTHING (STUDI KASUS : PT SEDONA HOLIDAYS MEDAN)](https://ejurnal.stmik-budidarma.ac.id/index.php/jurikom/article/view/354/332),” JURIKOM (Jurnal Riset Komputer), vol. 2, no. 6, Art. no. 6, Dec. 2015, doi: 10.30865/jurikom.v2i6.354.

[16] W. Wang and Y. Lu, “[Analysis of the Mean Absolute Error (MAE) and the Root Mean Square Error (RMSE) in Assessing Rounding Model](https://iopscience.iop.org/article/10.1088/1757-899X/324/1/012049/meta),” IOP Conf. Ser.: Mater. Sci. Eng., vol. 324, no. 1, p. 012049, Mar. 2018, doi: 10.1088/1757-899X/324/1/012049.
