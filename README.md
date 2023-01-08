# Prediksi-Saham-Dengan-Deep-Learning

Perkembangan machine learning kususnya deep learning semakin pesat saat ini. Deep learning dapat dipakai salah satunya dalam prediksi timeseries. Bahkan time series yang bersifat sangat lumpy atau volatile atau tidak stationar, seperti pergerakan harga saham. 

Pemrosesan data memiliki proporsi yang besar dalam machine learning. Pada kesempatan ini dibuat code untuk pemrosesan data, sehingga bisa di gunakan dalam machine learning aplikasi untuk  prediksi time series data. Pada sheet coding lain juga dicontohkan aplikasi penggunaan pada contoh model deep learning menggunakan LSTM. Pengguna code ini bisa membuat data untuk analisa timeseries sesuai keinginan dengan parameter antara lain panjang lag, output prediksi, dan batch. Lag merupakan panjang input data, sedangkan output panjang output data, serta batch merupakan jumlah batch saat dilakukan training model.

Code ini digunakan untuk melakukan prediksi data fianancial atau harga stock. Sumber data dapat berasal dari proses download API yfinance atau dari overwrite data excel.  Typical data pertama, akan tergenerate ketika dilakukan initiate object class ‘Preproses’. Dengan cara memasukan ticker data inputan yang ingin dianalisa timeseriesnya. Sedangngkan data excel dibuat untuk menambahkan variable- variable exogenous yang mungkin ingin dimasukan juga dalam feature mesin learning, yang itu sifatnya tidak dapat kita download melalui API atau data yang membutuhkan preproses atau penghitungan sebelum dipakai.

Terdapat dua kelas yakni class ‘Preproses’ dan ‘Prediction’. Object dari class ‘Preproses’ digunakan untuk membuat data training. Jika kita belum dan atau ingin membuat model, cukup menggunakan object ini. Sedangkan object dari class ‘Prediction’ merupakan inheritance dari class ‘Preproses’, sehingga selain dapat membuat data training, namun juga dapat digunakan untuk menghasilkan data input untuk keperluan prediksi setelah tersedia model nya. Parameter class ‘Prediksi’ yang tidak ada pada ‘Preproses’ adalah parameter start_pred, yakni tanggal awal prediksi. Seberapa panjang output prediksi tergantung parameter out.

Method penting dalam class ‘Preproses’ dan ‘Prediction’:
-	Preproses.df : menghasilakan data mentah time series, variable per colom. DataFrame object
-	Preproses.df_excel : melakukan overwrite data mentah (Preproses.df)  dari excel. DataFrame object
-	Preproses.feature  : menghasilakan data feature yang digunakan sebagai inputan model untuk proses training. Numpy object.
-	Preproses.label : menghasilkan data label yang digunakan sebagai output model untuk proses training. Numpy object.
-	Prediction.feature_pred : menghasilkan data input model yang telah jadi. Numpy object
-	Prediction.hasil_pred : menghasilkan output model berupa harga prediksi. Numpy object
-	Prediction.dframe_hasil : menghasilkan harga prediksi disertai dengan index waktu harga prediksi. DataFrame object.

Supaya efektif dalam melakukan prediksi timeseries data, berikut saran dalam pembuatan model machine learning atau deep learning :
-	Data training dan data validation harus dilakukan shuffle.  Bisa menggunakan library sklearn.model_selection.train_test_split. 
-	Tidak perlu menggunakan terlalu banyak Dense pada layer model. Serta tidak perlu menggunakan activation function pada Dense.
-	Karena hal diatas (point 2), perbanyak gunakan sequence layer LSTM atau GRU.

Contoh penggunaan model deep learning yang disajikan pada recipro, model untuk memprediksi harga saham adaro. Variable exogenous yang digunakan antara lain timeseries market dalam hal ini IHSG, serta beberapa technical data yang diproses melalu library finta dan disimpan dalam excel. Technical data ini antara lain volume, RSI, EMA, Bollinger Bands, MACD, Williams.
