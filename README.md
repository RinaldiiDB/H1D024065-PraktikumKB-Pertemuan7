# Klasifikasi Dataset Iris menggunakan Jaringan Syaraf Tiruan (JST)

Proyek ini adalah implementasi Jaringan Syaraf Tiruan (Artificial Neural Network) menggunakan TensorFlow dan Keras untuk mengklasifikasikan spesies bunga Iris berdasarkan fitur morfologinya.

## 📋 Deskripsi Proyek
Program ini memuat dataset Iris, melakukan prapemrosesan data, membangun model JST dengan beberapa layer *Dense*, melatih model, mengevaluasi performanya, dan menyediakan antarmuka sederhana untuk melakukan prediksi pada data baru.

## 🛠️ Prasyarat
Sebelum menjalankan program, pastikan Anda telah menginstal pustaka Python berikut:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## 📂 Struktur File
- `jst-iris.py`: Skrip utama yang berisi logika pemrosesan, pelatihan model, dan visualisasi.
- `iris.data`: Dataset Iris dalam format CSV (tanpa header).

## 🚀 Cara Menjalankan
1. Pastikan file `iris.data` berada di direktori yang sama dengan `jst-iris.py`.
2. Jalankan skrip menggunakan Python:
   ```bash
   python jst-iris.py
   ```
3. Program akan melatih model selama 50 epoch dan menampilkan grafik riwayat pelatihan serta *Confusion Matrix*.
4. Setelah visualisasi ditutup, program akan meminta input manual untuk memprediksi data baru.

## 🧠 Arsitektur Model
Model dibangun menggunakan API `Sequential` dari Keras dengan struktur sebagai berikut:
- **Input Layer**: Menyesuaikan dengan jumlah fitur (4 fitur).
- **Hidden Layer 1**: 1000 neuron, aktivasi ReLU.
- **Hidden Layer 2**: 500 neuron, aktivasi ReLU.
- **Hidden Layer 3**: 300 neuron, aktivasi ReLU.
- **Output Layer**: 3 neuron (Iris-setosa, Iris-versicolor, Iris-virginica), aktivasi Softmax.

## 📊 Visualisasi
- **Training History**: Menampilkan plot akurasi dan loss selama proses pelatihan.
- **Confusion Matrix**: Menunjukkan performa model dalam mengklasifikasikan data uji menggunakan heatmap dari Seaborn.

## 📝 Contoh Input Prediksi
Saat diminta memasukkan data baru, Anda dapat menggunakan nilai contoh berikut:
- Sepal length: 5.1
- Sepal width: 3.5
- Petal length: 1.4
- Petal width: 0.2
*(Hasil prediksi yang diharapkan: Iris-setosa)*
