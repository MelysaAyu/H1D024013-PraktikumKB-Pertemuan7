### Klasifikasi Spesies Bunga Iris menggunakan TensorFlow & Keras

---

## 📌 Tujuan
Menerapkan konsep Jaringan Syaraf Tiruan (JST) dalam kode Python menggunakan **TensorFlow** dan **Keras** untuk mengklasifikasikan spesies bunga Iris berdasarkan fitur morfologinya.

---

## 📂 Struktur File

```
H1D024013-PraktikumKB-Pertemuan7/
├── iris_jst.py           # Source code utama
├── iris.data             # Dataset Iris (150 data)
├── training_history.png  # Grafik loss & accuracy
├── confusion_matrix.png  # Confusion matrix hasil prediksi
└── README.md             # Dokumentasi ini
```

---

## 🗃️ Dataset

Dataset **Iris** berisi 150 sampel bunga dengan:
- **4 fitur input:** sepal length, sepal width, petal length, petal width (satuan cm)
- **3 kelas output:**
  | Kode | Spesies |
  |------|---------|
  | 0 | Iris-setosa |
  | 1 | Iris-versicolor |
  | 2 | Iris-virginica |

---

## 🧰 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| `tensorflow` / `keras` | Membangun dan melatih model JST |
| `pandas` | Membaca dan mengelola dataset |
| `numpy` | Operasi array/matriks |
| `scikit-learn` | Label encoding, split data, confusion matrix |
| `matplotlib` / `seaborn` | Visualisasi grafik dan heatmap |

Install semua library dengan:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

---

## 🧠 Arsitektur Model

```
Input Layer     →  4 neuron  (sepal & petal length/width)
Hidden Layer 1  →  1000 neuron  (aktivasi ReLU)
Hidden Layer 2  →  500 neuron   (aktivasi ReLU)
Hidden Layer 3  →  300 neuron   (aktivasi ReLU)
Output Layer    →  3 neuron     (aktivasi Softmax → 3 kelas)
```

> **ReLU** dipakai di hidden layer karena efektif mengatasi masalah vanishing gradient.
> **Softmax** dipakai di output layer karena menghasilkan probabilitas untuk tiap kelas (total = 1).

---

## 📋 Penjelasan Kode

### 1. Import Library
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```
Mengimpor semua library yang dibutuhkan. `Sequential` untuk membangun model berlapis, `Dense` untuk membuat layer fully connected, `LabelEncoder` untuk mengubah label teks menjadi angka.

---

### 2. Load Dataset
```python
dataset = pd.read_csv('iris.data', header=None, sep=',')
X = dataset.iloc[:, :-1].values  # 4 kolom fitur
y = dataset.iloc[:, -1].values   # kolom label
```
Dataset dibaca dari file CSV lokal. `X` berisi 4 kolom fitur (input), `y` berisi label spesies (output). `header=None` karena file tidak punya baris judul kolom.

---

### 3. Label Encoding
```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```
Label berupa teks (`'Iris-setosa'`, dll.) diubah jadi angka `0, 1, 2` agar bisa diproses model. Hasilnya:
- `Iris-setosa` → `0`
- `Iris-versicolor` → `1`
- `Iris-virginica` → `2`

---

### 4. Split Data
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Data dibagi menjadi **80% data latih** (120 data) dan **20% data uji** (30 data). `random_state=42` agar pembagian selalu sama setiap dijalankan.

---

### 5. Membuat Model
```python
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500,  activation='relu'),
    Dense(300,  activation='relu'),
    Dense(3,    activation='softmax')
])
model.summary()
```
Model dibuat secara berurutan (Sequential). `Input(shape=...)` menerima 4 fitur. Tiga hidden layer dengan neuron semakin kecil membantu model mengekstrak pola dari data secara bertahap. Output layer dengan 3 neuron menghasilkan probabilitas untuk 3 kelas.

---

### 6. Kompilasi Model
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- **optimizer `adam`**: Algoritma yang memperbarui bobot model secara adaptif — lebih cepat konvergen dibanding SGD biasa.
- **loss `sparse_categorical_crossentropy`**: Digunakan karena label berbentuk integer (0, 1, 2), bukan one-hot encoding.
- **metrics `accuracy`**: Memantau akurasi selama training.

---

### 7. Training Model
```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```
Model dilatih selama **50 epoch**. Setiap epoch, model melihat seluruh data latih dalam batch-batch 32 data. `validation_data` digunakan untuk memantau performa di data uji setiap epoch tanpa melatih model pada data tersebut.

---

### 8. Evaluasi
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
```
Mengukur performa akhir model pada data uji. Hasil percobaan menunjukkan **accuracy ≈ 96–100%**.

---

### 9. Visualisasi Training
```python
pd.DataFrame(history.history).plot(figsize=(10,6))
```
Menampilkan grafik perubahan nilai `loss` dan `accuracy` selama 50 epoch untuk training dan validasi. Berguna untuk mendeteksi apakah model mengalami overfitting.

---

### 10. Prediksi
```python
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
print("Prediksi:", predicted_classes)
print("Label Asli:", y_test)
```
`model.predict()` menghasilkan probabilitas tiap kelas. `.argmax(axis=1)` mengambil indeks kelas dengan probabilitas tertinggi sebagai hasil prediksi akhir.

---

### 11. Confusion Matrix
```python
cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
```
Confusion matrix menunjukkan perbandingan prediksi vs label asli per kelas. Angka di diagonal = prediksi **benar**, di luar diagonal = prediksi **salah**.

---

### 12. Prediksi Data Baru
```python
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    ...
    prediction = model.predict(new_data)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")
```
Fungsi interaktif yang memungkinkan pengguna memasukkan ukuran bunga baru dan langsung mendapat prediksi spesiesnya. `inverse_transform` mengubah angka kembali ke nama spesies asli.

---

## 📊 Hasil Percobaan

| Metrik | Nilai |
|--------|-------|
| Loss | ~0.07 |
| Accuracy | **96.67% – 100%** |

Contoh hasil confusion matrix menunjukkan hanya 0–1 data yang salah prediksi dari 30 data uji, umumnya terjadi antara kelas `Iris-versicolor` dan `Iris-virginica` karena keduanya memiliki fitur yang paling mirip.

---
