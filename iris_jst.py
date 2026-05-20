# PRAKTIKUM 7 - JARINGAN SYARAF TIRUAN 2
# Klasifikasi Spesies Bunga Iris menggunakan TensorFlow & Keras

# LANGKAH 1: Import library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# LANGKAH 2: Load dataset dari file lokal
dataset = pd.read_csv('iris.data', header=None, sep=',')
print(f"Shape dataset: {dataset.shape}")
print(dataset.head())

X = dataset.iloc[:, :-1].values  # 4 kolom fitur
y = dataset.iloc[:, -1].values   # kolom label

# LANGKAH 3: Label Encoder (string → angka)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("Kelas:", label_encoder.classes_)  # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# LANGKAH 4: Split data 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data latih: {X_train.shape[0]}, Data uji: {X_test.shape[0]}")

# LANGKAH 5: Buat arsitektur model
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500,  activation='relu'),
    Dense(300,  activation='relu'),
    Dense(3,    activation='softmax')
])
model.summary()

# LANGKAH 6: Kompilasi model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# LANGKAH 7: Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# LANGKAH 8: Evaluasi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nLoss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# LANGKAH 9: Visualisasi loss & accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Accuracy per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# LANGKAH 10: Prediksi
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
print("Prediksi :", predicted_classes)
print("Label Asli:", y_test)

# LANGKAH 11: Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# LANGKAH 12: Prediksi data baru
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width  = float(input("Masukkan sepal width : "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width  = float(input("Masukkan petal width : "))

    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()