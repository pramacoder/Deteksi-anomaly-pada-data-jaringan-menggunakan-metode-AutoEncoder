import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support, mean_squared_error, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Pipeline Setup & Data Loading
print("1. Memuat Data dari UNSW-NB15...")
train_df = pd.read_csv('UNSW_NB15_training-set.csv')
test_df = pd.read_csv('UNSW_NB15_testing-set.csv')

# Gabungkan data untuk keperluan preprocessing secara global
df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Preprocessing & Data Cleaning
print("2. Memulai Preprocessing Data...")

# a. Drop kolom ID yang tidak relevan
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# b. Hapus duplikat
df = df.drop_duplicates()

# c. Penanganan Missing Values (jika ada)
df = df.fillna(df.median(numeric_only=True))

# Identifikasi fitur kategorikal untuk di-encode
categorical_cols = ['proto', 'service', 'state']
label_encoders = {}

# d. Encoding Kategorikal menggunakan LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    # Ubah nilai ke string untuk menghindari error tipe campuran
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Simpan kolom target dan buat data numerik saja
y = df['label'].values  # 0 untuk normal, 1 untuk anomaly
attack_cat = df['attack_cat'].values if 'attack_cat' in df.columns else None

# Autencoder hanya perlu dilatih pada data numerik
features = df.drop(columns=['label', 'attack_cat'], errors='ignore')

# e. Seleksi Fitur (Opsional: Filter berdasarkan korelasi - Disini kita pertahankan 
#    semua fitur bersih sebagai input dasar karena AutoEncoder dapat mereduksi dimensionalitas).
feature_columns = features.columns

# f. Normalisasi menggunakan MinMaxScaler
print("   Normalisasi (Min-Max Scaling)...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

# 3. Data Splitting untuk AutoEncoder (Unsupervised Anomaly Detection)
# Kita melatih model HANYA pada data NORMAL (label == 0).
# Evaluasi dilakukan pada seluruh data test (Normal + Anomaly).

normal_indices = np.where(y == 0)[0]
anomaly_indices = np.where(y == 1)[0]

# Memisahkan 70% normal train, 15% val, 15% test
X_normal = X_scaled[normal_indices]
X_anomaly = X_scaled[anomaly_indices]

# Split validation dan test. Proporsi dihitung dari total normal agar mendekati 70:15:15
X_train_norm, X_temp_norm = train_test_split(X_normal, test_size=0.3, random_state=42)
X_val_norm, X_test_norm = train_test_split(X_temp_norm, test_size=0.5, random_state=42)

# Set Test: gabungkan sisa normal test dengan semua anomaly untuk simulasi test sungguhan
# (Atau sebagian anomaly untuk menyesuaikan proporsi). Di sini pakai semua anomaly (test-set).
X_test = np.vstack((X_test_norm, X_anomaly))
y_test = np.concatenate([np.zeros(len(X_test_norm)), np.ones(len(X_anomaly))])

# Acak X_test
test_indices = np.arange(len(X_test))
np.random.shuffle(test_indices)
X_test = X_test[test_indices]
y_test = y_test[test_indices]

print(f"   Dimensi Training Normal: {X_train_norm.shape}")
print(f"   Dimensi Validation Normal: {X_val_norm.shape}")
print(f"   Dimensi Test (Normal + Anomaly): {X_test.shape}")

# 4. Membangun Model AutoEncoder
print("3. Membangun Arsitektur AutoEncoder...")
input_dim = X_train_norm.shape[1]

# Encoder
input_layer = layers.Input(shape=(input_dim,))
enc = layers.Dense(32, activation="relu")(input_layer)
enc = layers.Dropout(0.2)(enc)
enc = layers.Dense(16, activation="relu")(enc)

# Latent Space
latent = layers.Dense(8, activation="relu")(enc)

# Decoder
dec = layers.Dense(16, activation="relu")(latent)
dec = layers.Dropout(0.2)(dec)
dec = layers.Dense(32, activation="relu")(dec)
output_layer = layers.Dense(input_dim, activation="sigmoid")(dec)  # output scaled (0-1)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# 5. Pelatihan dengan Early Stopping
print("4. Melatih Model AutoEncoder...")
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

history = autoencoder.fit(
    X_train_norm, X_train_norm,
    epochs=50,
    batch_size=256,
    validation_data=(X_val_norm, X_val_norm),
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluasi dan Penentuan Threshold Anomaly
print("5. Tahap Evaluasi Model...")

# Tentukan error threshold berdasarkan rekonstruksi data normal (mis. 95th percentile)
train_pred = autoencoder.predict(X_train_norm)
train_mse = np.mean(np.power(X_train_norm - train_pred, 2), axis=1)
threshold = np.percentile(train_mse, 95)
print(f"   Threshold Anomaly MSE ditentukan pada: {threshold:.6f}")

# Prediksi menggunakan data test
test_pred = autoencoder.predict(X_test)
test_mse = np.mean(np.power(X_test - test_pred, 2), axis=1)

# Jika MSE > Threshold, maka dikategorikan sebagai anomaly (1)
y_pred = (test_mse > threshold).astype(int)

# Hitung Metrik Secara Umum
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
fpr, tpr, _ = roc_curve(y_test, test_mse)
roc_auc = auc(fpr, tpr)

print("\n=== Laporan Evaluasi ===")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUC-ROC   : {roc_auc:.4f}")
print("========================")
print(classification_report(y_test, y_pred))

# [Opsional] Plot ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("Grafik ROC curve disimpan sebagai 'roc_curve.png'")
print("Pipeline Selesai!")
