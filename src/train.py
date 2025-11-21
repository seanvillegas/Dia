import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
from scipy.interpolate import PchipInterpolator
from tensorflow.keras.losses import Huber
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

# -------------------------------
# Configurable Parameters
# -------------------------------
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

SEQ_LENGTH = 6
FEATURES = ['mg/dl', 'normal', 'carbInput', 'insulinCarbRatio', 
            'bgInput', 'recommended.carb', 'recommended.correction', 
            'insulinSensitivityFactor', 'targetBloodGlucose', 'insulinOnBoard', 'basal_insulin',
            'delta_bg_5min', 'delta_bg_10min', 'delta_bg_15min']
EPOCHS = 50
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
PATIENCE = 8

# -------------------------------
# Custom Metrics
# -------------------------------
def glucose_weighted_mse(y_true, y_pred):
    weight = tf.where(y_true < 70, 2.0, tf.where(y_true > 180, 1.5, 1.0))
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))

def mard(y_true, y_pred):
    epsilon = 1e-6
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon)))

# -------------------------------
# Load and preprocess data with caching
# -------------------------------
INTERPOLATED_FILE = "merged_interpolated.parquet"

if os.path.exists(INTERPOLATED_FILE):
    print(f"Loading interpolated data from {INTERPOLATED_FILE}...")
    data = pd.read_parquet(INTERPOLATED_FILE)
else:
    print("Loading raw data and performing PCHIP interpolation per subject...")
    data = pd.read_csv("merged_model_data.csv", low_memory=False)
    
    # Keep group info for splitting later
    groups = data['subject_id'].values
    
    # Select needed columns + group keys
    data = data[FEATURES + ['subject_id', 'date']]
    data.reset_index(drop=True, inplace=True)

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Diagnostics: missing values before interpolation
    print("Missing values before interpolation per feature:")
    for col in FEATURES:
        missing_count = data[col].isnull().sum()
        total = len(data[col])
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing ({missing_count/total:.2%})")
        else:
            print(f"  {col}: No missing values")

    # Interpolate per subject
    interpolated_chunks = []
    unique_subjects = data['subject_id'].unique()
    print(f"Total subjects: {len(unique_subjects)}")

    for i, subject in enumerate(unique_subjects, 1):
        sub_df = data[data['subject_id'] == subject].copy()

        for col in FEATURES:
            if sub_df[col].isnull().any():
                valid = sub_df[col].dropna()
                if len(valid) >= 2:
                    interpolator = PchipInterpolator(valid.index, valid.values)
                    before_na = sub_df[col].isnull().sum()
                    sub_df.loc[:, col] = interpolator(np.arange(len(sub_df)))
                    after_na = sub_df[col].isnull().sum()
                    imputed = before_na - after_na
                    if imputed > 0:
                        print(f"    Subject {subject}: {imputed} values imputed in '{col}'")
        interpolated_chunks.append(sub_df)
        if i % 50 == 0 or i == len(unique_subjects):
            print(f"Interpolated subjects: {i}/{len(unique_subjects)}")

    data = pd.concat(interpolated_chunks, ignore_index=True)

    # Diagnostics: missing values after interpolation
    print("\nMissing values after interpolation per feature:")
    for col in FEATURES:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            print(f"  Warning: {col} still has {missing_count} missing values after interpolation!")
        else:
            print(f"  {col}: No missing values")

    data.to_parquet(INTERPOLATED_FILE)
    print(f"Saved interpolated data to {INTERPOLATED_FILE}")

# Re-extract groups for sequence splitting
groups = data['subject_id'].values

# Drop unused columns and convert type
data = data.drop(columns=['subject_id', 'date']).astype('float32')

print("Dropping rows with NaNs...")
data = data.dropna()
print(f"Final usable rows: {len(data)}")
groups = groups[data.index]

print("Scaling...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
print("Scaling done.")

# -------------------------------
# Sequence Creation
# -------------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length][0])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH)

def create_group_sequences(groups, seq_length):
    return np.array([groups[i + seq_length] for i in range(len(groups) - seq_length)])

group_seq = create_group_sequences(groups, SEQ_LENGTH)
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=group_seq))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
perm = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm], y_train[perm]

# -------------------------------
# Model
# -------------------------------
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, X.shape[2]), return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(DROPOUT_RATE),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(DROPOUT_RATE),
    Dense(1)
])

model.compile(
    loss=Huber(delta=15.0),
    optimizer='adam',
    metrics=[glucose_weighted_mse, mard]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save
model.save("lstm_gluc_mod.keras")
joblib.dump(scaler, 'scaler.save')

# Evaluate
test_loss, test_gmse, test_mard = model.evaluate(X_test, y_test)
print(f"Test Huber Loss: {test_loss:.4f} | gMSE: {test_gmse:.4f} | MARD: {test_mard:.4f}")

# -------------------------------
# Error vs Glucose Plot
# -------------------------------
preds = model.predict(X_test).flatten()
errors = preds - y_test

plt.figure(figsize=(10,6))
plt.scatter(y_test, errors, alpha=0.3, color='blue', edgecolor='k', linewidth=0.1)
plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
plt.xlabel('Actual Glucose (mg/dL)')
plt.ylabel('Prediction Error (mg/dL)')
plt.title('Prediction Error vs Actual Glucose')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("error_vs_glucose.png")
