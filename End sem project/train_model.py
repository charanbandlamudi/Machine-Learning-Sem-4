import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load spectrograms
spectrogram_dir = 'spectrograms'
spectrograms = []
for filename in os.listdir(spectrogram_dir):
    if filename.endswith('.npy'):
        spec = np.load(os.path.join(spectrogram_dir, filename))
        spectrograms.append(spec)

# Convert to numpy array and normalize
X = np.array(spectrograms)
X = X / 255.0  # Normalize to [0, 1]

# Load Excel data
excel_dir = 'excel_data'
excel_data = []
for filename in os.listdir(excel_dir):
    if filename.endswith('.xlsx'):
        df = pd.read_excel(os.path.join(excel_dir, filename))
        # Convert categorical columns to numeric
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.Categorical(df[col]).codes
        excel_data.append(df)

# Combine Excel data
y = pd.concat(excel_data, ignore_index=True)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(y.shape[1], activation='sigmoid')
])

# Compile model with appropriate loss and metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
print("\nModel Evaluation:")
test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(X_val, y_val)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save the model
model.save('spectrogram_model.h5')
print("\nModel saved as 'spectrogram_model.h5'")
print("Training history plot saved as 'training_history.png'") 