import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, MobileNetV2, DenseNet121
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import random

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories for spectrogram images and Excel files
SPECTROGRAM_DIR = "Audios and excels/Cleaned_Audios/spectrogram_images"
EXCEL_DIR = "Audios and excels/Cleaned_Excels"
IMG_SIZE = (224, 224)  # Image size for resizing the spectrogram images
BATCH_SIZE = 8  # Batch size for training
NUM_CLASSES = 3  # Number of classes for classification

# Function to load spectrogram images from the directory
def load_spectrograms():
    image_paths = glob.glob(os.path.join(SPECTROGRAM_DIR, "*.png"))  # Get all PNG files in the directory
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {SPECTROGRAM_DIR}")
    logging.info(f"Found {len(image_paths)} spectrogram images")
    
    images, filenames = [], []
    for path in image_paths:
        try:
            # Load and resize the image
            img = load_img(path, target_size=IMG_SIZE)
            img_array = img_to_array(img)  # Convert the image to a numpy array
            if img_array.shape != (*IMG_SIZE, 3):  # Ensure correct shape
                logging.warning(f"Image {path} has unexpected shape: {img_array.shape}")
                continue
            images.append(img_array)  # Add image to the list
            filenames.append(os.path.basename(path))  # Store the filename for later use
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
    
    if not images:
        raise ValueError("No valid images were loaded")
    return np.array(images), filenames  # Return the loaded images and filenames as arrays

# Function to load Excel data containing the clarity labels
def load_excel_data():
    excel_files = glob.glob(os.path.join(EXCEL_DIR, "*.xlsx"))  # Get all Excel files
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {EXCEL_DIR}")
    logging.info(f"Found {len(excel_files)} Excel files")
    
    all_data = []
    for file in excel_files:
        try:
            df = pd.read_excel(file)  # Read the Excel file into a dataframe
            if df.empty:
                logging.warning(f"Empty Excel file: {file}")
                continue
            df['source_file'] = os.path.basename(file)  # Add the source filename for reference
            all_data.append(df)  # Append to the list of data
            logging.info(f"Successfully loaded {file} with {len(df)} rows")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No valid Excel data was loaded")
    combined_data = pd.concat(all_data, ignore_index=True)  # Concatenate all data into one dataframe
    logging.info(f"Combined data shape: {combined_data.shape}")
    return combined_data

# Function to prepare target variables (clarity labels) for model training
def prepare_target_variables(excel_data, filenames):
    clarity_map = {'Low': 0, 'Medium': 1, 'High': 2}  # Mapping clarity levels to numeric values
    target_mapping = {}
    
    # Extract target values from the Excel data
    for idx, row in excel_data.iterrows():
        try:
            sample_num = str(row['source_file']).split('_')[1]  # Extract sample number from the filename
            clarity_value = row['Clarity'] if 'Clarity' in row else None  # Get clarity value
            if clarity_value in clarity_map:
                target_mapping[sample_num] = clarity_map[clarity_value]  # Map sample number to clarity
        except Exception as e:
            logging.warning(f"Error processing row {idx}: {e}")
            continue
    
    targets, valid_indices = [], []
    
    # Match target values with filenames
    for i, filename in enumerate(filenames):
        try:
            sample_num = filename.split('_')[0]  # Extract sample number from the filename
            if sample_num in target_mapping:
                targets.append(target_mapping[sample_num])  # Add target for valid sample
                valid_indices.append(i)  # Store the index of the valid sample
        except Exception as e:
            logging.warning(f"Error processing filename {filename}: {e}")
            continue
    
    if not targets:
        raise ValueError("No valid target values were found")
    return np.array(targets, dtype=np.int32), valid_indices

# Functions to create various CNN-based models

def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    base_model.trainable = False  # Freeze the base model layers
    return model

def create_alexnet_model():
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Additional CNN model creation functions (create_custom_cnn_model, create_mobilenetv2_model, etc.)

# Function to plot training history (accuracy and loss)
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png')  # Save plot
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)  # Generate confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name}.png')  # Save plot
    plt.close()

# Function to train and evaluate the model
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, class_weights):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=20,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ],
        verbose=2
    )
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate classification report and confusion matrix
    report = classification_report(y_test, y_pred_classes, target_names=['Low', 'Medium', 'High'])
    print(f"\n{model_name} Results:")
    print(report)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save plots and model
    plot_training_history(history, model_name)
    plot_confusion_matrix(y_test, y_pred_classes, ['Low', 'Medium', 'High'], model_name)
    model.save(f'spectrogram_model_{model_name}.h5')  # Save the trained model
    
    return val_acc, report, history.history['accuracy'][-1]

# Function to create a comparison table of model results
def create_comparison_table(results):
    # Create a DataFrame for comparison
    comparison_data = []
    
    for model_name, result in results.items():
        report_lines = result['report'].split('\n')  # Parse the classification report
        metrics = {}
        
        # Extract metrics for each class
        for line in report_lines[2:5]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    class_name = parts[0]
                    metrics[f'{class_name}_precision'] = float(parts[1])
                    metrics[f'{class_name}_recall'] = float(parts[2])
                    metrics[f'{class_name}_f1'] = float(parts[3])
        
        # Add overall metrics
        metrics['model'] = model_name
        metrics['accuracy'] = result['accuracy']
        
        comparison_data.append(metrics)
    
    df = pd.DataFrame(comparison_data)  # Convert list to DataFrame
    df = df.round(4)  # Round numeric values
    
    df.to_csv('model_comparison.csv', index=False)  # Save comparison table to CSV
    
    # Print detailed model comparison table
    print("\nModel Comparison Table:")
    print("=" * 100)
    print(f"{'Model':<10} {'Accuracy':<10} {'Low':<20} {'Medium':<20} {'High':<20}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<10} {row['accuracy']:<10.4f}", end='')
        for class_name in ['Low', 'Medium', 'High']:
            print(f" {row[f'{class_name}_precision']:<6.4f} {row[f'{class_name}_recall']:<6.4f} {row[f'{class_name}_f1']:<6.4f}", end='')
        print()
    
    print("=" * 100)
    print("\nDetailed comparison saved to 'model_comparison.csv'")
    
    return df

# Main function to execute the entire workflow
def main():
    print("Loading and preprocessing data...")
    X, filenames = load_spectrograms()
    X = X / 255.0  # Normalize image data
    excel_data = load_excel_data()
    y, valid_indices = prepare_target_variables(excel_data, filenames)
    X = X[valid_indices]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))  # Compute class weights to handle imbalance
    
    # Models to train
    models_to_train = {
        'VGG16': create_vgg16_model,
        'AlexNet': create_alexnet_model,
        'CustomCNN': create_custom_cnn_model,
        'MobileNetV2': create_mobilenetv2_model,
        'DenseNet121': create_densenet121_model,
        'ShallowCNN': create_shallow_cnn_model
    }
    
    results = {}
    for model_name, model_creator in models_to_train.items():
        print(f"\nTraining {model_name}...")
        model = model_creator()
        val_acc, report, train_acc = train_and_evaluate_model(
            model, model_name, X_train, y_train, X_test, y_test, class_weights_dict
        )
        results[model_name] = {'accuracy': val_acc, 'report': report, 'train_accuracy': train_acc}
    
    # Create and display comparison table
    comparison_df = create_comparison_table(results)
    
    # Print detailed classification reports
    print("\nDetailed Classification Reports:")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(result['report'])

if __name__ == "__main__":
    main() 
