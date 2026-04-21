import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import argparse
import os
import sys

# --- CLI args ---
parser = argparse.ArgumentParser(description='Evaluate ensemble DenseNet+Inception on held-out test set')
parser.add_argument('--data-csv',        default='anemia_dataset_cleaned.csv', help='Path to dataset CSV')
parser.add_argument('--densenet-model',  default='best_densenet_fold_1.keras', help='Path to DenseNet model')
parser.add_argument('--inception-model', default='best_inception_fold_1.keras', help='Path to Inception model')
parser.add_argument('--output-dir',      default='.', help='Directory to write outputs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

if not os.path.isfile(args.data_csv):
    print(f"Error: dataset CSV not found at {args.data_csv}")
    sys.exit(2)

df = pd.read_csv(args.data_csv)
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Using {len(test_df)} images for final ensemble evaluation.")

# --- 2. Load Trained Models ---
print("Loading trained models...")
try:
    if not os.path.isfile(args.densenet_model):
        raise FileNotFoundError(args.densenet_model)
    if not os.path.isfile(args.inception_model):
        raise FileNotFoundError(args.inception_model)
    model_densenet  = tf.keras.models.load_model(args.densenet_model)
    model_inception = tf.keras.models.load_model(args.inception_model)
    print("Both models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please make sure you have run the K-Fold scripts to generate the saved '.keras' files first.")
    sys.exit(1)

# --- 3. Create Separate Data Generators ---
# FIX: InceptionV3 was trained at 299x299 — must match here
test_datagen_densenet  = ImageDataGenerator(preprocessing_function=densenet_preprocess)
test_datagen_inception = ImageDataGenerator(preprocessing_function=inception_preprocess)

test_generator_densenet = test_datagen_densenet.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='label',
    target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)
test_generator_inception = test_datagen_inception.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='label',
    target_size=(299, 299), batch_size=32, class_mode='binary', shuffle=False
)

# --- 4. Get Predictions ---
print("\nGetting predictions from each model...")
preds_densenet  = model_densenet.predict(test_generator_densenet)
preds_inception = model_inception.predict(test_generator_inception)

# --- 5. Ensemble by Averaging ---
print("Averaging predictions to create ensemble result...")
ensemble_probs   = (preds_densenet + preds_inception) / 2.0
y_true           = test_generator_densenet.classes
ensemble_classes = (ensemble_probs > 0.5).astype("int32").flatten()

# --- 6. Evaluate ---
print("\n--- Final Ensemble Performance Report (DenseNet121 + InceptionV3) ---")

mse = mean_squared_error(y_true, ensemble_probs)
print(f"Aggregate Mean Squared Error (MSE): {mse:.4f}\n")

target_names = list(test_generator_densenet.class_indices.keys())
report = classification_report(y_true, ensemble_classes, target_names=target_names)
print("--- Aggregate Classification Report ---")
print(report)

print("--- Aggregate Confusion Matrix ---")
cm = confusion_matrix(y_true, ensemble_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Final Ensemble Confusion Matrix (DenseNet121 + InceptionV3)')
out_cm = os.path.join(args.output_dir, 'confusion_matrix_final_ensemble_two_models.png')
plt.savefig(out_cm)
print(f"\nFinal ensemble confusion matrix saved as '{out_cm}'")
