import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.utils import resample

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os
import sys

# --- Configuration ---
N_SPLITS = 5
EPOCHS = 30
FINE_TUNE_EPOCHS = 15

# --- CLI args & 1. Load the dataset ---
parser = argparse.ArgumentParser(description='Train DenseNet K-Fold on anemia dataset')
parser.add_argument('--data-csv', default='anemia_dataset_cleaned.csv', help='Path to dataset CSV')
parser.add_argument('--output-dir', default='.', help='Directory to write checkpoints and outputs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

if not os.path.isfile(args.data_csv):
    print(f"Error: dataset CSV not found at {args.data_csv}")
    sys.exit(2)

df = pd.read_csv(args.data_csv)
X = df['filepath']
y = df['label']

# --- K-Fold Cross-Validation Setup ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_no = 1
histories = []
all_scores = []
all_true_labels, all_pred_classes, all_pred_probs = [], [], []

# --- 2. Model Creation Function ---
def create_model():
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 3. K-Fold Training Loop ---
for train_index, test_index in skf.split(X, y):
    print(f"--- Starting DenseNet Fold {fold_no}/{N_SPLITS} ---")

    outer_train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    train_df, val_df = train_test_split(outer_train_df, test_size=0.1, random_state=42, stratify=outer_train_df['label'])

    df_majority = train_df[train_df.label == 'Non-Anemic']
    df_minority = train_df[train_df.label == 'Anemic']
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30, width_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_upsampled, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=True
    )
    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='filepath', y_col='label',
        target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
    )

    model = create_model()
    checkpoint_path = os.path.join(args.output_dir, f'best_densenet_fold_{fold_no}.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_generator, epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping], verbose=1
    )

    model.trainable = True
    for layer in model.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy', metrics=['accuracy']
    )
    history_fine_tune = model.fit(
        train_generator, epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping], verbose=1
    )

    model.load_weights(checkpoint_path)

    full_history = {key: history.history[key] + history_fine_tune.history[key] for key in history.history}
    histories.append(full_history)

    print(f"--- Evaluating Fold {fold_no} ---")
    loss, accuracy = model.evaluate(test_generator)
    all_scores.append(accuracy)

    y_true_fold = test_generator.classes
    y_pred_probs_fold = model.predict(test_generator)
    y_pred_classes_fold = (y_pred_probs_fold > 0.5).astype("int32").flatten()

    all_true_labels.extend(y_true_fold)
    all_pred_probs.extend(y_pred_probs_fold)
    all_pred_classes.extend(y_pred_classes_fold)

    fold_no += 1

# --- 4. Final Aggregated Results ---
print("\n--- DenseNet121 Cross-Validation Complete ---")
print(f"Average Accuracy: {np.mean(all_scores)*100:.2f}% (+/- {np.std(all_scores)*100:.2f}%)")

mse = mean_squared_error(all_true_labels, all_pred_probs)
print(f"Aggregate Mean Squared Error (MSE): {mse:.4f}\n")

print("--- Aggregate Classification Report ---")
target_names = list(test_generator.class_indices.keys())
report = classification_report(all_true_labels, all_pred_classes, target_names=target_names)
print(report)

print("--- Aggregate Confusion Matrix ---")
cm = confusion_matrix(all_true_labels, all_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Aggregate Confusion Matrix (DenseNet121 - All Folds)')
out_cm = os.path.join(args.output_dir, 'confusion_matrix_densenet.png')
plt.savefig(out_cm)
print(f"\nFinal confusion matrix plot saved as '{out_cm}'")

# --- FIX: Plot Average Learning Curves (corrected accumulation logic) ---
print("\n--- Generating Average Learning Curves ---")
max_epochs = max(len(h['loss']) for h in histories)
avg_train_acc  = np.zeros(max_epochs)
avg_val_acc    = np.zeros(max_epochs)
avg_train_loss = np.zeros(max_epochs)
avg_val_loss   = np.zeros(max_epochs)

for h in histories:
    # Pad then ACCUMULATE — do NOT divide inside the loop
    avg_train_acc  += np.pad(h['accuracy'],     (0, max_epochs - len(h['accuracy'])),     'edge')
    avg_val_acc    += np.pad(h['val_accuracy'], (0, max_epochs - len(h['val_accuracy'])), 'edge')
    avg_train_loss += np.pad(h['loss'],         (0, max_epochs - len(h['loss'])),         'edge')
    avg_val_loss   += np.pad(h['val_loss'],     (0, max_epochs - len(h['val_loss'])),     'edge')

# Divide ONCE after the loop
avg_train_acc  /= len(histories)
avg_val_acc    /= len(histories)
avg_train_loss /= len(histories)
avg_val_loss   /= len(histories)

epochs_range = range(1, max_epochs + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, avg_train_acc, label='Train Accuracy')
axes[0].plot(epochs_range, avg_val_acc,   label='Val Accuracy')
axes[0].set_title('DenseNet121 Average Accuracy (K-Fold)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(epochs_range, avg_train_loss, label='Train Loss')
axes[1].plot(epochs_range, avg_val_loss,   label='Val Loss')
axes[1].set_title('DenseNet121 Average Loss (K-Fold)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
out_lc = os.path.join(args.output_dir, 'learning_curves_densenet.png')
plt.savefig(out_lc)
print(f"Learning curves saved as '{out_lc}'")
