"""
setup_dataset.py
Run this ONCE to create anemia_dataset_cleaned.csv
Make sure all 184 images are inside the 'images' folder
inside your anemia_detection project folder.
"""

import os
import pandas as pd

# This looks for an 'images' folder inside your project folder
IMAGES_FOLDER = "images"
OUTPUT_CSV    = "anemia_dataset_cleaned.csv"

records = []

for filename in os.listdir(IMAGES_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    filepath = os.path.join(IMAGES_FOLDER, filename).replace("\\", "/")
    name = filename.lower()

    if name.startswith('img_1_'):
        label = 'Anemic'
    elif name.startswith('img_2_'):
        label = 'Non-Anemic'
    else:
        print(f"Skipping unknown file: {filename}")
        continue

    records.append({'filepath': filepath, 'label': label})

import random
random.seed(42)
random.shuffle(records)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*40}")
print(f"Total images : {len(df)}")
for label, count in df['label'].value_counts().items():
    print(f"  {label:15s}: {count}")
print(f"{'='*40}")
print(f"\nCSV saved: {OUTPUT_CSV}")
print("\nNow run:")
print("  python DenseNet.py --data-csv anemia_dataset_cleaned.csv --output-dir ./models")
