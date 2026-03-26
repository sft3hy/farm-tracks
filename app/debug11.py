"""Test instant loading from arrow files"""
from datasets import Dataset, concatenate_datasets
import glob
import time

start = time.time()
arrow_files = sorted(glob.glob("/root/.cache/huggingface/datasets/shi-labs___agriculture-vision/default-5eb8c5b2696fef73/0.0.0/3d7d6c5fbf08e3d05aff6d04d0ea9dcb846b4acb/agriculture-vision-train-*.arrow"))

ds_list = []
for f in arrow_files:
    ds_list.append(Dataset.from_file(f))

ds = concatenate_datasets(ds_list)
print(f"Loaded {len(ds)} rows from {len(arrow_files)} files in {time.time() - start:.2f} seconds!")
print("Columns:", ds.column_names)

print("Checking RGB images near index 431062...")
for i in range(431062, 431065):
    sample = ds[i]
    print(f"Index {i}: png column is {type(sample.get('png'))}")
    print(f"Index {i}: jpg column is {type(sample.get('jpg'))}")
