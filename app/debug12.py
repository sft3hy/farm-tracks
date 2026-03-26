import glob
from datasets import Dataset, concatenate_datasets
import numpy as np

print("Loading cached dataset...")
arrow_files = sorted(glob.glob("/root/.cache/huggingface/datasets/shi-labs___agriculture-vision/default-5eb8c5b2696fef73/0.0.0/3d7d6c5fbf08e3d05aff6d04d0ea9dcb846b4acb/agriculture-vision-train-*.arrow"))
ds = concatenate_datasets([Dataset.from_file(f) for f in arrow_files])

print("Scanning planter_skip masks...")
MASK_START = 680000
MAX_SCAN = 1000

found_nonzero = 0
for i in range(MASK_START, MASK_START + MAX_SCAN):
    key = ds[i]['__key__']
    if 'planter_skip' not in key:
        continue
    img = ds[i]['png']
    if img is None:
         continue
    arr = np.array(img)
    nonzero = np.count_nonzero(arr)
    if nonzero > 0:
         found_nonzero += 1
         # print(f"Index {i} ({key}): {nonzero} non-zero pixels out of {arr.size}")

print(f"Out of {MAX_SCAN} images checked, {found_nonzero} had anomalies.")
