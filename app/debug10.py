"""Test loading arrow directly"""
from datasets import load_dataset
import glob

# Get all arrow files in the train split
arrow_files = glob.glob("/root/.cache/huggingface/datasets/shi-labs___agriculture-vision/default-5eb8c5b2696fef73/0.0.0/3d7d6c5fbf08e3d05aff6d04d0ea9dcb846b4acb/agriculture-vision-train-*.arrow")

try:
    ds = load_dataset("arrow", data_files={"train": arrow_files}, split="train")
    print("SUCCESS: Loaded arrow files directly!")
    print(ds.column_names)
except Exception as e:
    print(f"FAILED: {e}")
