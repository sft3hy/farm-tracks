"""Test loading the exact config"""
from datasets import load_dataset
ds = load_dataset(
    "shi-labs/Agriculture-Vision",
    name="default-5eb8c5b2696fef73",
    split="train"
)
print("SUCCESS: Loaded specific config from cache!")
print(ds.column_names)

print("Checking RGB images near index 431062...")
for i in range(431062, 431065):
    sample = ds[i]
    print(f"Index {i}: png column is {type(sample.get('png'))}")
    print(f"Index {i}: jpg column is {type(sample.get('jpg'))}")
