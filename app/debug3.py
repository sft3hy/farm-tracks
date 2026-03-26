"""Quick check of column structure without custom features."""
from datasets import load_dataset

ds = load_dataset("shi-labs/Agriculture-Vision", split="train")
print(f"Total rows: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")

# Print first 5 samples
for i in range(5):
    sample = ds[i]
    print(f"\n--- Entry {i} ---")
    for k, v in sample.items():
        if v is None:
            print(f"  {k}: None")
        elif hasattr(v, 'size'):
            print(f"  {k}: Image({v.size})")
        elif isinstance(v, str):
            print(f"  {k}: '{v[:100]}'")
        else:
            print(f"  {k}: {type(v).__name__}")
