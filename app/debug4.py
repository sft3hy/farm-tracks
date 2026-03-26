"""Find where rgb and planter_skip entries are in the 1M-entry dataset."""
from datasets import load_dataset
from collections import Counter

ds = load_dataset("shi-labs/Agriculture-Vision", split="train")
total = len(ds)
print(f"Total: {total}")

# Sample at various offsets to find which ranges contain rgb/planter_skip
offsets = [0, 10000, 50000, 100000, 200000, 300000, 500000, 700000, 900000, 1000000]

for offset in offsets:
    if offset >= total:
        continue
    end = min(offset + 200, total)
    folders = Counter()
    for i in range(offset, end):
        key = ds[i]['__key__']
        parts = key.split('/')
        # Extract the folder structure (e.g., "train/images/rgb" or "val/masks")
        folder_path = '/'.join(parts[:-1])
        folders[folder_path] += 1
    print(f"\n--- Offset {offset}-{end} ---")
    for path, count in folders.most_common(10):
        print(f"  {path}: {count}")

# Binary search for first RGB entry
print("\n=== Binary searching for first 'rgb' entry ===")
lo, hi = 0, total - 1
first_rgb = None
while lo <= hi:
    mid = (lo + hi) // 2
    key = ds[mid]['__key__']
    if 'rgb' in key:
        first_rgb = mid
        hi = mid - 1
    else:
        # Check if any entry after mid has rgb
        found = False
        for check in range(mid, min(mid + 100, total)):
            if 'rgb' in ds[check]['__key__']:
                first_rgb = check
                hi = mid - 1
                found = True
                break
        if not found:
            lo = mid + 1

if first_rgb is not None:
    print(f"First RGB entry at index: {first_rgb}")
    print(f"Key: {ds[first_rgb]['__key__']}")
else:
    print("No RGB entry found!")

# Also search for planter_skip
print("\n=== Searching for 'planter_skip' ===")
for i in range(0, total, total // 20):
    key = ds[i]['__key__']
    if 'planter_skip' in key:
        print(f"Found planter_skip at index {i}: {key}")
        break
