"""Find exact ranges for train/images/rgb and train/labels/planter_skip."""
from datasets import load_dataset

ds = load_dataset("shi-labs/Agriculture-Vision", split="train")
total = len(ds)

# Find train/images/rgb by scanning from 400k to 700k  
print("Searching for train/images/rgb...")
for start in range(400000, 700000, 10000):
    key = ds[start]['__key__']
    if 'train/images/rgb' in key:
        print(f"Found train/images/rgb at ~{start}: {key}")
        # Now find exact start
        lo = max(start - 10000, 0)
        for i in range(lo, start + 1):
            if 'train/images/rgb' in ds[i]['__key__']:
                print(f"First train/images/rgb at index: {i}")
                print(f"Key: {ds[i]['__key__']}")
                break
        break

# Already know planter_skip is at ~687336, find exact range
print(f"\nplanter_skip at index 687336: {ds[687336]['__key__']}")

# Find first planter_skip
lo = 680000
for i in range(lo, 690000):
    if 'planter_skip' in ds[i]['__key__']:
        print(f"First planter_skip at index: {i}")
        print(f"Key: {ds[i]['__key__']}")
        break

# Find last planter_skip  
for i in range(690000, 710000):
    if 'planter_skip' not in ds[i]['__key__']:
        print(f"Last planter_skip at index: {i-1}")
        print(f"Next key: {ds[i]['__key__']}")
        break
