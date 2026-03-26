"""Lightweight diagnostic: dump unique key patterns and folder structures."""
from datasets import load_dataset, Features, Image as HFImage, Value
from collections import Counter

features = Features({
    'png': HFImage(),
    'jpg': HFImage(),
    '__key__': Value('string'),
    '__url__': Value('string')
})

ds = load_dataset("shi-labs/Agriculture-Vision", split="train", features=features, streaming=True)

# Just scan a bigger window and collect ALL unique folder paths
folder_counts = Counter()
url_set = set()
sample_keys_by_folder = {}

print("Scanning 2000 entries from offset 0...")
for i, sample in enumerate(ds):
    if i >= 2000:
        break
    key = sample.get('__key__', '')
    url = sample.get('__url__', '')
    parts = key.split('/')
    
    # Collect full path minus filename
    folder_path = '/'.join(parts[:-1]) if len(parts) > 1 else 'root'
    folder_counts[folder_path] += 1
    url_set.add(url)
    
    if folder_path not in sample_keys_by_folder:
        sample_keys_by_folder[folder_path] = key
    
    # Also check: does sample have jpg data?
    has_jpg = sample.get('jpg') is not None
    has_png = sample.get('png') is not None
    
    if i < 5:
        print(f"  [{i}] key={key}")
        print(f"       url={url}")
        print(f"       has_jpg={has_jpg}, has_png={has_png}")

print(f"\nUnique folder paths ({len(folder_counts)}):")
for path, count in folder_counts.most_common():
    print(f"  {path}: {count} entries")
    print(f"    Sample: {sample_keys_by_folder[path]}")

print(f"\nUnique URLs ({len(url_set)}):")
for u in sorted(url_set):
    print(f"  {u}")

print("\nDone!")
