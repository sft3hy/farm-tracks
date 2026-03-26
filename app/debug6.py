"""Test loading RGB images from the dataset and see if they are None"""
from datasets import load_dataset
ds = load_dataset("shi-labs/Agriculture-Vision", split="train")

print("Checking RGB images near index 431062...")
for i in range(431062, 431072):
    sample = ds[i]
    key = sample['__key__']
    if 'images/rgb' in key:
        png_val = sample.get('png')
        if png_val is None:
            print(f"Index {i}: png column is None!")
        else:
            print(f"Index {i}: png column is {type(png_val)} (size {getattr(png_val, 'size', 'unknown')})")
        
        jpg_val = sample.get('jpg')
        if jpg_val is not None:
             print(f"  Wait, jpg column exists? {type(jpg_val)}")
