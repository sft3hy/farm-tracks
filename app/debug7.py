from datasets import Features, Image as HFImage, Value, load_dataset
# Must exactly match train.py to get the same cache hash
features = Features({
    'jpg': HFImage(),
    'png': HFImage(),
    '__key__': Value('string'),
    '__url__': Value('string'),
})

try:
    ds = load_dataset(
        "shi-labs/Agriculture-Vision",
        split="train",
        features=features
    )
    print("SUCCESS: Loaded custom features dataset from cache!")
    print(ds.column_names)
except Exception as e:
    print(f"FAILED: {e}")
