import os
from datasets import load_dataset, Features, Image as HFImage, Value

def download_agriculture_vision():
    """
    Downloads and caches the Agriculture-Vision dataset from Hugging Face.
    This dataset is approximately 21GB.
    """
    print("🚀 Initializing Agriculture-Vision Dataset Download...")
    
    # Define features to ensure consistent loading
    features = Features({
        "png": HFImage(),
        "jpg": HFImage(),
        "__key__": Value("string"),
        "__url__": Value("string"),
    })

    print("📦 Fetching 'shi-labs/Agriculture-Vision' (train split)...")
    # This will download the dataset if it's not already in the HF cache
    dataset = load_dataset(
        "shi-labs/Agriculture-Vision", 
        split="train", 
        features=features,
        cache_dir=os.path.expanduser("~/.cache/huggingface/datasets")
    )
    
    print(f"✅ Download Complete! Total samples: {len(dataset)}")
    print(f"📍 Dataset cached at: {os.path.expanduser('~/.cache/huggingface/datasets')}")

if __name__ == "__main__":
    try:
        download_agriculture_vision()
    except Exception as e:
        print(f"❌ Error during download: {e}")
