import os
import shutil
from datasets import load_dataset, Features, Image as HFImage, Value

def check_disk_space(path):
    """Checks available disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (2**30)

def download_agriculture_vision(cache_dir=None, streaming=False):
    """
    Downloads and caches the Agriculture-Vision dataset from Hugging Face.
    This dataset is approximately 21GB compressed.
    """
    if cache_dir is None:
        cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))
    
    print(f"🚀 Initializing Agriculture-Vision Dataset Download (Streaming: {streaming})...")
    print(f"📍 Target Cache Directory: {cache_dir}")
    
    # Disk space check
    if not streaming:
        free_gb = check_disk_space(os.path.dirname(cache_dir.rstrip('/')))
        print(f"📊 Available Disk Space: {free_gb:.2f} GB")
        if free_gb < 50:
            print("⚠️  Warning: This dataset is 21GB compressed and may require 50GB+ for extraction.")
            print("   Consider using a larger partition or enabling streaming mode.")

    # Define features to ensure consistent loading
    features = Features({
        "png": HFImage(),
        "jpg": HFImage(),
        "__key__": Value("string"),
        "__url__": Value("string"),
    })

    print("📦 Fetching 'shi-labs/Agriculture-Vision' (train split)...")
    try:
        dataset = load_dataset(
            "shi-labs/Agriculture-Vision", 
            split="train", 
            features=features,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True
        )
        
        if streaming:
            print("✅ Streaming mode enabled. Accessing dataset without full download.")
            # Verify we can access the first item
            first_item = next(iter(dataset))
            print(f"✅ Connection successful! Sample key: {first_item.get('__key__')}")
        else:
            print(f"✅ Download Complete! Total samples: {len(dataset)}")
            print(f"📍 Dataset cached at: {cache_dir}")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        if "No space left on device" in str(e) or "error occurred while generating" in str(e):
            print("\n💡 Recommendation: It looks like you ran out of disk space.")
            print("Try one of the following:")
            print(f"1. Specify a larger volume: HF_HOME=/path/to/large/disk python {os.path.basename(__file__)}")
            print(f"2. Use streaming mode: Set streaming=True in {os.path.basename(__file__)}")

if __name__ == "__main__":
    # Enabled streaming mode to avoid disk space issues on EC2/SageMaker
    download_agriculture_vision(streaming=True)
