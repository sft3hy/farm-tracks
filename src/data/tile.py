import os
import rasterio
from rasterio.windows import Window
import numpy as np

def tile_large_image(input_path, output_dir, tile_size=512, overlap=64):
    """
    Slices a large GeoTIFF into overlapping tiles of size `tile_size` x `tile_size`.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    
    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        
        stride = tile_size - overlap
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Handle edge cases to not go out of bounds
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                
                # We want fixed size tiles. If edge is smaller, we can either pad or skip
                # Padding to tile_size
                window = Window(x, y, w, h)
                tile_data = src.read(window=window)
                
                # Check for padding
                if w < tile_size or h < tile_size:
                    padded = np.zeros((src.count, tile_size, tile_size), dtype=tile_data.dtype)
                    padded[:, :h, :w] = tile_data
                    tile_data = padded
                    
                # Setup output profile
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, src.transform)
                })
                
                out_path = os.path.join(output_dir, f"{basename}_tile_{x}_{y}.tif")
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(tile_data)
                    
    print(f"Finished tiling {input_path} to {output_dir}")

if __name__ == "__main__":
    # Example usage
    pass
