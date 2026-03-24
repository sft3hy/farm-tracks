import cv2
import numpy as np
from skimage.morphology import skeletonize
import json

def close_gaps(mask, kernel_size=5):
    """
    Applies morphological closing to fill small gaps in the tracks.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

def extract_skeleton(mask):
    """
    Reduces the track mask to a centerline skeleton.
    Requires input mask to be binary 0-1.
    """
    binary_mask = (mask > 127).astype(np.uint8)
    skeleton = skeletonize(binary_mask)
    return (skeleton * 255).astype(np.uint8)

def mask_to_polygons(mask):
    """
    Converts a binary mask to a list of polygons using cv2.findContours.
    Returns format suitable for GeoJSON.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Filter small artifacts
        if cv2.contourArea(contour) > 50:
            contour = contour.squeeze().tolist()
            if isinstance(contour[0], list):
                # Close the polygon
                contour.append(contour[0])
                polygons.append(contour)
                
    return polygons

def save_geojson(polygons, output_path):
    """
    Saves pixel polygons to a basic GeoJSON file.
    Note: Real GeoJSON requires geographic coordinates.
    """
    features = []
    for poly in polygons:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly]
            },
            "properties": {}
        })
        
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
