import rasterio
import numpy as np
import os

# Path to the directory containing the GeoTIFF files
DATA_DIR = "/Users/advaitdarbare/test_475/Raster_Burn_Data"

# List of the files you have access to
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
file_list.sort() # Sort alphabetically for cleaner output

def get_band_label(band_index, total_bands):
    """
    Returns the label for a band based on its position.
    Logic:
    - Band 1: Target (Burn Mask)
    - Last 7 Bands: Environmental Variables
    - Everything else: Spectral Bands
    """
    
    # Calculate position from the end
    # Example: If total is 11. Band 11 is last (offset 0). Band 5 is offset 6.
    offset_from_end = total_bands - band_index
    
    if offset_from_end == 6: return "Env: Landcover"
    if offset_from_end == 5: return "Env: Elevation"
    if offset_from_end == 4: return "Env: Slope"
    if offset_from_end == 3: return "Env: Aspect"
    if offset_from_end == 2: return "Env: Wind Speed"
    if offset_from_end == 1: return "Env: Wind Dir Sin"
    if offset_from_end == 0: return "Env: Wind Dir Cos"
    
    # If it's not the last 7, it represents a day of the fire
    return f"Day {band_index}"

def inspect_file(file_path):
    filename = os.path.basename(file_path)
    print(f"\n{'='*90}")
    print(f"FILE: {filename}")
    
    try:
        with rasterio.open(file_path) as src:
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Total Bands: {src.count}")
            print(f"{'-'*90}")
            # Table Header
            print(f"{'#':<4} | {'Label (Inferred)':<25} | {'Min':<12} | {'Max':<12} | {'Mean':<12}")
            print(f"{'-'*90}")
            
            # Read all data at once (might be heavy for huge files, but these are manageable)
            data = src.read()
            
            for i in range(src.count):
                band_idx = i + 1
                band_data = data[i]
                
                label = get_band_label(band_idx, src.count)
                b_min = band_data.min()
                b_max = band_data.max()
                b_mean = band_data.mean()
                
                print(f"{band_idx:<4} | {label:<25} | {b_min:<12.4f} | {b_max:<12.4f} | {b_mean:<12.4f}")
                
    except Exception as e:
        print(f"ERROR reading {filename}: {e}")

if __name__ == "__main__":
    print("Generating Band Analysis Table for all TIFFs...")
    for f in file_list:
        full_path = os.path.join(DATA_DIR, f)
        inspect_file(full_path)