import numpy as np
import rasterio
from pathlib import Path

RASTER_FILES = Path("/Users/advaitdarbare/test_475/Raster_Burn_Data")
PREPROCESSED_FILES = Path("/Users/advaitdarbare/test_475/processed_data")
CHIP_SIZE = 256
STRIDE = 128

# testing files
TESTING_SET = {'raster_final_horton.tif', 'raster_final_medicine.tif'}

# validation files
VALIDATION_SET = {'raster_final_salt.tif', 'raster_final_sawtooth.tif'}

# normalizing the bands 
def normalize_bands(band_index, total_bands):
    
    # checking if the band is not environmental feature
    if band_index < (total_bands - 7):
        return 1.0
    
    # getting the index of the environmental feature
    environment_index = band_index - (total_bands - 7)
    
    # need to check if these factors are correct
    # normalization factors for the environmental features
    normalization_factors = {
        0: 10.0,   # Landcover
        1: 3000.0, # Elevation
        2: 90.0,   # Slope
        3: 360.0,  # Aspect
        4: 10.0,   # Wind Speed
        5: 1.0,    # Wind Direction Sin
        6: 1.0     # Wind Direction Cos
    }
    
    return normalization_factors.get(environment_index, 1.0)

# getting the data split for the type of file
def get_filenames(filename):
    if filename in TESTING_SET:
        return "testing"
    elif filename in VALIDATION_SET:
        return "validation"
    else:
        return "training"


# function that will pad the raster fiels
def padding_rasterfiles(file_path, min_size):
    # open the raster file
    with rasterio.open(file_path) as src:
        data = src.read().astype(np.float32)
    

    band, height, width = data.shape
    
    # calculate the padding needed

    # padding if the height is les than 256
    pad_height = max(0, min_size - height)

    # padding if the width is less than 256
    pad_width = max(0, min_size - width)
    

    # apply padding if needed, thatz if the padding is less than 256 for the hieght or width
    if pad_height > 0 or pad_width > 0:
        data = np.pad(data, pad_width=((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    
    return data

# functuoion that will apply the normalization to the raster files
def normalize_raster(data):

    # get the number of bands
    band = data.shape[0]
    
    # for each of the bannd, normalizing each feature inthe data
    for band_index in range(band):

        # run the normalizing function, to get the factor
        norm_factor = normalize_bands(band_index, band)

        # if the factor is not 1, then normalize data
        if norm_factor != 1.0:
            data[band_index] /= norm_factor
    
    return data

# functuin taht is going to extract the chips from the files
def extract_chips(data, chip_size, stride):

    chips_positions_data = [] # trakc all the chips

    # get the height and width of the data
    _, height, width = data.shape

    # extracting the chips
    for y in range(0, height - chip_size + 1, stride):
        for x in range(0, width - chip_size + 1, stride):
            
            chip = data[:, y:y + chip_size, x:x + chip_size]
            chips_positions_data.append((chip, y, x))
    
    return chips_positions_data

# saving the chip files
def save_chips(chips, filename, save_dir):

    for i, (chip, y, x) in enumerate(chips):
        
        np.save(save_dir / f"{filename[:-4]}_{i}_{y}_{x}.npy", chip)

    return len(chips)


def process_file(filename):

    file_path = RASTER_FILES / filename
    
    # determine the split
    split = get_filenames(filename)
    save_dir = PREPROCESSED_FILES / split
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # calculate padding
    data = padding_rasterfiles(file_path, CHIP_SIZE)
    
    # normalizing the rasters
    data = normalize_raster(data)
    
    # extract chips
    chips = extract_chips(data, CHIP_SIZE, STRIDE)
    
    # saving chips
    num_chips = save_chips(chips, filename, save_dir)
    
    print(f"Processed {filename}: {num_chips} chips saved to {split} directory")


if __name__ == "__main__":
    PREPROCESSED_FILES.mkdir(parents=True, exist_ok=True)
    
    # get all the tif files
    tiff_files = [f.name for f in RASTER_FILES.glob("*.tif")]
    
    # Preprocess each tif file
    for tiff_file in tiff_files:
        process_file(tiff_file)