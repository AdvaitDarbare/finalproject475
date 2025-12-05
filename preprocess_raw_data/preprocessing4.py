import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path


def add_terrain(path):
    #### RASTER TERRAIN ###

    fire_path = path
    prefix = Path(path)
    directory = prefix.parent

    out_path = directory / "timeseries_landcover_terrain.tif"
    terrain_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Data_Bands/AZ_terrain.tif"
    # fire_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Backbone/backbone_multiband.tif"
    # out_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Backbone/backbone_all_static.tif"

    with rasterio.open(fire_path) as fire_src:
        fire_profile = fire_src.profile
        fire_transform = fire_src.transform
        fire_crs = fire_src.crs
        fire_height = fire_src.height
        fire_width = fire_src.width
        fire_band_count = fire_src.count

    #array for 3 terrain bands
    terrain_cropped = np.zeros((3, fire_height, fire_width), dtype=np.float32)

    with rasterio.open(terrain_path) as terr_src:
        for b in range(1, terr_src.count + 1):
            reproject(
                source=rasterio.band(terr_src, b),
                destination=terrain_cropped[b - 1],
                src_transform=terr_src.transform,
                src_crs=terr_src.crs,
                dst_transform=fire_transform,
                dst_crs=fire_crs,
                resampling=Resampling.bilinear
            )

    #add 3 additional bands
    fire_profile.update(count=fire_band_count + 3, dtype="float32")

    #write updated file
    with rasterio.open(fire_path) as fire_src, \
         rasterio.open(out_path, "w", **fire_profile) as dst:

        #write original bands first
        for i in range(fire_band_count):
            data = fire_src.read(i + 1)
            dst.write(data, i + 1)

            #keep naming for last band
            if i == fire_band_count - 1:
                dst.set_band_description(i + 1, "Landcover")

        #write terrain bands
        dst.write(terrain_cropped[0], fire_band_count + 1)
        dst.set_band_description(fire_band_count + 1, "Elevation")

        dst.write(terrain_cropped[1], fire_band_count + 2)
        dst.set_band_description(fire_band_count + 2, "Slope")

        dst.write(terrain_cropped[2], fire_band_count + 3)
        dst.set_band_description(fire_band_count + 3, "Aspect")

    return out_path