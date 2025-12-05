import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
import numpy as np
import os
from pathlib import Path


def add_landcover(path):
    ### RASTER AND CROP LANDCOVER TO BURNMD TIFF ###
    prefix = Path(path)
    directory = prefix.parent

    fire_path = path
    out_path = directory / "timeseries_landcover.tif"
    lc_path   = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Data_Bands/AZ_landcover.tif"
    # fire_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/backbone_timeseries.tif"
    # out_path  = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/backbone_multiband.tif"

    #open reference
    with rasterio.open(fire_path) as fire_src:
        fire_transform = fire_src.transform
        fire_crs = fire_src.crs
        fire_height = fire_src.height
        fire_width = fire_src.width
        fire_profile = fire_src.profile

    #create template
    lc_cropped = np.zeros((fire_height, fire_width), dtype=np.uint8)

    #reproject with matching properties to fire bounds
    with rasterio.open(lc_path) as lc_src:
        reproject(
            source=rasterio.band(lc_src, 1),
            destination=lc_cropped,
            src_transform=lc_src.transform,
            src_crs=lc_src.crs,
            dst_transform=fire_transform,
            dst_crs=fire_crs,
            resampling=Resampling.nearest
        )

    # add band
    fire_profile.update(count=fire_profile["count"] + 1)

    #save to new geotiff
    with rasterio.open(fire_path) as fire_src:
        with rasterio.open(out_path, "w", **fire_profile) as dst:
            # write existing fire bands
            for i in range(fire_src.count):
                dst.write(fire_src.read(i + 1), i + 1)

            #write landcover to last band
            dst.write(lc_cropped, fire_src.count + 1)
            dst.set_band_description(fire_src.count + 1, "Landcover")

    return out_path