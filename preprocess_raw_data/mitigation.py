import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from pathlib import Path


def shape_to_tif(shp_path, out_tif_path, target_tif):
    value = 1
    #load and drop bad geometries
    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]

    #open target raster to match properties
    with rasterio.open(target_tif) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)

    if gdf.crs != meta["crs"]:
        gdf = gdf.to_crs(meta["crs"])
    #get geomerty
    shapes = [(geom, value) for geom in gdf.geometry]

    #rasterize
    raster = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8"
    )

    #write band
    meta.update(count=1, dtype="uint8")
    with rasterio.open(out_tif_path, "w", **meta) as dst:
        dst.write(raster, 1)
    return out_tif_path


def append_band(target_tif, new_band_tif, out_tif_path):
    #add bands to tiff
    with rasterio.open(target_tif) as base:
        meta = base.meta.copy()
        base_data = base.read()

    with rasterio.open(new_band_tif) as nb:
        new_band = nb.read(1)

        if nb.shape != base_data.shape[1:]:
            raise ValueError("The new band must match dimensions of the target...")

    meta.update(count=base_data.shape[0] + 1)

    with rasterio.open(out_tif_path, "w", **meta) as dst:
        #write original bands
        for i in range(base_data.shape[0]):
            dst.write(base_data[i], i + 1)
        #appended band
        dst.write(new_band, base_data.shape[0] + 1)

    return out_tif_path


if __name__ == "__main__":
    shp = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Sawtooth/Sawtooth_LINES.shp"
    target_raster = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Sawtooth/raster_final_sawtooth.tif"
    temp_raster = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Sawtooth/line_rasterized.tif"
    final_output = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Sawtooth/Sawtooth_Full_Band_Mit.tif"

    shape_to_tif(shp, temp_raster, target_raster)

    append_band(target_raster, temp_raster, final_output)

    print("Complete :D")
