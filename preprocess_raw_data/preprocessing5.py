import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
from pathlib import Path


def add_wind(path):
    ### RASER WIND BANDS ###

    data_path = Path(path)
    wind_data = data_path.parent.name
    wind_data = wind_data.lower()
    directory = data_path.parent
    print(wind_data)

    filename = "AZ_wind_mean_" + str(wind_data) + ".tif"
    wind_raster_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Data_Bands/" + filename

    fire_raster_path = path
    output_path = directory / "raster_final.tif"
    # wind_raster_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Data_Bands/AZ_wind_mean_backbone.tif"
    # fire_raster_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Backbone/backbone_all_static.tif"
    # output_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Backbone/backbone_Full_Band.tif"

    with rasterio.open(fire_raster_path) as fire_src:
        fire_crs = fire_src.crs
        fire_bounds = fire_src.bounds
        fire_width = fire_src.width
        fire_height = fire_src.height
        fire_transform = fire_src.transform
        fire_data = fire_src.read()
        fire_band_count = fire_src.count

    with rasterio.open(wind_raster_path) as wind_src:
        #transform fire bounds to wind CRS for window
        fire_bounds_in_wind_crs = transform_bounds(fire_crs, wind_src.crs, fire_bounds.left, fire_bounds.bottom, fire_bounds.right, fire_bounds.top)

        window = from_bounds(*fire_bounds_in_wind_crs, transform=wind_src.transform)

        wind_speed = wind_src.read(1, window=window)
        wind_dir = wind_src.read(2, window=window)

        #matching fire raster size
        wind_speed_resampled = np.empty((fire_height, fire_width), dtype=np.float32)
        wind_dir_resampled = np.empty((fire_height, fire_width), dtype=np.float32)

        #reproject and resample to fire raster CRS, shape
        reproject(
            source=wind_speed,
            destination=wind_speed_resampled,
            src_transform=wind_src.window_transform(window),
            src_crs=wind_src.crs,
            dst_transform=fire_transform,
            dst_crs=fire_crs,
            resampling=Resampling.bilinear
        )
        reproject(
            source=wind_dir,
            destination=wind_dir_resampled,
            src_transform=wind_src.window_transform(window),
            src_crs=wind_src.crs,
            dst_transform=fire_transform,
            dst_crs=fire_crs,
            resampling=Resampling.bilinear
        )

        #covert direction to sin/cos
        wind_dir_rad = np.deg2rad(wind_dir_resampled)
        wind_dir_sin = np.sin(wind_dir_rad)
        wind_dir_cos = np.cos(wind_dir_rad)

    #stack bands
    new_bands = np.stack([wind_speed_resampled, wind_dir_sin, wind_dir_cos])
    combined_data = np.concatenate([fire_data, new_bands], axis=0)

    #save all rasters w/ names
    band_names = ["Landcover", "Elevation", "Slope", "Aspect",
                  "Wind_Speed", "Wind_Dir_Sin", "Wind_Dir_Cos"]

    with rasterio.open( #export to final geotiff
            output_path,
            "w",
            driver="GTiff",
            height=fire_height,
            width=fire_width,
            count=combined_data.shape[0],
            dtype=combined_data.dtype,
            crs=fire_crs,
            transform=fire_transform,
            compress="LZW"
    ) as dst:
        #Write all bands
        for i in range(combined_data.shape[0]):
            dst.write(combined_data[i], i + 1)

        #lavel environment bands
        for idx, name in enumerate(band_names, start=combined_data.shape[0] - 7 + 1):
            dst.set_band_description(idx, name)
