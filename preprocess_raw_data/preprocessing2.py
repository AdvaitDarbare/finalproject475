import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio import features
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path


def shape_to_tif(path):
    ### CONVERT .SHP FILE TO GEOTIFF WITH TIME SERIES ###
    prefix = Path(path)
    matches = list(prefix.glob("*POLYGONS.shp"))

    if not matches:
        raise FileNotFoundError("No POLYGONS.shp files found")

    shapefile_path = matches[0]
    output_path = path + "/timeseries.tif"
    # shapefile_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Backbone/Backbone_POLYGONS.shp"
    # output_tif_path = "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/backbone_timeseries.tif"
    pixel_size_m = 30  # meters
    date_col = 'PolygonDat'

    gdf = gpd.read_file(shapefile_path)

    #convert date column to datetime
    gdf[date_col] = pd.to_datetime(gdf[date_col])

    #handle crs
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3857")  # assume already in meters
    gdf_m = gdf.to_crs("EPSG:3857")

    gdf_m['date_only'] = gdf_m[date_col].dt.date
    latest_per_day = gdf_m.sort_values(date_col).groupby('date_only').tail(1)
    all_dates = sorted(latest_per_day['date_only'].unique())
    n_days = len(all_dates)

    minx, miny, maxx, maxy = gdf_m.total_bounds
    width = int((maxx - minx) / pixel_size_m)
    height = int((maxy - miny) / pixel_size_m)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    print(f"Raster dimensions: {width} x {height}")

    raster_stack = np.zeros((n_days, height, width), dtype=np.uint8)
    cumulative_mask = np.zeros((height, width), dtype=np.uint8)

    for i, day in enumerate(all_dates):
        day_polygons = latest_per_day[latest_per_day['date_only'] == day]

        if day_polygons.empty:
            print(f"No polygons for {day} — using previous days mask")
            raster_stack[i] = cumulative_mask.copy()
            continue

        day_polygons.columns = [c.lower() for c in day_polygons.columns]  #generalize names to lower

        for idx, row in day_polygons.iterrows():
            print(f"{day} → OBJECTID={row['objectid']}, GISAcres={row['gisacres']}, Shape_Area={row['shape_area']}")

        #rasterize days polygons
        day_raster = features.rasterize(
            ((geom, 1) for geom in day_polygons.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        #accumulate burned area as fire grows
        cumulative_mask |= day_raster

        #store fire mask for this day
        raster_stack[i] = cumulative_mask.copy()

        print(f"Cumulative rasterized {day} : {len(day_polygons)} polygon(s)")

    # for i, day in enumerate(all_dates):
    #     day_polygons = latest_per_day[latest_per_day['date_only'] == day]
    #
    #     if day_polygons.empty:
    #         print(f"No polygons for {day}")
    #         continue
    #
    #     for idx, row in day_polygons.iterrows():
    #         print(f"{day} → OBJECTID={row['OBJECTID']}, GISAcres={row['GISAcres']}, Shape_Area={row['Shape_Area']}")
    #
    #     shapes = ((geom, 1) for geom in day_polygons.geometry)
    #
    #     raster_stack[i] = features.rasterize(
    #         shapes=shapes,
    #         out_shape=(height, width),
    #         transform=transform,
    #         fill=0,
    #         dtype=np.uint8
    #     )
    #     print(f"Rasterized {day} → {len(day_polygons)} polygon(s)")


    # save multiband geotiff
    with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=n_days,
            dtype=raster_stack.dtype,
            crs="EPSG:3857",
            transform=transform,
    ) as dst:
        for i in range(n_days):
            dst.write(raster_stack[i], i + 1)
            dst.set_band_description(i + 1, str(all_dates[i]))

    print(f"Saved time-series GeoTIFF: {output_path}")
    return output_path
