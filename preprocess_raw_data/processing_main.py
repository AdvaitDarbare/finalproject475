### MAIN PROCESSING ###

import mitigation as pre2
import preprocessing3 as pre3
import preprocessing4 as pre4
import preprocessing5 as pre5


print("Begin Rastering Wildfire Burn Images")

burn_events = ["/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Basin",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Blue_River",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Salt",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Castle",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Constellation",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Griffin",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Horton",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Ikes",
               "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Medicine",
                "/Users/pyn/Desktop/AZ_wildfire/wildfire_poc/Sawtooth"]

for path in burn_events:
    print(f"Processing {path}")

    print("Attempting shape to tif...")
    phase1 = pre2.shape_to_tif(path)
    print("Completed shape to tif...")

    print("Attempting raster land cover data...")
    phase2 = pre3.add_landcover(phase1)
    print("Completed raster land cover data...")

    print("Attempting raster terrain data...")
    phase3 = pre4.add_terrain(phase2)
    print("Completed raster terrain data...")

    print("Attempting raster wind data...")
    phase4 = pre5.add_wind(phase3)
    print("Completed raster wind data...")

    print(f"Completed full raster {path}\n\n :D")

