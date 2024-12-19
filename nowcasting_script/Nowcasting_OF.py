# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:18:15 2024

@author: Avijit Majhi
"""

import os
import numpy as np
import shutil
import pickle
import utility
from datetime import datetime, timedelta
import pysteps
from pprint import pprint
import matplotlib.pyplot as plt
from contextily import add_basemap
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator,FuncFormatter

# Define the directories
root_path = "D:\\Geosciences_Project\\Nowcasting_OF\\Data"
unica_sg_dir = os.path.join(root_path, "UNICA_SG")  # X-band
civil_pro_c_dir = os.path.join(root_path, "Civil_Pro_C")  # C-band
metadata_X = utility.get_matadata(os.path.join(root_path,"radarmappatipo.tif"),type='X')
metadata_C = utility.get_matadata(os.path.join(root_path,"civilpromap.tiff"),type='C')

# choose on the event 
date = datetime.strptime("20191024_0415", "%Y%m%d_%H%M")

# Input parameter for loading X-band radar data
data_source_X = "UNICA_SG"
fn_pattern_X = "%Y%m%d_%H%M"
f_ext_X = ".png"
# load X-Band radar data
R_X , metadata_X = utility.import_files_by_date(date, root_path,data_source_X, fn_pattern_X,f_ext_X, metadata_X, timestep=5, num_prev_files=9)
## use of watershed techniques for noise removal

R_X_clean = np.empty_like(R_X)
# Iterate over the time dimension
for t in range(R_X.shape[0]):
    R_X_clean[t, :, :] = utility.noise_remove(R_X[t, :, :], type='Watershed')
# Digital Number to  reflectivity (Decibel)
R_X_dbz = utility.dn_to_dbz(R_X_clean)
R_X_R, metadata_X = pysteps.utils.conversion.to_rainrate(R_X_dbz,metadata_X)

# Log-transform the data [dBR]
image_R, metadata_X = pysteps.utils.transformation.dB_transform(R_X_R, metadata_X, threshold=0.01, zerovalue=-15.0)

# print the metadata
pprint(metadata_X)

oflow_method = pysteps.motion.get_method("LK")
V1 = oflow_method(image_R[-3:, :, :])

import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field, quiver

# Plot precipitation field
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the precipitation field
plot_precip_field(
    R_X_dbz[-1, :, :],  # Last time step of the reflectivity data
    ptype="intensity",
    geodata=metadata_X,
    units="dBZ",
    title="Lucas Kanade Optical Flow Method",
    ax=ax,
    colorscale="pysteps"
)

# Plot the motion field (quiver plot) on top of the precipitation field
quiver(
    V1,  # Motion field
    geodata=metadata_X,
    step=45,
    ax=ax
)



# Customize x and y axis labels in degree 
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pyproj

# Function to convert meter-based coordinates to degrees
def convert_meters_to_degrees(x, y, crs_proj, crs_geo="EPSG:4326"):
    """
    Convert coordinates from a projected CRS (meters) to geographic CRS (degrees).
    Parameters:
        x (float or ndarray): X-coordinate(s) in the projected CRS.
        y (float or ndarray): Y-coordinate(s) in the projected CRS.
        crs_proj (str): CRS of the input coordinates (e.g., 'EPSG:3857').
        crs_geo (str): Target geographic CRS (default is 'EPSG:4326').
    Returns:
        tuple: Converted (lon, lat) coordinates in degrees.
    """
    transformer = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)
    return transformer.transform(x, y)

def format_ticks(value, tick_number):
    """Convert tick values from meters to degrees and format them."""
    lon, lat = convert_meters_to_degrees(value, 0, crs_proj) if tick_number == 0 else convert_meters_to_degrees(0, value, crs_proj)
    return f"{lon:.2f}" if tick_number == 0 else f"{lat:.2f}"


# Define the CRS of the projected data
crs_proj = metadata_X["projection"]  # CRS in meters (e.g., 'EPSG:3857')
crs_geo = "EPSG:4326"  # Geographic CRS (degrees)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_ticks(x, 0)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_ticks(y, 1)))
# Reduce the number of ticks
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# Add a basemap using the original CRS
import contextily as cx
cx.add_basemap(ax, crs=metadata_X["projection"], source=cx.providers.OpenStreetMap.Mapnik)

# Show the plot
plt.show()






### Analysis with civil protection radar data 
# Input parameter for loading C band radar data
data_source_C = "Civil_Pro_C"
fn_pattern_C = "%Y%m%d%H%M"
f_ext_C = ".tiff"
# load C-Band radar data
R_C, metadata_C = utility.import_files_by_date(date, root_path,data_source_C, fn_pattern_C,f_ext_C, metadata_C, timestep=5, num_prev_files=9)


