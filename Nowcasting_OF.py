# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:18:15 2024
@author: Avijit Majhi

### important #####
run the code in 
radar_nowcasting_env
"""
import os
import numpy as np
import utility
from datetime import datetime, timedelta
import pysteps
# Define the directories
root_path = "D:\\Geosciences_Project\\Nowcasting_OF"

# Define data directory
data_path = os.path.join(root_path,"Data")
metadata_X = utility.get_matadata(os.path.join(data_path,"radarmappatipo.tif"),type='X')

# choose on the event 
date = datetime.strptime("20191024_0415", "%Y%m%d_%H%M")
# Input parameter for loading X-band radar data
data_source_X = "UNICA_SG"
f_ext_X = "png"
# load X-Band radar data
R_X , metadata_X = utility.import_files_by_date(date, data_path,data_source_X, f_ext_X, metadata_X, timestep=5, num_prev_files=9)

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
from pprint import pprint
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
utility.plot_modification(ax,metadata_X)

# ### Analysis with civil protection radar data 

# metadata_C = utility.get_matadata(os.path.join(data_path,"civilpromap.tiff"),type='C')
# # Input parameter for loading C band radar data
# data_source_C = "Civil_Pro_C"
# f_ext_C = "tiff"
# # load C-Band radar data
# R_C, metadata_C = utility.import_files_by_date(date, data_path,data_source_C, f_ext_C, metadata_C, timestep=5, num_prev_files=9)