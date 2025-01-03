# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:21:49 2025

@author: Avijit

Save evaluation metrics results into separate Excel files for each metric.
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from openpyxl import load_workbook
from pysteps.utils import conversion, transformation
from pysteps import nowcasts, motion, verification
import utility

def calculate_metrics(nowcasts, observed, metric_name, threshold=None):
    """
    Calculate evaluation metrics for each lead time.
    """
    if nowcasts is None or observed is None:
        return np.full(num_next_files, np.nan)  # Return NaN for missing data

    metric_func = verification.get_method("CSI" if "CSI" in metric_name else metric_name, type="deterministic")
    scores = []
    for i in range(nowcasts.shape[0]):
        if "CSI" in metric_name and threshold is not None:
            scores.append(metric_func(nowcasts[i, :, :], observed[i, :, :], thr=threshold)["CSI"])
        else:
            scores.append(metric_func(nowcasts[i, :, :], observed[i, :, :])["MAE"])
    return np.array(scores)

def save_results_to_excel(event_data, file_path, sheet_name):
    """
    Save event results to an Excel file, appending or creating a new sheet.
    """
    try:
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df = pd.DataFrame(event_data[1:], columns=event_data[0])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(file_path, mode="w") as writer:
                df = pd.DataFrame(event_data[1:], columns=event_data[0])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        
# Generate subsets
def generate_subsets(event_scans, num_prev_files, num_next_files):
    current_scans = []
    for i in range(len(event_scans)):
        if i >= num_prev_files and i + num_next_files < len(event_scans):
            current_scans.append(event_scans[i])
    return current_scans



# Constants
data_path = os.path.join(os.getcwd(), "Data")
results_dir = os.path.join(os.getcwd(), "Results")
metadata_X = utility.get_matadata(os.path.join(data_path, "radarmappatipo.tif"), type="X")
num_next_files = 6
timestep = 5
# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

optical_flow_methods = ["LK", "VET", "DARTS", "proesmans"]
nowcasting_methods = ["persistence", "extrapolation"]
# # Specify one method for optical flow and nowcasting method
# nc_method = nowcasting_methods[2]
# of_method = optical_flow_methods[0]

for of_method in tqdm(optical_flow_methods):
    if of_method == "DARTS":
        num_prev_files =  9
    else :
        num_prev_files =  3     
    for nc_method in tqdm(nowcasting_methods):
        event_results_mae = []
        event_results_csi_1 = []
        event_results_csi_5 = []
        # Events processing
        events = [event for event in os.listdir(os.path.join(data_path, "UNICA_SG")) if os.path.isdir(os.path.join(data_path, "UNICA_SG", event))]
        for event in tqdm(events):
            event_path = os.path.join(data_path, "UNICA_SG", event)
            event_scans = sorted([f for f in os.listdir(event_path) if f.endswith(".png")])
            subsets = generate_subsets(event_scans, num_prev_files, num_next_files)
            subset_csi_1_results=[]
            subset_csi_5_results=[]
            subset_mae_results=[]
            
            for current_scan in tqdm(subsets, desc=f"Subsets ({event})", leave=False):
                R_dn, metadata_dn = utility.import_files_by_date(
                    date=datetime.strptime(current_scan, "%Y%m%d_%H%M.png"),
                    root_path=data_path,
                    data_source="UNICA_SG",
                    event_subdir=event,
                    f_ext="png",
                    metadata=metadata_X,
                    num_prev_files=num_prev_files,
                    num_next_files=0,
                    timestep=timestep,
                )
                R_dn_clean = np.empty_like(R_dn)
                for t in range(R_dn.shape[0]):
                    R_dn_clean[t, :, :] = utility.noise_remove(R_dn[t, :, :], type="Watershed")
                R_dbz, metadata_dbz = utility.dn_to_dbz(R_dn_clean, metadata_dn)
                R_R, metadata_R = conversion.to_rainrate(R_dbz, metadata_dbz)
                R_dbr, metadata_dbr = transformation.dB_transform(R_R, metadata_R, threshold=0.1, zerovalue=-15.0)
                R_dbr[~np.isfinite(R_dbr)] = -15.0
        
                # Motion estimation
                
                if of_method == "LK":
                    oflow_method= motion.get_method("LK")
                    motion_field = oflow_method(R_dbr[-4:, :, :])
                elif of_method == "VET":
                    oflow_method= motion.get_method("VET")
                    motion_field = oflow_method(R_dbr[-3:, :, :])
                elif of_method == "DARTS":
                    oflow_method= motion.get_method("DARTS")
                    motion_field = oflow_method(R_dbr[-10:, :, :])
                elif of_method == "proesmans":
                    oflow_method= motion.get_method("proesmans")
                    motion_field = oflow_method(R_dbr[-2:, :, :])
        
                # Nowcasting computation
                R_nowcast = None
                n_leadtimes = num_next_files
                if nc_method == "persistence":
                    nowcast_func = nowcasts.get_method("extrapolation")
                    R_nowcast = nowcast_func(R_dbr[-1, :, :], motion_field, n_leadtimes, extrap_method="eulerian")
                    R_nowcast = transformation.dB_transform(R_nowcast, 
                                                            threshold=-10.0, 
                                                            inverse=True)[0]
                elif nc_method == "extrapolation":
                    nowcast_func = nowcasts.get_method("extrapolation")
                    R_nowcast = nowcast_func(R_dbr[-1, :, :], motion_field, n_leadtimes, extrap_method="semilagrangian")
                    R_nowcast = transformation.dB_transform(R_nowcast, 
                                                            threshold=-10.0, 
                                                            inverse=True)[0]
                elif nc_method == "sprog":
                    
                    nowcast_func = nowcasts.get_method("sprog")
                    R_nowcast = nowcast_func(R_dbr[-3:, :, :], 
                                             motion_field, 
                                             n_leadtimes, 
                                             n_cascade_levels=6,
                                             R_thr=-10.0)
                    R_nowcast = transformation.dB_transform(R_nowcast, 
                                                            threshold=-10.0, 
                                                            inverse=True)[0]
                elif nc_method == "anvil":
                    nowcast_func = nowcasts.get_method("anvil")
                    R_nowcast = nowcast_func(R_R[-4:, :, :], 
                                             motion_field, 
                                             n_leadtimes, ar_window_radius=25, ar_order=2)
        
                # Observed data
                R_O, metadata_O = utility.import_files_by_date(
                    date=datetime.strptime(current_scan, "%Y%m%d_%H%M.png"),
                    root_path=data_path,
                    data_source="UNICA_SG",
                    event_subdir=event,
                    f_ext="png",
                    metadata=metadata_X,
                    num_prev_files=None,
                    num_next_files=num_next_files,
                    timestep=timestep,
                )
                R_O_clean = np.empty_like(R_O)
                for t in range(R_O.shape[0]):
                    R_O_clean[t, :, :] = utility.noise_remove(R_O[t, :, :], type="Watershed")
                R_O_dbz, metadata_O_dbz = utility.dn_to_dbz(R_O_clean, metadata_O)
                R_O_R, metadata_O_R = conversion.to_rainrate(R_O_dbz, metadata_O_dbz)
                
                csi_1_results =calculate_metrics(R_nowcast, R_O_R, "CSI", 1)
                csi_5_results = calculate_metrics(R_nowcast, R_O_R, "CSI", 5)
                mae_results = calculate_metrics(R_nowcast, R_O_R, "MAE")
                
                subset_csi_1_results.append(csi_1_results)
                subset_csi_5_results.append(csi_5_results)
                subset_mae_results.append(mae_results)
            # Aggregate results
            if subset_csi_1_results:
                subset_csi_1_results = np.array(subset_csi_1_results)
                mean_metrics_csi_1 = np.nanmean(subset_csi_1_results, axis=0).flatten()
                event_results_csi_1.append([event] + mean_metrics_csi_1.tolist())
                
            # Aggregate results
            if subset_csi_5_results:
                subset_csi_5_results = np.array(subset_csi_5_results)
                mean_metrics_csi_5 = np.nanmean(subset_csi_5_results, axis=0).flatten()
                event_results_csi_5.append([event] + mean_metrics_csi_5.tolist())
                
            # Aggregate results
            if subset_mae_results:
                subset_mae_results = np.array(subset_mae_results)
                mean_metrics_mae = np.nanmean(subset_mae_results, axis=0).flatten()
                event_results_mae.append([event] + mean_metrics_mae.tolist())
        
        # Prepare and save sheet data
        csi_1_file_path = os.path.join(results_dir, f"CSI_1_results.xlsx")
        csi_5_file_path = os.path.join(results_dir, f"CSI_5_results.xlsx")
        mae_file_path = os.path.join(results_dir, f"MAE_results.xlsx")
        headers = ["Event"] + [f"Lead Time {i+1}" for i in range(num_next_files)]
        sheet_name = f"{of_method}_{nc_method}"
        save_results_to_excel([headers] + event_results_csi_1, csi_1_file_path, sheet_name)
        save_results_to_excel([headers] + event_results_csi_5, csi_5_file_path, sheet_name)
        save_results_to_excel([headers] + event_results_mae, mae_file_path, sheet_name)
    


