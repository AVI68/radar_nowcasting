#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:49:17 2024

@author: avijitmajhi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pysteps.utils import conversion, transformation
from pysteps import nowcasts, motion, verification
import utility

# Define constants
data_path = os.path.join(os.getcwd(), "Data")
results_dir = os.path.join(os.getcwd(), "Results")
# optical_flow_methods = {
#     "LK": 4,
#     "VET": 3,
#     "DARTS": 10,
#     "proesmans": 2
# }

optical_flow_methods = {
    "LK": 4,
    "DARTS": 10,
}
# nowcasting_methods = ["persistence", "extrapolation", "sprog", "anvil"]
nowcasting_methods = ["persistence", "extrapolation"]

evaluation_metrics = ["MAE", "CSI"]
thresholds = [0.5, 1.0, 5.0]
num_next_files = 6
timestep = 5

# Ensure results directory structure
os.makedirs(results_dir, exist_ok=True)
for of_method in optical_flow_methods.keys():
    os.makedirs(os.path.join(results_dir, "opticalflow", of_method), exist_ok=True)
for nc_method in nowcasting_methods:
    os.makedirs(os.path.join(results_dir, "nowcast", nc_method), exist_ok=True)

# Initialize combined metrics storage
combined_metrics = {nc_method: {metric: [] for metric in evaluation_metrics} for nc_method in nowcasting_methods}

# Function to calculate evaluation metrics
def calculate_metrics(nowcasts, observed, metric_name, thresholds=None):
    metric_func = verification.get_method(metric_name, type="deterministic")
    scores = []
    for i in range(nowcasts.shape[0]):
        if metric_name == "CSI" and thresholds:
            scores.append([metric_func(nowcasts[i], observed[i], thr=thr)["CSI"] for thr in thresholds])
        else:
            scores.append(metric_func(nowcasts[i], observed[i])["MAE"])
    return np.array(scores)

# Process each event
events = [event for event in os.listdir(os.path.join(data_path, "UNICA_SG")) if os.path.isdir(os.path.join(data_path, "UNICA_SG", event))]

for event in events:
    event_path = os.path.join(data_path, "UNICA_SG", event)
    event_scans = sorted([f for f in os.listdir(event_path) if f.endswith(".png")])

    for of_method, num_prev_files in optical_flow_methods.items():
        # Generate subsamples for the event
        subsets = utility.generate_subsets(event_scans, num_prev_files, num_next_files)

        # Store metrics for the event
        event_metrics = {nc_method: {metric: [] for metric in evaluation_metrics} for nc_method in nowcasting_methods}

        for current_scan, prev_scans, next_scans in subsets:
            # Load and preprocess radar data
            R_dn, metadata_dn = utility.import_files_by_date(
                date=datetime.strptime(current_scan, "%Y%m%d_%H%M"),
                root_path=data_path,
                data_source="UNICA_SG",
                f_ext="png",
                metadata=utility.get_matadata(os.path.join(data_path, "radarmappatipo.tif"), type="X"),
                num_prev_files=num_prev_files,
                num_next_files=num_next_files,
                timestep=timestep
            )
            R_dn_clean = utility.remove_noise(R_dn)
            R_dbr = utility.dn_to_dbz(R_dn_clean)
            R_R = conversion.to_rainrate(R_dbr)[0]

            # Estimate motion field
            motion_field = motion.get_method(of_method)(R_dbr[-num_prev_files:, :, :])

            # Perform nowcasts
            R_nowcasts = {}
            if "sprog" in nowcasting_methods:
                sprog = nowcasts.get_method("sprog")
                R_f_sp = sprog(R_dbr[-3:, :, :], motion_field, num_next_files)
                R_f_sp = transformation.dB_transform(R_f_sp, threshold=-10.0, inverse=True)[0]
                R_nowcasts["sprog"] = R_f_sp
            if "anvil" in nowcasting_methods:
                anvil = nowcasts.get_method("anvil")
                R_f_anvil = anvil(R_R[-4:, :, :], motion_field, num_next_files)
                R_nowcasts["anvil"] = R_f_anvil
            if "extrapolation" in nowcasting_methods:
                extrapolate = nowcasts.get_method("extrapolation")
                R_f_extrap = extrapolate(R_dbr[-1, :, :], motion_field, num_next_files, extrap_method="semilagrangian")
                R_f_extrap = transformation.dB_transform(R_f_extrap, threshold=-10.0, inverse=True)[0]
                R_nowcasts["extrapolation"] = R_f_extrap
            if "persistence" in nowcasting_methods:
                persistence = nowcasts.get_method("extrapolation")
                R_f_persist = persistence(R_dbr[-1, :, :], motion_field, num_next_files, extrap_method="eulerian")
                R_f_persist = transformation.dB_transform(R_f_persist, threshold=-10.0, inverse=True)[0]
                R_nowcasts["persistence"] = R_f_persist

            # Load and preprocess observed data
            R_observed, metadata_observed = utility.import_files_by_date(
                date=datetime.strptime(current_scan, "%Y%m%d_%H%M"),
                root_path=data_path,
                data_source="UNICA_SG",
                f_ext="png",
                metadata=utility.get_matadata(os.path.join(data_path, "radarmappatipo.tif"), type="X"),
                num_prev_files=0,
                num_next_files=num_next_files,
                timestep=timestep
            )
            R_observed_clean = utility.remove_noise(R_observed)
            R_observed_dbz = utility.dn_to_dbz(R_observed_clean)
            R_observed_rainrate = conversion.to_rainrate(R_observed_dbz)[0]

            # Evaluate metrics for each nowcasting method
            for nc_method, R_f in R_nowcasts.items():
                event_metrics[nc_method]["MAE"].append(calculate_metrics(R_f, R_observed_rainrate, "MAE"))
                event_metrics[nc_method]["CSI"].append(calculate_metrics(R_f, R_observed_rainrate, "CSI", thresholds))

                # Accumulate metrics for combined averages
                if not combined_metrics[nc_method]["MAE"]:
                    combined_metrics[nc_method]["MAE"] = event_metrics[nc_method]["MAE"]
                else:
                    combined_metrics[nc_method]["MAE"] = np.add(
                        combined_metrics[nc_method]["MAE"], event_metrics[nc_method]["MAE"]
                    )

                if not combined_metrics[nc_method]["CSI"]:
                    combined_metrics[nc_method]["CSI"] = event_metrics[nc_method]["CSI"]
                else:
                    combined_metrics[nc_method]["CSI"] = np.add(
                        combined_metrics[nc_method]["CSI"], event_metrics[nc_method]["CSI"]
                    )

        # Aggregate and plot metrics for the event
        for metric in evaluation_metrics:
            fig, axes = plt.subplots(4, 3, figsize=(15, 10))
            for idx, nc_method in enumerate(nowcasting_methods):
                ax = axes[idx // 3, idx % 3]
                avg_metric = np.mean(event_metrics[nc_method][metric], axis=0)
                ax.plot(range(1, num_next_files + 1), avg_metric)
                ax.set_title(f"{nc_method} - {metric}")
                ax.set_xlabel("Lead Time (5 min steps)")
                ax.set_ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "opticalflow", of_method, f"{event}_{metric}.png"))
            plt.close(fig)

# Generate combined metrics plots
for metric in evaluation_metrics:
    fig, ax = plt.subplots(figsize=(10, 8))
    for nc_method in nowcasting_methods:
        avg_metric = np.mean(combined_metrics[nc_method][metric], axis=0) / len(events)
        ax.plot(range(1, num_next_files + 1), avg_metric, label=nc_method)
    ax.set_title(f"Combined {metric} Across All Events")
    ax.set_xlabel("Lead Time (5 min steps)")
    ax.set_ylabel(metric)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"combined_{metric}.png"))
    plt.close(fig)
