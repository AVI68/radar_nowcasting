#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:49:17 2024

@author: avijitmajhi
"""

import pandas as pd
from datetime import datetime
import numpy as np
import pysteps

# Step 1: Read Event Details
events_df = pd.read_excel("events_info.xlsx")  # Excel file with event details
events = events_df.to_dict(orient="records")  # Convert to a list of dictionaries

# Define Optical Flow and Advection Schemes
optical_flows = {
    "LK": 3,       # Lucas-Kanade requires last 3 scans
    "DARTS": 10       # Variational Echo Tracking requires last 3 scans
}

advection_schemes = [
    "extrapolation",
    "sprog"
]

# Loop Through Events
results = []  # To store results for each event
for event in events:
    event_name = event["event_name"]
    total_scans = event["total_scans"]
    event_folder = f"{data_path}/{event_name}"  # Path to the event folder

    # List all scans inside the event folder
    event_scans = sorted([scan for scan in os.listdir(event_folder) if scan.endswith(".png")])

    # Step 4: Generate Subsets
    for optical_flow, num_prev_files in optical_flows.items():
        subsets = generate_subsets(event_scans, num_prev_files, num_next_files=12)

        for current_scan, prev_scans, next_scans in subsets:
            # Step 5: Load Data
            radar_data, metadata = import_files_by_date(
                date=datetime.strptime(current_scan, "%Y%m%d_%H%M"),
                root_path=data_path,
                data_source=event_name,
                f_ext="png",
                metadata={"projection": "EPSG:3857"},
                num_prev_files=len(prev_scans),
                num_next_files=len(next_scans),
                timestep=5
            )

            # Step 6: Apply Optical Flow
            motion_field = pysteps.motion.get_method(optical_flow)(radar_data[-len(prev_scans):])

            for advection in advection_schemes:
                # Step 7: Perform Nowcasting
                nowcast_method = pysteps.nowcasts.get_method(advection)
                nowcast = nowcast_method(
                    radar_data[-1], motion_field, 12, metadata=metadata
                )

                # Step 8: Evaluate and Store Results
                # Example: Compute MAE (you can add CSI and thresholds later)
                ground_truth = radar_data[-len(next_scans):]
                mae = np.mean(np.abs(nowcast - ground_truth))

                results.append({
                    "event": event_name,
                    "optical_flow": optical_flow,
                    "advection": advection,
                    "mae": mae,
                    "nowcast": nowcast
                })
