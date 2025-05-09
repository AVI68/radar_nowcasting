# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:26:32 2024

run the code "radar_nowcasting_env" environment

@author: Avijit Majhi
"""
import os
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from osgeo import gdal, gdalconst, osr
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver
import pyproj
import contextily as cx
from scipy.ndimage import median_filter
from scipy.signal import correlate2d
import cv2

# Provided utility functions
def list_files_in_folder(folder_path):
    """
    Lists all files in a given folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of full paths to each file in the folder.
    """
    # List comprehension to filter files and construct their full paths
    return [
        os.path.join(folder_path, f)  # Join the folder path with the file name to get the full path
        for f in os.listdir(folder_path)  # Iterate over all entries in the folder
        if os.path.isfile(os.path.join(folder_path, f))  # Include only files (not directories)
    ]

def count_files(root_path, file_format=None):
    """
    Counts the number of files with a specific format in a directory 
    or the number of folders if no format is provided.

    Parameters:
        root_path (str): Path to the root directory.
        file_format (str, optional): File extension to filter (e.g., 'png', 'tiff').
                                     If None, count the number of folders.

    Returns:
        int: Count of files matching the format or the number of folders.
    """
    if not os.path.isdir(root_path):
        raise ValueError(f"The path {root_path} is not a valid directory.")

    if file_format:  # Count files with the specified format
        file_count = sum(
            1 for f in os.listdir(root_path)
            if f.endswith(f".{file_format}") and os.path.isfile(os.path.join(root_path, f))
        )
        return file_count
    else:  # Count the number of folders
        folder_count = sum(
            1 for f in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, f))
        )
        return folder_count

def get_file_format(file_path):
    """
    Get the file format (extension) of the given file path.
    
    Parameters:
        file_path (str): Path to the file.
    
    Returns:
        File format (str): File format (extension), e.g., 'png', 'tiff', etc.
    """
    # Extract the file extension from the file path
    _, file_extension = os.path.splitext(file_path)
    
    # Convert the file extension to lowercase and remove the leading dot
    file_format = file_extension.lower().lstrip('.')
    
    return file_format
def get_matadata(file_path, type='X'):
    """
    Read the image file to get the metadata 
    
    Parameters:
        file_path (str): Path to the image file (geotiff image)
        type (str): 'X','C','Other'
    Returns:
        metadata (dict) : dictionary contains metadata
    """
    file_format = get_file_format(file_path)
    if file_format == 'tiff' or 'tif':
        if type == 'X':
            try:
                f = gdal.Open(file_path, gdalconst.GA_ReadOnly)
                # sr = osr.SpatialReference()
                # pr = f.GetProjection()
                # sr.ImportFromWkt(pr)
                # projdef = sr.ExportToProj4()
                gt = f.GetGeoTransform()
                projdef ='+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 +x_0=1500000 +y_0=0 +ellps=intl +towgs84=-104.1,-49.1,-9.9,0.971,-2.917,0.714,-11.68 +units=m +no_defs +type=crs'
                metadata = {
                    "projection": projdef,
                    "cartesian_unit": "m",
                    "x1": gt[0],
                    "y1": gt[3] + gt[5] * f.RasterYSize,
                    "x2": gt[0] + gt[1] * f.RasterXSize,
                    "y2": gt[3],
                    "xpixelsize": abs(gt[1]),
                    "ypixelsize": abs(gt[5]),
                    "yorigin": "upper" if gt[5] < 0 else "lower",
                    "institution": "UNICA",
                    "unit": "DN",
                    "transform": "None",
                    "accutime": 1,
                    "zr_a": 53.89,
                    "zr_b": 0.86,
                    "product":"png",
                    "zerovalue" : 0,
                    "threshold" :0,
                }
                return metadata
            except FileNotFoundError:
                print(f"Error: File not found at '{file_path}'.")
            except ValueError as e:
                print(f"Error: {e}")
            return None
        elif type == 'C':
           try:
               f = gdal.Open(file_path, gdalconst.GA_ReadOnly)
               sr = osr.SpatialReference()
               pr = f.GetProjection()
               sr.ImportFromWkt(pr)
               projdef = sr.ExportToProj4()
               gt = f.GetGeoTransform()
               metadata = {
                   "projection": projdef,
                   "x1": gt[0],
                   "y1": gt[3] + gt[5] * f.RasterYSize,
                   "x2": gt[0] + gt[1] * f.RasterXSize,
                   "y2": gt[3],
                   "xpixelsize": abs(gt[1]),
                   "ypixelsize": abs(gt[5]),
                   "yorigin": "upper" if gt[5] < 0 else "lower",
                   "institution": "Italain Civil Protection",
                   "product":"tiff",
                   "cartesian_unit": "degree",
                   "unit": "dBZ",
                   "transform": "dB",
                   "accutime": 5,
                   "zr_a": 200,
                   "zr_b": 1.6,
                   "zerovalue" : 0,
                   "threshold" :0,
               }
               return metadata
           except FileNotFoundError:
               print(f"Error: File not found at '{file_path}'.")
           except ValueError as e:
               print(f"Error: {e}")
           return None
        elif type =='Other':
           try:
               f = gdal.Open(file_path, gdalconst.GA_ReadOnly)
               sr = osr.SpatialReference()
               pr = f.GetProjection()
               sr.ImportFromWkt(pr)
               projdef = sr.ExportToProj4()
               gt = f.GetGeoTransform()
               metadata = {
                   "projection": projdef,
                   "x1": gt[0],
                   "y1": gt[3] + gt[5] * f.RasterYSize,
                   "x2": gt[0] + gt[1] * f.RasterXSize,
                   "y2": gt[3],
                   "xpixelsize": abs(gt[1]),
                   "ypixelsize": abs(gt[5]),
                   "yorigin": "upper" if gt[5] < 0 else "lower",
                   "institution": "Unknown",
                   "product":"tiff",
                   "cartesian_unit": "degree",
                   "unit": [],
                   "transform": [],
                   "accutime": [],
                   "zr_a": [],
                   "zr_b":[],
                   "zerovalue" : [],
                   "threshold" :[],
               }
               return metadata
           except FileNotFoundError:
               print(f"Error: File not found at '{file_path}'.")
           except ValueError as e:
               print(f"Error: {e}")
           return None
    else:
        print(f"Error: File format is '{file_format}',Supported format is 'tiff' (geotiff file).")
        
def extract_datetime_from_filename(filename):
    """
   Extract date-time information from the given filename.
   
   Parameters:
       filename (str): Name of the file.
   
   Returns:
       datetime: Date-time information extracted from the filename.
   """
    file_format = get_file_format(filename)
    if file_format == 'png':
        datetime_info = filename.split(".")[0].split("\\")[-1]
        
        # date_time_info = pd.to_datetime(datetime_info, format='%Y%m%d_%H%M')
        date_time_info = datetime.strptime(datetime_info, '%Y%m%d_%H%M')
        
    elif file_format == 'tiff':
    
        datetime_info = filename.split(".")[0].split("_")[-1]

        date_time_info = datetime.strptime(datetime_info, '%Y%m%d%H%M')
        # date_time_info = pd.to_datetime(datetime_info, format='%Y%m%d%H%M')
    return date_time_info
def extract_filename_from_datetime(t, frmt):
    """
    Extract filename information from the given datetime.
    
    Parameters:
        t (datetime): datetime object
        frmt (str): format of the file ("png" or "tiff")
    
    Returns:
        str: filename information extracted from the Date-time.
    """
    # Ensure the datetime is in UTC
    if t.tzinfo is None:
       t = t.replace(tzinfo=timezone.utc)
    else:
       t = t.astimezone(timezone.utc)
    # Convert the datetime to Unix epoch time in milliseconds
    unix_epoch_time = int(t.timestamp() * 1000)
    
    # Format the datetime for the filename
    if frmt == 'png':
        date_format = t.strftime('%Y%m%d_%H%M')
        filename = f"{date_format}.png"
    elif frmt == 'tiff':
        date_format = t.strftime('%Y%m%d%H%M')
        filename = f"VMI_{unix_epoch_time}_{date_format}.tiff"
    else:
        raise ValueError("Unsupported format. Please use 'png' or 'tiff'.")
    
    return filename
      
def import_files_by_date(
    date, 
    root_path, 
    data_source, 
    event_subdir, 
    f_ext, 
    metadata, 
    num_prev_files=None, 
    num_next_files=0, 
    timestep=5
):
    """
    Import radar files from a fixed event-specific subdirectory and update metadata with timestamps.

    Parameters:
        date (datetime): The target datetime for the radar files.
        root_path (str): The root directory containing radar files.
        data_source (str): Subdirectory under the root path (e.g., "UNICA_SG" or "Civil_Pro_C").
        event_subdir (str): Subdirectory under data_source (format "%Y%m%d_%H%M")).
        f_ext (str): File extension (e.g., "png" or "tiff").
        metadata (dict): The metadata dictionary to be updated.
        timestep (int): Time interval (in minutes) between consecutive files.
        num_prev_files (int, optional): Number of files to retrieve before the target date. 
            If None, previous files are not included.
        num_next_files (int): Number of files to retrieve after the target date.

    Returns:
        tuple:
            - radar_data (numpy.ndarray): Array containing the loaded radar data.
            - metadata (dict): Updated metadata including timestamps.
    """
    radar_data = []
    timestamps = []

    # Load previous files (before the target date) if num_prev_files is not None
    if num_prev_files is not None:
        for i in range(num_prev_files, -1, -1):
            file_time = date - timedelta(minutes=i * timestep)
            timestamps.append(file_time)

            # Generate the filename using extract_filename_from_datetime
            filename = extract_filename_from_datetime(file_time, f_ext)

            # Construct the file path
            file_path = os.path.join(root_path, data_source, event_subdir, filename)

            # Check if the file exists and read it
            if os.path.exists(file_path):
                radar_image = read_image(file_path)
                radar_data.append(radar_image)
            else:
                print(f"Warning: No file found at {file_path}")

    # Load next files (after the target date)
    for i in range(1, num_next_files + 1):  # Start from 1 to exclude the current scan
        file_time = date + timedelta(minutes=i * timestep)
        timestamps.append(file_time)

        # Generate the filename using extract_filename_from_datetime
        filename = extract_filename_from_datetime(file_time, f_ext)

        # Construct the file path
        file_path = os.path.join(root_path, data_source, event_subdir, filename)

        # Check if the file exists and read it
        if os.path.exists(file_path):
            radar_image = read_image(file_path)
            radar_data.append(radar_image)
        else:
            print(f"Warning: No file found at {file_path}")

    # Convert the radar data list to a NumPy array
    if radar_data:
        radar_data = np.stack(radar_data, axis=0)
    else:
        raise FileNotFoundError(f"No radar files were found for the event in {os.path.join(root_path, data_source, event_subdir)}.")

    # Update metadata with timestamps
    metadata['timestamps'] = np.array(timestamps, dtype=object)
    metadata['accutime'] = timestep
    return radar_data, metadata

def read_image(file_path):
    """
    Read the image file.
    Parameters:
        file_path (str): Path to the image file.
    Returns:
        numpy.ndarray: Array containing the pixel values or None in case of error.
    """
    file_format = get_file_format(file_path)
    if file_format == 'png':
        try:
            img = Image.open(file_path)
            
            # Ensure the image has 2 channels (LA: Luminance + Alpha)
            if img.mode != 'LA':
                raise ValueError("PNG image must have exactly 2 channels (Luminance and Alpha).")
            
            img_array = np.asarray(img)
            mask = img_array[:, :, 1] / 255  # Normalize alpha (mask) values to 0-1
            mask = mask.astype(np.uint8)     # Convert to 0 or 1 (uint8)
            intensity = img_array[:, :, 0]   # Luminance values
            image = intensity * mask         # Apply the mask to intensity
            return image
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'.")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return None

    elif file_format == 'tiff':
        try:
            with rasterio.open(file_path) as src:
                image = src.read(1)  # Read the first band of the TIFF file
            return image
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return None

    else:
        raise ValueError("Unsupported file format. Supported formats are 'png' and 'tiff'.")
def dn_to_dbz(image_data,metadata):
    """
    Convert pixel values from digital number (DN) to reflectivity in dBZ.
    
    Parameters:
        image_data (numpy.ndarray): Array containing the pixel values.
    
    Returns:
        numpy.ndarray: Array containing the reflectivity values in dBZ.
    """
    reflectivity_dbz = (image_data / 2.55) - 100 + 91.4
    metadata["unit"]='dBZ'
    metadata["transform"]='dB'
    return reflectivity_dbz, metadata

def dbz_to_dn(reflectivity_dbz, metadata):
    """
    Convert reflectivity values in dBZ back to digital number (DN).

    Parameters:
        reflectivity_dbz (numpy.ndarray): Array containing the reflectivity values in dBZ.

    Returns:
        numpy.ndarray: Array containing pixel values as digital numbers.
    """
    dn = (reflectivity_dbz - 91.4 + 100) * 2.55
    metadata["unit"] = "DN"
    metadata["transform"] = "linear"
    return dn.astype("uint8"), metadata

 
def noise_remove(image, type='Watershed'):
    if type == 'Median':
        # Median filter specific parameters
        kernel_size = 3
        
        # Apply a median filter
        filtered_image = median_filter(image, size=kernel_size)

    elif type == 'Gaussian':
        # Gaussian filter specific parameters
        kernel_size = (5, 5)
        sigmaX = 1.5
        
        # Apply Gaussian filtering to smooth the image
        filtered_image = cv2.GaussianBlur(image, kernel_size, sigmaX)

    elif type == 'Watershed':
        # Watershed segmentation for cluster detection
        # Convert image to binary
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove small noise using morphological operations
        
        kernel = np.ones((3, 3), np.uint8)
        
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        

        # Background area
        sure_bg = cv2.dilate(opening, kernel, iterations=5)

        # Foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)

        # Unknown region (noise)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add 1 to all labels so that sure background is not zero
        markers = markers + 1

        # Mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Application of watershed algorithm
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        image[markers == -1] = [0]  # Mark boundary
        

        # Convert to filtered image, keeping only cluster regions
        filtered_image = np.zeros_like(image)
        filtered_image[markers > 1] = image[markers > 1]
        
        # Center region noise removal based on cluster size
        # Define the center region (504:518 pixels)
        center_region = filtered_image[500:520, 500:520]
    
        # Convert center region to binary for contour detection
        _, center_binary = cv2.threshold(center_region, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    
        # Detect contours (clusters) in the center region
        contours, _ = cv2.findContours(center_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Iterate over each detected contour and check the cluster size
        for contour in contours:
            cluster_size = cv2.contourArea(contour)
    
            # If the cluster size is between 5 and 15 pixels, it's considered noise
            if 0< cluster_size < 20:
                # Remove the cluster by setting the corresponding pixel values to zero
                cv2.drawContours(center_region, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    
        # Replace the modified center region back into the original image
        filtered_image[500:520, 500:520] = center_region

    else:
        raise ValueError(f"Unknown noise removal type: {type}")
    
    return filtered_image

def clip_domain_epsg4326(data, metadata, extent):
    """
    Clip a 2D or 3D EPSG:4326 array using the given geographic extent.

    Parameters:
        data: np.ndarray
            Shape can be either (T, H, W) or (H, W)
        metadata: dict
            Pysteps-style metadata
        extent: tuple
            (lon_min, lon_max, lat_min, lat_max)

    Returns:
        clipped_data: np.ndarray
        clipped_metadata: dict
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # Determine dimensionality
    if data.ndim == 3:
        has_time = True
        _, ny, nx = data.shape
    elif data.ndim == 2:
        has_time = False
        ny, nx = data.shape
    else:
        raise ValueError("Input data must be 2D or 3D with time as the first dimension.")

    # Generate coordinate arrays
    lons = np.linspace(metadata["x1"], metadata["x2"], nx)
    lats = np.linspace(metadata["y1"], metadata["y2"], ny)

    # Because yorigin='upper', latitude array goes top -> bottom
    if metadata["yorigin"] == "upper":
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
    else:
        lat_mask = (lats <= lat_max) & (lats >= lat_min)

    lon_mask = (lons >= lon_min) & (lons <= lon_max)

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    if lat_indices.size == 0 or lon_indices.size == 0:
        raise ValueError("No overlap between data and clipping extent.")

    # Perform clipping
    if has_time:
        clipped = data[:, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    else:
        clipped = data[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]

    # Update metadata
    new_metadata = metadata.copy()
    new_metadata["x1"] = float(lons[lon_indices[0]])
    new_metadata["x2"] = float(lons[lon_indices[-1]])
    new_metadata["y1"] = float(lats[lat_indices[0]])
    new_metadata["y2"] = float(lats[lat_indices[-1]])

    return clipped, new_metadata
def create_gif(R, metadata, units="dBZ", title="Precipitation Field GIF", duration=0.5, loop=True):
    """
    Create and display a GIF from radar data along the time axis.

    Parameters:
        R (numpy.ndarray): 3D array of radar reflectivity data (time, x, y).
        metadata (dict): Metadata containing geospatial information.
        units (str): Units for the plot (e.g., 'dBZ').
        title (str): Title for the GIF frames.
        duration (float): Duration of each frame in seconds.
        loop (bool): If True, the GIF will loop indefinitely.

    Returns:
        None: Displays the GIF in the Jupyter Notebook.
    """
    import os
    import matplotlib.pyplot as plt
    from pysteps.visualization import plot_precip_field
    import imageio
    from IPython.display import Image, display

    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frame_files = []
    map_kwargs = {"drawlonlatlines": True}

    try:
        # Generate frames for each time step
        for t in range(R.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 8),layout="constrained"); ax.axis("off")

            # Plot precipitation field
            plot_precip_field(
                R[t, :, :], 
                ptype="intensity", 
                geodata=metadata, 
                units=units, 
                title=f"{title} - {metadata['timestamps'][t].strftime('%d/%m/%Y %H:%M')}", 
                ax=ax, 
                map_kwargs=map_kwargs,
                colorscale="pysteps"
            )
            # Save frame
            frame_file = os.path.join(temp_dir, f"frame_{t}.png")
            plt.savefig(frame_file)
            frame_files.append(frame_file)
            plt.close(fig)

        # Save GIF in the current working directory
        gif_path = os.path.join(os.getcwd(), "precipitation_field.gif")
        loop_count = 0 if loop else 1  # 0 means infinite loop
        with imageio.get_writer(gif_path, mode="I", duration=duration, loop=loop_count) as writer:
            for frame_file in frame_files:
                image = imageio.v3.imread(frame_file)
                writer.append_data(image)

        # Display GIF in notebook
        display(Image(gif_path))

    finally:
        # Cleanup temporary files
        for frame_file in frame_files:
            os.remove(frame_file)
        os.rmdir(temp_dir)
