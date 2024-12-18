# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:26:32 2024

run the code "avi_ppt_dwnsc" environment

@author: Avijit Majhi
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import rasterio
import rasterio
from pysteps.utils import conversion, transformation
from osgeo import gdal, osr, gdalconst
from pysteps import io, motion, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver
from typing import List, Dict, Any, Tuple
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from pysteps.utils.spectral import rapsd
from pprint import pprint
from tqdm import tqdm
import pickle
import cv2
from sklearn.cluster import DBSCAN, KMeans
from scipy.ndimage import median_filter
import pywt
from contextily import add_basemap
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator,FuncFormatter
import contextily
from scipy.signal import correlate2d

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
                    "unit": "dBZ",
                    "transform": "dB",
                    "accutime": 1,
                    "zr_a": 200,
                    "zr_b": 1.6,
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
        
        
        
def import_files_by_date(date, root_path, data_source, fn_pattern,f_ext, metadata, timestep=5, num_prev_files=9):
    """
    Import radar files from a fixed event-specific subdirectory and update metadata with timestamps.

    Parameters:
        date (datetime): The target datetime for the radar files.
        root_path (str): The root directory containing radar files.
        subdir (str): Subdirectory under the root path (e.g., "UNICA_SG").
        fn_pattern (str): The filename pattern (e.g., "%Y%m%d_%H%M.png").
        metadata (dict): The metadata dictionary to be updated.
        timestep (int): Time interval (in minutes) between consecutive files.
        num_prev_files (int): Number of files to retrieve before the target date.

    Returns:
        tuple: 
            - R (numpy.ndarray): Array containing the loaded radar data.
            - metadata (dict): Updated metadata including timestamps.
    """
    
    import glob 
    radar_data = []
    timestamps = []

    # Determine the event-specific subdirectory from the input date
    event_subdir = date.strftime("%Y%m%d_%H%M")

    for i in range(num_prev_files):
        # Compute the timestamp for each file
        file_time = date - timedelta(minutes=i * timestep)
        timestamps.append(file_time)

        # Conditional logic for subdirectory-specific behavior
        if data_source == "Civil_Pro_C":
            # Construct the wildcard search pattern for Civil_Pro_C
            file_timestamp = file_time.strftime(fn_pattern)  # E.g., "201910240320"
            search_pattern = os.path.join(root_path, data_source, event_subdir, f"VMI_*_{file_timestamp}{f_ext}")
        elif data_source == "UNICA_SG":
            # Construct the file path for UNICA_SG
            file_timestamp = file_time.strftime(fn_pattern)  # E.g., "201910240415"
            search_pattern = os.path.join(root_path, data_source, event_subdir, f"{file_timestamp}{f_ext}")
        else:
            raise ValueError(f"Unknown subdirectory: {data_source}")

        # Use glob to find the matching file
        matching_files = glob.glob(search_pattern)

        if matching_files:
            # Assume only one match is valid
            file_path = matching_files[0]
            # Read the radar data using utility.read_image
            radar_image = read_image(file_path)
            radar_data.append(radar_image)
        else:
            print(f"Warning: No file found matching {search_pattern}")

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
def dn_to_dbz(image_data):
    """
    Convert pixel values from digital number (DN) to reflectivity in dBZ.
    
    Parameters:
        image_data (numpy.ndarray): Array containing the pixel values.
    
    Returns:
        numpy.ndarray: Array containing the reflectivity values in dBZ.
    """
    reflectivity_dbz = (image_data / 2.55) - 100 + 91.4
    return reflectivity_dbz
def noise_remove(image, type='Watershed'):
    if type == 'DBSCAN':
        # DBSCAN specific parameters
        eps = 3
        min_samples = 10
        
        # Convert image to a list of points [x, y, intensity]
        points = []
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                intensity = image[x, y]
                if intensity > 0:  # Only consider non-zero intensity pixels
                    points.append([x, y, intensity])
    
        # Convert to a numpy array for processing
        points = np.array(points)
    
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(points)
    
        # Labels for each point in the dataset (-1 represents noise)
        labels = dbscan.labels_
    
        # Remove noise points from the original image
        filtered_image = np.zeros_like(image)
        for (point, label) in zip(points, labels):
            x, y, intensity = point
            if label != -1:  # Keep only non-noise points
                filtered_image[int(x), int(y)] = intensity

    elif type == 'Median':
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

    elif type == 'Bilateral':
        # Bilateral filter specific parameters
        d = 9
        sigmaColor = 75
        sigmaSpace = 75
        
        # Apply bilateral filtering
        filtered_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    elif type == 'NonLocalMeans':
        # Non-local means denoising specific parameters
        h = 10
        templateWindowSize = 7
        searchWindowSize = 21
        
        # Apply non-local means denoising
        filtered_image = cv2.fastNlMeansDenoising(image, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

    elif type == 'Wavelet':
        # Wavelet denoising specific parameters
        value = 10
        
        # Wavelet-based denoising using PyWavelets library
        coeffs = pywt.wavedec2(image, 'db1', level=2)
        coeffs[1:] = [(pywt.threshold(c, value=value, mode='soft') for c in coeff) for coeff in coeffs[1:]]
        filtered_image = pywt.waverec2(coeffs, 'db1')

    elif type == 'Morphological':
        # Morphological operations specific parameters
        kernel_size = (3, 3)
        iterations = 2
        
        # Apply morphological operations like opening to remove small noise spots
        kernel = np.ones(kernel_size, np.uint8)
        filtered_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

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
        
        
    
    elif type == 'KMeans':
        # K-means clustering specific parameters
        k = 2
        
        # K-means clustering for cluster detection and noise removal
        # Reshape image into a 2D array of pixels and 1D intensity
        pixel_values = image.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)

        #  Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert labels to the segmented image
        segmented_image = labels.reshape(image.shape)

        # Choose the largest cluster as the foreground (e.g., clusters with most pixels)
        unique, counts = np.unique(segmented_image, return_counts=True)
        largest_cluster_label = unique[np.argmax(counts)]

        # Create an image with only the largest cluster and remove others
        filtered_image = np.zeros_like(image)
        filtered_image[segmented_image == largest_cluster_label] = image[segmented_image == largest_cluster_label]

    elif type == 'ConnectedComponents':
        # Connected Components specific parameters
        min_cluster_size = 50
        
        #  Detect clusters using connected components or contours
        # Apply a binary threshold to get binary image
        _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        
        #  Find connected components (or contours) in the binary image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # Create an empty image to store filtered clusters
        filtered_image = np.zeros_like(image)
        
        # Filter clusters based on size
        for label in range(1, num_labels):  # Skip label 0 as it's the background
            # Get size of the cluster
            cluster_size = stats[label, cv2.CC_STAT_AREA]
            
            if cluster_size >= min_cluster_size:  # Only keep clusters larger than min_cluster_size
                # Copy the cluster's original intensity values to the filtered image
                filtered_image[labels == label] = image[labels == label]

    else:
        raise ValueError(f"Unknown noise removal type: {type}")
    
    return filtered_image
def calculate_rainfall_intensity(reflectivity_dbz):
    """
    Calculate the total rainfall depth from reflectivity values.
    
    Parameters:
        reflectivity_dbz (numpy.ndarray): Array containing the reflectivity values in dBZ.
        
    Returns:
        float: Total rainfall intensity in mm/hr.
    """
    RR = conversion.to_rainrate(reflectivity_dbz)
    return RR
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
def reproject_large_image(large_image_path, small_image_crs, output_path):
    """
    Reproject the large image to the same projection system as the small image.

    Parameters:
        large_image_path (str): Path to the large image file.
        small_image_crs (str or dict): CRS of the small image.
        output_path (str): Path to save the reprojected image.
    """
    with rasterio.open(large_image_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, small_image_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': small_image_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=small_image_crs,
                    resampling=Resampling.nearest)                
# find the centre of the Radar Image
def find_center_of_image(image_path):
    """
    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: Center point coordinates (x, y).
    """
    with rasterio.open(image_path) as img:
        bounds = img.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
    return center_x, center_y

# create mask 
def create_mask(small_image_path):
    """
    Parameters:
        small_image_path (str): Path to the image file.

    Returns:
        mask (array) : small image mask
    """
    with rasterio.open(small_image_path) as src:
        X_small_img = src.read(1)  # Read sample X band TIFF file
        height,width = X_small_img.shape
        center_x,center_y = width//2,height//2
        radius = 512
        mask = np.zeros((height,width),dtype=np.uint8)
        pil_mask=Image.new('L',(width,height),0)
        draw =ImageDraw.Draw(pil_mask)
        # Draw a white filled circle at the center
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius,
                      center_y + radius), fill=255)
        # Convert the PIL mask back to a numpy array
        mask = np.array(pil_mask)
    return mask
# Creation of Buffer box
def create_buffered_bbox(center_x, center_y, buffer_distance):
    
    """
    Create a buffered bounding box around the center point.

    Parameters:
        center_x (float): X coordinate of the center point.
        center_y (float): Y coordinate of the center point.
        buffer_distance (float): Buffer distance in meters.

    Returns:
        tuple: Buffered bounding box (left, bottom, right, top).
    """
    buffered_bbox = box(center_x - buffer_distance, center_y - buffer_distance,
                        center_x + buffer_distance, center_y + buffer_distance)
    return buffered_bbox.bounds
# Crop the large image using the buffered bounding box
def crop_large_image_with_buffer(large_image_path, buffered_bbox, output_path):
    """
    Crop the large image using the buffered bounding box and save the result.

    Parameters:
        large_image_path (str): Path to the large image file.
        buffered_bbox (tuple): Buffered bounding box (left, bottom, right, top).
        output_path (str): Path to save the cropped image.
    """
    with rasterio.open(large_image_path) as src:
        # Convert buffered bounds to pixel coordinates
        xmin, ymin = src.index(buffered_bbox[0], buffered_bbox[3])
        xmax, ymax = src.index(buffered_bbox[2], buffered_bbox[1])

        # Ensure bounds are within the image boundaries
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(src.width, xmax)
        ymax = min(src.height, ymax)

        # Calculate the window in the larger image corresponding to the buffered extent
        window = Window.from_slices((xmin, xmax),(ymin, ymax))

        # Read the data from the larger image corresponding to the window
        data = src.read(window=window)
        profile = src.profile

        # Update the profile with the new window extent
        profile.update({
            'width': window.width,
            'height': window.height,
            'transform': src.window_transform(window)
        })

        # Write the cropped data to a new GeoTIFF file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
         


    
# Creating a function to plot the histogram with customizable bin size and log scale option

def plot_histogram(array, title, dt, save_dir):
    """
    Plots the intensity histogram of an image array in dBZ, with customizable bin size and log scale option.

    Parameters:
    - image_array: np.array, array containing the image values in dBZ
    - bin_size: int, the size of each bin in dBZ (default is 5 dBZ)
    - log_scale: bool, whether to use logarithmic scale for the frequency (default is False)
    """
    
    bin_size=10
    log_scale=True
    # Replace values below 0 with NaN
    array = np.where(array < 0, np.nan, array)
    # Remove NaN values from the arrays
    image_array = array[np.isfinite(array)]
    # Debug: Check if any non-finite values are removed
    print(f"Total elements: {array.size}, Finite elements: {image_array.size}")

    if image_array.size == 0:
        print(f"Warning: {title} contains no finite values to plot.")
        return
    bins = np.arange(0, 60 + bin_size, bin_size)  # Creating bins based on bin size
    
    # Create the histogram
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(image_array, bins=bins, color='blue', alpha=0.7, edgecolor='black', log=log_scale)
    # Set titles and labels
    plt.title(f'{title} at {dt}')
    plt.xlabel('Rain Intensity (dBZ)')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))
    plt.xlim(0, 60)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{title}_hist_{dt.strftime('%Y%m%d_%H%M')}.png")) 
    plt.close()



def plot_histogram_comparison(array1, array2, title1, title2, dt, save_dir):
    """
    Plots the intensity histograms of two arrays in dBZ, on the same axis for bin-by-bin comparison.

    Parameters:
    - array1, array2: np.array, arrays containing the image values in dBZ
    - title: str, the title for the plot
    - dt: datetime object, the timestamp for the plot (used in the filename)
    - save_dir: str, directory to save the plot image
    """
    
    bin_size = 10
    log_scale = True
    
    # Replace values below 0 with NaN for both arrays
    array1 = np.where(array1 < 0, np.nan, array1)
    array2 = np.where(array2 < 0, np.nan, array2)
    
    # Remove NaN values from the arrays
    image_array1 = array1[np.isfinite(array1)]
    image_array2 = array2[np.isfinite(array2)]
    
    # Debugging: Check if any non-finite values were removed
    print(f"Array 1 - Total elements: {array1.size}, Finite elements: {image_array1.size}")
    print(f"Array 2 - Total elements: {array2.size}, Finite elements: {image_array2.size}")

    if image_array1.size == 0 or image_array2.size == 0:
        print(f"Warning: One of the arrays contains no finite values to plot.")
        return
    
    # Create common bins based on the bin size
    bins = np.arange(0, 60 + bin_size, bin_size)
    
    # Create the histogram plot
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(image_array1, bins=bins, color='blue', alpha=0.5, edgecolor='black', log=log_scale, label='Array 1')
    plt.hist(image_array2, bins=bins, color='red', alpha=0.5, edgecolor='black', log=log_scale, label='Array 2')
    
    # Set titles and labels
    plt.title(f'Rain Intensity Comparison: {title1} vs {title2} on {dt}')
    plt.xlabel('Rain Intensity (dBZ)')
    plt.ylabel('Frequency' + (' (log scale)' if log_scale else ''))
    plt.xlim(0, 60)
    plt.grid(True)
    plt.legend()

    # Save the plot
    file_name = f"hist_combined_{dt.strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(os.path.join(save_dir, file_name)) 
    plt.close()

  
    

def plotting_positions(data, a=0.5):
    """Calculate plotting positions for empirical CDF."""
    n = len(data)
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)

def plot_empirical_cdf(array_C,array_X, title1,title2, save_dir,dt):
    # Ensure the inputs are numpy arrays
    array_C = np.asarray(array_C)
    array_X[array_X <0] = np.nan
    array_X = np.asarray(array_X)

    # Filter and sort finite values
    xint = np.sort(array_C[np.isfinite(array_C)])
    yint = np.sort(array_X[np.isfinite(array_X)])

    # Calculate empirical CDF values
    a = 0.5
    x_cdf = plotting_positions(xint, a)
    y_cdf = plotting_positions(yint, a)

    # Plotting the results
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(xint, x_cdf, label=title1, color='blue')
    plt.plot(yint, y_cdf, label=title2, color='green')
    plt.xlabel('Sorted Data')
    plt.ylabel('CDF')
    plt.legend()
    plt.title(f'ECDF of {title1} vs. {title1} on {dt}')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'ECDF_comparison_{dt.strftime('%Y%m%d_%H%M')}.png')) 
    plt.close()
    
    

# get the X-band filenames based on the time range
def get_xband_filenames(rainfall_data, timestamp, x):
    xband_filenames = []
    
    # Loop over the past 'x' minutes starting from 'timestamp'
    for minute_offset in range(x):
        # Calculate the timestamp for each minute back
        target_time = timestamp - pd.Timedelta(minutes=minute_offset)
        
        # Try to find the X-band file in rainfall_data for the target_time
        matching_row = rainfall_data[rainfall_data['datetime'] == target_time]
        
        if not matching_row.empty:
            # If the matching datetime is found, append the filename
            xband_filenames.append(matching_row['filename'].values[0])
        else:
            # If no file is found, repeat the last available file directory
            if xband_filenames:
                xband_filenames.append(xband_filenames[-1])  # Repeat the last available file
            else:
                print(f"Warning: No X-band data found for {target_time}. Unable to fetch the previous files.")

    return xband_filenames
def normalized_cross_corr(array1, array2):
    # Replace NaN values with 0 for correlation purposes
    array1 = np.nan_to_num(array1)
    array2 = np.nan_to_num(array2)
    # Compute normalized cross-correlation
    ncc = correlate2d(array1, array2, boundary='symm', mode='same')
    return ncc

def pearson_corr(array1,array2):
    # Replace NaN values with 0 for correlation purposes
    array1 = np.nan_to_num(array1)
    array2 = np.nan_to_num(array2)
    corr = np.corrcoef(array1.flatten(), array2.flatten())[0, 1]
    return corr

# Function to adjust the X-band and C-band frames
def adjust_band_values(cband_frame, xband_frame):
    # Find the max of both the frames, ignoring NaNs and zeros
    max_c = np.nanmax(cband_frame[cband_frame != 0])
    max_x = np.nanmax(xband_frame[xband_frame != 0])

    # Calculate the ratio of max_c / max_x
    if max_x != 0:
        ratio = max_c / max_x
    else:
        print("Warning: Maximum X-band value is zero, setting ratio to 1.")
        ratio = 1  # Default to 1 if max_x is zero

    # Adjust the X-band values using the ratio, ignoring NaNs, zeros and negative values
    x_band_adj = np.where((~np.isnan(xband_frame)) & (xband_frame > 0), 
                          xband_frame * ratio, xband_frame)
    
    threshold =30
    
    x_band_adj = np.where((x_band_adj >= threshold),
                             x_band_adj, np.nan)
    

    # # Calculate the percentile threshold for the adjusted X-band (ignoring zeros and NaNs)
    # valid_x_band_adj = x_band_adj[x_band_adj > 0]
    # threshold = np.percentile(valid_x_band_adj, percentile=0.001)
    
    
    # Create two frames from the C-band data
    # - c_band_within: C-band values that fall within the X-band range
    # - c_band_outside: C-band values that fall outside (below) the X-band range
    c_band_within = np.where((cband_frame >= threshold),
                             cband_frame, np.nan)
    c_band_outside = np.where(cband_frame < threshold, cband_frame, np.nan)

    return c_band_within, x_band_adj, c_band_outside, ratio    


# for plotting rainfal intensity map   
def precip_intensity_plots(R,metadata, title, ax, out_dir, dt):
    
    """
    Plot precipitation intensity map by adding basemap.
    
    Parameters:
        R (array): rainfall intensity in dBZ
        metadata (dict) : metadata of the R file
        title (str) : Title for the plot
        ax : for providing the axis (useful for subplot)
        dt : timestamps
    """
    
    
    R[R == -9998] =np.nan
    R[R == -9999] =np.nan
    R[R <0] = np.nan
    def format_func(value, tick_number):
        return f"{int(value):,}"  # Format to one decimal place    
    
    # Concatenate title with datetime
    full_title = f"{title} of {dt.strftime('%Y-%m-%d %H:%M:')}"
     
    ax = plot_precip_field(R, ptype="intensity", geodata=metadata,units="dBZ", 
                           title=full_title, ax=ax,colorscale='pysteps')

    # Format the x and y axis labels
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Reduce the number of x-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Reduce the number of y-axis ticks
    add_basemap(ax, crs=metadata["projection"], 
                source=contextily.providers.OpenStreetMap.Mapnik)
    plt.savefig(os.path.join(out_dir, f"{title}_{dt.strftime('%Y%m%d_%H%M')}.png")) 
    
# # reconstruction of PC radar using TDM -Power spectra merging
# reconstruction of PC radar using TDM -Power spectra merging
def reconstruct_rain(c_band_adj, x_band_adj, output_dir):
    # Replace values below 0 with NaN
    
    c_band_adj = np.where(np.isnan(c_band_adj), 0, c_band_adj)
    x_band_adj = np.where(np.isnan(x_band_adj), 0, x_band_adj)
    Ztdm = x_band_adj
    Zpc = c_band_adj
    r = 30000
    msize = int(x_band_adj.shape[0] / 2)
    h = np.floor((r * np.cos(np.pi / 4)) / 60).astype("int")
    # PC frame within the square inscribed into the TDM radius domain
    Zpc_cal_rep = c_band_adj[msize-h:-(msize-h), msize-h:-(msize-h)]
    # TDM frame within the square inscribed into the TDM radius domain
    ZtdmQ = x_band_adj[msize-h:-(msize-h), msize-h:-(msize-h)]
    #######################################################
    # Power spectra merging using Radial Average method
    # Radial average PC spectrum
    P1pc, k1pc = rapsd(Zpc_cal_rep, fft_method=np.fft, return_freq=True, d=1/18)
    k1pc = np.abs(k1pc)  # all frequencies should be positive
    # Radial average TDM spectrum
    P1tdm, k1tdm = rapsd(ZtdmQ, fft_method=np.fft, return_freq=True, d=1/18)
    k1tdm = np.abs(k1tdm)  # all frequencies should be positive
    ##### merging of Spectrums  ################################
    wl_cut = 1
    k1mrg = k1pc.copy()
    P1mrg = P1pc.copy()
    k1mrg[k1tdm > wl_cut] = k1tdm[k1tdm > wl_cut].copy()
    P1mrg[k1tdm > wl_cut] = P1tdm[k1tdm > wl_cut].copy()
    ########################################################
    ########################################################
    # Plot 1D spectra using rapsd 
    plt.figure(figsize=(8,6),dpi=300)
    plt.plot(k1tdm, 10 * np.log10(P1tdm), "-.r", label='TDM')
    plt.plot(k1pc, 10 * np.log10(P1pc), "-.b", label='PC')
    plt.plot(k1mrg, 10 * np.log10(P1mrg), "--g", label='merge')
    plt.xscale("log", base=2)
    plt.xlim([0.001, 15])
    plt.legend(fontsize=8)
    plt.xlabel('Frequency [$km^{-1}$]', fontsize=8)
    plt.ylabel('10 $\\times$ log($\\overline{P}$)', fontsize=8)
    plt.grid(linewidth=0.5)
    plt.rc("axes", linewidth=1)
    plt.savefig(os.path.join(output_dir, 'Power_Spectral_with_merge_spectra.png'))
    plt.close() 
    ############### Ensemble method #############################
    # Frequencies for the whole TDM domain
    lx = x_band_adj.shape[0]
    ly = x_band_adj.shape[1]
    kItdm = np.fft.fftfreq(lx, d=1/18)
    kJtdm = np.fft.fftfreq(ly, d=1/18)
    ktdm = np.sqrt(kItdm[:, None] ** 2 + kJtdm[None, :] ** 2)
    # 2D merge spectrum generation
    PSbld = np.interp(ktdm, k1mrg, P1mrg)
    # GENERATION OF DOWNSCALED FIELDS
    # Array allocation for saving fields
    Nens = 4  # number of ensemble to generate
    Zens = np.zeros(shape=(lx, ly, Nens))
    # Cycle for the ensemble number
    for n in range(Nens):
        # Random phase generation
        fg = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(*PSbld.shape))
        fg[0, 0] = 0
        # Random phase application
        FTbld = np.sqrt(PSbld) * fg
        # Inverse transform of the Fourier spectrum
        IFTbld = np.fft.ifft2(FTbld).real  # g is Gaussian
        IFTbld /= IFTbld.std()  # Fixed the unit variance
        # Exponential transform of the field
        Ibld = np.exp(IFTbld)
        # PC field aggregation at UniCA resolution
        scale = 18
        
        Prep = convolve2d(Zpc, np.ones(shape=(scale, scale)) * 1/(scale**2), mode='same')
        # Reconstructed field aggregation
        Pagg = convolve2d(Ibld, np.ones(shape=(scale, scale)) * 1/(scale**2), mode="same")
        # Enforce consistency with initial PC field
        eps = np.nanmax(Ztdm) / np.nanmax(Zpc)
        Ibld = Prep * Ibld * eps / Pagg
        Ibld[Ibld < 0] = 0
        Zens[:, :, n] = Ibld
    Zmrg = Zens[:, :, 0]
    # Ensemble average calculation
    Zmean = np.mean(Zens, axis=2)
    # Calculation of the 1D spectrum of the generated field
    P1mrg, k1mrg = rapsd(Zmrg, fft_method=np.fft, return_freq=True, d=1/18)
    k1mrg = np.abs(k1mrg)
    # Calculation of the 1D spectrum of the generated ensemble field
    P1bld, k1bld = rapsd(Zmean, fft_method=np.fft, return_freq=True, d=1/18)
    k1bld = np.abs(k1bld)
    # 1D SPECTRUM PLOT
    plt.figure(figsize=(8, 6),dpi=300)
    plt.plot(k1pc, 10 * np.log10(P1pc), "-.b", label='PC - Zmean = ' + "%8.6f" % Zpc.mean() + " [dBZ]", linewidth=1, markersize=8)
    plt.plot(k1tdm, 10 * np.log10(P1tdm), "-.r", label='TDM - Zmean = ' + "%8.6f" % np.nanmean(Ztdm) + " [dBZ]", linewidth=1, markersize=8)
    plt.plot(k1mrg, 10 * np.log10(P1mrg), "--g", label='Merge- Zmean = ' + "%8.6f" % Zmrg.mean() + " [dBZ]", linewidth=1, markersize=8)
    plt.plot(k1bld, 10 * np.log10(P1bld), "--m", label='Ensemble_merge - Zmean = ' + "%8.6f" % Zmean.mean() + " [dBZ]", linewidth=2)
    plt.xlim([0.001, 15])
    plt.xscale("log", base=2)
    plt.legend(fontsize=8)
    plt.xlabel('Frequency [$km^{-1}$]', fontsize=8)
    plt.ylabel('10 $\\times$ log($\\overline{P}$)', fontsize=8)
    plt.grid(linewidth=0.5)
    plt.rc("axes", linewidth=1)
    plt.savefig(os.path.join(output_dir, 'Power_Spectral_with_reconstructed_intensity_map.png'))
    plt.close() 
    return Zmrg,Zmean