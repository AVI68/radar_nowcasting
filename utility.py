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
import pywt


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
      
        
def import_files_by_date(date, root_path, data_source, f_ext, metadata, num_prev_files, timestep=5):
    """
    Import radar files from a fixed event-specific subdirectory and update metadata with timestamps.

    Parameters:
        date (datetime): The target datetime for the radar files.
        root_path (str): The root directory containing radar files.
        data_source (str): Subdirectory under the root path (e.g., "UNICA_SG" or "Civil_Pro_C").
        f_ext (str): File extension (e.g., "png" or "tiff").
        metadata (dict): The metadata dictionary to be updated.
        timestep (int): Time interval (in minutes) between consecutive files.
        num_prev_files (int): Number of files to retrieve before the target date.

    Returns:
        tuple:
            - radar_data (numpy.ndarray): Array containing the loaded radar data.
            - metadata (dict): Updated metadata including timestamps.
    """
    radar_data = []
    timestamps = []

    # Determine the event-specific subdirectory from the input date
    event_subdir = date.strftime("%Y%m%d_%H%M")

    for i in range(num_prev_files,-1,-1):
        # Compute the timestamp for each file
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


# for plotting rainfal intensity map and save in an output directory
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
    R[R <-10] = np.nan 
    
    # Concatenate title with datetime
    full_title = f"{title} of {dt.strftime('%Y-%m-%d %H:%M:')}"
     
    ax = plot_precip_field(R, ptype="intensity", geodata=metadata,units="dBZ", 
                           title=full_title, ax=ax,colorscale='pysteps')
    plot_modification(ax, metadata)
    plt.savefig(os.path.join(out_dir, f"{title}_{dt.strftime('%Y%m%d_%H%M')}.png")) 
    
def plot_modification(ax, metadata, crs_geo="EPSG:4326", tick_step=5):
    """
    Modify an existing plot's axis with geographic labels, basemap, and tick formatting.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object to modify.
        metadata (dict): Metadata containing projection information.
        crs_geo (str): Geographic CRS for converting coordinates (default is 'EPSG:4326').
        tick_step (int): Step size for major ticks on both axes.
    """
    
    import pyproj
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import contextily as cx
    # Extract CRS projection from metadata
    crs_proj = metadata["projection"]

    # Function to convert meter-based coordinates to degrees
    def convert_meters_to_degrees(x, y, crs_proj, crs_geo="EPSG:4326"):
        transformer = pyproj.Transformer.from_crs(crs_proj, crs_geo, always_xy=True)
        return transformer.transform(x, y)

    # Formatter for ticks
    def format_ticks(value, tick_number, axis):
        """
        Format tick values to display degrees with compass directions (E, W, N, S).
        
        Parameters:
            value (float): The coordinate value (in meters).
            tick_number (int): Tick index (not used, required by FuncFormatter).
            axis (str): The axis being formatted ('x' for longitude, 'y' for latitude).
        
        Returns:
            str: Formatted tick label with degrees and compass directions.
        """
        # Convert meters to degrees
        lon, lat = convert_meters_to_degrees(value, 0, crs_proj) if axis == "x" else convert_meters_to_degrees(0, value, crs_proj)
        
        # Determine compass direction
        if axis == "x":  # Longitude
            direction = "E" if lon >= 0 else "W"
            lon = abs(lon)  # Take absolute value for formatting
            return f"{lon:.2f}° {direction}"
        else:  # Latitude
            direction = "N" if lat >= 0 else "S"
            lat = abs(lat)  # Take absolute value for formatting
        return f"{lat:.2f}° {direction}"


    # Format x and y axis labels to degrees
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_ticks(x, 0, "x")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_ticks(y, 0, "y")))

    # Reduce the number of ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_step))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_step))
    # Rotate the Y-axis tick labels by 90 degrees
    ax.tick_params(axis='y',labelrotation=90)

    # Add a basemap
    cx.add_basemap(ax, crs=crs_proj, source=cx.providers.OpenStreetMap.Mapnik)
    

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