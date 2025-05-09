import os
import numpy as np
import h5py
from PIL import Image
from datetime import datetime, timedelta, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_valid_radar_types():
    """Return the standardized list of valid radar types"""
    return ["X-Band", "C-Band", "X-Band-dualpol"]
def generate_radar_filepaths(data_path, radar_type, data_type, ele_info, dt, scan_interval=5):
    """
    Generate radar data file path for the specified datetime only.
    
    Parameters:
        data_path (str): Root data directory path
        radar_type (str): One of ["X-Band-dualpol", "C-Band", "X-Band"]
        data_type (str): Varies by radar_type:
            - "X-Band-dualpol": ["OZW","CZW","PZW","PUW"]
            - "C-Band": ["CAPPI1"..."CAPPI8", "VMI", "SRT1", "SRI"]
            - "X-Band": ["data"]
        ele_info (str/None): Elevation information:
            - "X-Band-dualpol": "810" for OZW, "805" for CZW, "801"-"805" for PZW/PUW
            - "C-Band": 1-8 for CAPPI, None for VMI/SRT1/SRI
            - "X-Band": None
        dt (datetime): Datetime object for the target time
        scan_interval (int): Minutes between scans (used for rounding)
        
    Returns:
        str: Generated file path or None if invalid
    """
    # Validate inputs
    if radar_type not in get_valid_radar_types():
        raise ValueError(f"Invalid radar_type. Must be one of {get_valid_radar_types()}")
    
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Round to nearest scan interval
    minutes = (dt.minute // scan_interval) * scan_interval
    rounded_dt = dt.replace(minute=minutes, second=0, microsecond=0)
    
    # Generate single file path
    try:
        if radar_type == "X-Band-dualpol":
            return _generate_xband_dualpol_path(data_path, data_type, ele_info, rounded_dt)
        elif radar_type == "C-Band":
            return _generate_cband_path(data_path, data_type, ele_info, rounded_dt)
        elif radar_type == "X-Band":
            return _generate_xband_path(data_path, data_type, rounded_dt)
    except Exception as e:
        logging.error(f"Failed to generate path for {rounded_dt}: {str(e)}")
        return None

def _generate_xband_dualpol_path(data_path, data_type, ele_info, dt):
    """Generate path for X-Band-dualpol data"""
    # Validate data type
    valid_data_types = ["OZW", "CZW", "PZW", "PUW"]
    if data_type not in valid_data_types:
        raise ValueError(f"For X-Band-dualpol, data_type must be one of {valid_data_types}")
    
    # Validate elevation info
    if data_type in ["OZW", "CZW"]:
        valid_ele = ["810"] if data_type == "OZW" else ["805"]
    else:  # PZW or PUW
        valid_ele = [f"80{i}" for i in range(1, 6)]
    
    if ele_info not in valid_ele:
        raise ValueError(f"For {data_type}, ele_info must be one of {valid_ele}")
    
    # Format date and time strings
    year_short = dt.strftime("%y")
    day_of_year = dt.strftime("%j")
    date_str = f"{year_short}{day_of_year}"
    time_str = dt.strftime("%H%M") + "0"
    
    # Generate filename and path
    filename = f"{data_type}{date_str}{time_str}L.{ele_info}.h5"
    dir_path = os.path.join(data_path, data_type, f"{data_type}{date_str}")
    return os.path.join(dir_path, filename)


def _generate_cband_path(data_path, data_type, ele_info, dt):
    """Generate path for C-Band data"""
    # Validate data type
    valid_data_types = [f"CAPPI{i}" for i in range(1, 9)] + ["VMI", "SRT1", "SRI"]
    if data_type not in valid_data_types:
        raise ValueError(f"For C-Band, data_type must be one of {valid_data_types}")
    
    # Validate elevation info
    if data_type.startswith("CAPPI"):
        if ele_info is None or not ele_info.isdigit() or int(ele_info) not in range(1, 9):
            raise ValueError("For CAPPI, ele_info must be 1-8")
    else:  # VMI, SRT1, SRI
        if ele_info is not None:
            raise ValueError(f"For {data_type}, ele_info must be None")
    
    # Convert datetime to Unix epoch milliseconds
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    unix_time = int(dt.timestamp() * 1000)
    
    # Generate filename and path
    filename = f"{data_type}_{unix_time}.h5"
    dir_path = os.path.join(data_path, data_type, 
                          dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
    return os.path.join(dir_path, filename)


def _generate_xband_path(data_path, data_type, dt):
    """Generate path for X-Band data"""
    if data_type != "data":
        raise ValueError("For X-Band, data_type must be 'data'")
    
    # Generate filename and path
    filename = dt.strftime("%Y%m%d_%H%M.png")
    dir_path = os.path.join(data_path, data_type, 
                          dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
    return os.path.join(dir_path, filename)

def get_metadata(radar_type):
    """
    Create metadata dictionary for pysteps 
    
    Parameters:
        radar_type (str): One of ["X-Band-dualpol", "C-Band", "X-Band"]
    
    Returns:
        dict: Dictionary containing metadata
        
    """
    if radar_type not in get_valid_radar_types():
        raise ValueError(f"Invalid radar_type. Must be one of {get_valid_radar_types()}")
    if radar_type == 'X-Band':
        return {
            "projection": '+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 +x_0=1500000 +y_0=0 +ellps=intl +towgs84=-104.1,-49.1,-9.9,0.971,-2.917,0.714,-11.68 +units=m +no_defs +type=crs',
            "cartesian_unit": "m",
            "x1": 1478699.0,
            "y1": 4311475.0,
            "x2": 1540139.0,
            "y2": 4372915.0,
            "xpixelsize": 60,
            "ypixelsize": 60,
            "yorigin": 'upper',
            "institution": "UNICA-GAVINO",
            "unit": "DN",
            "transform": None,  
            "accutime": 1,
            "zr_a": 200,
            "zr_b": 1.6,
            "product": "png",
            "zerovalue": 0,
            "threshold": 0,
        }
    elif radar_type == 'C-Band':
        return {
            "projection":'EPSG:4326',
            "x1": 4.523391511849696,
            "y1": 35.06508577670113,
            "x2": 20.476608488150305,
            "y2": 47.84898921631927,
            "xpixelsize": 0.013294347480250508,
            "ypixelsize": 0.00913135959972724,
            "yorigin": "upper",
            "institution": "Italian Civil Protection",  
            "product": "h5",
            "cartesian_unit": "degree",
            "unit": "dBZ",
            "transform": "dB",
            "accutime": 5,
            "zr_a": 200,
            "zr_b": 1.6,
            "zerovalue": -30,
            "threshold": -30
        }
    elif radar_type == 'X-Band-dualpol':  
        return {
            "projection": '+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 +x_0=1500000 +y_0=0 +ellps=intl +towgs84=-104.1,-49.1,-9.9,0.971,-2.917,0.714,-11.68 +units=m +no_defs',
            "cartesian_unit": "m",
            "x1": 1390630.7804467857,
            "y1": 4226918.564180394,
            "x2": 1630630.7804467857,
            "y2": 4466918.564180394,
            "xpixelsize": 250,
            "ypixelsize": 250,
            "yorigin": "upper",
            "institution": "UNICA-PERSER",
            "unit": "dBZ",
            "transform": "dB",
            "accutime": 5,
            "zr_a": 200,
            "zr_b": 1.6,
            "product": "h5",
            "zerovalue": -30,
            "threshold": -30,
        }
    else:
        raise ValueError(f"Unknown radar type: {radar_type}. Must be one of ['X-Band-dualpol', 'C-Band', 'X-Band']")

def read_image(file_path, radar_type):
    """
    Read radar image data from file with proper validation.
    
    Parameters:
        file_path (str): Path to radar data file
        radar_type (str): Must be one of get_valid_radar_types()
        
    Returns:
        np.ma.MaskedArray: Masked array of radar data or None if error
    """
    try:
        if radar_type not in get_valid_radar_types():
            raise ValueError(f"Invalid radar_type. Must be one of {get_valid_radar_types()}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if radar_type == "X-Band":
            return _read_xband_image(file_path)
        elif radar_type == "C-Band":
            return _read_cband_image(file_path)
        elif radar_type == "X-Band-dualpol":
            return _read_xband_dualpol_image(file_path)
            
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {str(e)}")
        return None
def _read_xband_image(file_path):
    """Read X-Band PNG image"""
    try:
        with Image.open(file_path) as img:
            if img.mode != 'LA':
                raise ValueError("PNG must have Luminance+Alpha channels")
            
            img_array = np.asarray(img)
            mask = (img_array[:, :, 1] / 255 > 0.5).astype(np.uint8)
            return img_array[:, :, 0] * mask
            
    except Exception as e:
        logging.error(f"X-Band read error: {str(e)}")
        return None

def _read_cband_image(file_path):
    """Read C-Band HDF5 image"""
    try:
        with h5py.File(file_path, 'r') as file:
            image = file['radar_data'][:]
            invalid_mask = (np.isnan(image)) | (image == -9999) | (image == -9998)
            return np.ma.masked_where(invalid_mask, image)
            
    except Exception as e:
        logging.error(f"C-Band read error: {str(e)}")
        return None

def _read_xband_dualpol_image(file_path):
    """Read X-Band-dualpol HDF5 image"""
    try:
        with h5py.File(file_path, 'r') as file:
            dataset1 = file['dataset1']
            data1 = dataset1['data1']
            data = data1['data'][:]
            
            what_data1 = data1['what']
            gain = what_data1.attrs.get('gain', 1.0)
            offset = what_data1.attrs.get('offset', 0.0)
            nodata = what_data1.attrs.get('nodata', np.nan)
            undetect = what_data1.attrs.get('undetect', np.nan)
            
            image = data * gain + offset
            return np.ma.masked_where(
                (image == nodata) | (image == undetect) | np.isnan(image), 
                image
            )
            
    except Exception as e:
        logging.error(f"X-Band-dualpol read error: {str(e)}")
        return None
def fname2dt(filepath, radar_type):
    """
    Extract datetime from filename based on radar type.

    Parameters:
        filepath (str): Full path to the radar file
        radar_type (str): One of ["X-Band-dualpol", "C-Band", "X-Band"]

    Returns:
        datetime: Extracted datetime object
    """
    fname = os.path.basename(filepath)

    if radar_type == "C-Band":
        # Format: "PRODUCT_TIMESTAMP.h5"
        ts = int(fname.split('_')[1].split('.')[0])
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    elif radar_type == "X-Band-dualpol":
        # Format: "XXXyydddHHMM0L.ELV.h5", where XXX is a 3-letter prefix
        time_str = fname[3:12]  # Skip the 3-letter prefix and take next 9 characters
        year = 2000 + int(time_str[:2])
        doy = int(time_str[2:5])
        hour = int(time_str[5:7])
        minute = int(time_str[7:9])
        return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute)

    elif radar_type == "X-Band":
        # Format: "YYYYMMDD_HHMM.png"
        fname_no_ext = fname.split('.')[0]
        return datetime.strptime(fname_no_ext, "%Y%m%d_%H%M")

    else:
        raise ValueError(f"Unsupported radar_type: {radar_type}")
def import_files_by_date(
    root_path,
    radar_type,
    data_type,
    ele_info,
    dt,
    scan_interval=5,
    num_prev_files=None,
    num_next_files=None
):
    """
    Import radar files by date and stack them into a 3D array with updated metadata.
    
    Parameters:
        root_path (str): Root directory containing radar data
        radar_type (str): One of ["X-Band-dualpol", "C-Band", "X-Band"]
        data_type (str): Product type specific to each radar
        ele_info (str/None): Elevation information
        dt (datetime): Target datetime
        scan_interval (int): Minutes between scans (default 5)
        num_prev_files (int/None): Number of previous files to include (None or 0 for none)
        num_next_files (int/None): Number of next files to include (None or 0 for none)
        
    Returns:
        tuple: (radar_data, metadata) where:
            - radar_data: 3D numpy array of stacked radar images (time, y, x)
            - metadata: Updated metadata dictionary with timestamps
    """
    # Validate input
    if num_prev_files is not None and num_prev_files < 0:
        raise ValueError("num_prev_files must be None or a non-negative integer.")
    if num_next_files is not None and num_next_files < 0:
        raise ValueError("num_next_files must be None or a non-negative integer.")
        
    # Determine radar-specific data path
    if radar_type == "C-Band":
        data_path = os.path.join(root_path, "Data_Civil_Protection")
    elif radar_type == "X-Band":
        data_path = os.path.join(root_path, "Data_gavino")
    elif radar_type == "X-Band-dualpol":
        data_path = os.path.join(root_path, "Data_perser")
    else:
        raise ValueError(f"Invalid radar_type: {radar_type}")

    # Handle None/0 cases for file counts
    num_prev = num_prev_files if isinstance(num_prev_files, int) and num_prev_files > 0 else 0
    num_next = num_next_files if isinstance(num_next_files, int) and num_next_files > 0 else 0

    metadata = get_metadata(radar_type)

    # Round datetime to nearest scan_interval
    minutes = (dt.minute // scan_interval) * scan_interval
    rounded_dt = dt.replace(minute=minutes, second=0, microsecond=0)

    # Create list of timestamps to check
    datetimes_to_check = [
        rounded_dt + timedelta(minutes=i * scan_interval)
        for i in range(-num_prev, num_next + 1)
    ]

    radar_data = []
    timestamps = []

    for check_dt in datetimes_to_check:
        try:
            filepath = generate_radar_filepaths(
                data_path=data_path,
                radar_type=radar_type,
                data_type=data_type,
                ele_info=ele_info,
                dt=check_dt,
                scan_interval=scan_interval
            )

            if not filepath or not os.path.exists(filepath):
                logging.warning(f"File not found for {check_dt}: {filepath}")
                continue

            image = read_image(filepath, radar_type)
            if image is not None:
                radar_data.append(image)
                timestamps.append(fname2dt(filepath, radar_type))

        except Exception as e:
            logging.error(f"Error processing {check_dt}: {str(e)}")
            continue

    if not radar_data:
        raise FileNotFoundError(f"No valid radar files found for {dt}")

    radar_data = np.stack(radar_data, axis=0)

    metadata['timestamps'] = np.array(timestamps, dtype=object)
    metadata['accutime'] = scan_interval
    metadata['n_timesteps'] = len(timestamps)

    return radar_data, metadata