# radar_nowcasting

This repository focuses on analyzing X-Band weather radar data for nowcasting using **Pysteps** (Python framework for short-term ensemble prediction systems). The methods utilized include:

- **Motion Estimation Techniques**:
  - Lucas-Kanade (LK)
  - Variational Echo Tracking (VET)
  - Dynamic and Adaptive Radar Tracking of Storms (DARTS)
  - Anisotropic Diffusion Method (Proesmans et al., 1994)

- **Nowcasting Schemes**:
  - Extrapolation
  - Deterministic nowcasting with S-PROG
  - ANVIL nowcast

- **Verification Metrics**:
  - Mean Absolute Error (MAE)
  - Critical Success Index (CSI)
  - Fractional Skill Score (FSS)

Additionally, the **Power Spectral Density (PSD)** has been calculated to analyze the power spectra of various nowcasting results.

This repository contains code and resources for radar-based precipitation nowcasting. Follow the steps below to clone the repository and set up the required environment on your local system.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Usage](#usage)
5. [Sample Data](#sample-data)
6. [Contributing](#contributing)
7. [Funding](#funding)
8. [References](#references)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: [Download and install Git](https://git-scm.com/downloads).
- **Anaconda/Miniconda**: [Download and install Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

---

## Installation

### Step 1: Clone the Repository
To clone this repository to your local directory, open a terminal and run the following command:
```bash
git clone https://github.com/AVI68/radar_nowcasting.git
```


### Step2: Navigate to the Repository
Change the directory to the cloned repository:
```bash
cd radar_nowcasting
```

## Setting Up the Environment

### Step 1: Create the Conda Environment
The environment.yml file is provided in the repository to create a Conda environment with all the necessary dependencies. Run the following command to create the environment:
```bash
conda env create -f environment.yml
```

### Step 2: Activate the Environment
Activate the newly created environment:

```bash
conda activate radar_nowcasting_env
```

### Step 3: Adjust Cartopy for Basemap Reproduction
By default, Cartopy may have resolution limitations incompatible with this dataset. We use Contextily to overlay basemaps. To avoid conflicts:

```bash
conda deactivate cartopy
```

## Usage

### Step 4: Run the Jupyter Notebook
Run the provided Jupyter Notebook to reproduce the results:
```bash
jupyter notebook
```

## Sample Data
The repository includes a Data directory containing sample datasets required to run the Jupyter Notebook. Make sure the Data directory is in the root of the repository after cloning. The notebook is configured to automatically load data from this directory.
### Data Structure
The Data directory is structured as follows:
```bash
Data/
├── radarmappatipo.tif
└── UNICA_SG/
    ├── yyyymmdd_HHMM/
    │   ├── yyyymmdd_HHMM.png
    │   ├── ...
    │   └── additional_images.png
    ├── another_date_time_folder/
    │   ├── another_image.png
    └── ...
```
### Data Description
1. radarmappatipo.tif: Radar base map file used for visualization and analysis.
2. UNICA_SG: Directory containing radar images organized in subdirectories by timestamp in the format yyyymmdd_HHMM (e.g., 20240101_1200).
	Each timestamped subdirectory contains .png images relevant to that radar capture.
Ensure that the data remains in the specified structure to avoid path issues when running the notebook.

Ensure that the data remains in the Data directory to avoid path issues when running the notebook.


## Contributing
Craeted by Avijit Majhi, PhD. Scholar University of Cagliari, Department of Civil, Environmental and Architectural Engineering.
Contributions are welcome! Feel free to fork this repository, make your changes, and submit a pull request.
## funding
Funded by the GeoSciences IR, WP4, UNICA 03 - Risk monitoring and management project under NRRP, European Union – NextGenerationEU 

## Reference publications

Pulkkinen, S., D. Nerini, A. Perez Hortal, C. Velasco-Forero, U. Germann, A. Seed, and L. Foresti, 2019: Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0). Geosci. Model Dev., 12 (10), 4185–4219, doi:10.5194/gmd-12-4185-2019, https://gmd.copernicus.org/articles/12/4185/2019/
