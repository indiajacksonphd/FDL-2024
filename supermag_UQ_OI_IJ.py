"""A module that plots the SECs + GP mean and standard deviation for SuperMAG stations
            ==> will be later deployed on our website

Author: Opal Issan (oissan@ucsd.edu) PhD @ ucsd
        India Jackson (indiajacksonphd@gmail.com) Posdoc @ gsu


Last Modified: August 14th, 2024
"""

import numpy as np
import tkinter
import cartopy.crs as ccrs
from supermag_api import *
import GPy
from sec import T_df, get_mesh
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
from mycolorpy import colorlist as mcp
from google.cloud import storage
import datetime

font = {'family': 'serif',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

storage_client = storage.Client()

def upload_to_gcloud(project_name, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def get_current_datetime():
    # Returns a formatted datetime string, e.g., '20230901_150505'
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
# Function to download CSV from Google Cloud Storage
def download_csv_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket and returns it as a pandas DataFrame."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    content = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(content), header = None)
    return df

def get_last_uploaded_object(project_name, bucket_name, prefix):
    """Retrieve the most recently uploaded object in a specified 'directory' within a bucket."""
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # List all objects that have the prefix

    # Initialize a variable to track the most recent object
    last_uploaded = None
    last_uploaded_time = None

    for blob in blobs:
        if last_uploaded_time is None or blob.time_created > last_uploaded_time:
            last_uploaded = blob.name
            last_uploaded_time = blob.time_created

    return last_uploaded, last_uploaded_time

# Function to convert DataFrame to NPY
def df_to_npy(df):
    """Convert a pandas DataFrame to a numpy array."""
    return df.to_numpy()

def split_lat_lon_df(df):
    """Split lat/lon DataFrame and return separate NPY arrays."""
    glat = df.iloc[1:]['glat'].astype(float).to_numpy()
    glon = df.iloc[1:]['glon'].astype(float).to_numpy()
    return glat, glon

def save_predictions(pred_lon, pred_lat, mean_, sd_, n_lat, n_lon, output_path):
    """
    Extracts predictions and saves them to a CSV file including standard deviations.

    Parameters:
    pred_lon (np.ndarray): 3D array of predicted longitudes.
    pred_lat (np.ndarray): 3D array of predicted latitudes.
    mean_ (np.ndarray): 1D array of predicted mean values.
    sd_ (np.ndarray): 1D array of predicted standard deviations.
    n_lat (int): Number of latitude grid points.
    n_lon (int): Number of longitude grid points.
    output_path (str): Path to save the CSV file.
    """
    # Reshape predictions and standard deviations
    pred_lon_2d = pred_lon[:, :, 0]
    pred_lat_2d = pred_lat[:, :, 0]
    mean_reshaped_1 = np.reshape(mean_[:n_lat * n_lon], (n_lat, n_lon), "C")
    mean_reshaped_2 = np.reshape(mean_[n_lat * n_lon:2 * n_lat * n_lon], (n_lat, n_lon), "C")
    sd_reshaped_1 = np.reshape(sd_[:n_lat * n_lon], (n_lat, n_lon), "C")
    sd_reshaped_2 = np.reshape(sd_[n_lat * n_lon:2 * n_lat * n_lon], (n_lat, n_lon), "C")
    
    # Flatten the arrays for saving
    pred_lon_flat = pred_lon_2d.flatten()
    pred_lat_flat = pred_lat_2d.flatten()
    mean_flat_1 = mean_reshaped_1.flatten()
    mean_flat_2 = mean_reshaped_2.flatten()
    sd_flat_1 = sd_reshaped_1.flatten()
    sd_flat_2 = sd_reshaped_2.flatten()
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Longitude': pred_lon_flat,
        'Latitude': pred_lat_flat,
        'Mean for Bn': mean_flat_1,
        'Mean for Be': mean_flat_2,
        'SD for Bn': sd_flat_1,
        'SD for Be': sd_flat_2
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)

############################################################################## Web Portal #########################################################################
'''
last_uploaded_inference_dbn, last_uploaded_inference_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/202405120000/dbn_geo')
last_uploaded_inference_dbe, last_uploaded_inference_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/202405120000/dbe_geo')
last_uploaded_inference_dbz, last_uploaded_inference_dbz_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/202405120000/dbz_geo')
'''
############################################################################## Web Portal #########################################################################



########################################################################################## Stage 1 ##############################################################################################
last_uploaded_inference_ace_dbn, last_uploaded_inference_ace_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_1/ace_predictions/dbn_geo')
last_uploaded_inference_ace_dbe, last_uploaded_inference_ace_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_1/ace_predictions/dbe_geo')

last_uploaded_inference_dscovr_dbn, last_uploaded_inference_dscovr_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_1/dscovr_predictions/dbn_geo')
last_uploaded_inference_dscovr_dbe, last_uploaded_inference_dscovr_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_1/dscovr_predictions/dbe_geo')

########################################################################################## Stage 1 ##############################################################################################

########################################################################################## Stage 2 ##############################################################################################
'''
last_uploaded_inference_ace_dbn, last_uploaded_inference_ace_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_2/ace_predictions/dbn_geo')
last_uploaded_inference_ace_dbe, last_uploaded_inference_ace_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_2/ace_predictions/dbe_geo')

last_uploaded_inference_dscovr_dbn, last_uploaded_inference_dscovr_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_2/dscovr_predictions/dbn_geo')
last_uploaded_inference_dscovr_dbe, last_uploaded_inference_dscovr_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_2/dscovr_predictions/dbe_geo')
'''
########################################################################################## Stage 2 ##############################################################################################

########################################################################################## Stage 3 ##############################################################################################
'''
last_uploaded_inference_ace_dbn, last_uploaded_inference_ace_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_3/ace_predictions/dbn_geo')
last_uploaded_inference_ace_dbe, last_uploaded_inference_ace_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_3/ace_predictions/dbe_geo')

last_uploaded_inference_dscovr_dbn, last_uploaded_inference_dscovr_dbn_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_3/dscovr_predictions/dbn_geo')
last_uploaded_inference_dscovr_dbe, last_uploaded_inference_dscovr_dbe_time = get_last_uploaded_object('hl-geo', 'geocloak2024', 'inference_outputs/DAGGER/stage_3/dscovr_predictions/dbe_geo')
'''
########################################################################################## Stage 3 ##############################################################################################

'''
csv_files = {
    last_uploaded_inference_dbn: "data_Bn.npy",
    last_uploaded_inference_dbe: "data_Be.npy",
    last_uploaded_inference_dbz: "data_Bz.npy",
}
'''

csv_files = {
    last_uploaded_inference_ace_dbn: "data_Bn.npy",
    last_uploaded_inference_ace_dbe: "data_Be.npy",
    # last_uploaded_inference_dbz: "data_Bz.npy",
}

last_uploaded_inference_lat_lon, last_uploaded_inference_lat_lon = get_last_uploaded_object('hl-geo', 'india-jackson-1', 'formatted_data/SuperMAG/supermag_processed')

lat_lon_csv = "formatted_data/SuperMAG/supermag_processed/stns.csv"
npy_data = {}
nan_mask = []
# Download and convert each CSV file
for csv_file, npy_var in csv_files.items():
    df = download_csv_from_gcs('geocloak2024', csv_file)
    # mask nan rows, storing the indices
    nan_mask.append(df.isnull().any(axis=1))
    npy_data[npy_var] = df_to_npy(df)

nan_mask = np.any(nan_mask, axis=0)

# Assign the NumPy arrays to variables
data_Bn = npy_data["data_Bn.npy"]
data_Bn = data_Bn[~nan_mask]
data_Be = npy_data["data_Be.npy"]
data_Be = data_Be[~nan_mask]

'''
data_Bz = npy_data["data_Bz.npy"]
data_Bz = data_Bz[~nan_mask]
'''
lat_lon_df = download_csv_from_gcs('geocloak2024', lat_lon_csv)
lat_lon_df.columns = lat_lon_df.iloc[0]
geo_lat, geo_lon = split_lat_lon_df(lat_lon_df)
geo_lat = geo_lat[~nan_mask]
geo_lon = geo_lon[~nan_mask]

# Flatten the arrays to 1D
data_Bn_flat = data_Bn.flatten()
data_Be_flat = data_Be.flatten()
geo_lat_flat = geo_lat.flatten()
geo_lon_flat = geo_lon.flatten()

# Create the DataFrame
df_flat = pd.DataFrame({
    'Bn': data_Bn_flat,
    'Be': data_Be_flat,
    'Latitude': geo_lat_flat,
    'Longitude': geo_lon_flat
})

# Print the DataFrame
current_datetime = get_current_datetime()
df_flat.to_csv(f'inputs_{current_datetime}.csv')

########################################################################################## Upload New Input Data ##############################################################################################
upload_to_gcloud('hl-geo', 'india-jackson-1', f'inputs_{current_datetime}.csv', f'uncertainty_vm_test/inputs/inputs_{current_datetime}.csv')
########################################################################################## Upload New Input Data ##############################################################################################

# set up constants for SECs
R_earth = 6371  # in km
R_ionosphere = R_earth + 100  # in km

# setup the SECs "node" grid
# n_lon and n_lat are free parameters but are limited to n_lon*n_lat ~ number of stations
n_lon, n_lat = 35, 35
secs_lat_lon_r, lat_sec, lon_sec = get_mesh(n_lon=n_lon, n_lat=n_lat, radius=R_ionosphere)

# setup the SuperMAG stations grid
obs_lat_lon_r = np.vstack((geo_lat, geo_lon, R_earth * np.ones(len(geo_lon)))).T

# observations in a vector
# B_obs = np.append(np.append(data_Bn, data_Be), data_Bz)
B_obs = np.append(data_Bn, data_Be)
B_obs = np.reshape(B_obs, (len(B_obs), 1))

# get T matrix for SECs
# T_mat = T_df(obs_loc=obs_lat_lon_r, sec_loc=secs_lat_lon_r)
T_mat = T_df(obs_loc=obs_lat_lon_r, sec_loc=secs_lat_lon_r, include_Bz=False)

# setup GP kernel + its hyperparameters
kernel = GPy.kern.Linear(input_dim=np.shape(T_mat)[1], variances=1)

# create simple GP model
model = GPy.models.GPRegression(T_mat, B_obs, kernel)
# optimize GP hyperparameters
model.optimize(messages=True)

# predicted grid
n_lat, n_lon = 100, 200
pred_lat_lon_r, pred_lat, pred_lon = get_mesh(n_lon=n_lon, n_lat=n_lat, radius=R_earth,
                                              lat_max=80, lat_min=-80, endpoint_lon=True)
# predict via GP
mean_, sd_ = model.predict(Xnew=T_df(obs_loc=pred_lat_lon_r, sec_loc=secs_lat_lon_r))

# Example usage assuming you have pred_lon, pred_lat, mean_, sd_ available
save_predictions(pred_lon, pred_lat, mean_, sd_, n_lat, n_lon, f"predictions_{current_datetime}.csv")
########################################################################################## Upload New Predictions Data ##############################################################################################
upload_to_gcloud('hl-geo', 'india-jackson-1', f'predictions_{current_datetime}.csv', f'uncertainty_vm_test/predictions/predictions_{current_datetime}.csv')
########################################################################################## Upload New Predictions Data ##############################################################################################

# plot results
plt.style.use('dark_background')
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(pred_lon[:, :, 0], pred_lat[:, :, 0], np.reshape(mean_[:n_lat * n_lon], (n_lat, n_lon), "C"),
                  alpha=0.6,
                  transform=ccrs.PlateCarree())

ax.scatter(geo_lon, geo_lat, c=data_Bn, vmin=np.min(mean_[:n_lat * n_lon]), vmax=np.max(mean_[:n_lat * n_lon]),
           s=10, cmap='viridis', transform=ccrs.PlateCarree())
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("mean $B_{n}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig(f'secgp_mean_Bn_{current_datetime}.png', bbox_inches='tight', dpi=72)
########################################################################################## Upload New Bn Mean Graph ##############################################################################################
upload_to_gcloud('hl-geo', 'india-jackson-1', f'secgp_mean_Bn_{current_datetime}.png', f'uncertainty_vm_test/graphs/dbn_geo_mean/secgp_mean_Bn_{current_datetime}.png')
########################################################################################## Upload New Bn Mean Graph ##############################################################################################

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(pred_lon[:, :, 0], pred_lat[:, :, 0],
                  np.reshape(mean_[n_lat * n_lon:2 * n_lat * n_lon], (n_lat, n_lon), "C"),
                  alpha=0.6,
                  transform=ccrs.PlateCarree())
ax.scatter(geo_lon, geo_lat, c=data_Be, vmin=np.min(mean_[n_lat * n_lon:2 * n_lat * n_lon]),
           vmax=np.max(mean_[n_lat * n_lon:2 * n_lat * n_lon]), s=10, cmap='viridis', transform=ccrs.PlateCarree())
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("mean $B_{e}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig(f"secgp_mean_Be_{current_datetime}.png", bbox_inches='tight', dpi=72)
########################################################################################## Upload New Be Mean Graph ##############################################################################################
upload_to_gcloud('hl-geo', 'india-jackson-1', f'secgp_mean_Be_{current_datetime}.png', f'uncertainty_vm_test/graphs/dbe_geo_mean/secgp_mean_Be_{current_datetime}.png')
########################################################################################## Upload New Bn Mean Graph ##############################################################################################

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(pred_lon[:, :, 0], pred_lat[:, :, 0], np.reshape(np.sqrt(sd_)[:n_lat * n_lon], (n_lat, n_lon), "C"),
                  alpha=0.6,
                  cmap="plasma",
                  transform=ccrs.PlateCarree())

ax.scatter(lon_sec[:, :, 0], lat_sec[:, :, 0], c="blue", s=7, marker="x")
ax.scatter(geo_lon, geo_lat, c="red", s=7)
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("standard deviation $B_{n}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig(f"secgp_sd_Bn_{current_datetime}.png", bbox_inches='tight', dpi=72)
########################################################################################## Upload New Bn Sd Graph ##############################################################################################
upload_to_gcloud('hl-geo', 'india-jackson-1', f'secgp_sd_Bn_{current_datetime}.png', f'uncertainty_vm_test/graphs/dbn_geo_sd/secgp_sd_Bn_{current_datetime}.png')
########################################################################################## Upload New Bn Sd Graph ##############################################################################################

# plot results
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
pos = ax.contourf(pred_lon[:, :, 0], pred_lat[:, :, 0],
                  np.reshape(np.sqrt(sd_)[n_lat * n_lon:2 * n_lat * n_lon], (n_lat, n_lon), "C"),
                  alpha=0.6,
                  cmap="plasma",
                  transform=ccrs.PlateCarree())

ax.scatter(lon_sec[:, :, 0], lat_sec[:, :, 0], c="blue", s=7, marker="x")
ax.scatter(geo_lon, geo_lat, c="red", s=7)
cbar = fig.colorbar(pos)
cbar.ax.set_ylabel("standard deviation $B_{e}$ [nT]", rotation=90)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-80, -40, 0, 40, 80])
ax.set_ylim(-80, 80)
ax.set_xlabel("longitude [deg]")
ax.set_ylabel("latitude [deg]")
plt.tight_layout()
plt.savefig(f"secgp_sd_Be_{current_datetime}.png", bbox_inches='tight', dpi=72)
########################################################################################## Upload New Be Sd Graph ##############################################################################################
upload_to_gcloud('hl-geo', 'india-jackson-1', f'secgp_sd_Be_{current_datetime}.png', f'uncertainty_vm_test/graphs/dbe_geo_sd/secgp_sd_Be_{current_datetime}.png')
########################################################################################## Upload New Be Sd Graph ##############################################################################################


