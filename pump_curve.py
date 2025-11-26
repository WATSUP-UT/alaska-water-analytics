"""
Pump Curve Analysis
Generates:
1. Raw Head vs Flow Scatter
2. Clustered Head vs Flow
3. Time Series System Pressure (Thursdays)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import matplotlib.ticker as ticker
import statistics
import numpy as np
import bmondata

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP OUTPUT DIRECTORY
# ==========================================

output_folder = 'Pump_Curve_Outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# ==========================================
# 2. DATA ACQUISITION
# ==========================================

print("Fetching data from BMON...")
server = bmondata.Server('https://anthc.bmon.org', store_key='temporary-key')

sensors = [
    ('70B3D5CD0002060F_Pressure', 'WST Height, ft'), 
    ('70B3D5CD0002063D_Pressure', 'Distribution System Pressure, psi'), 
    ('A81758FFFE067721_pulseAbs', 'Master Meter Flow Rate, GPM')
]

df = server.sensor_readings(
    sensors,
    start_ts='2022-07-23 00:45 am',
    end_ts='2024-11-19 11:59 pm', averaging='30min'
)

df.reset_index(inplace=True)
df = df.rename(columns={'index': 'Timestamp'})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# ==========================================
# 3. DATA PRE-PROCESSING & CLEANING
# ==========================================

print("Processing data and removing outliers...")

# --- Manual Offset Adjustments ---
height = 28
date = datetime(2024, 6, 25, 18, 16, 0)
close_date = min(df['Timestamp'], key=lambda d: abs(d - date))
index_end_date = df.index[df['Timestamp'] == close_date][0]

if close_date > date:
    index_end_date = index_end_date - 1 

date2 = datetime(2024, 6, 26, 11, 17, 0)
close_date2 = min(df['Timestamp'], key=lambda d: abs(d - date2))
index_start_date = df.index[df['Timestamp'] == close_date2][0]

if close_date2 > date2:
    index_start_date = index_start_date - 1 

df.loc[0:index_end_date-1, 'WST Height, ft'] += 0.77
df.loc[index_end_date+1:index_start_date-1, 'WST Height, ft'] += 0.58
df.loc[index_end_date, 'WST Height, ft'] = (df.loc[index_end_date-1, 'WST Height, ft'] + df.loc[index_end_date+1, 'WST Height, ft']) / 2
df.loc[index_start_date, 'WST Height, ft'] = (df.loc[index_start_date-1, 'WST Height, ft'] + df.loc[index_start_date+1, 'WST Height, ft']) / 2

# --- Outlier Removal ---

# WST Height
df = df[(df['WST Height, ft'] > 0) & (df['WST Height, ft'] < height)]

# Pressure
df = df[df['Distribution System Pressure, psi'] > 0]
std_p = statistics.stdev(df['Distribution System Pressure, psi'])
median_p = statistics.median(df['Distribution System Pressure, psi'])
df_ub_p = median_p + 2 * std_p 
df_lb_p = median_p - 2 * std_p 
df = df[(df['Distribution System Pressure, psi'] > df_lb_p) & (df['Distribution System Pressure, psi'] < df_ub_p)]

# Flow Rate
df = df[df['Master Meter Flow Rate, GPM'] > 0]
std_f = statistics.stdev(df['Master Meter Flow Rate, GPM'])
median_f = statistics.median(df['Master Meter Flow Rate, GPM'])
df_ub_f = median_f + 2 * std_f 
df_lb_f = median_f - 2 * std_f 
df = df[(df['Master Meter Flow Rate, GPM'] > df_lb_f) & (df['Master Meter Flow Rate, GPM'] < df_ub_f)]

# --- Calculations ---
df['pressure_head_ft'] = df['Distribution System Pressure, psi'] / 0.433
df['head_diff_ft'] = df['pressure_head_ft'] - df['WST Height, ft']

# Clean NA
df_filt = df.dropna(how='any').reset_index(drop=True)

# SI Unit Conversion
df_filt['head_diff_m'] = df_filt['head_diff_ft'] * 0.3048
df_filt['Master Meter Flow Rate_m3hr'] = df_filt['Master Meter Flow Rate, GPM'] * 0.227124

# ==========================================
# 4. CURVE FITTING & CLUSTERING (SI UNITS)
# ==========================================

print("Performing curve fitting and clustering...")

# Initial Fit Logic based on IQR
df_IQR = df_filt[(df_filt['head_diff_m'] > 52*0.3) & (df_filt['head_diff_m'] < 70*0.3)]
x_IQR = df_IQR['Master Meter Flow Rate_m3hr']
y_IQR = df_IQR['head_diff_m']

# Base Polynomial Fit
a, b, c1 = np.polyfit(x_IQR, y_IQR, 2)

# Range Definitions
flows = np.arange(5, 20, .5)
speed = np.arange(0.51, 1.21, 0.04)

# Calculate Base Curve
y_est_flows = a * flows**2 + b * flows + c1

# Generate Ensemble Curves
ensemble_data = []
for i in speed: 
    # Affine Laws
    est_flows = flows * i
    est_head = i**2 * y_est_flows
    
    # Fit new curve for this speed
    a_new, b_new, c_new = np.polyfit(est_flows, est_head, 2)
    ensemble_data.append({'speed': i, 'a': a_new, 'b': b_new, 'c': c_new})

df_fit = pd.DataFrame(ensemble_data)

# --- Vectorized Clustering (Faster than looping) ---
# 1. Extract coefficients
coeffs = df_fit[['a', 'b', 'c']].values  # Shape: (n_curves, 3)

# 2. Extract Data
x_data = df_filt['Master Meter Flow Rate_m3hr'].values
y_data = df_filt['head_diff_m'].values

# 3. Calculate predicted Y for every point against every curve
# Shape x_data: (n_points, 1) to broadcast against (n_curves, )
# Result: (n_points, n_curves) matrix of errors
x_matrix = x_data[:, np.newaxis]
y_est_matrix = coeffs[:, 0] * x_matrix**2 + coeffs[:, 1] * x_matrix + coeffs[:, 2]
error_matrix = np.abs(y_est_matrix - y_data[:, np.newaxis])

# 4. Find index of minimum error for each point
df_filt['cluster'] = np.argmin(error_matrix, axis=1)

# ==========================================
# 5. PLOTTING
# ==========================================

print("Generating plots...")

# Styling Constants
unique_colors = ['#00a9b7', '#f8971f', '#9cadb7', '#bf5700'] * 5 
unique_shapes = ['o', 'o', 'o', 'o', 's', 's', 's', 's', '*', '*', '*', '*', 'v', 'v', 'v', 'v', 'X', 'X', 'X', 'X']

# --- Plot 1: Raw Scatter (Head vs Flow) ---
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlabel('Flow Rate [' + r'$m^3$' + '/hr]', fontsize=40)
ax.set_ylabel('Head [m]', fontsize=40)
ax.xaxis.set_major_locator(ticker.MaxNLocator(12, integer=True))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.plot(df_filt['Master Meter Flow Rate_m3hr'], df_filt['head_diff_m'], 'o', color='#f8971f')

save_path1 = os.path.join(output_folder, '1_Raw_Head_vs_Flow.png')
plt.savefig(save_path1, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {save_path1}")


# --- Plot 2: Clustered Scatter (Head vs Flow) ---
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlabel('Flow Rate [' + r'$m^3$' + '/hr]', fontsize=40)
ax.set_ylabel('Head [m]', fontsize=40)
ax.xaxis.set_major_locator(ticker.MaxNLocator(12, integer=True))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

group_label = df_filt.groupby('cluster')

for grlbl, df3 in group_label:
    # Ensure we don't go out of bounds on style lists
    c_idx = int(grlbl) % len(unique_colors)
    s_idx = int(grlbl) % len(unique_shapes)
    
    plt.plot(df3['Master Meter Flow Rate_m3hr'], 
             df3['head_diff_m'], 
             marker=unique_shapes[s_idx], 
             linestyle='None', 
             label=grlbl, 
             color=unique_colors[c_idx])

save_path2 = os.path.join(output_folder, '2_Clustered_Head_vs_Flow.png')
plt.savefig(save_path2, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {save_path2}")


# --- Plot 3: Time Series (System Pressure on Thursdays) ---
# Filter for Time
df_filt['time_str'] = df_filt['Timestamp'].dt.time.astype(str)
df_day = df_filt[df_filt['time_str'] <= '00:15:00'].copy()

# Filter for Day (Thursday)
# Note: Python's weekday(): Monday=0, Thursday=3
df_day['day_of_week'] = df_day['Timestamp'].dt.weekday
df_day = df_day[df_day['day_of_week'] == 3].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_ylabel('System Pressure [m]', fontsize=40)
ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
plt.xticks(rotation=0, fontsize=30)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
plt.yticks(fontsize=30)

# Iterate through rows for specific markers (Time series scatter)
for index, row in df_day.iterrows():
    grlbl = int(row['cluster'])
    c_idx = grlbl % len(unique_colors)
    s_idx = grlbl % len(unique_shapes)
    
    # Calculate pressure in meters
    pressure_m = row['Distribution System Pressure, psi'] * 0.70324961490205
    
    plt.plot(row['Timestamp'], 
             pressure_m, 
             marker=unique_shapes[s_idx], 
             ms=30, 
             color=unique_colors[c_idx])

save_path3 = os.path.join(output_folder, '3_System_Pressure_Time_Series.png')
plt.savefig(save_path3, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {save_path3}")

print(f"All plots saved in folder: {output_folder}")