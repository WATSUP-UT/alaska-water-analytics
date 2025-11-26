import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import statistics
from math import pi
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import bmondata

# Suppress pandas chained assignment warnings
pd.options.mode.chained_assignment = None

# ==========================================
# 0. SETUP OUTPUT DIRECTORY
# ==========================================

output_folder = 'Backwash_Outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# ==========================================
# 1. DATA ACQUISITION & PRE-PROCESSING
# ==========================================

print("Fetching and processing data...")
server = bmondata.Server('https://anthc.bmon.org', store_key='temporary-key')

# --- Initial Outlier Bound Calculations (Training Period) ---
# WST Height
sensors_WST = [('70B3D5CD0002060F_Pressure', 'WST Height, ft')]
df_temp = server.sensor_readings(sensors_WST, start_ts='2022-7-1 12:00 am', end_ts='2024-10-22 11:59 pm', averaging='30min')
df_temp.dropna(inplace=True)
std_WST = statistics.stdev(df_temp['WST Height, ft'])
median_WST = statistics.median(df_temp['WST Height, ft'])
df_ub_WST = median_WST + 2 * std_WST
df_lb_WST = median_WST - 2 * std_WST

# Pressure
sensors_P = [('70B3D5CD0002063D_Pressure', 'Distribution System Pressure, psi')]
df_temp = server.sensor_readings(sensors_P, start_ts='2022-7-1 12:00 am', end_ts='2024-10-22 11:59 pm', averaging='30min')
df_temp.dropna(inplace=True)
std_P = statistics.stdev(df_temp['Distribution System Pressure, psi'])
median_P = statistics.median(df_temp['Distribution System Pressure, psi'])
df_ub_P = median_P + 2 * std_P
df_lb_P = median_P - 2 * std_P

# Master Meter Flow
sensors_MM = [('A81758FFFE067721_pulseAbs', 'Master Meter Flow Rate, GPM')]
df_temp = server.sensor_readings(sensors_MM, start_ts='2022-7-1 12:00 am', end_ts='2024-10-22 11:59 pm', averaging='30min')
df_temp.dropna(inplace=True)
std_MM = statistics.stdev(df_temp['Master Meter Flow Rate, GPM'])
median_MM = statistics.median(df_temp['Master Meter Flow Rate, GPM'])
df_ub_MM = median_MM + 2 * std_MM
df_lb_MM = median_MM - 2 * std_MM

# Filtered Water Flow
sensors_FF = [('A81758FFFE072F64_pulseAbs', 'Filtered Water Flow Rate, GPM')]
df_temp = server.sensor_readings(sensors_FF, start_ts='2022-7-1 12:00 am', end_ts='2024-10-22 11:59 pm', averaging='30min')
df_temp.dropna(inplace=True)
std_FF = statistics.stdev(df_temp['Filtered Water Flow Rate, GPM'])
median_FF = statistics.median(df_temp['Filtered Water Flow Rate, GPM'])
df_ub_FF = median_FF + 2 * std_FF
df_lb_FF = median_FF - 2 * std_FF

# --- Main Analysis Data Fetch ---
sensors = [('70B3D5CD0002060F_Pressure', 'WST Height, ft'), 
           ('70B3D5CD0002063D_Pressure', 'Distribution System Pressure, psi'),
           ('A81758FFFE067721_pulseAbs', 'Master Meter Flow Rate, GPM'), 
           ('A81758FFFE072F64_pulseAbs', 'Filtered Water Flow Rate, GPM')]

df = server.sensor_readings(
    sensors,
    start_ts='2024-7-1 12:00 am',
    end_ts='2024-10-22 11:59 pm', averaging='15min'
)

df.reset_index(inplace=True)
df = df.rename(columns={'index': 'Timestamp'})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

raw_df = df.copy()

# --- Iterative Outlier Replacement ---
height = 28
for index, row in df.iterrows():
    if index == 0: 
        continue    
    # WST Logic
    if (df.loc[index, 'WST Height, ft'] < 0 or 
        df.loc[index, 'WST Height, ft'] > height or 
        df.loc[index, 'WST Height, ft'] < df_lb_WST or 
        df.loc[index, 'WST Height, ft'] > df_ub_WST):
        df.loc[index, 'WST Height, ft'] = df.loc[index-1, 'WST Height, ft']
    
    # Pressure Logic
    if (df.loc[index, 'Distribution System Pressure, psi'] < 0 or 
        df.loc[index, 'Distribution System Pressure, psi'] < df_lb_P or 
        df.loc[index, 'Distribution System Pressure, psi'] > df_ub_P):
        df.loc[index, 'Distribution System Pressure, psi'] = df.loc[index-1, 'Distribution System Pressure, psi']
    
    # Master Meter Logic
    if (df.loc[index, 'Master Meter Flow Rate, GPM'] < 0 or 
        df.loc[index, 'Master Meter Flow Rate, GPM'] < df_lb_MM or 
        df.loc[index, 'Master Meter Flow Rate, GPM'] > df_ub_MM):
        df.loc[index, 'Master Meter Flow Rate, GPM'] = df.loc[index-1, 'Master Meter Flow Rate, GPM']
    
    # Filtered Flow Logic
    if (df.loc[index, 'Filtered Water Flow Rate, GPM'] < 0 or 
        df.loc[index, 'Filtered Water Flow Rate, GPM'] < df_lb_FF or 
        df.loc[index, 'Filtered Water Flow Rate, GPM'] > df_ub_FF):
        df.loc[index, 'Filtered Water Flow Rate, GPM'] = 0

# Baseline stats
df_nobw = raw_df.copy()   
df_nobw['diff'] = ''
for index, row in df_nobw.iterrows():
    if index == 0:
        df_nobw.loc[index, 'diff'] = 0
    else:
        df_nobw.loc[index, 'diff'] = df_nobw.loc[index, 'WST Height, ft'] - df_nobw.loc[index-1, 'WST Height, ft']

df_nobw = df_nobw[df_nobw['Filtered Water Flow Rate, GPM'] > 30] 
diff_mean_nobw = statistics.mean(df_nobw['diff'])

# Check for missing data
START_DATE = datetime(2024, 7, 1, 00, 15, 0)
END_DATE = datetime(2024, 10, 22, 23, 59, 0)
timerange = pd.date_range(start=START_DATE, end=END_DATE, freq='15min')
dates = timerange.to_frame(index=False).rename(columns={0: 'Timestamp2'})
data_check = pd.concat([dates, df], axis=1)
df = df.dropna().reset_index(drop=True)
mb = df.copy()

# ==========================================
# 2. CUSUM ALGORITHM
# ==========================================

mb['diff'] = ''
for index, row in mb.iterrows():
    if index == 0:
        mb.loc[index, 'diff'] = 0
    else:
        mb.loc[index, 'diff'] = mb.loc[index, 'WST Height, ft'] - mb.loc[index-1, 'WST Height, ft']

diff_std = statistics.stdev(mb['diff'])
diff_mean = statistics.mean(mb['diff'])

k = 0.1/2 
mb['xi-u'] = mb['diff'] - diff_mean_nobw
mb['C'] = ''
for index, row in mb.iterrows():
    if index == 0:
        mb.loc[index, 'C'] = mb.loc[index, 'xi-u']
    else:
        mb.loc[index, 'C'] = mb.loc[index-1, 'C'] + mb.loc[index, 'xi-u']

mb['C+'] = ''
mb['C-'] = ''

for index, row in mb.iterrows():
    if index == 0:
        mb.loc[index, 'C+'] = max(0, (row['diff'] - diff_mean_nobw - k + 0))
        mb.loc[index, 'C-'] = max(0, (diff_mean_nobw - k - row['diff'] + 0))
    else:
        mb.loc[index, 'C+'] = max(0, (row['diff'] - diff_mean_nobw - k + mb.loc[index-1, 'C+']))
        mb.loc[index, 'C-'] = max(0, (diff_mean_nobw - k - row['diff'] + mb.loc[index-1, 'C-']))

mb['cusum'] = ''
mb['nC-'] = ''
countminus = 1
for index, row in mb.iterrows():
    if mb.loc[index, 'C-'] > 0:
        mb.loc[index, 'nC-'] = countminus
        mb.loc[index, 'cusum'] = countminus
        countminus = countminus + 1
    else:
        countminus = 1
        mb.loc[index, 'nC-'] = 0
        mb.loc[index, 'cusum'] = 0

# ==========================================
# 3. MASS BALANCE CALCULATION
# ==========================================

# Variables
radius = 39  # ft
ft3_gal = 7.481  # ft3 to gallons conversion
area = radius * radius * pi * ft3_gal

mb['Vol(t)_WSTH_c'] = ''
mb['Vol(t)_mb_c'] = ''
mb['error_c'] = ''
mb['demand_c'] = ''
mb['Vbw_c'] = ''

index = 0
while index < len(mb)-1:
    for row in mb:
        if index == 0:
            mb.loc[index, 'Vol(t)_WSTH_c'] = area * mb.loc[index, 'WST Height, ft']
            index = index + 1
            continue
        if index == len(mb)-1:
            break
        
        if mb.loc[index, 'cusum'] >= 1:  # backwash!
            prev_index = index-1
            tankdrop = abs(mb.loc[index-1, 'WST Height, ft'] - mb.loc[index, 'WST Height, ft'])
            Vbw = (area * tankdrop) - (15 * 0.5 * (mb.loc[index, 'Master Meter Flow Rate, GPM'] + mb.loc[prev_index, 'Master Meter Flow Rate, GPM']))
            mb.loc[index, 'Vol(t)_WSTH_c'] = area * mb.loc[index, 'WST Height, ft']
            Vin_gal = 15 * 0.5 * (mb.loc[index, 'Filtered Water Flow Rate, GPM'] + mb.loc[prev_index, 'Filtered Water Flow Rate, GPM'])
            Vout_gal = 15 * 0.5 * (mb.loc[index, 'Master Meter Flow Rate, GPM'] + mb.loc[prev_index, 'Master Meter Flow Rate, GPM'])
            mb.loc[index, 'demand_c'] = Vout_gal
            mb.loc[index, 'Vol(t)_mb_c'] = mb.loc[prev_index, 'Vol(t)_WSTH_c'] + Vin_gal - Vout_gal - Vbw
            mb.loc[index, 'error_c'] = mb.loc[index, 'Vol(t)_mb_c'] - mb.loc[index, 'Vol(t)_WSTH_c']
            mb.loc[index, 'Vbw_c'] = Vbw
            index = index + 1
        else:  # NOT BACKWASH!
            Vbw = 0
            prev_index = index-1
            mb.loc[index, 'Vol(t)_WSTH_c'] = area * mb.loc[index, 'WST Height, ft']
            Vin_gal = 15 * 0.5 * (mb.loc[index, 'Filtered Water Flow Rate, GPM'] + mb.loc[prev_index, 'Filtered Water Flow Rate, GPM'])
            Vout_gal = 15 * 0.5 * (mb.loc[index, 'Master Meter Flow Rate, GPM'] + mb.loc[prev_index, 'Master Meter Flow Rate, GPM'])
            mb.loc[index, 'demand_c'] = Vout_gal
            mb.loc[index, 'Vol(t)_mb_c'] = mb.loc[prev_index, 'Vol(t)_WSTH_c'] + Vin_gal - Vout_gal - Vbw
            mb.loc[index, 'error_c'] = mb.loc[index, 'Vol(t)_mb_c'] - mb.loc[index, 'Vol(t)_WSTH_c']
            mb.loc[index, 'Vbw_c'] = 0
            index = index + 1

# ==========================================
# 4. EVENT IDENTIFICATION 
# ==========================================

n = len(mb)
bw_data = [] 
index = 0
while index < (n-1):
    if index == 0 or index == (len(mb)-1):
        index = index + 1
        continue
    if mb.loc[index, 'Filtered Water Flow Rate, GPM'] < 30 and mb.loc[index-1, 'Filtered Water Flow Rate, GPM'] > 30:
        time_start = mb.loc[index, 'Timestamp']
        while mb.loc[index, 'Filtered Water Flow Rate, GPM'] < 30:
            if index >= (n-1):
                index = index + 1
                break
            if mb.loc[index, 'cusum'] == 1:
                time_bw_start = mb.loc[index, 'Timestamp']
                Vbw_tot = mb.loc[index, 'Vbw_c']
                index = index + 1
                while mb.loc[index, 'cusum'] > 1:
                    Vbw_tot = Vbw_tot + mb.loc[index, 'cusum']
                    index = index + 1
                time_bw_end = mb.loc[index-1, 'Timestamp']
            else: 
                index = index + 1
        if index >= (n-1):
            index = index + 1
            break
        
        time_end = mb.loc[index, 'Timestamp'] 
        temp = {'time_start': time_start, 'time_bw_start': time_bw_start,'time_bw_end': time_bw_end, 'time_end': time_end, 'bw_vol': Vbw_tot}
        
        bw_data.append(temp)
        
        time_start = np.nan
        time_bw_start = np.nan
        time_bw_end = np.nan
        time_end = np.nan
        Vbw_tot = 0
    else:
        index = index + 1

# Convert list to DataFrame at the end
if bw_data:
    bw = pd.DataFrame(bw_data)
else:
    # Fallback if no events detected to match expected structure
    bw = pd.DataFrame(columns=['time_start', 'time_bw_start', 'time_bw_end', 'time_end', 'bw_vol'])

FF_zero = bw.drop(['time_bw_start', 'time_bw_end', 'bw_vol'], axis=1)
FF_zero = FF_zero.reset_index(drop=True)
bw = bw.dropna().reset_index(drop=True)

# ==========================================
# 5. DATA PREPARATION FOR PLOTTING
# ==========================================

# Convert to SI Units
bw['bw_vol_m3'] =  bw['bw_vol'] * 0.00378541

# Date handling
bw['time_start'] = pd.to_datetime(bw['time_start'])
bw['time_bw_start'] = pd.to_datetime(bw['time_bw_start'])
bw['time_bw_end'] = pd.to_datetime(bw['time_bw_end'])
bw['time_end'] = pd.to_datetime(bw['time_end'])

FF_zero['time_start'] = pd.to_datetime(FF_zero['time_start'])
FF_zero['time_end'] = pd.to_datetime(FF_zero['time_end'])

# ==========================================
# 6. PLOTTING 
# ==========================================

# Setup Figure
fig, axes = plt.subplots(figsize=(25, 10))
axes.set_ylabel('Backwash Volume [' + r'$m^3$' + ']', fontsize=40)
plt.xticks(rotation=15, fontsize=30)
plt.yticks(fontsize=30)

# Plot Phase 2 (Orange lines)
c = 0
for index, row in bw.iterrows():
    c = index    
    plt.hlines(y=bw.loc[c, 'bw_vol_m3'], xmin=bw.loc[c, 'time_bw_start'], xmax=bw.loc[c, 'time_bw_end'], color='#bf5700', linewidth=1)
    plt.vlines(ymin=0, ymax=bw.loc[c, 'bw_vol_m3'], x=bw.loc[c, 'time_bw_start'], color='#bf5700', linewidth=1.5)
    plt.vlines(ymin=0, ymax=bw.loc[c, 'bw_vol_m3'], x=bw.loc[c, 'time_bw_end'], color='#bf5700', linewidth=1.5)

# Plot Phase 1 (Grey Spans)
c = 0
for index, row in FF_zero.iterrows():    
    c = index
    axes.axvspan(xmin=FF_zero.loc[c, 'time_start'], xmax=FF_zero.loc[c, 'time_end'], color='#9cadb7', alpha=0.5)

axes.set_ylim(0) 

# Legend
orange_patch = mpatches.Patch(color='#bf5700', label='phase 2: backwash event')
grey_patch = mpatches.Patch(color='#9cadb7', label='phase 1: backwash process duration')
plt.legend(handles=[grey_patch, orange_patch], fontsize=25, frameon=False)

# Axis Formatting
axes.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
axes.tick_params(axis='x', labelrotation=0, labelsize=30)
axes.set_xlabel('2024', size=40)
axes.xaxis.set_major_locator(MaxNLocator(nbins=5))

plt.tight_layout()

# Save Plot
plot_filename = os.path.join(output_folder, 'Backwash.png')
plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
plt.close() # Ensure no display
print(f"Plot saved to: {plot_filename}")

# ==========================================
# 7. SUMMARY STATISTICS 
# ==========================================

backwash_events = bw[['time_bw_start', 'time_bw_end', 'bw_vol_m3']].copy()
process_duration = FF_zero[['time_start', 'time_end']].copy()

stats_filename = os.path.join(output_folder, 'summary_statistics.txt')

with open(stats_filename, 'w') as f:
    f.write("=== DATA SUMMARY ===\n")
    f.write(f"Number of backwash events (Phase 2): {len(backwash_events)}\n")
    f.write(f"Number of process duration periods (Phase 1): {len(process_duration)}\n")
    f.write(f"Total backwash volume: {backwash_events['bw_vol_m3'].sum():.2f} m3\n")
    f.write(f"Average backwash volume: {backwash_events['bw_vol_m3'].mean():.2f} m3\n")
    f.write(f"Date range: {bw['time_start'].min()} to {bw['time_end'].max()}\n\n")

    f.write("=== BACKWASH EVENTS DATA (All Rows) ===\n")
    f.write(backwash_events.to_string())
    f.write("\n\n")
    
    f.write("=== PROCESS DURATION DATA (All Rows) ===\n")
    f.write(process_duration.to_string())

print(f"Summary statistics saved to: {stats_filename}")
print(f"All outputs saved in folder: {output_folder}")