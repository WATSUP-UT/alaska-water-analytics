# -*- coding: utf-8 -*-
"""
Unalakleet Raw Data Sample Plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import bmondata
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. SETUP OUTPUT DIRECTORY
# ==========================================

output_folder = 'Raw_Data_Sample_Outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# ==========================================
# 2. DATA ACQUISITION & PROCESSING
# ==========================================

print("Fetching and processing data...")
server = bmondata.Server('https://anthc.bmon.org', store_key='temporary-key')

sensors = [
    ('70B3D5CD0002060F_Pressure', 'WST Height, ft'), 
    ('70B3D5CD0002063D_Pressure', 'Distribution System Pressure, psi'), 
    ('A81758FFFE067721_pulseAbs', 'Master Meter Flow Rate, GPM'), 
    ('A81758FFFE072F64_pulseAbs', 'Filtered Water Flow Rate, GPM')
]

df = server.sensor_readings(
    sensors,
    start_ts='2024-07-01 00:45 am',
    end_ts='2024-10-31 11:59 pm', averaging='30min'
)

df.reset_index(inplace=True)
df = df.rename(columns={'index': 'Timestamp'})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# SI Unit Conversions
df['Master Meter Flow Rate, m3/hr'] = df['Master Meter Flow Rate, GPM'] * 0.227124
df['WST Height, m'] = df['WST Height, ft'] * 0.3048
df['Distribution System Pressure, kPa'] = df['Distribution System Pressure, psi'] * 6.89475729
df['Filtered Water Flow Rate, m3/hr'] = df['Filtered Water Flow Rate, GPM'] * 0.227124

# Month column for potential labeling
df['month'] = pd.to_datetime(df['Timestamp']).dt.strftime('%B')
df['month'] = df['month'].astype(str)

# ==========================================
# 3. PLOTTING
# ==========================================

print("Generating plot...")

# Stacked plots configuration
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(15, 20))
plt.tight_layout(pad=0.5)

# 1. Supply Flow
ax1.set_ylabel('Supply Flow,' + "\n" + r'$m^3$' + '/hr', size=40)
plot_1 = ax1.plot(df['Timestamp'], df['Master Meter Flow Rate, m3/hr'], '-', label='Supply Flow, m3/hr', color='#00a9b7')
ax1.tick_params(axis='y', labelcolor='black', labelsize=30)

# 2. Tank Level
ax2.set_ylabel('Tank Level, m', size=40)
plot_2 = ax2.plot(df['Timestamp'], df['WST Height, m'], '-', label='WST Level, m', color='#f8971f')
ax2.tick_params(axis='y', labelcolor='black', labelsize=30)

# 3. Pressure
ax3.set_ylabel('Pressure, kPa', size=40)
plot_3 = ax3.plot(df['Timestamp'], df['Distribution System Pressure, kPa'], '-', label='Distribution System Pressure, kPa', color='#9cadb7')
ax3.tick_params(axis='y', labelcolor='black', labelsize=30)

# 4. Treated Flow
ax4.set_ylabel('Treated Flow,' + "\n" + r'$m^3$' + '/hr', size=40)
plot_4 = ax4.plot(df['Timestamp'], df['Filtered Water Flow Rate, m3/hr'], '-', label='Treated Flow, m3/hr', color='#bf5700')
ax4.tick_params(axis='y', labelcolor='black', labelsize=30)

# Axis Formatting
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax4.tick_params(axis='x', labelrotation=0, labelsize=30)
ax4.set_xlabel('2024', size=40)
ax4.xaxis.set_major_locator(MaxNLocator(nbins=5))

# Save Plot
plot_filename = os.path.join(output_folder, 'Raw_Data_Sample.png')
plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
plt.close()
print(f"Plot saved to: {plot_filename}")

# ==========================================
# 4. DATA EXPORT
# ==========================================

# Create a DataFrame with only the columns that are plotted
plot_data = df[['Timestamp', 
                'Master Meter Flow Rate, m3/hr',
                'WST Height, m', 
                'Distribution System Pressure, kPa',
                'Filtered Water Flow Rate, m3/hr']].copy()

# Rename columns to match the plot labels exactly
plot_data = plot_data.rename(columns={
    'Master Meter Flow Rate, m3/hr': 'Supply Flow, m3/hr',
    'WST Height, m': 'Tank Level, m',
    'Distribution System Pressure, kPa': 'Pressure, kPa', 
    'Filtered Water Flow Rate, m3/hr': 'Treated Flow, m3/hr'
})

# Save to CSV file
csv_filename = os.path.join(output_folder, 'water_system_timeseries_data.csv')
plot_data.to_csv(csv_filename, index=False)

print(f"Time series data saved to: {csv_filename}")
print(f"All outputs saved in folder: {output_folder}")