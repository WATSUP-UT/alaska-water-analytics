"""
Water Demand Pattern Analysis (BMON Data Source)
------------------------------------------------
Fetches 30-minute interval flow data directly from the BMON server 
and generates diurnal demand curves segregated by day of the week.

Generates:
1. Median Diurnal Demand Profile (Combined Week)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
import bmondata

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# -- Server & Sensor Config --
BMON_URL = 'https://anthc.bmon.org'
STORE_KEY = 'temporary-key'

# List of sensors to fetch: (ID, Label)
SENSORS = [
    ('A81758FFFE067721_pulseAbs', 'Master Meter Flow Rate, GPM')
]

# -- Time Range --
START_DATE = '2022-07-23 00:00 am'
END_DATE = '2024-11-19 11:59 pm'
AVERAGING_INTERVAL = '30min'

# -- Output Config --
OUTPUT_FOLDER = 'System_Demand_Outputs'
OUTPUT_FILENAME = 'system_demand.png'

# -- Unit Conversion --
GPM_TO_M3HR = 0.227124707

# -- Plotting Config --
COLORS = {
    "Monday": "#00a9b7",      # Cyan-ish
    "Tuesday": "#f8971f",     # Orange
    "Wednesday": "#9cadb7",   # Grey-Blue
    "Thursday": "#bf5700",    # Dark Orange
    "Friday": "purple",       # Purple
    "Saturday": "brown",      # Brown
    "Sunday": "#ff69b4"       # Pink
}

# Explicit order for plotting
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Ensure output directory exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created directory: {OUTPUT_FOLDER}")

# ==========================================
# 2. DATA ACQUISITION & PROCESSING
# ==========================================

def fetch_and_process_data():
    """
    Fetches data from BMON, processes timestamps, and calculates median profiles.
    """
    print("Fetching data from BMON...")
    server = bmondata.Server(BMON_URL, store_key=STORE_KEY)
    
    # Fetch sensor readings
    df = server.sensor_readings(
        SENSORS,
        start_ts=START_DATE,
        end_ts=END_DATE,
        averaging=AVERAGING_INTERVAL
    )
    
    # Standardize Index/Columns
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Timestamp'})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sensor Column Name 
    flow_col = SENSORS[0][1] 
    
    # Ensure numeric (handle non-numeric errors, but keep values)
    df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
    print("Processing data (Calculating SI units)...")
    
    # Create derived columns for grouping
    df['day_of_week'] = df['Timestamp'].dt.day_name()
    
    # Create 'fractional_hour' (e.g., 08:30 -> 8.5)
    df['fractional_hour'] = df['Timestamp'].dt.hour + (df['Timestamp'].dt.minute / 60.0)
    
    # Conversion: GPM -> m3/hr
    df['flow_m3hr'] = df[flow_col] * GPM_TO_M3HR
    
    # Set categorical order for days
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=DAYS_ORDER, ordered=True)
    
    return df

def calculate_median_profiles(df):
    """
    Groups data by Fractional Hour and Day of Week to find the median flow.
    """
    print("Calculating median demand profiles...")
    
    grouped = df.groupby(['day_of_week', 'fractional_hour'], observed=True)['flow_m3hr'].median().reset_index()
    
    return grouped

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_combined_demand(df_grouped):
    """
    Plots the superimposed diurnal curves for all days of the week.
    """
    print("Generating Combined Demand Plot...")
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Loop through days in order
    for day in DAYS_ORDER:
        day_data = df_grouped[df_grouped['day_of_week'] == day]
        
        if day_data.empty:
            continue
            
        ax.plot(day_data['fractional_hour'], 
                day_data['flow_m3hr'], 
                marker='.', 
                linestyle='-', 
                linewidth=2,
                markersize=8,
                label=day, 
                color=COLORS.get(day, 'black'),
                alpha=0.8)

    # Formatting Axes
    ax.set_xlabel('Time of Day [Hour]', fontsize=30, labelpad=15)
    ax.set_ylabel(r'Demand [$m^3/hr$]', fontsize=30, labelpad=15)
    

    ax.set_xlim(-0.5, 25)
    
    # Ticks every 4 hours
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    # Horizontal ticks
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    
    # Legend
    ax.legend(fontsize=18, loc='upper left', frameon=True, framealpha=0.9)
    
    # Grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Save
    save_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to: {save_path}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        # 1. Fetch & Process 
        df_raw = fetch_and_process_data()
        
        # 2. Calculate Statistics
        df_profiles = calculate_median_profiles(df_raw)
        
        # 3. Plot
        plot_combined_demand(df_profiles)
        
        print("Analysis complete.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please ensure you have 'bmondata' installed and the server URL is accessible.")