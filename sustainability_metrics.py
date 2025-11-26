"""
Water Tank Sustainability Metrics Analysis
------------------------------------------
Calculates Reliability (15-min), Resiliency & Vulnerability (4-hr), 
and Event Statistics (4-hr). Plots vertical bands for detected events.

Outputs (saved in 'sustainability_metrics_result/'):
1. 'Communitiy_metrics.xlsx'
2. Individual time-series plots with vertical event bands.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. Configuration & Setup
# ==========================================
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12})

OUTPUT_DIR = "sustainability_metrics_result"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

COMMUNITIES_INFO = {
    "Hydaburg":       {"sensor": ('A81758FFFE067D45_analog', 'WST Height, ft'), "height": 27.2},
    "Saints Mary":    {"sensor": ('345463', 'WST Height, ft'),                  "height": 22.7},
    "Arctic Village": {"sensor": ('389441', 'WST Height, ft'),                  "height": 22.0},
    "Tooksook Bay":   {"sensor": ('A81758FFFE067D6B_analog', 'WST Height, ft'), "height": 16.2},
    "Golovin":        {"sensor": ('187816', 'WST Height, ft'),                  "height": 14.6},
    "Fort Yukon":     {"sensor": ('54604', 'WST Height, ft'),                   "height": 19.4},
    "Kake":           {"sensor": ('A81758FFFE060522_analog', 'WST Height, ft'), "height": 41.0},
    "Kipnuk":         {"sensor": ('A81758FFFE072ED1_analog', 'WST Height, ft'), "height": 15.0},
    "Mertarvik":      {"sensor": ('491165', 'WST Height, ft'),                  "height": 12.5},
    "Unalakleet":     {"sensor": ('70B3D5CD0002060F_Pressure', 'WST Height, ft'),"height": 27.5}
}

# ==========================================
# 2. Data Handling
# ==========================================

def load_and_align_data(server, sensor_tuple, start_ts, end_ts, freq='15min'):
    """Fetch data from BMON and align to a strict time grid."""
    print(f"  Fetching data...")
    try:
        df = server.sensor_readings([sensor_tuple], start_ts=start_ts, end_ts=end_ts, averaging=freq)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Timestamp'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        print(f"  Error fetching data: {e}")
        return pd.DataFrame()

    if df.empty: return df

    # Align to perfect grid
    start_date = df['Timestamp'].min()
    end_date = df['Timestamp'].max()
    timerange = pd.date_range(start=start_date, end=end_date, freq=freq)
    dates_df = pd.DataFrame({'Timestamp': timerange})
    
    df_aligned = pd.merge(dates_df, df, on='Timestamp', how='left')
    
    # Fill gaps
    sensor_col = sensor_tuple[1]
    df_aligned[sensor_col] = df_aligned[sensor_col].ffill().bfill()
    
    return df_aligned

def apply_unalakleet_correction(df, sensor_col):
    """Correction logic."""
    date_threshold = datetime(2024, 6, 26, 11, 17, 0)
    
    close_date = min(df['Timestamp'], key=lambda d: abs(d - date_threshold))
    idx_candidates = df.index[df['Timestamp'] == close_date]
    
    if len(idx_candidates) == 0:
        return df
        
    index_date = idx_candidates[0]
    if close_date > date_threshold:
        index_date = max(0, index_date - 1)
        
    df.loc[0:index_date, sensor_col] = df.loc[0:index_date, sensor_col] - 0.39
    print("  Applied Unalakleet sensor correction.")
    return df

def clean_data_pipeline(df, sensor_col, community_name, tank_height):
    """Applies physical bounds and statistical cleaning."""
    df_clean = df.copy()

    # 1. Site Specific
    if community_name == "Unalakleet":
        df_clean = apply_unalakleet_correction(df_clean, sensor_col)

    # 2. Physical Bounds
    outlier_mask_phy = (df_clean[sensor_col] < 0) | (df_clean[sensor_col] > tank_height)
    if outlier_mask_phy.sum() > 0:
        df_clean.loc[outlier_mask_phy, sensor_col] = np.nan
        df_clean[sensor_col] = df_clean[sensor_col].ffill().bfill()
        print(f"  Cleaned {outlier_mask_phy.sum()} physical outliers.")

    # 3. Statistical (Rolling Median)
    window_size = 480 # ~5 days
    threshold = 2
    
    roll_med = df_clean[sensor_col].rolling(window=window_size, center=True).median()
    roll_std = df_clean[sensor_col].rolling(window=window_size, center=True).std()
    
    diff = np.abs(df_clean[sensor_col] - roll_med)
    limit = threshold * roll_std
    
    limit = limit.ffill().bfill()
    diff = diff.fillna(0)

    outlier_mask_stat = diff > limit
    if outlier_mask_stat.sum() > 0:
        df_clean.loc[outlier_mask_stat, sensor_col] = np.nan
        df_clean[sensor_col] = df_clean[sensor_col].ffill().bfill()
        print(f"  Cleaned {outlier_mask_stat.sum()} statistical outliers.")

    return df_clean

# ==========================================
# 3. Metric Calculation
# ==========================================

def calculate_indices(df, sensor_col, tank_height, crit_ratio=1/3):
    """
    Calculates indices and returns the event periods detected 
    by the 4-hour sampling logic.
    """
    # 1. Setup and Reliability 
    df_calc = df.copy()
    df_calc['normalized'] = df_calc[sensor_col] / tank_height
    crit = crit_ratio
    df_calc['D'] = (df_calc['normalized'] <= crit).astype(int)

    count_d0 = (df_calc['D'] == 0).sum()
    total_points = len(df_calc)
    RI = count_d0 / total_points if total_points > 0 else 0

    # 2. Filter to fixed times (The 4-hour subset)
    df_calc['time_only'] = df_calc['Timestamp'].dt.strftime('%H:%M:%S')
    mask_times = df_calc['time_only'].isin([
        '00:15:00', '04:15:00', '08:15:00',
        '12:15:00', '16:15:00', '20:15:00'
    ])
    df_times = df_calc[mask_times].copy()
    df_times.sort_values('Timestamp', inplace=True)
    df_times.reset_index(drop=True, inplace=True)

    # 3. Resiliency and Vulnerability (Using df_times)
    D_0, D_1 = 0, 0
    for i in range(1, len(df_times)):
        # D decreasing (1 -> 0) implies recovery
        if df_times.loc[i, 'D'] < df_times.loc[i-1, 'D']:
            D_0 += 1
        if df_times.loc[i, 'D'] == 1:
            D_1 += 1
    
    if len(df_times) > 0 and df_times.loc[len(df_times)-1, 'D'] == 1:
        D_1 += 1

    Res = D_0 / D_1 if D_1 > 0 else float('inf')

    if D_1 > 0:
        D_sum = sum((crit - df_times.loc[i, 'normalized']) 
                    for i in range(len(df_times)) if df_times.loc[i, 'D'] == 1)
        Vun = (D_sum / D_1) / crit
    else:
        Vun = 0

    # 4. Event Statistics 
    df_times['below'] = (df_times['normalized'] <= crit).astype(int)
    below_periods = []
    in_event = False
    start_time = None

    for i in range(len(df_times)):
        if df_times.loc[i, 'below'] == 1 and not in_event:
            in_event = True
            start_time = df_times.loc[i, 'Timestamp']
        elif df_times.loc[i, 'below'] == 0 and in_event:
            end_time = df_times.loc[i, 'Timestamp']
            below_periods.append((start_time, end_time))
            in_event = False

    if in_event:  # ends in failure state
        below_periods.append((start_time, df_times.loc[len(df_times)-1, 'Timestamp']))

    if below_periods:
        durations_hr = [(end - start).total_seconds() / 3600 for start, end in below_periods]
        num_events = len(below_periods)
        avg_duration = np.mean(durations_hr)
        total_duration = np.sum(durations_hr)
    else:
        num_events, avg_duration, total_duration = 0, 0, 0

    return {
        'RI': RI, 
        'Res': Res, 
        'Vun': Vun,
        'Num_Events': num_events,
        'Avg_Duration_Hr': avg_duration,
        'Total_Duration_Hr': total_duration,
        'MaxHeight': df_calc[sensor_col].max(),
        'Event_Periods': below_periods
    }

# ==========================================
# 4. Visualization
# ==========================================

def plot_final_analysis(df, sensor_col, community_name, tank_height, event_periods, crit_ratio=1/3):
    """
    Saves a time-series plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Main Data Line 
    ax.plot(df['Timestamp'], df[sensor_col], label='Cleaned Water Level', color='#1f77b4', linewidth=1)
    
    # 2. Critical Threshold
    crit_level = crit_ratio * tank_height
    ax.axhline(crit_level, color='#d62728', linestyle='--', label='Critical Level', linewidth=1.5)
    
    # 3. Highlight Events 
    # We use the specific start/end times returned by the calculation logic
    added_label = False
    for start, end in event_periods:
        if not added_label:
            ax.axvspan(start, end, color='gray', alpha=0.4, label='Below Critical Event')
            added_label = True
        else:
            ax.axvspan(start, end, color='gray', alpha=0.4)

    ax.set_ylabel('Water Level (ft)')
    ax.set_title(f"{community_name}: Water Level Analysis")
    ax.set_ylim(0, tank_height * 1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    ax.legend()
    
    safe_name = community_name.replace(" ", "_")
    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}_Analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    print(">>> Connecting to BMON Server...")
    try:
        import bmondata
        server = bmondata.Server('https://anthc.bmon.org', store_key='temporary-key')
    except ImportError:
        print("Error: 'bmondata' library not found.")
        return

    all_results = []
    
    # Time range
    start_ts, end_ts = '2022-12-31 10:00 pm', '2025-1-1 01:00 am'

    for community_name, info in COMMUNITIES_INFO.items():
        print(f"\nProcessing {community_name}...")
        df = load_and_align_data(server, info["sensor"], start_ts, end_ts)
        
        if df.empty:
            print(f"  Skipping {community_name} (No data)")
            continue

        # Clean
        df_clean = clean_data_pipeline(df, info["sensor"][1], community_name, info["height"])
        
        # Calculate (Returns metrics + event list)
        idx = calculate_indices(df_clean, info["sensor"][1], info["height"])
        
        # Print Stats to Console
        print(f"  {community_name}: Number of below-critical events = {idx['Num_Events']}")
        print(f"  {community_name}: Average duration below critical (hours) = {idx['Avg_Duration_Hr']:.2f}")
        print(f"  {community_name}: Total time below critical (hours) = {idx['Total_Duration_Hr']:.2f}")
        print(f"  {community_name}: Highest Water Level after cleaning: {idx['MaxHeight']:.2f} ft")
        
        # Store for Excel 
        all_results.append({
            'Community': community_name,
            'Reliability': idx['RI'],
            'Vulnerability': idx['Vun'],
            'Resiliency': idx['Res'],
            'Num_Below_Events': idx['Num_Events'],
            'Avg_Duration_Hr': idx['Avg_Duration_Hr'],
            'Total_Duration_Hr': idx['Total_Duration_Hr'],
            'Max_Height_Ft': idx['MaxHeight']
        })
        
        # Plot 
        plot_final_analysis(df_clean, info["sensor"][1], community_name, info["height"], idx['Event_Periods'])

    if not all_results:
        print("No results generated.")
        return

    # --- Save Excel Report ---
    print(f"\nSaving Excel Report to '{OUTPUT_DIR}'...")
    df_all = pd.DataFrame(all_results)
    
    # Replace infinite/NaN with dashed line for special cases (Perfect Reliability)
    df_export = df_all.copy()
    df_export['Resiliency'] = df_export['Resiliency'].apply(lambda x: '-' if np.isinf(x) or pd.isna(x) else x)
    df_export['Vulnerability'] = df_export['Vulnerability'].apply(lambda x: '-' if pd.isna(x) or x == 0 and df_export.loc[df_export['Vulnerability'].index[df_export['Vulnerability'] == x], 'Reliability'].values[0] == 1 else x)
    
    # Logic check: If Reliability is 1.0, Vun and Res should be '-'
    mask_perfect = df_export['Reliability'] == 1.0
    df_export.loc[mask_perfect, 'Resiliency'] = '-'
    df_export.loc[mask_perfect, 'Vulnerability'] = '-'

    # Reorder columns
    cols = ['Community', 'Reliability', 'Vulnerability', 'Resiliency',  
            'Num_Below_Events', 'Avg_Duration_Hr', 'Total_Duration_Hr', 'Max_Height_Ft']
    df_export = df_export[cols]
    
    save_path = os.path.join(OUTPUT_DIR, "Communitiy_metrics.xlsx")
    df_export.to_excel(save_path, index=False)
    print(f"  Saved '{save_path}'")
    print("\n>>> Processing Complete.")

if __name__ == "__main__":
    main()