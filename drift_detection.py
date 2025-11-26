import pandas as pd
import numpy as np
import time
import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.tsa.seasonal import STL

# ==========================================
# 1. Data Handling
# ==========================================

def load_real_data(backup_file="server_data.csv"):
    """
    Load data from BMON server. 
    If server works -> Save to backup_file.
    If server fails -> Load from backup_file.
    """
    df = None
    
    # Try Server First
    try:
        print("Attempting to connect to BMON server...")
        import bmondata
        server = bmondata.Server('https://anthc.bmon.org', store_key='temporary-key')
        sensors = [('A81758FFFE067721_pulseAbs', 'Master Meter Flow Rate, GPM')]

        df = server.sensor_readings(
            sensors,
            start_ts='2022-07-01 12:00 am',
            end_ts='2024-10-27 11:59 pm', 
            averaging='30min'
        )

        # Process the data 
        df.reset_index(inplace=True)
        df = df.rename(columns={'index': 'Timestamp'})
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # # Save backup for future reproducibility/failures
        # print(f"Server download successful. Saving backup to {backup_file}")
        # df.to_csv(backup_file, index=False)
        
    except Exception as e:
        print(f"Server connection failed: {e}")
        print(f"Attempting to load from local backup: {backup_file}")
        
        if os.path.exists(backup_file):
            df = pd.read_csv(backup_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        else:
            raise RuntimeError("No server connection and no local backup file found. Cannot proceed.")

    START_DATE = df['Timestamp'].min()
    END_DATE = df['Timestamp'].max()
    timerange = pd.date_range(start=START_DATE, end=END_DATE, freq='30min')
    
    dates = timerange.to_frame(index=False)
    dates = dates.rename(columns={0: 'Timestamp'})
    dates = dates.set_index('Timestamp')
    
    df_data = df.copy()
    df_data = df_data.set_index('Timestamp')
    
    df = pd.concat([dates, df_data], axis=1)
    df = df.reset_index(drop=False)
    df = df.rename(columns={'index': 'Timestamp'}) # reset_index might name it 'index'

    # Use forward fill then backward fill
    df['Master Meter Flow Rate, GPM'] = df['Master Meter Flow Rate, GPM'].ffill()
    df['Master Meter Flow Rate, GPM'] = df['Master Meter Flow Rate, GPM'].bfill()
    
    return df

# ==========================================
# 2. STL Decomposition
# ==========================================

def apply_stl_decomposition(df, column_name='Master Meter Flow Rate, GPM'):
    """
    Apply STL decomposition
    """
    df_stl = df.copy()
    df_stl = df_stl.set_index('Timestamp')
    
    stl_daily = STL(df_stl[column_name], 
                   period=48,  
                   seasonal=13,  
                   trend=None,  
                   robust=True)
    
    result_daily = stl_daily.fit()
    
    df_stl['trend'] = result_daily.trend
    df_stl['seasonal'] = result_daily.seasonal
    df_stl['residual'] = result_daily.resid
    
    df_stl = df_stl.reset_index()
    
    return df_stl

# ==========================================
# 3. Benchmark Definition
# ==========================================

def get_known_events():
    """
    Hardcoded benchmark dates
    """
    benchmark_dates = [
        "2023-09-11", "2023-10-24", "2023-11-07", "2023-12-12", 
        "2024-01-01", "2024-05-06", "2024-06-07", "2024-07-16", "2024-10-16"
    ]
    
    benchmark_anomalies = []
    for date_str in benchmark_dates:
        date = pd.Timestamp(date_str)
        start_time = date.replace(hour=0, minute=0, second=0)
        end_time = date.replace(hour=23, minute=59, second=59)
        midpoint = start_time + (end_time - start_time) / 2
        
        benchmark_anomalies.append({
            'label': date.strftime('%m/%d'),
            'start_time': start_time,
            'end_time': end_time,
            'midpoint': midpoint
        })
    return benchmark_anomalies

# ==========================================
# 4. Main Detection Logic
# ==========================================

def moving_window_detection(
    historical_window_days=90, 
    stats_window_days=14, 
    k_factor=1.5, 
    h_factor=4.0, 
    min_threshold=0.85,
    hours_to_simulate=984,
    event_min_duration=12,
    event_significance_threshold=0.6
):
    print(f"\n===  Moving Window Anomaly Detection ===\n")
    
    output_dir = "Drift_Detection_Outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load Data
    print("Loading historical data...")
    historical_data = load_real_data()
    print("Initial data columns:", historical_data.columns.tolist())
    
    # Step 2: Split data
    cutoff_date = pd.Timestamp("2024-05-01")
    real_time_start = cutoff_date
    historical_end = real_time_start - pd.Timedelta(seconds=1)
    
    historical_data_part = historical_data[
        historical_data['Timestamp'] <= historical_end
    ].copy()
    
    real_time_data = historical_data[
        (historical_data['Timestamp'] >= real_time_start) & 
        (historical_data['Timestamp'] <= real_time_start + pd.Timedelta(hours=hours_to_simulate))
    ].copy()
    
    print(f"Historical data: {len(historical_data_part)} points ending at {historical_data_part['Timestamp'].max()}")
    print(f"Real-time data: {len(real_time_data)} points from {real_time_data['Timestamp'].min()} to {real_time_data['Timestamp'].max()}")
    
    # Step 3: Historical Analysis Setup
    points_per_day = 48
    historical_window_points = historical_window_days * points_per_day
    stats_window_points = stats_window_days * points_per_day
    
    if len(historical_data_part) < historical_window_points:
        print(f"Warning: Historical data contains fewer points than requested window")
        initial_window = historical_data_part.copy()
    else:
        initial_window = historical_data_part.tail(historical_window_points).copy()
        
    # Step 6: Real-time detection setup
    print("\nInitializing real-time detection...")
    current_data = initial_window.copy()
    
    anomaly_sequence_active = False
    points_over_zero = 0
    points_over_threshold = 0
    current_sequence_start = None
    detected_events = []
    real_time_results = []
    
    # Initialize CUSUM state
    
    # Step 7: Process each point
    print("\nProcessing real-time data points...")
    
    for i, (_, row) in enumerate(real_time_data.iterrows()):
        new_timestamp = row['Timestamp']
        new_flow_value = row['Master Meter Flow Rate, GPM']
        
        # Printing every point
        print(f"  Processing point {i+1}/{len(real_time_data)} at {new_timestamp}...")
        
        new_row = pd.DataFrame({
            'Timestamp': [new_timestamp],
            'Master Meter Flow Rate, GPM': [new_flow_value]
        })
        
        current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        if len(current_data) > historical_window_points:
            current_data = current_data.iloc[-historical_window_points:].reset_index(drop=True)
        
        # Apply STL
        start_time = time.time()
        df_stl = apply_stl_decomposition(current_data)
        decomp_time = time.time() - start_time
        
        df_stl['trend_delta'] = df_stl['trend'].diff()
        
        latest_idx = len(df_stl) - 1
        latest = df_stl.iloc[latest_idx]
        latest_delta = latest['trend_delta'] if not pd.isna(latest['trend_delta']) else 0
        
        # Stats
        stats_start_idx = max(0, latest_idx - stats_window_points)
        stats_end_idx = latest_idx - 1
        
        delta_values = df_stl['trend_delta'].iloc[stats_start_idx:stats_end_idx+1].dropna().values
        if len(delta_values) > 0:
            delta_mean = np.mean(delta_values)
            delta_std = np.std(delta_values)
        else:
            delta_mean = 0
            delta_std = 0.01
            
        k = k_factor * delta_std
        H = max(h_factor * delta_std, min_threshold)
        
        # CUSUM Calculation
        if i == 0:
            C_plus = max(0, latest_delta - delta_mean - k)
            C_minus = max(0, delta_mean - k - latest_delta)
        else:
            prev_result = real_time_results[-1]
            prev_C_plus = prev_result['C_plus']
            prev_C_minus = prev_result['C_minus']
            
            C_plus = max(0, latest_delta - delta_mean - k + prev_C_plus)
            C_minus = max(0, delta_mean - k - latest_delta + prev_C_minus)
            
        is_over_zero = (C_plus > 0) or (C_minus > 0)
        is_over_threshold = (C_plus > H) or (C_minus > H)
        
        # Event tracking logic
        if is_over_zero:
            if not anomaly_sequence_active:
                anomaly_sequence_active = True
                current_sequence_start = new_timestamp
                points_over_zero = 1
                points_over_threshold = 1 if is_over_threshold else 0
            else:
                points_over_zero += 1
                if is_over_threshold:
                    points_over_threshold += 1
                
                if (points_over_threshold >= event_min_duration and 
                    points_over_threshold / points_over_zero >= event_significance_threshold):
                    if current_sequence_start not in detected_events:
                        detected_events.append(current_sequence_start)
                        hours_since_start = (current_sequence_start - real_time_start).total_seconds() / 3600
                        print(f"    EVENT DETECTED at {current_sequence_start}!")
                        print(f"      Detected {hours_since_start:.2f} hours after monitoring start")
                        print(f"      Significance: {points_over_threshold / points_over_zero:.2f}")
        else:
            if anomaly_sequence_active:
                anomaly_sequence_active = False
                points_over_zero = 0
                points_over_threshold = 0
                
        result = {
            'Timestamp': new_timestamp,
            'Flow_Rate': new_flow_value,
            'Trend_Delta': latest_delta,
            'Delta_Mean': delta_mean,
            'Delta_Std': delta_std,
            'k': k,
            'H': H,
            'C_plus': C_plus,
            'C_minus': C_minus,
            'Is_Over_Zero': is_over_zero,
            'Is_Over_Threshold': is_over_threshold,
            'Points_Over_Zero': points_over_zero if anomaly_sequence_active else 0,
            'Points_Over_Threshold': points_over_threshold if anomaly_sequence_active else 0,
            'Significance': points_over_threshold / points_over_zero if anomaly_sequence_active and points_over_zero > 0 else 0,
            'Is_Event': current_sequence_start in detected_events if anomaly_sequence_active else False,
            'Processing_Time': decomp_time
        }
        real_time_results.append(result)

    df_real_time_results = pd.DataFrame(real_time_results)
    
    print(f"\nDetected {len(detected_events)} significant events in real-time data")
    for date in detected_events:
        print(f"  Real-time event at {date}")
        
    df_real_time_results.to_csv(f"{output_dir}/real_time_results.csv", index=False)
    print(f"Saved results to {output_dir}/real_time_results.csv")
    
    return f"{output_dir}/real_time_results.csv", detected_events

# ==========================================
# 5. Visualization
# ==========================================

def create_flow_visualization(results_file, benchmark_anomalies, output_dir="Drift_Detection_Outputs"):
    print(f"Creating visualization from {results_file}...")
    df_results = pd.read_csv(results_file)
    df_results['Timestamp'] = pd.to_datetime(df_results['Timestamp'])
    
    # Reconstruct Event Logic for Plotting
    detected_events = []
    cusum_over_zero_starts = []
    
    if 'Is_Event' in df_results.columns:
        df_results['event_start'] = (df_results['Is_Event'] != df_results['Is_Event'].shift(1)) & (df_results['Is_Event'] == True)
        detected_events = df_results[df_results['event_start'] == True]['Timestamp'].tolist()
        
        if 'C_plus' in df_results.columns and 'C_minus' in df_results.columns:
            for event_start in detected_events:
                try:
                    event_idx = df_results[df_results['Timestamp'] == event_start].index[0]
                    seq_start_idx = event_idx
                    while seq_start_idx > 0:
                        prev_idx = seq_start_idx - 1
                        prev_point = df_results.iloc[prev_idx]
                        if prev_point['C_plus'] <= 0 and prev_point['C_minus'] <= 0:
                            break
                        seq_start_idx = prev_idx
                    if seq_start_idx < event_idx:
                        cusum_start = df_results.iloc[seq_start_idx]['Timestamp']
                        cusum_over_zero_starts.append(cusum_start)
                    else:
                        cusum_over_zero_starts.append(event_start)
                except:
                    pass

    # Plotting
    plt.rcParams.update({'font.size': 40})
    fig, ax = plt.subplots(figsize=(15, 10))
    
    flow_col = 'Master Meter Flow Rate, GPM' if 'Master Meter Flow Rate, GPM' in df_results.columns else 'Flow_Rate'
    if flow_col in df_results.columns:
        df_results['Flow Rate, m3/hr'] = df_results[flow_col] * 0.227124
        ax.plot(df_results['Timestamp'], df_results['Flow Rate, m3/hr'], '-', 
                label='Flow Rate, m3/hr', color='#00a9b7', linewidth=1.5)
        y_min = df_results['Flow Rate, m3/hr'].min()
        y_max = df_results['Flow Rate, m3/hr'].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.15)
        
    # Benchmarks
    plot_start = df_results['Timestamp'].min()
    plot_end = df_results['Timestamp'].max()
    
    for anomaly in benchmark_anomalies:
        if anomaly['start_time'] >= plot_start and anomaly['start_time'] <= plot_end:
            ax.axvspan(anomaly['start_time'], anomaly['end_time'],
                      alpha=0.4, color='#F8971F', zorder=1)
            ax.annotate(f"{anomaly.get('label', 'Benchmark')}", 
                      xy=(anomaly['midpoint'], df_results['Flow Rate, m3/hr'].max() * 0.95),
                      xytext=(0, 5), textcoords='offset points',
                      ha='center', va='bottom', fontsize=14, color='#F8971F')
                      
    # Anomaly Sequences
    for i, start_time in enumerate(cusum_over_zero_starts):
        if i < len(detected_events):
            end_time = detected_events[i]
            ax.axvspan(start_time, end_time, alpha=0.3, color='grey', zorder=2)
            
    # Detected Events
    for date in detected_events:
        ax.axvline(x=date, color='#BF5700', linewidth=2, alpha=0.8, zorder=3)
        
    # Legend
    legend_elements = [
        Line2D([0], [0], color='#00a9b7', lw=2, label='Flow Rate['+ r'$m^3$'+ '/hr]'),
        Line2D([0], [0], color='#BF5700', lw=2, label='Event Detection'),
        Patch(facecolor='grey', alpha=0.3, label='Anomaly Sequence'),
        Patch(facecolor='#F8971F', alpha=0.4, label='Benchmark Anomalies')
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=2, 
              fontsize=16, frameon=False)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_ylabel('Flow Rate['+ r'$m^3$'+ '/hr]', fontsize=40)
    ax.set_xlabel('Date', fontsize=40)
    ax.tick_params(axis='y', labelcolor='black', labelsize=30)
    ax.tick_params(axis='x', labelrotation=0, labelsize=25)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flow_rate_m3hr.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_dir}/flow_rate_m3hr.png")

if __name__ == "__main__":
    results_csv, events = moving_window_detection(
        historical_window_days=90,
        stats_window_days=14,
        k_factor=1.5,
        h_factor=4.0,
        min_threshold=0.85,
        hours_to_simulate=984,
        event_min_duration=12,
        event_significance_threshold=0.6
    )
    
    benchmarks = get_known_events()
    create_flow_visualization(results_csv, benchmarks)
    print("\nAnalysis Complete.")