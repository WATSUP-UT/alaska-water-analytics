# Transforming Unused Data into Actionable Insights for Enhanced Water Operations in Remote Alaska

This repository contains the source code and analysis scripts associated with the research paper:

**Transforming Unused Data into Actionable Insights for Enhanced Water Operations in Remote Alaska**  
_R. Cantrell, H. Ghamkhar, L. Sela_

The codebase provides a suite of tools for transforming high‑resolution sensor data from small, remote Alaskan community water systems into directly actionable insights for operators. Methods are designed for **resource‑constrained utilities** and are integrated into an interactive Streamlit dashboard to support day‑to‑day decision making.

The framework focuses on four core operational tasks:

1. Deriving and tracking **pressure pump curves** from routine operations.  
2. Establishing **demand patterns** and detecting **operational drifts** in community demand.  
3. Detecting and quantifying **unmeasured internal water losses** (e.g., filter backwash events).  
4. Evaluating **sustainability metrics** (Reliability, Resiliency, Vulnerability) across multiple communities.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Access](#data-access)
- [Usage](#usage)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Remote Alaska communities operate small and very small community water systems (CWS) under **harsh climate, workforce shortages, and limited budgets**. The Alaska Native Tribal Health Consortium (ANTHC) has deployed a Remote Monitoring Program, providing online measurements of key processes (flows, levels, pressures), but much of this data remains **underutilized “dark data”**.

This repository implements the data‑processing and analysis pipeline used in the paper to:

- Clean and impute raw sensor data.  
- Extract **pump performance curves** and track operating ranges over time.  
- Build **diurnal and seasonal demand profiles** and detect real‑time drifts using STL + adaptive CUSUM.  
- Detect **filter backwash events** and estimate their volume via CUSUM + mass balance.  
- Compute **sustainability metrics** for tank operations across multiple communities over multi‑year periods.  
- Expose all of the above through an **operator‑friendly dashboard** built in Streamlit.

The methods are demonstrated primarily on the Unalakleet water system and then scaled to additional communities for the sustainability analysis.

---

## Repository Structure

| File / Directory           | Description                                                                                                                          |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `raw_data.py`             | Fetches, and visualizes raw time‑series data (flow, tank level, pressure).                                                           |
| `pump_curve.py`           | Derives pump curves from routine operations using inferred head–flow pairs and pump affinity laws; tracks pump operating ranges.    |
| `system_demand.py`        | Builds diurnal demand profiles (median patterns) by day of week from supply flow data.                |
| `drift_detection.py`      | Implements STL + adaptive two‑sided CUSUM for real‑time detection of sustained drifts in baseline demand.                           |
| `backwash.py`             | Detects and quantifies backwash events using CUSUM on tank level changes and a mass‑balance calculation of backwash volumes.        |
| `sustainability_metrics.py` | Computes Reliability, Resiliency, and Vulnerability indices for tank level time series (per community, per analysis period).      |
| `requirements.txt`        | Python dependencies required to run the analysis and dashboard.                                                                      |
| [Dashboard](https://hanighamkhar-alaska-dashboard-main-yksxm3.streamlit.app/)| Streamlit dashboard that integrates all methods for interactive exploration by operators and remote technical support.      |


---

## Installation

It is recommended to use a virtual environment.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv venv          # or: python3 -m venv venv
source venv/bin/activate     # Mac/Linux
# .env\Scriptsctivate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data Access

These scripts rely on the **bmondata** library to access sensor data from ANTHC’s Remote Monitoring Program (BMON server).

- **Source**: `https://anthc.bmon.org`  
- **Sensors typically used** (15–30 min resolution):  
  - Treated water inflow to storage tank.  
  - Water storage tank level.  
  - Supply flow rate to the distribution system (used as community demand).  
  - Distribution system pressure head.

Configuration notes:

- Scripts assume a valid `store_key` for `bmondata.Server(...)` is available.  
- Some scripts (e.g., `drift_detection.py`) can optionally read from local CSV backups when the server is not reachable.

---


## Usage

Each script is designed to be run independently after configuring community IDs, date ranges, and sensor tags inside the script or via command‑line arguments (if implemented).

Example:

```bash
# Pump curve analysis
python pump_curve.py

# Demand profiling
python system_demand.py

# Real‑time / batch drift detection
python drift_detection.py

# Backwash detection and volume estimation
python backwash.py

# Sustainability metrics across communities
python sustainability_metrics.py

# Launch Streamlit dashboard
https://hanighamkhar-alaska-dashboard-main-yksxm3.streamlit.app/
```

Outputs (plots, CSV/Excel reports, and derived metrics) are written to dedicated sub‑directories such as `Pump_Curve_Outputs/`, `Demand_Outputs/`, `Backwash_Outputs/`, and `Sustainability_Outputs/` (adjust names to your implementation).

---

## Reproducing the Paper Results

A typical workflow to reproduce the main figures in the paper is:

1. **Preprocess raw data** for Unalakleet (and other communities) with `raw_data.py`.  
2. **Derive pump curves** and weekly operating curves with `pump_curve.py` (Figures analogous to pump curve and weekly head plots).  
3. **Generate demand patterns** and **run drift detection** with `system_demand.py` and `drift_detection.py` (diurnal profiles and drift events).  
4. **Detect backwash events** and compute volumes with `backwash.py` (backwash frequency/volume plots).  
5. **Compute sustainability metrics** across communities using `sustainability_metrics.py` (heatmaps and time series of tank performance).  
6. Use the [live dashboard](https://hanighamkhar-alaska-dashboard-main-yksxm3.streamlit.app/) to explore all results interactively from the operator perspective.

Adjust paths, to match your deployment.

---

## Citation

If you use this code or data in your work, please cite the associated paper:

```bibtex
@article{Cantrell2025UnusedData,
  title  = {Transforming Unused Data into Actionable Insights for Enhanced Water Operations in Remote Alaska},
  author = {Cantrell, Rebecca and Ghamkhar, Hani and Sela, Lina},
  year   = {2025},
  url    = {to_BE_ADDED},
  note   = {Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.


