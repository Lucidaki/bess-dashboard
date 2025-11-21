"""
Phase 2: Digital Twin Configuration
Stores the physical and commercial constraints extracted from the
Northwold Storage Asset Optimisation Agreement.
"""

# --- Physical Limits ---
ASSET_NAME = "Northwold Solar Farm (Hall Farm)"

# Power Constraints (Asymmetric)
P_IMP_MAX_MW = 4.2   # Maximum Import (Charge) Rate
P_EXP_MAX_MW = 7.5   # Maximum Export (Discharge) Rate

# Energy Capacity
CAPACITY_MWH = 8.4   # Usable Energy Capacity

# Efficiency
EFF_ROUND_TRIP = 0.87
import numpy as np
EFF_ONE_WAY = np.sqrt(EFF_ROUND_TRIP) # Assumed symmetric losses on charge/discharge

# State of Charge Limits (Safety Buffers)
SOC_MIN_PCT = 0.05  # 5%
SOC_MAX_PCT = 0.95  # 95%
SOC_MIN_MWH = CAPACITY_MWH * SOC_MIN_PCT
SOC_MAX_MWH = CAPACITY_MWH * SOC_MAX_PCT

# --- Warranty Constraints ---
# Critical Limit: The battery cannot discharge more than this amount per day.
CYCLES_PER_DAY = 1.5
MAX_DAILY_THROUGHPUT_MWH = CAPACITY_MWH * CYCLES_PER_DAY # 12.6 MWh

# --- Optimization Constants ---
DT_HOURS = 0.5 # Time step (30 minutes)