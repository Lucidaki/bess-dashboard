import pandas as pd

# Constants from Northwold Agreement
CAPACITY_MWH = 8.4
WARRANTY_CYCLES_DAILY = 1.5
WARRANTY_DEGRADATION_ANNUAL_PCT = 2.5  # Lower bound of 2.5-3% estimate

# Degradation Factor Calculation
# We assume the 2.5% annual loss is based on fully utilizing the 1.5 cycle warranty every day.
# Annual Max Cycles = 1.5 * 365 = 547.5 cycles
DEGRADATION_PER_CYCLE_PCT = WARRANTY_DEGRADATION_ANNUAL_PCT / (WARRANTY_CYCLES_DAILY * 365)

def calculate_metrics(master_file='Master_BESS_Analysis_Sept_2025.csv',
                      optimized_file='Optimized_Results.csv'):
    print(f"--- Phase 5: Analyzing Cycles & Degradation ---")

    # Load master file for actual data
    master_df = pd.read_csv(master_file)
    if 'Unnamed: 0' in master_df.columns:
        master_df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
    master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp'])
    master_df['Date'] = master_df['Timestamp'].dt.date

    # Load optimized results
    opt_df = pd.read_csv(optimized_file)
    opt_df['Timestamp'] = pd.to_datetime(opt_df['Timestamp'])

    # Merge the datasets
    merged_df = pd.merge(master_df, opt_df, on='Timestamp', how='inner')

    num_days = merged_df['Date'].nunique()
    print(f"Analysis Period: {num_days} days")

    # Find the actual battery MWh column (handle the newline in column name)
    actual_col = None
    for col in master_df.columns:
        if 'Battery MWh' in col and 'Output' in col:
            actual_col = col
            break

    if actual_col is None:
        print("Warning: Could not find actual battery MWh column")
        actual_col = 'Physical_Power_MW'  # Fallback

    scenarios = {
        "Actual (Aggregator)": actual_col,
        "Optimized (Daily Switching)": "Optimised_Net_MWh"
    }

    results = []

    for name, col in scenarios.items():
        if col not in merged_df.columns:
            print(f"Warning: Column '{col}' not found, skipping {name}")
            continue

        # 1. Calculate Total Discharge (MWh)
        # In the source file, Positive = Discharge (Export), Negative = Charge (Import)
        # We only sum positive values to get Discharge Throughput
        discharge_mask = merged_df[col] > 0
        total_discharge_mwh = merged_df.loc[discharge_mask, col].sum()

        # 2. Calculate Cycles
        # Cycle Definition: Total Discharge / Usable Capacity
        total_cycles = total_discharge_mwh / CAPACITY_MWH

        # 3. Average Daily Cycles
        avg_daily_cycles = total_cycles / num_days

        # 4. Estimated Degradation
        est_degradation = total_cycles * DEGRADATION_PER_CYCLE_PCT

        results.append({
            "Scenario": name,
            "Total Discharge (MWh)": round(total_discharge_mwh, 2),
            "Total Cycles": round(total_cycles, 2),
            "Avg Cycles/Day": round(avg_daily_cycles, 2),
            "Est. Degradation (%)": round(est_degradation, 4),
            "Warranty Status": "OK" if avg_daily_cycles <= WARRANTY_CYCLES_DAILY else "WARRANTY VOID"
        })

    # Convert to DataFrame for display
    results_df = pd.DataFrame(results)

    print("\n--- Cycle & Degradation Results ---")
    print(results_df.to_string(index=False))

    # Export
    results_df.to_csv("Northwold_Cycle_Degradation_Report.csv", index=False)
    print("\nReport saved to: Northwold_Cycle_Degradation_Report.csv")

    return results_df

if __name__ == "__main__":
    # Run analysis using our existing files
    calculate_metrics('Master_BESS_Analysis_Sept_2025.csv', 'Optimized_Results.csv')