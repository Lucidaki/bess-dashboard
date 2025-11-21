import pandas as pd
import numpy as np
from scipy.optimize import linprog
import digital_twin_config as config  # Imports the constants file

def solve_dispatch_multimarket(buy_prices, sell_prices, ffr_mask, start_soc, market_names=None):
    """
    Solves the linear optimization problem for multi-market trading.

    Args:
        buy_prices (array): Vector of best buy prices across markets for each period.
        sell_prices (array): Vector of best sell prices across markets for each period.
        ffr_mask (array): Boolean vector. True = Battery Locked in FFR (0 MW).
        start_soc (float): Starting energy in MWh.
        market_names (array): Optional array indicating which market offers best price per period.

    Returns:
        charge_mw, discharge_mw, soc_profile, revenue_by_period (arrays)
    """
    T = len(buy_prices)
    dt = config.DT_HOURS

    # Variables: [Charge_0..T, Discharge_0..T, SoC_0..T]
    num_vars = 3 * T
    idx_chg = np.arange(0, T)
    idx_dis = np.arange(T, 2*T)
    idx_soc = np.arange(2*T, 3*T)

    # Objective: Maximize Revenue with different buy/sell prices
    # Minimize: (Charge_Cost@buy_prices - Discharge_Revenue@sell_prices)
    c = np.zeros(num_vars)
    c[idx_chg] = buy_prices * dt  # Cost to charge at buy prices
    c[idx_dis] = -sell_prices * dt  # Revenue from discharge at sell prices

    # Bounds
    bounds = []
    for i in range(num_vars):
        t_idx = i % T
        if i in idx_chg:
            # If locked in FFR, Max Charge is 0. Else config limit.
            ub = 0 if ffr_mask[t_idx] else config.P_IMP_MAX_MW
            bounds.append((0, ub))
        elif i in idx_dis:
            # If locked in FFR, Max Discharge is 0. Else config limit.
            ub = 0 if ffr_mask[t_idx] else config.P_EXP_MAX_MW
            bounds.append((0, ub))
        elif i in idx_soc:
            bounds.append((config.SOC_MIN_MWH, config.SOC_MAX_MWH))

    # Equality Constraints: SoC Physics
    A_eq = np.zeros((T, num_vars))
    b_eq = np.zeros(T)

    for t in range(T):
        A_eq[t, idx_soc[t]] = 1
        if t == 0:
            b_eq[t] += start_soc
        else:
            A_eq[t, idx_soc[t-1]] = -1

        A_eq[t, idx_chg[t]] = -config.EFF_ONE_WAY * dt
        A_eq[t, idx_dis[t]] = (1/config.EFF_ONE_WAY) * dt

    # Inequality Constraint: Daily Cycle Limit
    A_ub = np.zeros((1, num_vars))
    b_ub = np.zeros(1)
    A_ub[0, idx_dis] = dt
    b_ub[0] = config.MAX_DAILY_THROUGHPUT_MWH

    # Run Solver
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        charge_profile = res.x[idx_chg]
        discharge_profile = res.x[idx_dis]
        soc_profile = res.x[idx_soc]

        # Calculate revenue by period
        revenue_by_period = (discharge_profile * sell_prices * dt -
                           charge_profile * buy_prices * dt)

        return charge_profile, discharge_profile, soc_profile, revenue_by_period
    else:
        return np.zeros(T), np.zeros(T), np.full(T, start_soc), np.zeros(T)

def get_best_market_prices(day_df):
    """
    Analyzes all available market prices and returns best buy/sell opportunities.

    Args:
        day_df: DataFrame with market price columns

    Returns:
        buy_prices, sell_prices, buy_markets, sell_markets
    """
    # Extract all available market prices
    market_columns = {
        'EPEX': 'Day Ahead Price (EPEX)',
        'ISEM': 'GB-ISEM Intraday 1 Price',
        'SSP': 'SSP',
        'SBP': 'SBP',
        'DA_HH': 'DA HH Price'
    }

    # Create price matrix
    price_matrix = pd.DataFrame()
    available_markets = []

    for market_name, col_name in market_columns.items():
        if col_name in day_df.columns:
            prices = day_df[col_name].fillna(method='ffill').fillna(method='bfill')
            if not prices.isna().all():
                price_matrix[market_name] = prices.values
                available_markets.append(market_name)

    if price_matrix.empty:
        # Fallback to EPEX only if no markets available
        return (day_df['Day Ahead Price (EPEX)'].values,
                day_df['Day Ahead Price (EPEX)'].values,
                ['EPEX'] * len(day_df),
                ['EPEX'] * len(day_df))

    # Find best prices for each period
    buy_prices = price_matrix.min(axis=1).values
    sell_prices = price_matrix.max(axis=1).values
    buy_markets = price_matrix.idxmin(axis=1).values
    sell_markets = price_matrix.idxmax(axis=1).values

    return buy_prices, sell_prices, buy_markets, sell_markets

def calculate_market_strategy_value(day_df, start_soc, strategy_type='multimarket'):
    """
    Calculates the value of different market strategies.

    Args:
        day_df: DataFrame with market data
        start_soc: Starting state of charge
        strategy_type: 'multimarket', 'epex_only', or 'sffr'

    Returns:
        Total value and dispatch profile
    """
    if strategy_type == 'sffr':
        # SFFR strategy - fixed revenue, no dispatch
        sffr_prices = day_df['SFFR Clearing Price'].fillna(0).values
        total_value = np.sum(7.0 * sffr_prices * 0.5)
        return total_value, np.zeros(len(day_df)), np.zeros(len(day_df)), np.full(len(day_df), start_soc)

    elif strategy_type == 'epex_only':
        # Original EPEX-only strategy
        prices = day_df['Day Ahead Price (EPEX)'].values
        dummy_mask = np.full(len(day_df), False)
        chg, dis, soc, _ = solve_dispatch_multimarket(prices, prices, dummy_mask, start_soc)
        total_value = np.sum(dis * prices * 0.5 - chg * prices * 0.5)
        return total_value, chg, dis, soc

    else:  # multimarket
        # Multi-market optimization
        buy_prices, sell_prices, buy_markets, sell_markets = get_best_market_prices(day_df)
        dummy_mask = np.full(len(day_df), False)
        chg, dis, soc, revenue = solve_dispatch_multimarket(buy_prices, sell_prices, dummy_mask, start_soc)
        total_value = np.sum(revenue)
        return total_value, chg, dis, soc

def run_phase_3_multimarket(input_file):
    """
    Enhanced Phase 3 with multi-market optimization capabilities.
    """
    print("--- Phase 3: Running Multi-Market Counterfactual Analysis ---")
    df = pd.read_csv(input_file)

    # Handle the timestamp column
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)

    # Ensure Datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date

    # Placeholders for results
    results = []
    curr_soc_daily = 0.5 * config.CAPACITY_MWH  # Start month at 50%
    curr_soc_efa = 0.5 * config.CAPACITY_MWH
    curr_soc_multi = 0.5 * config.CAPACITY_MWH

    unique_dates = df['Date'].unique()

    for date in unique_dates:
        day_df = df[df['Date'] == date].copy()

        if len(day_df) != 48:
            continue  # Skip partial days

        # Get multi-market prices
        buy_prices, sell_prices, buy_markets, sell_markets = get_best_market_prices(day_df)
        sffr_price = day_df['SFFR Clearing Price'].fillna(0).values
        epex_prices = day_df['Day Ahead Price (EPEX)'].values

        # --- Strategy 1: Daily Switching (EPEX-only for comparison) ---
        ffr_daily_val = np.sum(7.0 * sffr_price * 0.5)
        dummy_mask = np.full(48, False)
        chg_daily, dis_daily, soc_daily, _ = solve_dispatch_multimarket(
            epex_prices, epex_prices, dummy_mask, curr_soc_daily
        )
        arb_daily_val = np.sum(dis_daily * epex_prices * 0.5 - chg_daily * epex_prices * 0.5)

        if ffr_daily_val > arb_daily_val:
            strategy_daily = "SFFR"
            daily_rev = ffr_daily_val
            chg_daily, dis_daily = np.zeros(48), np.zeros(48)
            soc_daily = np.full(48, curr_soc_daily)
        else:
            strategy_daily = "EPEX"
            daily_rev = arb_daily_val

        # --- Strategy 2: EFA Block Switching (EPEX-only) ---
        chg_efa = np.zeros(48)
        dis_efa = np.zeros(48)
        soc_efa = np.zeros(48)
        efa_strategy = []
        temp_soc = curr_soc_efa

        for efa_block in range(12):
            start_idx = efa_block * 4
            end_idx = start_idx + 4

            ffr_efa_val = np.sum(7.0 * sffr_price[start_idx:end_idx] * 0.5)
            efa_mask = np.full(4, False)
            chg_temp, dis_temp, soc_temp, _ = solve_dispatch_multimarket(
                epex_prices[start_idx:end_idx],
                epex_prices[start_idx:end_idx],
                efa_mask, temp_soc
            )
            arb_efa_val = np.sum(dis_temp * epex_prices[start_idx:end_idx] * 0.5 -
                                chg_temp * epex_prices[start_idx:end_idx] * 0.5)

            if ffr_efa_val > arb_efa_val:
                chg_efa[start_idx:end_idx] = 0
                dis_efa[start_idx:end_idx] = 0
                soc_efa[start_idx:end_idx] = temp_soc
                efa_strategy.extend(['SFFR'] * 4)
            else:
                chg_efa[start_idx:end_idx] = chg_temp
                dis_efa[start_idx:end_idx] = dis_temp
                soc_efa[start_idx:end_idx] = soc_temp
                efa_strategy.extend(['EPEX'] * 4)
                temp_soc = soc_temp[-1]

        # --- Strategy 3: Multi-Market Optimization ---
        # Compare SFFR vs multi-market arbitrage for the whole day
        multi_arb_val, chg_multi, dis_multi, soc_multi_profile = calculate_market_strategy_value(
            day_df, curr_soc_multi, 'multimarket'
        )

        if ffr_daily_val > multi_arb_val:
            strategy_multi = "SFFR"
            multi_rev = ffr_daily_val
            chg_multi, dis_multi = np.zeros(48), np.zeros(48)
            soc_multi_profile = np.full(48, curr_soc_multi)
            selected_markets = ['SFFR'] * 48
        else:
            strategy_multi = "Multi-Market"
            multi_rev = multi_arb_val
            # Determine which markets were used
            selected_markets = []
            for i in range(48):
                if chg_multi[i] > 0:
                    selected_markets.append(f"Buy-{buy_markets[i]}")
                elif dis_multi[i] > 0:
                    selected_markets.append(f"Sell-{sell_markets[i]}")
                else:
                    selected_markets.append("Idle")

        # Store Results for this day
        for i in range(48):
            results.append({
                'Timestamp': day_df.iloc[i]['Timestamp'],
                # Original EPEX-only daily strategy
                'Optimised_Net_MWh_Daily': (dis_daily[i] - chg_daily[i]) * 0.5,
                'Optimised_SoC_Daily': soc_daily[i],
                'Strategy_Selected_Daily': strategy_daily,
                'Optimised_Revenue_Daily': daily_rev / 48 if strategy_daily == 'SFFR' else
                                          (dis_daily[i]-chg_daily[i])*0.5*epex_prices[i],
                # EPEX-only EFA strategy
                'Optimised_Net_MWh_EFA': (dis_efa[i] - chg_efa[i]) * 0.5,
                'Optimised_SoC_EFA': soc_efa[i],
                'Strategy_Selected_EFA': efa_strategy[i],
                'Optimised_Revenue_EFA': (7.0 * sffr_price[i] * 0.5) if efa_strategy[i] == 'SFFR' else
                                        (dis_efa[i]-chg_efa[i])*0.5*epex_prices[i],
                # New multi-market strategy
                'Optimised_Net_MWh_Multi': (dis_multi[i] - chg_multi[i]) * 0.5,
                'Optimised_SoC_Multi': soc_multi_profile[i],
                'Strategy_Selected_Multi': strategy_multi,
                'Market_Used_Multi': selected_markets[i],
                'Optimised_Revenue_Multi': multi_rev / 48 if strategy_multi == 'SFFR' else
                                          (dis_multi[i] * sell_prices[i] - chg_multi[i] * buy_prices[i]) * 0.5,
                # Market spreads
                'Market_Spread': sell_prices[i] - buy_prices[i],
                'Best_Buy_Price': buy_prices[i],
                'Best_Sell_Price': sell_prices[i],
                'Best_Buy_Market': buy_markets[i],
                'Best_Sell_Market': sell_markets[i]
            })

        curr_soc_daily = soc_daily[-1]
        curr_soc_efa = soc_efa[-1]
        curr_soc_multi = soc_multi_profile[-1]

    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    print("\n=== Multi-Market Optimization Results ===")
    print(f"Total Revenue (EPEX Daily): £{results_df['Optimised_Revenue_Daily'].sum():,.2f}")
    print(f"Total Revenue (EPEX EFA): £{results_df['Optimised_Revenue_EFA'].sum():,.2f}")
    print(f"Total Revenue (Multi-Market): £{results_df['Optimised_Revenue_Multi'].sum():,.2f}")

    improvement = ((results_df['Optimised_Revenue_Multi'].sum() /
                   results_df['Optimised_Revenue_Daily'].sum() - 1) * 100)
    print(f"Multi-Market Improvement: {improvement:.1f}%")

    # Market usage statistics
    if 'Market_Used_Multi' in results_df.columns:
        market_usage = results_df['Market_Used_Multi'].value_counts()
        print("\n=== Market Usage (Multi-Market Strategy) ===")
        print(market_usage.head(10))

    print("\nOptimization Complete - Multi-market strategies calculated.")
    return results_df

if __name__ == "__main__":
    results_df = run_phase_3_multimarket('Master_BESS_Analysis_Sept_2025.csv')
    results_df.to_csv('Optimized_Results_MultiMarket.csv', index=False)
    print(f"Multi-market optimization results saved to: Optimized_Results_MultiMarket.csv")