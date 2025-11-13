# CSV File Format Guide

## Required CSV Formats for Upload

### SCADA Data CSV

**Required Columns:**
- `timestamp` - Date/time of measurement (will be converted to UTC)
- `power_mw` - Power in MW (positive = discharge/export, negative = charge/import)
- `soc_percent` - State of Charge in percent (0-100)

**Example:**
```csv
timestamp,power_mw,soc_percent
15/10/2025 00:00:00,-1.5,45.2
15/10/2025 00:10:00,-1.8,47.3
15/10/2025 00:20:00,-2.0,49.5
```

**Supported Timestamp Formats:**
- `15/10/2025 00:00:00` (DD/MM/YYYY HH:MM:SS)
- `15-10-2025 00:00` (DD-MM-YYYY HH:MM)
- `2025-10-15 00:00:00` (YYYY-MM-DD HH:MM:SS)
- `2025-10-15T00:00:00` (ISO8601)
- `2025-10-15T00:00:00Z` (ISO8601 with Z)

**Notes:**
- Column names are **case-insensitive** (Timestamp, TIMESTAMP, timestamp all work)
- Leading/trailing spaces in column names are automatically trimmed
- BOM characters are handled automatically
- Timestamps are automatically converted to UTC during processing

---

### Market Price Data CSV

**Required Columns:**
- `timestamp` - Date/time of price (will be converted to UTC)
- `price_gbp_mwh` - Price in GBP per MWh (**negative prices are valid!**)
- `market_type` - Type of price (e.g., day_ahead, imbalance)

**Example:**
```csv
timestamp,price_gbp_mwh,market_type
15/10/2025 00:00:00,45.50,day_ahead
15/10/2025 00:30:00,48.20,day_ahead
15/10/2025 01:00:00,52.10,day_ahead
```

**Market Types:**
- `day_ahead` - Day-ahead market prices (N2EX)
- `imbalance` - Imbalance prices (Balancing Mechanism)
- `intraday` - Intraday market prices (optional)

**Notes:**
- Column names are **case-insensitive**
- Market type values are case-insensitive (day_ahead, Day_Ahead, DAY_AHEAD all work)
- For UK market: Use 30-minute settlement periods (48 periods per day)
- Timestamps must align with SCADA data (will be aligned during processing)

**Understanding Negative Prices:**

Negative electricity prices are **economically valid** and occur when:
- Excess renewable generation (wind/solar) floods the market
- Grid operators need to reduce generation quickly
- Producers pay consumers to take electricity

**Economic Impact on BESS:**
- ✅ **Charging during negative prices**: You get PAID to charge (consume electricity) - excellent opportunity!
- ❌ **Discharging during negative prices**: You PAY to discharge (export electricity) - avoid this!

The optimization algorithm understands negative prices and will:
- Maximize charging when prices are negative (earn money while charging)
- Avoid discharging when prices are negative (prevent paying to export)
- Calculate correct revenue: negative price × positive power (discharge) = negative revenue (cost)

**Price Bounds:**
- Minimum: £-1,000/MWh (configured in market_constraints.yaml)
- Maximum: £6,000/MWh (configured in market_constraints.yaml)
- Prices outside these bounds will be flagged in the Data Quality report

---

## Data Quality Requirements

Your uploaded files must meet these criteria:

### Completeness
- **Minimum 80%** of expected records present
- Gaps ≤60 minutes can be auto-interpolated if completeness ≥95%

### Continuity
- Maximum single gap: 2 hours
- Total gap time: ≤15% of time range

### Bounds
- Power: Within asset power limits (configured per asset)
- SoC: 0-100%
- Prices: Within market constraints (£-1,000 to £6,000/MWh for UK)

### Energy Reconciliation (SCADA only)
- Power integration must match SoC changes within ±5% tolerance
- Helps detect metering issues or data errors

---

## Processing Pipeline

When you upload your CSV files, they go through these steps:

1. **Load & Normalize**
   - Read CSV with BOM handling
   - Lowercase column names
   - Trim whitespace

2. **Validate**
   - Check required columns present
   - Parse timestamps
   - Validate data types

3. **Clean & Resample**
   - SCADA: Resample to 30-minute periods (if at higher frequency)
   - Remove duplicate timestamps
   - Align SCADA and market data timestamps

4. **Quality Check**
   - Calculate Data Quality (DQ) score (4 components)
   - Apply auto-remediation if enabled and DQ < 80%
   - Generate DQ report

5. **Output**
   - Save canonical files with `timestamp_utc` column
   - Files saved to: `data/canonical/`
   - Format: `scada_{asset}_{date}_{timestamp}.csv`

---

## Common Issues & Solutions

### Issue: "Missing required columns"
**Solution:** Check your CSV has columns named `timestamp`, `power_mw`, `soc_percent` (for SCADA) or `timestamp`, `price_gbp_mwh`, `market_type` (for market). Column names are case-insensitive but must be spelled correctly.

### Issue: "Failed to parse timestamps"
**Solution:** Ensure your timestamp format matches one of the supported formats listed above. The most common formats are DD/MM/YYYY HH:MM:SS or ISO8601.

### Issue: "Data quality score too low"
**Solution:**
- Enable auto-remediation to fix small gaps and bounds violations
- Check for large data gaps (>60 minutes)
- Verify SoC values are between 0-100%
- Verify power values are within asset limits

### Issue: "Energy reconciliation failed"
**Solution:** This indicates power measurements don't integrate correctly to SoC changes. Check:
- Power sign convention (positive = discharge, negative = charge)
- SoC meter calibration
- Timestamps alignment

---

## Example Files

Sample CSV files are available in the test data directory:
- `data/raw/Scada_csv.csv` - Example SCADA data
- `data/raw/Market_price_csv.csv` - Example market data

You can use these as templates for your own data.
