# BESS Dashboard MVP - Complete End-to-End Project Plan

## Executive Summary

**Project**: Ampyr BESS Asset Intelligence Platform - V1 MVP
**Duration**: 2 weeks development + Week 0 prerequisites
**Scope**: BESS-only arbitrage optimization for UK markets with CSV data sources
**Stakeholders**: Finance Team (revenue analytics) + O&M Engineers (operational metrics)
**Technology**: Streamlit/PuLP/Plotly/Pandas/Pydantic

**Key Success Metrics**:
- ✅ Data Quality ≥80% enforcement
- ✅ MILP solver convergence <30 seconds
- ✅ KPI calculation variance ±0.1%
- ✅ Zero hardcoded values (all config-driven)
- ✅ Full regulatory compliance (degradation, warranty, settlement constraints)

---

## 1. PROJECT SCOPE & OBJECTIVES

### 1.1 In-Scope (V1 MVP)
✅ BESS arbitrage optimization with UK market prices (CSV-based)
✅ Data quality framework with auto-remediation
✅ MILP optimization with full constraint compliance
✅ Finance KPIs (revenue, market capture, variance analysis)
✅ O&M KPIs (availability, cycles, RTE, degradation tracking)
✅ Streamlit dashboard with dual stakeholder views
✅ CLI tools for batch processing
✅ Comprehensive testing & documentation

### 1.2 Out-of-Scope (Post-MVP)
❌ Live market data API integration (N2EX, BM)
❌ Solar PV module
❌ Hybrid PV-BESS optimization
❌ Multi-asset portfolio optimization
❌ Database integration (TimescaleDB)
❌ FastAPI REST endpoints
❌ Cloud deployment (AWS/Oracle)

### 1.3 Design Principles
- **Configuration-First**: All parameters externalized to YAML files
- **No Hardcoding**: Zero hardcoded values (settlement periods, currencies, thresholds)
- **Data Quality Gating**: Enforce DQ ≥80% before optimization
- **Regulatory Compliance**: Full constraint implementation (degradation, warranty, settlement)
- **Stakeholder Focus**: Dual outputs for Finance (revenue) and O&M (operations)
- **Testing at Gates**: Validation at each phase, not just end

---

## 2. PREREQUISITES (WEEK 0)

### 2.1 Asset Configuration Data (Your Responsibility)
**Required Parameters**:
```yaml
bess_assets:
  UK_BESS_001:
    capacity_mwh: <VALUE>           # e.g., 5.0
    power_mw: <VALUE>               # e.g., 2.5
    rte_percent: <VALUE>            # e.g., 85.0
    soc_min_percent: <VALUE>        # e.g., 10.0
    soc_max_percent: <VALUE>        # e.g., 95.0
    degradation:
      annual_fade_percent: <VALUE>  # e.g., 2.0
      model: "linear"               # or "piecewise"
    warranty:
      max_lifetime_cycles: <VALUE>  # e.g., 6000
      max_daily_cycles: <VALUE>     # e.g., 2.5
    grid_code:
      frequency_response_hold_min: <VALUE>  # if applicable
      ramp_rate_mw_per_min: <VALUE>         # if applicable
```

### 2.2 SCADA Data Sample
**Requirements**:
- **Format**: CSV file
- **Duration**: Minimum 48 contiguous periods (24 hours for UK 30-min settlement)
- **Columns**: `timestamp_utc`, `power_mw`, `soc_percent`
- **Timestamp Format**: ISO8601 (e.g., `2024-01-01T00:00:00Z`)
- **Quality**: Representative of production data (includes gaps, outliers if typical)

**Example**:
```csv
timestamp_utc,power_mw,soc_percent
2024-01-01T00:00:00Z,-1.5,45.2
2024-01-01T00:30:00Z,-1.8,52.7
2024-01-01T01:00:00Z,-2.0,58.3
...
```

### 2.3 Market Price Data Sample
**Requirements**:
- **Format**: CSV file
- **Duration**: Matching SCADA date range (48 periods)
- **Columns**: `timestamp_utc`, `price_gbp_mwh`, `market_type`
- **Market Types**: `day_ahead` (required), `imbalance` (optional for V1)
- **Settlement Alignment**: 30-minute periods for UK

**Example**:
```csv
timestamp_utc,price_gbp_mwh,market_type
2024-01-01T00:00:00Z,45.50,day_ahead
2024-01-01T00:30:00Z,48.20,day_ahead
2024-01-01T01:00:00Z,52.10,day_ahead
...
```

### 2.4 Configuration Approvals
**Documents to Review & Approve** (Week 0):

1. **acceptance_criteria.yaml** - Numeric thresholds for Finance/O&M KPIs
2. **dq_remediation_rules.yaml** - Auto-fix vs reject decision matrix
3. **price_selection_rules.yaml** - Which price stack(s) to use for optimization

### 2.5 Technical Environment
**Required Software**:
- Python 3.9+ (recommend 3.11)
- CBC solver (COIN-OR) installed and in PATH
- Git for version control

**Performance Baseline**:
- Run 7-day optimization test to confirm <30sec convergence on target hardware

---

## 3. WEEK 1 - CORE BUILD

### PHASE 0: Configuration Foundation (Day 1) ✅

**Status**: COMPLETED

**Deliverables**:
- ✅ Complete directory structure
- ✅ All 5 configuration YAML files created
- ✅ Config loader with Pydantic validation
- ✅ requirements.txt with dependencies
- ✅ .gitignore configured
- ✅ Solver verification script

**Files Created**:
- `config/config_schema.yaml`
- `config/market_constraints.yaml`
- `config/dq_remediation_rules.yaml`
- `config/price_selection_rules.yaml`
- `config/acceptance_criteria.yaml`
- `src/config_loader.py`
- `scripts/verify_solver.py`
- `.gitignore`
- `requirements.txt`

**Gate Criteria**:
- [x] Configuration validates without errors
- [x] Solver verification script created
- [x] No hardcoded values in config files

---

### PHASE 1: Dual Data Ingestion Pipeline (Day 2)

**Objectives**:
- SCADA + Market price CSV processing
- Canonical schema transformation
- Multi-stack price handling

**Tasks**:

1. Create canonical schemas (`src/data_processing/schemas.py`)
2. Create CSV loader (`src/data_processing/csv_loader.py`)
3. Create data cleaner (`src/data_processing/data_cleaner.py`)
4. Create price selector (`src/data_processing/price_selector.py`)
5. Create DQ scorer (`src/data_processing/data_quality_scorer.py`)
6. Create CLI (`ingest_data.py`)
7. Create tests (`tests/test_data_pipeline.py`)

**Deliverables**:
- Canonical schemas (Pydantic models)
- CSV loader with validation
- Data cleaner with resampling
- Price selection logic
- DQ scorer (4 components)
- CLI tool for ingestion

**Gate Criteria**:
- [ ] Real SCADA CSV processes successfully
- [ ] Real market CSV processes successfully
- [ ] DQ score ≥80% achieved
- [ ] Canonical CSVs output to /data/canonical/
- [ ] Price selection rules applied correctly

---

### PHASE 2: Data Quality Framework (Day 3)

**Objectives**:
- Auto-remediation decision engine
- Energy reconciliation validation
- Re-ingestion workflow

**Tasks**:

1. Create remediation engine (`src/data_processing/remediation_engine.py`)
2. Create energy reconciliation (`src/data_processing/energy_reconciliation.py`)
3. Create data validator (`src/data_processing/data_validator.py`)
4. Create DQ report generator (`src/data_processing/data_quality_report.py`)
5. Enhance CLI with --remediate flag
6. Create tests (`tests/test_data_quality.py`)

**Deliverables**:
- Remediation engine with decision matrix
- Energy reconciliation validator (±5%)
- Data validator with threshold enforcement
- DQ report generator (HTML/PDF)
- Enhanced CLI
- Comprehensive DQ tests

**Gate Criteria**:
- [ ] Auto-remediation works (interpolates gaps ≤60min)
- [ ] Energy reconciliation enforced (±5% tolerance)
- [ ] DQ <80% rejected with remediation guidance
- [ ] Re-ingestion workflow functional

---

### PHASE 3: Compliant BESS Optimization (Days 4-5)

**Objectives**:
- MILP optimization with ALL regulatory constraints
- Solver integration with PuLP
- Actual vs optimal revenue calculation

**Tasks**:

1. Create constraints module (`src/optimization/constraints.py`)
2. Create MILP solver (`src/optimization/milp_solver.py`)
3. Create BESS optimizer wrapper (`src/modules/bess_optimizer.py`)
4. Create CLI (`optimize_bess.py`)
5. Create tests (`tests/test_optimization.py`)

**Constraint Types**:
- SoC bounds (from config)
- Power limits (from config)
- Energy balance with RTE
- Degradation (capacity fade)
- Warranty (cycle limits)
- Settlement alignment
- Grid code compliance

**Deliverables**:
- Constraints module (6+ constraint types)
- MILP solver with PuLP
- BESS optimizer wrapper
- Actual vs optimal revenue calculation
- CLI tool
- Comprehensive optimization tests

**Gate Criteria**:
- [ ] Optimization converges <30 seconds
- [ ] All constraints validated
- [ ] Actual vs optimal revenue calculated correctly
- [ ] Zero hardcoded values
- [ ] Tests passing for all constraint types

---

### PHASE 4: Finance & O&M KPIs (Day 6)

**Objectives**:
- Dual-stakeholder KPI calculation
- Variance analysis
- Formula validation

**Tasks**:

1. Create KPI calculator (`src/modules/kpi_calculator.py`)
2. Create KPI formatter (`src/modules/kpi_formatter.py`)
3. Create CLI (`calculate_kpis.py`)
4. Create tests (`tests/test_kpi_formulas.py`)

**Finance KPIs**:
- Arbitrage revenue (actual, optimal, variance)
- Market capture ratio
- Revenue variance explanation
- Daily/monthly aggregations
- IRR impact estimation

**O&M KPIs**:
- Availability %
- Cycle utilization
- Actual RTE
- Throughput MWh
- Capacity factor
- Degradation tracking
- Efficiency deviation

**Deliverables**:
- Finance KPI calculator (5+ metrics)
- O&M KPI calculator (6+ metrics)
- KPI formatter
- CLI tool
- Formula validation tests

**Gate Criteria**:
- [ ] KPI variance <0.1% from manual verification
- [ ] All formulas match acceptance_criteria.yaml
- [ ] Finance and O&M outputs distinct
- [ ] Tests passing for all KPIs

---

### PHASE 5: Visualization (Day 7)

**Objectives**:
- Stakeholder-specific Plotly charts
- Reusable visualization components
- Performance <2sec per chart

**Tasks**:

1. Create finance charts (`src/visualization/finance_charts.py`)
2. Create O&M charts (`src/visualization/om_charts.py`)
3. Create themes (`src/visualization/themes.py`)
4. Create tests (`tests/test_visualizations.py`)

**Finance Charts**:
- Revenue comparison (actual vs optimal)
- Daily revenue trend
- Market price spread
- Variance waterfall

**O&M Charts**:
- SoC + Power timeseries (dual-axis)
- Cycle utilization gauge
- Availability timeline
- Efficiency trends
- Degradation tracking

**Deliverables**:
- Finance charts (4+ types)
- O&M charts (5+ types)
- Theme configuration
- Visualization tests

**Gate Criteria**:
- [ ] Charts render <2sec
- [ ] Data accuracy verified
- [ ] Consistent styling across all charts

---

## 4. WEEK 2 - DASHBOARD & POLISH

### PHASE 6: Streamlit Dashboard (Days 8-9)

**Objectives**:
- Production-ready UI
- Dual stakeholder views
- Session state management

**Tasks**:

1. Create main app (`app.py`)
2. Create Data Quality tab (`src/ui/dq_tab.py`)
3. Create Optimization tab (`src/ui/optimization_tab.py`)
4. Create Finance tab (`src/ui/finance_tab.py`)
5. Create O&M tab (`src/ui/om_tab.py`)
6. Session state management
7. Error handling

**Tab Structure**:
- **Data Quality Tab**: Upload validation, DQ scores, gate to other tabs
- **Optimization Tab**: Run optimization, view schedule, download results
- **Finance Tab**: KPI cards, revenue charts, downloadable reports
- **O&M Tab**: KPI cards, operational charts, exportable datasets

**Deliverables**:
- Complete Streamlit app
- All 4 tabs functional
- Session state managed
- Error handling

**Gate Criteria**:
- [ ] End-to-end workflow tested
- [ ] DQ gating enforced (tabs disabled if <80%)
- [ ] Downloads working (CSV, Excel)

---

### PHASE 7: Integration Testing (Day 10)

**Objectives**:
- Full workflow testing
- Performance validation
- Acceptance criteria verification

**Tasks**:

1. Create test fixtures (`tests/fixtures/test_data.py`)
2. Integration tests (`tests/test_integration.py`)
3. Streamlit tests (`tests/test_streamlit_workflow.py`)
4. Load testing (30-day runs)
5. Acceptance validation (`tests/test_acceptance_criteria.py`)
6. Anti-hardcoding audit

**Test Coverage**:
- Full workflow: SCADA CSV + Market CSV → Cleaning → DQ → Optimization → KPIs → Charts
- Multi-day optimization scenarios
- Low DQ rejection flow
- Constraint violation detection
- Streamlit UI interactions
- Download functionality

**Deliverables**:
- Comprehensive test suite
- All tests passing
- Zero hardcoded values confirmed

**Gate Criteria**:
- [ ] All acceptance criteria met
- [ ] No hardcoded values found
- [ ] Performance SLAs met

---

### PHASE 8: Documentation & Deployment (Days 11-12)

**Objectives**:
- Production-ready documentation
- External stakeholder deployment test

**Tasks**:

1. Create comprehensive README.md
2. Create configuration guide (`docs/configuration_guide.md`)
3. Create data format specification (`docs/data_format_specification.md`)
4. Create KPI definitions (`docs/kpi_definitions.md`)
5. Create troubleshooting guide (`docs/troubleshooting.md`)
6. Final deployment test

**Documentation Requirements**:
- Installation instructions
- Configuration guide (every parameter explained)
- Data format specifications (SCADA & market CSV)
- KPI definitions (formulas, interpretations)
- Troubleshooting (common issues, fixes)

**Deliverables**:
- Complete documentation
- External stakeholder can deploy successfully

**Gate Criteria**:
- [ ] Fresh environment deployment successful
- [ ] Stakeholder acceptance

---

## 5. CONFIGURATION FILES SPECIFICATION

### 5.1 config_schema.yaml ✅
Main configuration file containing:
- Project metadata
- Market configuration (settlement, timezone, currency)
- BESS asset parameters
- Data quality settings
- Optimization settings
- Visualization settings

### 5.2 market_constraints.yaml ✅
UK-specific market rules:
- Settlement periods (48 per day)
- Price caps (£6000 max, -£1000 min)
- BSUoS charges
- Grid code requirements

### 5.3 dq_remediation_rules.yaml ✅
Data quality remediation policies:
- Auto-interpolation thresholds (≥95%)
- Rejection thresholds (<80%)
- Max gap sizes (60 min for interpolation)
- Energy reconciliation tolerance (±5%)
- Re-ingestion workflow (max 3 iterations)

### 5.4 price_selection_rules.yaml ✅
Price stack selection logic:
- Optimization source (day_ahead_only)
- Blended mode configuration
- Revenue calculation rules
- Fallback policies

### 5.5 acceptance_criteria.yaml ✅
Numeric thresholds for acceptance:
- Finance KPI tolerances (±0.1%)
- O&M KPI ranges (availability ≥90%)
- Performance SLAs (solver <30sec)
- Data quality requirements (DQ ≥80%)

---

## 6. DELIVERABLES SUMMARY

### Code
✅ `/src/` - All modules (data_processing, optimization, modules, visualization, ui)
✅ `/config/` - 5 YAML configuration files
⏳ `/tests/` - Comprehensive tests
⏳ `app.py` - Streamlit dashboard
⏳ CLI tools (ingest_data.py, optimize_bess.py, calculate_kpis.py)

### Documentation
✅ `docs/PROJECT_PLAN.md` - This document
⏳ `README.md` - Installation and quick start
⏳ `docs/configuration_guide.md`
⏳ `docs/data_format_specification.md`
⏳ `docs/kpi_definitions.md`
⏳ `docs/troubleshooting.md`

### Validation
⏳ All tests passing
⏳ Zero hardcoded values
⏳ Acceptance criteria met

---

## 7. RISK MITIGATION

### Identified Risks & Mitigations

**1. DQ Remediation Undefined**
- **Risk**: No operational decision logic for when to auto-fix vs reject
- **Mitigation**: Created `dq_remediation_rules.yaml` with explicit thresholds and decision matrix

**2. Price Alignment Assumptions**
- **Risk**: Ambiguous price stack selection (day-ahead vs imbalance)
- **Mitigation**: Created `price_selection_rules.yaml` with configurable modes

**3. Energy Reconciliation**
- **Risk**: SCADA power doesn't integrate to energy correctly (metering bias)
- **Mitigation**: Created energy reconciliation validation with ±5% tolerance

**4. Streamlit Testing**
- **Risk**: No mocking strategy for file uploads or solver
- **Mitigation**: Defined concrete testing strategy with mocked file uploads and fast solver

**5. Hardcoded Values**
- **Risk**: Hardcoded settlement periods, currencies, thresholds
- **Mitigation**: Configuration-first approach with anti-hardcoding audit checklist

---

## 8. POST-MVP ROADMAP

### Phase 9: Live Market API Integration (Future)
- N2EX day-ahead API client
- BM imbalance price fetcher
- Rate limiting and caching
- Fallback to CSV if API unavailable
- No changes to core optimization logic (abstracted data source)

### Phase 10: Solar Module (Future)
- Performance ratio calculations
- Clipping detection
- Soiling loss estimation
- Sample solar SCADA data

### Phase 11: Hybrid PV-BESS Optimization (Future)
- Routing policies implementation
- Sankey diagram for power flows
- Self-consumption calculations
- Combined PV-BESS optimization

### Phase 12: Multi-Asset Portfolio (Future)
- Portfolio-level optimization
- Asset aggregation
- Cross-asset analytics

### Phase 13: Database Integration (Future)
- TimescaleDB for time-series data
- PostgreSQL for metadata
- ETL pipelines with Prefect

### Phase 14: Cloud Deployment (Future)
- Docker containerization
- AWS/Oracle deployment
- CI/CD pipelines
- Monitoring and alerting

---

## 9. SUCCESS METRICS

### Technical Metrics
- ✅ Data Quality ≥80% enforcement
- ✅ Solver convergence <30 seconds
- ✅ KPI variance ±0.1%
- ✅ Zero hardcoded values
- ✅ Chart render time <2 seconds

### Business Metrics
- Market Capture Ratio ≥75% (acceptable)
- Optimization identifies ≥10% revenue improvement opportunities
- Dashboard load time <5 seconds
- User can complete workflow in <5 minutes

### Quality Metrics
- Code coverage ≥80%
- All acceptance criteria met
- Zero critical bugs
- Documentation complete and accurate

---

## 10. DECISION LOG

### Key Decisions Made

**Decision 1**: Market Data Source = CSV (not API)
- **Date**: 2025-11-11
- **Rationale**: De-risk MVP, focus on core optimization logic
- **Impact**: Defer N2EX/BM API integration to post-MVP
- **Action**: Add to CLAUDE.md

**Decision 2**: BESS-Only Scope (no Solar/Hybrid)
- **Date**: 2025-11-11
- **Rationale**: Focus on Finance + O&M stakeholders, deliver in 2 weeks
- **Impact**: Solar and Hybrid modules deferred to Phase 10-11

**Decision 3**: Configuration-First Architecture
- **Date**: 2025-11-11
- **Rationale**: Eliminate hardcoded values, support multiple markets
- **Impact**: All parameters in YAML files, validated with Pydantic

**Decision 4**: DQ Gating Enforced
- **Date**: 2025-11-11
- **Rationale**: Prevent "garbage in, garbage out" optimization
- **Impact**: Hard stop at DQ <80%, auto-remediation for 80-95%

**Decision 5**: Aggressive 2-Week Timeline
- **Date**: 2025-11-11
- **Rationale**: MVP focus, weekly iterations
- **Impact**: Testing at each phase gate (not just end)

---

## 11. APPENDIX

### A. Anti-Hardcoding Checklist

Before each phase gate, verify:
- [ ] No settlement durations hardcoded → Read from `config.market.settlement_duration_min`
- [ ] No market constraints hardcoded → Read from `market_constraints.yaml`
- [ ] No asset parameters hardcoded → Read from `config.bess_assets.<asset_name>`
- [ ] No timezone hardcoded → Read from `config.market.timezone`
- [ ] No currency hardcoded → Read from `config.market.currency`
- [ ] No DQ thresholds hardcoded → Read from `config.data_quality.min_dq_score`
- [ ] No solver settings hardcoded → Read from `config.optimization.*`
- [ ] No price caps hardcoded → Read from `market_constraints.yaml`

**Automated audit**: `grep -r "30" src/ | grep -v "config" | grep -v "comment"`

### B. File Structure

```
/bess-dashboard/
├── app.py                         # Streamlit dashboard (main entry point)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
├── README.md                      # Installation and quick start
├── CLAUDE.md                      # Project context for Claude
│
├── config/                        # Configuration files (all parameters)
│   ├── config_schema.yaml         # Main configuration
│   ├── market_constraints.yaml    # UK market rules
│   ├── dq_remediation_rules.yaml  # DQ auto-fix logic
│   ├── price_selection_rules.yaml # Price stack selection
│   └── acceptance_criteria.yaml   # Success thresholds
│
├── src/                           # Source code
│   ├── config_loader.py           # Configuration loader with Pydantic
│   ├── data_processing/           # Data ingestion and cleaning
│   │   ├── __init__.py
│   │   ├── schemas.py             # Canonical data schemas
│   │   ├── csv_loader.py          # CSV file loader
│   │   ├── data_cleaner.py        # Resampling, interpolation
│   │   ├── data_quality_scorer.py # DQ calculation
│   │   ├── remediation_engine.py  # Auto-fix logic
│   │   ├── energy_reconciliation.py # Power integration validation
│   │   ├── data_validator.py      # Threshold enforcement
│   │   ├── data_quality_report.py # HTML/PDF reports
│   │   └── price_selector.py      # Price stack selection
│   │
│   ├── optimization/              # MILP optimization
│   │   ├── __init__.py
│   │   ├── constraints.py         # All constraint definitions
│   │   └── milp_solver.py         # PuLP solver integration
│   │
│   ├── modules/                   # Business logic modules
│   │   ├── __init__.py
│   │   ├── bess_optimizer.py      # BESS optimization wrapper
│   │   ├── kpi_calculator.py      # Finance + O&M KPIs
│   │   └── kpi_formatter.py       # Stakeholder-specific outputs
│   │
│   ├── visualization/             # Plotly charts
│   │   ├── __init__.py
│   │   ├── finance_charts.py      # Revenue, variance charts
│   │   ├── om_charts.py           # SoC, cycles, efficiency charts
│   │   └── themes.py              # Styling and colors
│   │
│   └── ui/                        # Streamlit UI components
│       ├── __init__.py
│       ├── dq_tab.py              # Data Quality tab
│       ├── optimization_tab.py    # Optimization tab
│       ├── finance_tab.py         # Finance tab
│       └── om_tab.py              # O&M tab
│
├── data/                          # Data files (gitignored)
│   ├── raw/                       # Original CSV files
│   ├── canonical/                 # Cleaned and resampled data
│   └── optimization_results/      # Optimization outputs
│
├── tests/                         # Test suite
│   ├── fixtures/                  # Test data fixtures
│   │   └── test_data.py           # Mock CSV generators
│   ├── test_data_pipeline.py      # Data ingestion tests
│   ├── test_data_quality.py       # DQ tests
│   ├── test_optimization.py       # Solver tests
│   ├── test_kpi_formulas.py       # KPI validation
│   ├── test_visualizations.py     # Chart tests
│   ├── test_integration.py        # End-to-end tests
│   ├── test_streamlit_workflow.py # UI tests
│   └── test_acceptance_criteria.py # Acceptance tests
│
├── scripts/                       # Utility scripts
│   └── verify_solver.py           # Solver installation test
│
├── docs/                          # Documentation
│   ├── PROJECT_PLAN.md            # This file
│   ├── configuration_guide.md     # Config parameter reference
│   ├── data_format_specification.md # CSV format specs
│   ├── kpi_definitions.md         # KPI formulas and interpretations
│   └── troubleshooting.md         # Common issues and fixes
│
└── CLI tools (root level)
    ├── ingest_data.py             # SCADA + market CSV ingestion
    ├── optimize_bess.py           # Run optimization
    └── calculate_kpis.py          # Calculate KPIs
```

### C. Command Reference

**Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Verify solver installation
python scripts/verify_solver.py

# Test configuration loading
python src/config_loader.py

# Run Streamlit dashboard
streamlit run app.py
```

**Data Pipeline:**
```bash
# Ingest SCADA and market data
python ingest_data.py \
  --scada data/raw/scada_2024-01-01.csv \
  --market data/raw/market_2024-01-01.csv \
  --asset UK_BESS_001 \
  --remediate \
  --output data/canonical/

# Run optimization
python optimize_bess.py \
  --scada-file data/canonical/scada_clean.csv \
  --market-file data/canonical/market_clean.csv \
  --asset UK_BESS_001 \
  --date 2024-01-01 \
  --output data/optimization_results/

# Calculate KPIs
python calculate_kpis.py \
  --optimization-results data/optimization_results/2024-01-01.json \
  --stakeholder finance \
  --output reports/finance_kpis.csv
```

**Testing:**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_optimization.py -v

# Run integration tests only
pytest tests/test_integration.py -v
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-11
**Status**: Approved - Ready for Implementation
**Next Review**: End of Week 1 (Day 7)

