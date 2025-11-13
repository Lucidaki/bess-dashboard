# PROJECT CONTEXT & CORE DIRECTIVES

## Project Overview
**Ampyr Asset Intelligence Platform** - Multi-asset performance management and revenue optimization platform for Solar PV, BESS, and hybrid renewable energy projects across global markets (Europe, USA, UK, Australia)

**Technology Stack (V1 MVP)**: Streamlit/FastAPI/PuLP/Plotly/CSV
**Technology Stack (V2+)**: React/FastAPI Microservices/PostgreSQL+TimescaleDB/Parquet/S3/Prefect
**Architecture**: Modular microservice architecture with hot-swappable components
**Deployment**: Local (V1 MVP) → Cloud AWS/Oracle (V2+)

## SYSTEM-LEVEL OPERATING PRINCIPLES

### Core Implementation Philosophy
- DIRECT IMPLEMENTATION ONLY: Generate complete, working code for energy optimization and analysis
- NO PARTIAL IMPLEMENTATIONS: Eliminate mocks, stubs, TODOs, or placeholder functions in optimization algorithms
- SOLUTION-FIRST THINKING: Think at SYSTEM level for multi-asset optimization, then linearize into actionable strategies
- TOKEN OPTIMIZATION: Focus on energy arbitrage calculations and KPI implementations
- MODULAR EXCELLENCE: Each module operates independently with stable APIs and consistent data contracts

### Multi-Dimensional Analysis Framework
When encountering energy optimization requirements:
1. **Observer 1**: MILP optimization feasibility and constraint modeling
2. **Observer 2**: Data quality and temporal resolution handling
3. **Observer 3**: Revenue maximization and market capture opportunities
4. **Observer 4**: PV-BESS hybrid routing and energy flow optimization
5. **Synthesis**: Unified implementation with canonical data schemas

## ANTI-PATTERN ELIMINATION

### Prohibited Implementation Patterns
- Incomplete optimization models without full constraint sets
- Mock SCADA data or placeholder market prices
- Partial KPI calculations missing critical metrics
- Deferred data cleaning or normalization steps
- Incomplete hybrid routing logic implementations

### Prohibited Communication Patterns
- Vague references to "typical BESS performance"
- Generic energy market explanations without specific context
- Hedging on optimization results without confidence intervals
- Excessive discussion of theoretical vs practical implementation
- Omitting data quality score impacts on results

### Null Space Pattern Exclusion
Eliminate patterns that consume tokens without advancing implementation:
- Generic renewable energy background information
- Historical context about energy markets unless directly relevant
- Multiple solver options without performance benchmarks
- Theoretical optimization approaches without practical application

## DYNAMIC MODE ADAPTATION

### Context-Driven Behavior Switching

**DATA INGESTION MODE** (Triggered by raw SCADA/market data)
- Canonical schema transformation (UTC timestamps, MW power, % SoC)
- Resampling policy application (10/15/60-min → 30-min)
- Data quality scoring and validation
- Timestamp continuity and bounds checking

**OPTIMIZATION MODE** (Triggered by arbitrage analysis requests)
- MILP solver configuration with full constraint sets
- RTE, cycle limits, import/export caps implementation
- Day-ahead and intraday price integration
- Actual vs optimal revenue computation

**HYBRID ANALYSIS MODE** (Triggered by PV-BESS interaction queries)
- Routing policy implementation (PV Surplus/Grid Import/Export Priority)
- Power flow visualization preparation
- Self-consumption and curtailment avoidance calculations
- Sankey diagram data structuring

**REPORTING MODE** (Triggered by stakeholder report requests)
- KPI aggregation across assets and timeframes
- Performance ratio and capacity factor calculations
- Market capture ratio and arbitrage efficiency metrics
- Automated insight generation with AI agents

## PROJECT-SPECIFIC GUIDELINES

### Essential Commands

#### Development (V1 MVP)
```bash
# Data Ingestion Pipeline
python ingest_data.py --scada data/raw/Scada_csv.csv --market data/raw/Market_price_csv.csv --asset UK_BESS_001 --remediate

# BESS Optimization
python optimize_bess.py --scada-file data/canonical/scada_UK_BESS_001_2025-10-14.csv --market-file data/canonical/market_UK_BESS_001_2025-10-14.csv --asset UK_BESS_001

# KPI Calculation
python calculate_kpis.py --summary-file data/optimization_results/summary_UK_BESS_001_2025-10-14.json --schedule-file data/optimization_results/schedule_UK_BESS_001_2025-10-14.csv --stakeholder both

# Streamlit Dashboard (✅ Complete with CSV upload integration)
streamlit run app.py
```

#### Database (V2+)
```bash
python migrate_timescale.py  # Initialize TimescaleDB hypertables
python etl/ingest_scada.py --source api --asset all  # Ingest SCADA data
python etl/market_prices.py --market epex --interval 30min  # Fetch market data
```

#### Testing
```bash
pytest tests/test_optimization.py -v  # Test MILP solver
pytest tests/test_data_quality.py -v  # Test DQ scoring
pytest tests/test_hybrid_routing.py -v  # Test routing policies
```

### File Structure & Boundaries
**SAFE TO MODIFY**:
- `/src/modules/` - Core functional modules (Solar, BESS, Hybrid, AI, KPI)
- `/src/optimization/` - MILP solver and constraint definitions
- `/src/data_processing/` - Cleaning and normalization pipelines
- `/src/visualization/` - Plotly charts and Sankey diagrams
- `/src/api/` - FastAPI endpoints
- `/config/` - Asset configurations and market parameters
- `/tests/` - Unit and integration tests

**NEVER MODIFY**:
- `/data/raw/` - Original SCADA and market data
- `/data/canonical/` - Processed canonical datasets
- `/models/trained/` - Trained AI models
- `/lib/solvers/` - HiGHS/PuLP solver binaries
- `/.env` - API keys and database credentials

### Code Style & Architecture Standards

**Naming Conventions**:
- Variables: snake_case (e.g., `soc_percentage`, `arbitrage_revenue`)
- Functions: snake_case with descriptive verbs (e.g., `calculate_performance_ratio()`)
- Classes: PascalCase (e.g., `BESSOptimizer`, `HybridRouter`)
- Constants: SCREAMING_SNAKE_CASE (e.g., `MAX_CYCLES_PER_DAY`, `RTE_EFFICIENCY`)
- Files: snake_case (e.g., `bess_module.py`, `data_quality_scorer.py`)

**Architecture Patterns**:
- Microservice modules with versioned APIs
- Event-driven data pipelines with Prefect/Airflow
- Repository pattern for data access
- Strategy pattern for routing policies
- Observer pattern for KPI updates

**Data Standards**:
- **Power Convention**: Positive = discharge/export, Negative = charge/import
- **Time Zone**: All timestamps in UTC (ISO8601)
- **Energy Units**: MW for power, MWh for energy
- **Currency**: GBP/MWh for UK, EUR/MWh for EU, USD/MWh for US
- **Efficiency**: Percentage format (0-100)
- **Price Bounds**: -£1,000 to £6,000/MWh (UK) - **Negative prices are economically valid**
  - Negative prices occur when excess renewable generation floods the market
  - BESS gets PAID to charge during negative prices (excellent opportunity)
  - BESS PAYS to discharge during negative prices (avoid this)
  - Validation: Use `market_constraints.yaml` price caps, NOT hard-coded positive checks

**Critical Data Structure Patterns**:
- **Pydantic v2**: Always use `.model_dump()` for serialization, NOT `.to_dict()`
- **ConfigLoader**: Use `load_all_configs()` then access dict, NOT individual getters
- **Optimization Summary**: Flat structure matching CLI tools (optimize_bess.py format)
  - Top-level: `actual_revenue_gbp`, `optimal_revenue_gbp`, `cycles_used`, etc.
  - Nested: `actual_performance` dict for detailed metrics
  - NOT nested: ~~`actual['revenue_gbp']`~~ or ~~`optimal['revenue_gbp']`~~
- **DQ Reports**: Use schema fields from `schemas.py`
  - Top-level metadata: `valid_periods`, `total_periods`, `missing_periods`
  - Component details from `issues` lists, NOT non-existent nested fields
- **BESSOptimizer**: Parameter is `solver_timeout_sec` NOT `timeout`
- **CSV Column Names**: Input requires `timestamp`, system converts to `timestamp_utc` internally

## TOOL CALL OPTIMIZATION

### Batching Strategy
Group operations by:
- **Data Pipeline**: Raw ingestion → Cleaning → Normalization → Storage
- **Optimization Runs**: Constraint setup → Solver execution → Results extraction
- **KPI Calculations**: Asset-level → Portfolio aggregation → Reporting
- **Visualization**: Data preparation → Chart generation → Export

### Parallel Execution Identification
Execute simultaneously when operations:
- Process different assets independently
- Calculate non-dependent KPIs
- Generate visualizations for multiple timeframes
- Run optimization scenarios with different parameters

## QUALITY ASSURANCE METRICS

### Success Indicators
- ✅ DQ Score ≥ 80% for all optimization inputs
- ✅ MILP solver convergence within 30 seconds
- ✅ Market Capture Ratio calculation accuracy ±0.1%
- ✅ Complete PV-BESS power flow traceability
- ✅ All canonical schema validations passing
- ✅ Zero data loss in resampling operations

### Failure Recognition
- ❌ Optimization without complete constraint validation
- ❌ KPI calculations on low-quality data (DQ < 80%)
- ❌ Missing timestamp continuity checks
- ❌ Incomplete hybrid routing logic
- ❌ Revenue calculations without market price validation
- ❌ Solver timeout without fallback strategy

## METACOGNITIVE PROCESSING

### Optimization Pattern Recognition
1. **Constraint Hierarchy**: Hard constraints (physics) → Soft constraints (economics)
2. **Solver Tuning**: Gap tolerance vs computation time tradeoffs
3. **Data Quality Impact**: Quantify revenue uncertainty from DQ scores
4. **Market Regime Detection**: Identify arbitrage opportunity patterns
5. **Hybrid Coordination**: Optimize PV-BESS interaction patterns

### Energy Market Context Awareness
- Track day-ahead vs intraday price spreads
- Monitor ancillary service opportunity windows
- Identify seasonal patterns in PV generation
- Recognize BESS degradation impact on optimization
- Adapt to market rule changes and grid codes

## MODULE-SPECIFIC IMPLEMENTATIONS

### Solar Module
```python
# Key functions to implement
def calculate_performance_ratio(ac_power, irradiance, capacity):
    """PR = AC Energy / (Irradiance × Nominal Capacity × Time)"""
    pass

def detect_clipping(dc_power, ac_power, inverter_capacity):
    """Identify periods where DC > AC due to inverter limits"""
    pass

def calculate_soiling_losses(expected_pr, actual_pr):
    """Soiling Factor = Actual PR / Expected PR"""
    pass
```

### BESS Module
```python
# Core optimization constraints
def setup_milp_constraints(model, params):
    """
    - SoC bounds: soc_min <= soc[t] <= soc_max
    - Power limits: -p_charge_max <= power[t] <= p_discharge_max
    - Energy balance: soc[t+1] = soc[t] - power[t] * dt / capacity
    - RTE losses: discharge_energy = charge_energy * rte
    - Cycle limits: daily_cycles <= max_cycles_per_day
    """
    pass

def calculate_arbitrage_revenue(schedule, prices):
    """Revenue = Σ(power[t] × price[t] × dt) for all t"""
    pass
```

### Hybrid Module
```python
# Routing policies
def route_pv_surplus_only(pv_generation, grid_demand, bess_soc):
    """BESS charges only from excess PV after grid demand"""
    pass

def route_grid_import_allowed(pv_generation, prices, bess_constraints):
    """Strategic charging based on price spreads"""
    pass

def calculate_self_consumption(pv_to_bess, total_pv):
    """Self-Consumption = PV→BESS Energy / Total PV Generation"""
    pass
```

## TESTING & VALIDATION PROTOCOLS

### Unit Testing Requirements
```python
# Test optimization accuracy
def test_milp_solver_convergence():
    """Verify solver finds optimal solution for known scenarios"""
    
def test_data_quality_scoring():
    """Validate DQ score calculation with edge cases"""
    
def test_kpi_calculations():
    """Ensure KPI formulas match industry standards"""
```

### Integration Testing
- End-to-end data pipeline from SCADA to KPI dashboard
- Multi-asset portfolio optimization scenarios
- Hybrid routing under various PV generation profiles
- Market price API integration and fallback handling

### Validation Metrics
- Optimization gap < 1% from theoretical optimum
- KPI calculation variance < 0.1% from manual verification
- Data pipeline latency < 5 minutes for hourly updates
- Visualization render time < 2 seconds per chart

## DEPLOYMENT & OPERATIONS

### V1 MVP Deployment
```bash
# Local Streamlit deployment
streamlit run app.py --server.port 8501 --server.address localhost

# Data refresh
python scripts/refresh_data.py --date today --assets all
```

### V2 Platform Deployment
```bash
# Docker compose for microservices
docker-compose up -d

# Database migrations
alembic upgrade head

# ETL pipeline scheduling
prefect deployment create --name daily-ingestion
```

### Monitoring & Alerting
- Data quality below threshold (DQ < 80%)
- Optimization solver timeout or non-convergence
- Market data feed interruption
- Anomalous KPI deviations (>3σ from mean)
- Storage capacity warnings (>80% full)

## AI AGENT CONFIGURATIONS

### AI Agent 1 - Insight Engine
```python
# Prompt template for daily summaries
INSIGHT_PROMPT = """
Analyze the following KPIs for {asset_name} on {date}:
- Arbitrage Efficiency: {arb_efficiency}%
- Market Capture: {market_capture}%
- Cycles Used: {cycles}/{max_cycles}

Identify top 3 optimization opportunities and anomalies.
"""
```

### AI Agent 2 - Operational Co-Pilot
```python
# What-if scenario analysis
SCENARIO_PROMPT = """
Given current market conditions:
- Peak spread: £{peak_spread}/MWh
- Off-peak price: £{offpeak_price}/MWh
- Current SoC: {current_soc}%

Recommend optimal dispatch strategy for next 24 hours.
"""
```

## CUSTOM PROJECT INSTRUCTIONS

### Critical Business Rules
1. **UK BESS Assets**: Must comply with BM Unit registration requirements
2. **Degradation Modeling**: Apply 2% annual capacity fade to optimization
3. **Warranty Constraints**: Limit to manufacturer-specified cycle counts
4. **Grid Codes**: Implement frequency response hold requirements
5. **Settlement Periods**: Align all UK calculations to 30-minute periods

### Stakeholder-Specific Outputs
- **Finance Team**: Focus on IRR impact and revenue variance explanations
- **Asset Managers**: Emphasize availability, performance ratios, and aggregator benchmarking
- **O&M Engineers**: Provide detailed drill-downs and exportable datasets
- **Executive Reports**: High-level KPIs with trend analysis and market positioning

### Market-Specific Configurations
- **UK**: N2EX day-ahead, BM imbalance prices, BSUoS charges
- **EU**: EPEX SPOT, country-specific grid fees
- **US**: ISO/RTO locational marginal prices (LMP)
- **Australia**: NEM spot prices, FCAS markets

---

## MVP IMPLEMENTATION DECISIONS

### V1 MVP Scope (Approved: 2025-11-11)

**In-Scope**:
- ✅ BESS-only optimization (Solar/Hybrid deferred to V2)
- ✅ UK market focus (30-minute settlement periods)
- ✅ CSV-based data sources (SCADA + market prices)
- ✅ Dual stakeholder focus: Finance (revenue analytics) + O&M (operational metrics)
- ✅ Configuration-first architecture (zero hardcoded values)
- ✅ Data quality gating (DQ ≥80% enforcement)
- ✅ Full regulatory compliance (degradation, warranty, settlement constraints)

**Out-of-Scope (Post-MVP)**:
- ❌ Live market data API integration (N2EX, BM) - deferred to Phase 9
- ❌ Solar PV module - deferred to Phase 10
- ❌ Hybrid PV-BESS optimization - deferred to Phase 11
- ❌ Multi-asset portfolio optimization - deferred to Phase 12
- ❌ Database integration (TimescaleDB) - deferred to Phase 13
- ❌ FastAPI REST endpoints
- ❌ Cloud deployment (AWS/Oracle) - deferred to Phase 14

### Critical Design Decisions

**1. Market Data Source: CSV (Not API)**
- **Decision Date**: 2025-11-11
- **Rationale**: De-risk MVP, focus on core optimization logic, eliminate external dependencies
- **Implementation**:
  - SCADA data provided as CSV (timestamp_utc, power_mw, soc_percent)
  - Market price data provided as CSV (timestamp_utc, price_gbp_mwh, market_type)
  - Live API integration deferred to post-MVP enhancement
  - Data abstraction layer allows easy API integration later (config.market.data_source: "csv" → "api")
- **Files**:
  - `config/price_selection_rules.yaml` - Defines price stack selection logic
  - `src/data_processing/csv_loader.py` - Generic CSV ingestion
  - Future: `src/data_processing/api_client.py` - N2EX/BM API integration

**2. Configuration-First Architecture**
- **Decision Date**: 2025-11-11
- **Rationale**: Eliminate all hardcoded values, support multiple markets, enable rapid configuration changes
- **Implementation**:
  - All parameters externalized to 5 YAML files (config_schema, market_constraints, dq_remediation_rules, price_selection_rules, acceptance_criteria)
  - Pydantic validation for all configurations
  - Anti-hardcoding audit checklist enforced at each phase gate
- **Files**:
  - `config/*.yaml` - All configuration files
  - `src/config_loader.py` - Pydantic-based configuration loader

**3. Data Quality Gating with Auto-Remediation**
- **Decision Date**: 2025-11-11
- **Rationale**: Prevent "garbage in, garbage out" optimization, provide operational decision logic
- **Implementation**:
  - DQ score ≥80% required to proceed with optimization
  - Auto-interpolation for gaps ≤60 minutes if completeness ≥95%
  - Hard reject if completeness <80%
  - Energy reconciliation validation (±5% tolerance for power integration)
  - Re-ingestion workflow with max 3 iterations
- **Files**:
  - `config/dq_remediation_rules.yaml` - Decision matrix for auto-fix vs reject
  - `src/data_processing/remediation_engine.py` - Auto-remediation logic
  - `src/data_processing/energy_reconciliation.py` - Power-to-energy validation

**4. Dual-Stakeholder KPI Focus**
- **Decision Date**: 2025-11-11
- **Rationale**: Finance team needs revenue analytics, O&M needs operational metrics
- **Implementation Status**: ✅ Complete (Phase 4)
- **Implementation**:
  - Finance KPIs: Market Capture Ratio, revenue variance, IRR impact, price spread captured, letter grading (A-F)
  - O&M KPIs: Availability, cycle utilization, actual RTE, capacity factor, idle time, degradation tracking, letter grading (A-F)
  - 36 total KPIs (19 finance + 17 O&M)
  - Separate report generation for each stakeholder
  - Downloadable reports in CSV and JSON formats
- **Files**:
  - `config/acceptance_criteria.yaml` - Numeric thresholds for each stakeholder
  - `src/modules/finance_kpis.py` - Finance KPI calculation engine (11 metrics)
  - `src/modules/om_kpis.py` - O&M KPI calculation engine (11 metrics)
  - `calculate_kpis.py` - CLI tool for KPI calculation with dual-stakeholder support
  - Dashboard tabs: Deferred to Phase 6 (Streamlit)

**5. MILP Optimization with Full Constraint Set**
- **Decision Date**: 2025-11-11
- **Rationale**: Maximize arbitrage revenue while respecting all physical, regulatory, and warranty constraints
- **Implementation Status**: ✅ Complete (Phase 3)
- **Implementation**:
  - PuLP-based MILP solver with CBC backend
  - Asymmetric power limits (4.2 MW import, 7.5 MW export)
  - SoC bounds (5-95%)
  - RTE losses (87% charging efficiency)
  - Daily cycle limit (1.5 cycles/day for 15-year warranty)
  - Charge/discharge mutual exclusivity
  - Sub-second solve times (<1 second for 90 periods)
- **Files**:
  - `src/optimization/bess_optimizer.py` - MILP solver with all constraints
  - `optimize_bess.py` - CLI tool for optimization runs
  - Solver: PULP_CBC_CMD (included with PuLP)

**6. Testing at Phase Gates (Not Just End)**
- **Decision Date**: 2025-11-11
- **Rationale**: Catch issues early, validate assumptions incrementally, prevent cascading failures
- **Implementation Status**: ⚠️ Manual testing complete, unit tests pending
- **Implementation**:
  - Manual CLI testing for each phase (Phases 0-4 complete)
  - Real data validation with UK BESS asset
  - Unit tests deferred to post-MVP
  - Integration tests deferred to Phase 7
- **Files**:
  - `tests/` directory created (unit tests pending)
  - Manual testing logs in PROJECT_PLAN.md

### Data Conventions for MVP

**SCADA CSV Format**:
```csv
timestamp_utc,power_mw,soc_percent
2024-01-01T00:00:00Z,-1.5,45.2
2024-01-01T00:30:00Z,-1.8,52.7
```

**Market Price CSV Format**:
```csv
timestamp_utc,price_gbp_mwh,market_type
2024-01-01T00:00:00Z,45.50,day_ahead
2024-01-01T00:30:00Z,48.20,day_ahead
```

**Power Convention**: Positive = discharge/export, Negative = charge/import (unchanged)
**Timestamp Format**: ISO8601 UTC (e.g., 2024-01-01T00:00:00Z)
**Settlement Periods**: 48 per day (30-minute periods for UK)
**Price Stack**: day_ahead (required), imbalance (optional for V1)

### Timeline

- **Week 0**: Prerequisites (BESS parameters, SCADA/market CSV samples, config approvals)
- **Week 1**: Core build (data pipeline, optimization, KPIs, visualization)
- **Week 2**: Dashboard, testing, documentation, deployment

### Success Criteria (Phases 0-4 Complete)

- ✅ DQ ≥80% enforcement - Achieved (SCADA: 89.3%, Market: 100%)
- ✅ Solver convergence <30 seconds - Achieved (0.13 seconds for 90 periods)
- ✅ KPI calculations accurate - Achieved (36 KPIs with letter grading)
- ✅ Zero hardcoded values - Achieved (all parameters in YAML configs)
- ✅ Finance + O&M acceptance criteria framework - Achieved
- ✅ Asymmetric power constraints - Achieved (4.2 MW / 7.5 MW)
- ✅ RTE losses modeled - Achieved (87% efficiency)
- ✅ Daily cycle limit enforced - Achieved (1.5 cycles/day)
- ✅ Configuration-first architecture validated - Achieved

**Test Results Summary**:
- Optimization: £2,326 optimal revenue vs £-106 actual (£2,433 opportunity)
- Market Capture: -4.6% (Grade F) indicates idle BESS missed opportunities
- Cycle Utilization: 65.3% (underutilized)
- Solve Time: 0.13 seconds (sub-second performance)

See `docs/PROJECT_PLAN.md` for detailed implementation plan and progress logs.

---

## CURRENT MVP IMPLEMENTATION STATUS (2025-11-11)

### Completed Phases (0-4)

**Phase 0: Configuration Foundation** ✅
- 5 YAML configuration files (config_schema, market_constraints, dq_remediation_rules, price_selection_rules, acceptance_criteria)
- Pydantic-based config loader with validation
- Zero hardcoded values achieved

**Phase 1: Data Ingestion Pipeline** ✅
- CSV loader with BOM handling and flexible timestamp parsing
- 10-min → 30-min resampling for UK settlement periods
- Day-ahead price selection
- 4-component DQ scoring (completeness, continuity, bounds, energy reconciliation)
- CLI tool: `ingest_data.py`

**Phase 2: Data Quality Framework** ✅
- Remediation engine with auto-interpolation
- Energy reconciliation module with diagnostic error analysis
- Re-ingestion workflow (max 3 iterations)
- `--remediate` flag support

**Phase 3: BESS Optimization** ✅
- MILP solver with PuLP/CBC
- Full constraint set: SoC bounds, asymmetric power limits, RTE losses, daily cycle limit
- Actual vs optimal revenue comparison
- CLI tool: `optimize_bess.py`
- Solve time: 0.13 seconds for 90 periods

**Phase 4: KPI Calculations** ✅
- Finance KPIs: 19 metrics (market capture, revenue variance, IRR impact, price spread)
- O&M KPIs: 17 metrics (availability, cycle utilization, RTE, capacity factor, idle time)
- Letter grading system (A-F) for both stakeholders
- CLI tool: `calculate_kpis.py`
- Multi-format outputs: CSV + JSON

### Remaining Phases (5-8)

**Phase 5: Visualization** ⏳ Not Started
- Plotly charts: power profiles, SoC curves, price spreads
- Actual vs optimal comparison charts
- Market opportunity heatmaps

**Phase 6: Streamlit Dashboard** ⏳ Not Started
- Finance tab with revenue analytics
- O&M tab with operational metrics
- Interactive visualizations
- Multi-asset support

**Phase 7: Integration Testing** ⏳ Not Started
- End-to-end pipeline testing
- Multi-day optimization scenarios
- Unit test suite

**Phase 8: Documentation & Deployment** ⏳ Not Started
- User guides for stakeholders
- API documentation
- Deployment scripts

### Key Deliverables Created

**CLI Tools (3)**:
- `ingest_data.py` - Data ingestion with DQ gating
- `optimize_bess.py` - MILP arbitrage optimization
- `calculate_kpis.py` - Finance & O&M KPI calculation

**Core Modules (11)**:
- Configuration: `config_loader.py`
- Data Processing: `csv_loader.py`, `data_cleaner.py`, `price_selector.py`, `data_quality_scorer.py`, `remediation_engine.py`, `energy_reconciliation.py`
- Optimization: `bess_optimizer.py`
- KPIs: `finance_kpis.py`, `om_kpis.py`
- Schemas: `schemas.py`

**Configuration Files (5)**:
- `config/config_schema.yaml`
- `config/market_constraints.yaml`
- `config/dq_remediation_rules.yaml`
- `config/price_selection_rules.yaml`
- `config/acceptance_criteria.yaml`

**Total Lines of Code**: ~3,500 lines (estimated)

---

**ACTIVATION PROTOCOL**: This configuration optimizes Claude for the Ampyr Asset Intelligence Platform development. All responses will demonstrate deep understanding of energy markets, optimization algorithms, and modular architecture patterns. Implementation will be direct, complete, and production-ready with full consideration of data quality, performance metrics, and stakeholder requirements.