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
streamlit run app.py  # Launch Streamlit interface
python optimize_bess.py --asset UK_BESS_001 --date 2024-01-01  # Run optimization
python calculate_kpis.py --module bess --period daily  # Calculate KPIs
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

**ACTIVATION PROTOCOL**: This configuration optimizes Claude for the Ampyr Asset Intelligence Platform development. All responses will demonstrate deep understanding of energy markets, optimization algorithms, and modular architecture patterns. Implementation will be direct, complete, and production-ready with full consideration of data quality, performance metrics, and stakeholder requirements.