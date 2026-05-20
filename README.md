# CCTS & EU CBAM Enterprise Climate Risk Engine

An advanced, highly-optimized enterprise compliance and forecasting engine. This tool utilizes a high-performance **WOA-VMD-sLSTM-Transformer** architecture to forecast carbon prices, stochastic Monte Carlo simulations to project risk (VaR/CVaR), and rigid recursive algorithms to calculate cradle-to-gate compliance liabilities under the **EU CBAM Definitive Regime** and India's **CCTS**.

## Table of Contents
1. [Core Features](#1-core-features)
2. [Installation & Setup](#2-installation)
3. [Required Data](#3-required-data)
4. [Usage Guide](#4-usage-guide)
   - [Phase A & B: Data Pipeline & Decomposition](#phase-a--b)
   - [Phase C: Deep Learning Core](#phase-c)
   - [Phase D: Stochastic Risk Engine](#phase-d)
   - [Phase E & F: EU CBAM Definitive Compliance](#phase-e--f)
5. [Architecture & Models](#5-architecture)

---

## 1. Core Features <a id="1-core-features"></a>

- **WOA-VMD Signal Decomposition**: Uses a custom Numba-accelerated Whale Optimization Algorithm (WOA) to tune Variational Mode Decomposition (VMD), effectively isolating noise from financial signals.
- **sLSTM & Transformer Forecaster**: Implements a custom Scalar LSTM (sLSTM) with exponential gating and memory normalizers, routed into a Multi-Head Transformer Encoder to accurately forecast carbon prices.
- **Monte Carlo Jump-Diffusion Risk Engine**: Simulates 10,000 regulatory price paths using Geometric Brownian Motion and EGARCH volatility models to output 95% and 99% VaR and CVaR.
- **EU CBAM Definitive Regime Compliance**:
  - Calculates recursive **Cradle-to-Gate** embedded emissions using Directed Acyclic Graphs (DAGs).
  - Dynamically scales **SEFA** (Specific Embedded Free Allocations) to prevent gross liability overestimation.
  - Applies strictly audited 5% Materiality Thresholds, Geopolitical Arbitrage defaults, and up to 30% punitive data markups.
- **XML Payload Serialization**: Automatically exports multi-gas (CO2, N2O, PFC) compliance declarations perfectly aligned with the EU CBAM Registry's XSD format.

---

## 2. Installation & Setup <a id="2-installation"></a>

This engine relies on high-performance libraries (`numba`, `duckdb`, `polars`, `lxml`, `torch`).

```bash
# Clone the repository
git clone <repository_url>
cd INDIA-CCTS-CARBON-RISK-ENGINE

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Required Data <a id="3-required-data"></a>

The pipeline requires both macro-financial data and facility-specific operational data.

### Global Market Proxies (Auto-Fetched)
The engine automatically downloads the following training data using Yahoo Finance via `data/fetchers.py`:
- **Primary Market Proxy**: `KEUA` (EU Carbon Allowance ETF)
- **Macro Drivers**: `KRBN` (Global Carbon), Brent Crude Oil (`BZ=F`), NIFTY 50 (`^NSEI`), USD/INR, Coal India.

### Manufacturer Compliance Data (Required for CBAM/CCTS)
To run the compliance risk engines, you must provide:
- **Production Quantities**: Exported vs. Domestic production in metric tonnes.
- **Sector**: e.g., Steel, Cement, Aluminum, Fertilizer.
- **Precursor Supply Chain**: Mass inputs of upstream materials (e.g., Iron Ore -> Pig Iron -> Steel) for cradle-to-gate accounting.
- **Actual Emissions**: Direct process emissions, fuel usage, and multi-gas metrics ($N_2O$, $PFCs$).
- **Domestic Carbon Costs**: Total Effective Price paid domestically (to qualify for Article 9 deductions).

---

## 4. Usage Guide <a id="4-usage-guide"></a>

The repository is built in distinct, decoupled phases. You can execute them sequentially from the root directory.

### Phase A & B: Data Pipeline & Decomposition <a id="phase-a--b"></a>
Downloads macro proxies, calculates normalized log returns via DuckDB/Polars, and isolates intrinsic mode functions via VMD.

```bash
PYTHONPATH=. python data/fetchers.py
PYTHONPATH=. python data/preprocessing.py
PYTHONPATH=. python models/decomposition/pipeline.py
```
*Outputs: `data/cache/*.parquet`, `data/processed/master_dataset.parquet`, `data/processed/decomposed_features.parquet`*

### Phase C: Deep Learning Core <a id="phase-c"></a>
Trains the custom sLSTM-Transformer model on the decomposed financial signals to predict future price drifts.

```bash
PYTHONPATH=. python models/forecasting/trainer.py
```
*Outputs: Evaluates R² and MAPE metrics against baseline ARIMA-GARCH models.*

### Phase D: Stochastic Risk Engine <a id="phase-d"></a>
Extracts EGARCH volatility and runs a Numba-accelerated Monte Carlo simulation to estimate Value at Risk (VaR).

```bash
PYTHONPATH=. python risk/orchestrator.py
```

### Phase E & F: EU CBAM Definitive Compliance <a id="phase-e--f"></a>
Integrates price forecasts and recursive DAG mathematics to determine domestic CCTS penalties and definitive EU CBAM liabilities. Also validates and generates the XML payload.

```bash
PYTHONPATH=. python compliance/cbam/integration.py
```
*Outputs: `reports/cbam_declaration.xml` and terminal logs detailing the exact financial risk in INR and EUR.*

**Note**: You can track continuous execution logs and audit trails natively in `reports/progress_report.md`.

---

## 5. Architecture & Models <a id="5-architecture"></a>

### Project Directory Structure
```
.
├── compliance/          # CCTS & EU CBAM logic (DAGs, SEFA, XML serialization)
├── config/              # Static parameters, Benchmarks, Emission Factors
├── data/                # High-speed fetchers (yfinance) and preprocessors (DuckDB/Polars)
├── logs/                # Automated execution logs
├── models/              # Numba-WOA, VMD, PyTorch sLSTM, and Transformer layers
├── reports/             # Generated XML payloads and Markdown progress tracking
├── risk/                # Monte Carlo Jump-Diffusion, EGARCH, and VaR/CVaR calculators
├── tests/               # Comprehensive Pytest suites validating mathematical logic
└── utils/               # Common logging and reporting utilities
```

### Tech Stack Details
- **Data Layer**: DuckDB, Polars, Parquet (Fast, zero-copy memory manipulation)
- **Math/Optimization Layer**: Numba, NumPy, SciPy (JIT-compiled loops for extreme speed)
- **Deep Learning Layer**: PyTorch (sLSTM, Multi-Head Attention)
- **Serialization Layer**: lxml (XSD-compliant parsing and generation)
