# CCTS Carbon Risk Engine — Complete Technical Blueprint

## Table of Contents
1. [The Problem We're Solving](#1-the-problem)
2. [Data: What We Need & Where It Comes From](#2-data)
3. [Data Pipeline: Ingestion → Cleaning → Features](#3-pipeline)
4. [Model Architecture: WOA-VMD-sLSTM-Transformer](#4-model)
5. [Risk Engine: Monte Carlo + CVaR](#5-risk)
6. [Compliance Modules: GEI + CBAM + Watchdog](#6-compliance)
7. [Phased Build Plan: Start Small, Scale Later](#7-phases)
8. [Full Hyperparameter Reference](#8-hyperparams)
9. [Project Directory Structure](#9-structure)

---

## 1. The Problem We're Solving <a id="1-the-problem"></a>

India's CCTS mandates 490+ industrial facilities to meet Greenhouse Gas Emission Intensity (GEI) targets. Facilities that fail pay a penalty of **2× the average CCC market price per tonne of deficit**. No software currently provides:

- **Forward carbon price curves** for the Indian market (IEX hasn't launched CCC trading yet)
- **Probability of compliance breach** per facility
- **Optimal hedge timing** (when to buy CCCs vs invest in physical abatement)

Our engine solves all three using a hybrid deep-learning price forecaster + stochastic risk simulator.

### Why Not Just ARIMA-GARCH?

| Problem | ARIMA-GARCH | Our Hybrid |
|---|---|---|
| Non-linear regime shifts | ❌ Assumes linear | ✅ Transformer attention |
| Noisy raw price data | ❌ Feeds raw series | ✅ VMD decomposes first |
| Hyperparameter tuning | ❌ Manual grid search | ✅ WOA auto-optimizes |
| Long-range dependencies | ❌ Fixed lag structure | ✅ Self-attention (any distance) |
| Short-term volatility | ❌ Basic GARCH | ✅ sLSTM local memory |
| **Reported R²** | **~0.78** | **0.9862** |
| **Reported MAPE** | **>3.0%** | **0.56%** |

---

## 2. Data: What We Need & Where It Comes From <a id="2-data"></a>

### 2.1 Phase 1 Data (Available NOW — Free)

Since IEX CCC trading hasn't launched, we train on global carbon market proxies and transfer-learn.

#### Primary Training Data: EU ETS Futures

| Field | Detail |
|---|---|
| **What** | EU Carbon Allowance (EUA) daily close prices |
| **Ticker** | `KEUA` (KraneShares EU Carbon ETF) via yfinance |
| **Backup** | `KRBN` (KraneShares Global Carbon Strategy ETF) |
| **Period** | Jan 2018 – Present (~8 years, ~2000 trading days) |
| **Why** | Most liquid, most mature carbon market. Best proxy for price dynamics. |
| **How to get it** | `yfinance.download("KEUA", start="2018-01-01")` |

#### Secondary Training Data: China ETS Pilots

| Field | Detail |
|---|---|
| **What** | Hubei & Shenzhen pilot ETS daily settlement prices |
| **Source** | China Carbon Trading Network / Wind Financial Terminal |
| **Period** | 2014 – Present |
| **Why** | Structurally closest to India's CCTS (intensity-based, emerging market, policy-driven jumps) |
| **How to get it** | Web scraping with Selenium from tanpaifang.com, or CSV download from academic datasets |

#### Exogenous Macro Features (X variables for ARIMAX)

| Feature | Ticker / Source | Rationale |
|---|---|---|
| Brent Crude Oil | `BZ=F` via yfinance | Fuel cost → emission intensity |
| NIFTY 50 Index | `^NSEI` via yfinance | Indian economic activity proxy |
| USD/INR Exchange | `USDINR=X` via yfinance | Import cost of abatement tech |
| Coal India Price | `COALINDIA.NS` via yfinance | Domestic fuel cost driver |
| India Power Demand | CEA monthly reports (PDF scrape) | Grid emission factor proxy |
| EU EUA Futures Curve | ICE ECX (if available) | Global carbon sentiment |

### 2.2 Phase 2 Data (Months 5-12 — From Pilot Clients)

| Data | Source | Frequency |
|---|---|---|
| Fuel consumption by type | Client Excel/invoice | Monthly |
| Production output (tonnes) | Client ERP/manual | Monthly |
| Grid electricity purchased | DISCOM bills | Monthly |
| CEMS readings (if available) | Client CEMS PDF exports | Quarterly |
| PAT cycle historical data | BEE portal scrape | One-time |

### 2.3 Phase 3 Data (Month 12+ — IEX Live)

| Data | Source | Frequency |
|---|---|---|
| CCC spot price | IEX API / web scrape | Daily |
| CCC order book depth | IEX | Intraday |
| CCC volume & open interest | IEX | Daily |
| CERC price band updates | CERC gazette notifications | Quarterly |

---

## 3. Data Pipeline: Ingestion → Cleaning → Features <a id="3-pipeline"></a>

### 3.1 Pipeline Architecture

```
Raw Sources                Processing                    Output
─────────────             ─────────────                 ────────
yfinance API ──┐
               ├──→ [Fetcher] ──→ [Aligner] ──→ [Cache]
Web Scraper  ──┘         │            │             │
                         ▼            ▼             ▼
                    Raw CSVs    Aligned DataFrame   Parquet
                         │
                         ▼
                  [Log Returns] ──→ [Normalization] ──→ [Windowing]
                                        │                   │
                                        ▼                   ▼
                                   MinMax(0,1)        (X, y) pairs
                                   or Z-score       lookback=30 days
                                                    horizon=1/5 days
                                        │
                                        ▼
                              [Train / Val / Test Split]
                                 70%   15%    15%
                              Chronological (no shuffle!)
```

### 3.2 Step-by-Step Pipeline Process

**Step 1: Fetch** — Download daily close prices for all tickers. Cache to `data/cache/` as CSV. Retry logic with 3 attempts. Fallback to cached data if API fails.

**Step 2: Align** — All series aligned to same trading calendar using forward-fill. Drop rows where primary series (carbon price) is NaN.

**Step 3: Log Returns** — Calculate `r_t = ln(P_t / P_{t-1})` for all price series. This makes the data stationary (required for GARCH and helpful for neural nets).

**Step 4: Normalize** — MinMax scaling to [0, 1] for neural network inputs. Save scaler parameters for inverse transform at prediction time.

**Step 5: Window** — Create sliding windows: `X = [r_{t-30}, ..., r_{t-1}]`, `y = r_t` (or `r_{t:t+5}` for 5-day horizon). No data leakage — windows are strictly causal.

**Step 6: Split** — Chronological split: first 70% train, next 15% validation, last 15% test. Never shuffle time-series data.

### 3.3 Anti-Leakage Safeguards

- VMD decomposition runs on training window ONLY, then applies learned modes to val/test
- Normalization fitted on train split only, then `.transform()` on val/test
- No future information in any feature engineering step

---

## 4. Model Architecture: WOA-VMD-sLSTM-Transformer <a id="4-model"></a>

The model follows a **"Decompose → Extract → Attend → Reconstruct"** paradigm.

### 4.1 Stage 1: WOA-Optimized VMD Decomposition

**What VMD Does**: Takes a noisy carbon price series and decomposes it into K separate "modes" (sub-signals), each capturing a different frequency band — from slow trends to fast noise.

**Why not just feed raw data?** Raw carbon prices are chaotic — a superposition of long-term policy trends, seasonal compliance cycles, and high-frequency trading noise. Neural networks struggle with this mixture. By separating the frequencies first, each sub-network can specialize.

**How VMD works mathematically:**

VMD solves a constrained optimization:
```
min  Σ_k ||∂_t [σ(t) * u_k(t)] * e^{-jω_k t}||²
s.t. Σ_k u_k(t) = f(t)    [reconstruction constraint]
```

Where:
- `u_k(t)` = mode k (the sub-signal we're extracting)
- `ω_k` = center frequency of mode k
- `f(t)` = original price series
- `α` = penalty factor (controls bandwidth of modes)
- `K` = number of modes to extract

**The problem**: VMD has two critical parameters — `K` (how many modes) and `α` (penalty factor). Wrong values → mode mixing (bad decomposition). Manual tuning is unreliable.

**Solution: Whale Optimization Algorithm (WOA)**

WOA is a meta-heuristic optimizer inspired by humpback whale bubble-net hunting. It searches for the optimal `(K, α)` pair by:

1. **Initialize** 30 "whale" agents, each with random `(K, α)` values
2. **Evaluate** each whale: run VMD with its parameters, measure quality via **envelope entropy** (lower = better separation)
3. **Update** whale positions using three strategies:
   - **Encircling prey**: Whales move toward the best solution found so far
   - **Bubble-net attack**: Whales spiral toward prey (exploitation)
   - **Random search**: Whales explore new `(K, α)` regions (exploration)
4. **Repeat** for 50 iterations
5. **Output**: Optimal `K*` and `α*`

**Envelope entropy objective function:**
```python
def objective(K, alpha, price_series):
    imfs = VMD(price_series, K=K, alpha=alpha)
    entropies = [sample_entropy(imf) for imf in imfs]
    return sum(entropies)  # minimize total entropy
```

**WOA Config:**
```
Population:      30 whales
Max iterations:  50
K search range:  [3, 10]   (integers)
α search range:  [100, 5000] (continuous)
Objective:       Minimize envelope entropy
```

**Output of Stage 1**: K clean IMF sub-signals, e.g., for K=6:
- IMF₁: Long-term policy trend (very smooth)
- IMF₂: Annual compliance cycle
- IMF₃: Quarterly seasonal pattern
- IMF₄: Monthly trading dynamics
- IMF₅: Weekly volatility
- IMF₆: Daily noise / high-frequency jumps

### 4.2 Stage 2: sLSTM Local Feature Extraction

**What sLSTM does**: Each IMF sub-signal is fed into its own sLSTM encoder. The sLSTM captures **local, short-term patterns** within each frequency band.

**Why sLSTM over standard LSTM?**

Standard LSTM uses sigmoid gates (output range 0-1), which compresses information. The sLSTM (from the 2024 xLSTM paper by Sepp Hochreiter) uses **exponential gates**, providing a much larger dynamic range for memory management.

**sLSTM Cell Equations:**
```
# Exponential gates (key innovation)
i_t = exp(W_i · x_t + R_i · h_{t-1} + b_i)    # input gate
f_t = exp(W_f · x_t + R_f · h_{t-1} + b_f)    # forget gate

# Standard components
z_t = tanh(W_z · x_t + R_z · h_{t-1} + b_z)   # cell input
o_t = σ(W_o · x_t + R_o · h_{t-1} + b_o)      # output gate

# Normalizer (prevents explosion from exp gates)
n_t = f_t · n_{t-1} + i_t

# Cell state update
c_t = f_t · c_{t-1} + i_t · z_t

# Hidden state (normalized)
h_t = o_t · (c_t / n_t)
```

**Architecture per IMF:**
```
IMF_k (seq_len=30) → sLSTM Layer 1 (128 units) → Dropout(0.2)
                    → sLSTM Layer 2 (128 units) → Dropout(0.2)
                    → Output: h_k ∈ R^128 (local feature vector)
```

**sLSTM Config:**
```
Input dimension:    1 (univariate IMF)
Hidden dimension:   128
Number of layers:   2 (stacked)
Dropout:            0.2
Gating:             Exponential (with normalizer)
Sequence length:    30 (lookback window)
```

### 4.3 Stage 3: Multi-Head Transformer Global Attention

**What the Transformer does**: Takes the K local feature vectors from the sLSTM encoders and models **global dependencies** between them — how the trend, seasonal, and noise components interact.

**Why Transformer after sLSTM?**

- sLSTM processes each IMF **independently** and **sequentially** — great for local patterns within a single frequency
- Transformer processes all IMFs **jointly** and in **parallel** — great for cross-frequency dependencies (e.g., how a policy trend shift in IMF₁ eventually cascades into volatility in IMF₅)

**Transformer Architecture:**
```
K feature vectors [h₁, h₂, ..., h_K] ∈ R^{K×128}
       │
       ▼
[Positional Encoding] — sinusoidal, encodes IMF ordering
       │
       ▼
[Multi-Head Self-Attention] × 3 layers
  ├─ 8 attention heads
  ├─ d_model = 128
  ├─ d_ff = 512 (feed-forward)
  ├─ GELU activation
  ├─ Layer Normalization
  └─ Residual connections
       │
       ▼
[Global Pooling] — aggregate across K modes
       │
       ▼
[Linear Head] — R^128 → R^K (one prediction per IMF)
       │
       ▼
[Reconstruction] — Σ predictions = final price forecast
```

**Transformer Config:**
```
d_model:              128
Number of heads:      8
Encoder layers:       3
Feed-forward dim:     512
Dropout:              0.1
Activation:           GELU
Positional encoding:  Sinusoidal
```

### 4.4 Stage 4: Reconstruction & Inverse Transform

```python
# Per-IMF predictions from Transformer
pred_imf_1, pred_imf_2, ..., pred_imf_K = transformer_output

# Reconstruct full price prediction
pred_return = sum(pred_imf_1, ..., pred_imf_K)

# Inverse normalize
pred_return_real = scaler.inverse_transform(pred_return)

# Convert log-return back to price
pred_price = last_known_price * exp(pred_return_real)
```

### 4.5 Training Configuration

```
Optimizer:          Adam (lr=1e-4, weight_decay=1e-5)
LR Scheduler:      Cosine Annealing (T_max=200, η_min=1e-6)
Loss Function:     MSE (Mean Squared Error)
Batch Size:        64
Max Epochs:        200
Early Stopping:    Patience=15 (monitor val_loss)
Gradient Clipping:  max_norm=1.0
Device:            CUDA if available, else CPU
```

### 4.6 Complete Forward Pass Summary

```
Raw Carbon Price Series (2000+ daily observations)
       │
       ▼
[WOA Optimizer] → finds optimal K=6, α=2000
       │
       ▼
[VMD Decomposition] → 6 clean IMF sub-signals
       │
       ▼
[Per-IMF Windowing] → 6 × (batch, seq_len=30, 1) tensors
       │
       ▼
[6 × sLSTM Encoders] → 6 × (batch, 128) local features
       │
       ▼
[Stack] → (batch, 6, 128) tensor
       │
       ▼
[Transformer Encoder] → (batch, 6, 128) attended features
       │
       ▼
[Linear Heads] → 6 × (batch, 1) per-IMF predictions
       │
       ▼
[Sum] → (batch, 1) final price prediction
       │
       ▼
[Inverse Transform] → predicted carbon price in INR
```

---

## 5. Risk Engine: Monte Carlo + CVaR <a id="5-risk"></a>

### 5.1 EGARCH Volatility Model

The Transformer gives us a point forecast. But for risk management, we need the **distribution** of possible outcomes. EGARCH models the time-varying volatility.

```
ln(σ²_t) = ω + α|z_{t-1}| + γ·z_{t-1} + β·ln(σ²_{t-1})
```

Where `γ` captures asymmetry — negative shocks (regulatory tightening) cause bigger volatility spikes than positive shocks.

**Implementation**: Use the `arch` Python library:
```python
from arch import arch_model
model = arch_model(returns, mean='ARX', vol='EGARCH', p=1, q=1, dist='studentst')
result = model.fit()
conditional_vol = result.conditional_volatility
```

### 5.2 Monte Carlo GBM + Jump-Diffusion

With the Transformer's drift forecast (μ) and EGARCH's volatility (σ), we simulate 10,000 price paths:

```
dS = μ·S·dt + σ_t·S·dW + J·S·dN(λ)
```

Where:
- `μ` = drift from Transformer forecast
- `σ_t` = conditional volatility from EGARCH
- `dW` = Wiener process (Brownian motion)
- `J` = jump magnitude ~ N(μ_j, σ_j), calibrated from historical 3σ events
- `dN(λ)` = Poisson process with intensity λ (jumps per year)

### 5.3 VaR and CVaR (Expected Shortfall)

From the 10,000 simulated terminal prices, compute:

```python
losses = deficit_tonnes * simulated_prices * penalty_multiplier
VaR_99 = np.percentile(losses, 99)
CVaR_99 = losses[losses >= VaR_99].mean()  # Expected Shortfall
```

CVaR answers: "If things go really badly (top 1% worst scenarios), how much should the CFO reserve?"

---

## 6. Compliance Modules <a id="6-compliance"></a>

### 6.1 GEI Calculator
```python
GEI = total_emissions / production_output
# Where:
total_emissions = Σ(fuel_qty × emission_factor) + (grid_electricity × grid_EF)
# Emission factors from BEE Schedule I
# Grid EF from CEA state-level data
```

### 6.2 Watchdog — Breach Probability
```python
P_breach = count(simulated_GEI > target_GEI) / n_simulations
if P_breach > 0.05:  # 5% risk tolerance
    trigger_alert("PREEMPTIVE HEDGE RECOMMENDED")
```

### 6.3 CBAM Module
Maps facility GEI output → EU CBAM certificate format for steel/cement/aluminium exporters.

---

## 7. Phased Build Plan: Start Small, Scale Later <a id="7-phases"></a>

### Phase A — MVP Data + Baseline (Week 1-2)
**Goal**: Working data pipeline + ARIMA-GARCH baseline to beat later.

| Task | Details | Output |
|---|---|---|
| Set up project structure | Directories, config, requirements | Skeleton repo |
| EU ETS data fetcher | yfinance KEUA/KRBN with caching | `data/cache/eu_ets.csv` |
| Macro features fetcher | NIFTY, Brent, Coal India, USD/INR | `data/cache/macro.csv` |
| Preprocessing pipeline | Log returns, normalization, windowing | `data/preprocessing.py` |
| ARIMA-GARCH baseline | Using `statsmodels` + `arch` | Baseline R², MAPE scores |

**Start with JUST these 2 files working end-to-end:**
1. `data/fetchers.py` — downloads and caches everything
2. `data/preprocessing.py` — produces train/val/test tensors

### Phase B — Signal Decomposition (Week 2-3)
**Goal**: VMD working, WOA optimizing it automatically.

| Task | Details | Output |
|---|---|---|
| VMD wrapper | Using `vmdpy` library | Decomposed IMFs |
| Envelope entropy | Sample entropy for each IMF | Quality metric |
| WOA optimizer | 30 whales, 50 iterations | Optimal K*, α* |
| Visualization | Plot all K modes separately | Validation plots |

**Start with JUST VMD manually tuned (K=5, α=2000), then add WOA.**

### Phase C — Deep Learning Core (Week 3-5)
**Goal**: sLSTM + Transformer beating the ARIMA-GARCH baseline.

| Task | Details | Output |
|---|---|---|
| sLSTM cell | Exponential gating, normalizer | `models/slstm.py` |
| Transformer encoder | Multi-head attention, 3 layers | `models/transformer.py` |
| Hybrid pipeline | VMD → sLSTM → Transformer → reconstruct | `models/hybrid.py` |
| Training loop | Adam, early stopping, checkpoints | Trained model weights |
| Benchmark | Compare R², MAPE vs all baselines | Results table |

**Start with standard LSTM first, validate pipeline works, THEN swap in sLSTM.**

### Phase D — Risk Engine (Week 5-6)
**Goal**: Monte Carlo + CVaR producing CFO-grade risk numbers.

| Task | Details | Output |
|---|---|---|
| EGARCH model | `arch` library, Student-t residuals | Conditional volatility |
| Monte Carlo engine | 10K paths, GBM + jump-diffusion | Price distributions |
| VaR/CVaR | 95% and 99% confidence | Risk metrics |
| Watchdog | Breach probability tracker | Alert system |

### Phase E — Compliance + Dashboard (Week 6-8)
**Goal**: Working Streamlit demo with one cement plant scenario.

| Task | Details | Output |
|---|---|---|
| GEI calculator | BEE sector formulas | Compliance status |
| Penalty engine | 2× avg price × deficit | Cost projections |
| CBAM module | EU declaration format | Export compliance |
| Streamlit dashboard | 4-page interactive app | Live URL |

---

## 8. Full Hyperparameter Reference <a id="8-hyperparams"></a>

### WOA (Whale Optimization Algorithm)
| Parameter | Value | Description |
|---|---|---|
| `population_size` | 30 | Number of whale search agents |
| `max_iterations` | 50 | Optimization cycles |
| `a_start` | 2.0 | Exploration coefficient (start) |
| `a_end` | 0.0 | Exploration coefficient (end) |
| `b_spiral` | 1.0 | Spiral shape constant |

### VMD (Variational Mode Decomposition)
| Parameter | Value | Description |
|---|---|---|
| `K_range` | [3, 10] | Search range for number of modes |
| `alpha_range` | [100, 5000] | Search range for penalty factor |
| `tau` | 0.0 | Noise tolerance (0 = no noise) |
| `tol` | 1e-7 | Convergence tolerance |
| `max_iter` | 500 | Max VMD iterations |

### sLSTM (Scalar LSTM with Exponential Gating)
| Parameter | Value | Description |
|---|---|---|
| `input_dim` | 1 | Univariate IMF input |
| `hidden_dim` | 128 | Hidden state size |
| `num_layers` | 2 | Stacked sLSTM layers |
| `dropout` | 0.2 | Between-layer dropout |
| `gating` | exponential | exp() gates + normalizer |

### Transformer Encoder
| Parameter | Value | Description |
|---|---|---|
| `d_model` | 128 | Embedding dimension |
| `nhead` | 8 | Attention heads |
| `num_layers` | 3 | Encoder layers |
| `dim_feedforward` | 512 | FFN hidden size |
| `dropout` | 0.1 | Attention + FFN dropout |
| `activation` | GELU | Non-linearity |

### Training
| Parameter | Value | Description |
|---|---|---|
| `lookback` | 30 | Input sequence length (days) |
| `horizon` | 1 | Forecast steps ahead |
| `batch_size` | 64 | Training batch size |
| `lr` | 1e-4 | Initial learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `max_epochs` | 200 | Maximum training epochs |
| `patience` | 15 | Early stopping patience |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `scheduler` | Cosine Annealing | LR decay schedule |
| `train/val/test` | 70/15/15 | Chronological split |

### Monte Carlo Simulation
| Parameter | Value | Description |
|---|---|---|
| `n_simulations` | 10,000 | Number of price paths |
| `n_steps` | 252 | Trading days (1 year) |
| `confidence_levels` | [0.95, 0.99] | VaR/CVaR thresholds |
| `jump_threshold` | 3σ | What counts as a "jump" |

### EGARCH
| Parameter | Value | Description |
|---|---|---|
| `p` | 1 | GARCH lag order |
| `q` | 1 | ARCH lag order |
| `distribution` | Student-t | Heavy-tailed residuals |
| `mean_model` | ARX | With exogenous inputs |

---

## 9. Project Directory Structure <a id="9-structure"></a>

```
carbon_system/
├── config/
│   ├── model_config.py        # All hyperparameters (Section 8)
│   ├── sectors.py             # BEE 7-sector GEI targets
│   └── data_sources.py        # Tickers, API configs
│
├── data/
│   ├── fetchers.py            # yfinance + web scraping
│   ├── preprocessing.py       # Returns, normalize, window
│   ├── regulatory.py          # BEE/CERC static data
│   └── cache/                 # Downloaded CSVs/Parquets
│
├── models/
│   ├── decomposition/
│   │   ├── vmd.py             # VMD wrapper
│   │   ├── woa_optimizer.py   # Whale Optimization
│   │   └── pipeline.py        # WOA→VMD orchestration
│   │
│   └── forecasting/
│       ├── slstm.py           # sLSTM cell + stacked layers
│       ├── transformer.py     # Multi-head Transformer encoder
│       ├── hybrid_model.py    # Full pipeline model
│       └── trainer.py         # Training loop
│
├── risk/
│   ├── egarch.py              # EGARCH volatility
│   ├── monte_carlo.py         # GBM + jump-diffusion MC
│   ├── var_cvar.py            # VaR / CVaR computation
│   └── watchdog.py            # Breach probability alerts
│
├── compliance/
│   ├── gei_calculator.py      # Gate-to-gate GEI formula
│   ├── cbam_module.py         # EU CBAM declarations
│   └── penalty_engine.py      # Penalty cost projections
│
├── dashboard/
│   └── app.py                 # Streamlit MVP
│
├── tests/                     # pytest test suite
├── notebooks/                 # Validation & visualization
├── main.py                    # CLI entry point
└── requirements.txt           # Python dependencies
```

### Required Python Packages
```
torch>=2.1.0
vmdpy>=0.2
arch>=6.0
yfinance>=0.2.31
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
streamlit>=1.30
plotly>=5.18
statsmodels>=0.14
tensorboard>=2.15
PyEMD>=1.5
```

---

## Performance Targets

| Metric | ARIMA-GARCH | Std LSTM | Our Hybrid Target |
|---|---|---|---|
| R² | ~0.78 | ~0.66 | **>0.98** |
| MAPE | >3.0% | ~4.08% | **<0.70%** |
| MAE | Baseline | ~2.75 | **<1.00** |
| RMSE | Baseline | ~3.60 | **<1.20** |

---

*Document version: 1.0 | Generated: May 16, 2026 | Based on: Carbon Pricing Model Evaluation, CCTS Risk Engine, and Indian Carbon Market Quantitative Model specification documents.*
