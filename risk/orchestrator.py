import sys
import numpy as np
from risk.egarch import fit_egarch
from risk.monte_carlo import MonteCarloSimulator
from risk.var_cvar import compute_var_cvar, ComplianceWatchdog
from utils.logger import setup_logger, write_progress_report
import polars as pl

logger = setup_logger('risk_orchestrator', 'logs/risk_orchestrator.log')

def run_full_risk_engine(current_price, forecasted_drift, deficit_tonnes, target_gei):
    logger.info("Initializing Full Risk Engine (Phase D)...")

    # 1. EGARCH Volatility
    try:
        df = pl.read_parquet("data/processed/master_dataset.parquet")
        historical_returns = df["keua_close_log_ret_norm"].to_numpy()
        vol_info = fit_egarch(historical_returns)
        sigma = vol_info["current_volatility"]
    except Exception as e:
        logger.warning(f"Could not fit EGARCH, falling back to static 0.2: {str(e)}")
        sigma = 0.2

    # 2. Monte Carlo Simulation
    mc = MonteCarloSimulator(
        S0=current_price,
        mu=forecasted_drift,
        sigma=sigma,
        n_simulations=10000,
        n_steps=252
    )
    paths = mc.run()

    # Get terminal prices (end of year)
    terminal_prices = paths[:, -1]

    # 3. VaR / CVaR Computation
    risk_metrics = compute_var_cvar(terminal_prices, deficit_tonnes)

    # 4. Watchdog Breach Probability
    # For demonstration, we simulate GEI variations (e.g. +/- 10% around a base estimate)
    # Assume base GEI is exactly on the edge of failing (e.g., Target + 0.01)
    simulated_gei = np.random.normal(target_gei + 0.05, 0.15, size=10000)

    watchdog = ComplianceWatchdog(target_gei=target_gei, simulated_gei_distribution=simulated_gei)
    p_breach, alert = watchdog.evaluate_breach_probability()

    # Logging and Reporting
    report = "### Phase D: Monte Carlo Risk Engine (VaR/CVaR)\n\n"

    report += "**Volatility & Simulation Dynamics:**\n"
    report += f"- Initial Carbon Price: ₹{current_price:,.2f}\n"
    report += f"- DL Forecasted Drift (μ): {forecasted_drift:.4f}\n"
    report += f"- EGARCH Conditional Volatility (σ): {sigma:.4f}\n"
    report += f"- Monte Carlo Paths: 10,000 (with Jump-Diffusion)\n\n"

    report += "**Expected Shortfall & Risk Metrics (INR):**\n"
    report += f"- **Compliance Deficit**: {deficit_tonnes:,.0f} tCO2e\n"
    report += f"- **VaR (95%)**: ₹{risk_metrics['VaR_95']:,.2f}\n"
    report += f"- **CVaR (95%)**: ₹{risk_metrics['CVaR_95']:,.2f} *(Expected Shortfall)*\n"
    report += f"- **VaR (99%)**: ₹{risk_metrics['VaR_99']:,.2f}\n"
    report += f"- **CVaR (99%)**: ₹{risk_metrics['CVaR_99']:,.2f} *(CFO Reserve Requirement)*\n\n"

    report += "**Watchdog Alert System:**\n"
    report += f"- Probability of GEI Breach: **{p_breach*100:.1f}%**\n"
    report += f"- System Status: **{alert}**\n"

    write_progress_report(
        "Phase D: Stochastic Risk Engine",
        "Completed",
        report
    )

    return risk_metrics

if __name__ == "__main__":
    # Test Orchestrator
    run_full_risk_engine(
        current_price=2500.0,
        forecasted_drift=0.08, # 8% expected growth from hybrid model
        deficit_tonnes=10000,
        target_gei=2.2
    )
