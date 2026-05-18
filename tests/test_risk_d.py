import pytest
import numpy as np
from risk.var_cvar import compute_var_cvar, ComplianceWatchdog
from risk.monte_carlo import MonteCarloSimulator

def test_monte_carlo_shape():
    sim = MonteCarloSimulator(2500, 0.05, 0.2, n_simulations=100, n_steps=10)
    paths = sim.run()
    # 100 paths, 11 steps (0 to 10)
    assert paths.shape == (100, 11)
    # Starts at S0
    assert np.all(paths[:, 0] == 2500)

def test_var_cvar_computation():
    sim_prices = np.array([1000, 1500, 2000, 2500, 3000])
    deficit = 10
    multiplier = 1.0

    # Losses = [10000, 15000, 20000, 25000, 30000]
    metrics = compute_var_cvar(sim_prices, deficit, penalty_multiplier=multiplier, confidence_levels=[0.80])

    # 80th percentile of [10k, 15k, 20k, 25k, 30k] -> ~26000 depending on np.percentile interp
    var = metrics["VaR_80"]
    cvar = metrics["CVaR_80"]

    assert var > 20000
    assert cvar >= var

def test_watchdog_safe():
    target_gei = 2.2
    # Simulated GEIs all below target
    sim_gei = np.array([2.0, 2.1, 1.9, 1.5, 2.15])

    watchdog = ComplianceWatchdog(target_gei, sim_gei)
    prob, msg = watchdog.evaluate_breach_probability()

    assert prob == 0.0
    assert msg == "SAFE"

def test_watchdog_breach():
    target_gei = 2.2
    # Simulated GEIs mostly above target
    sim_gei = np.array([2.3, 2.4, 2.1, 2.5, 2.6])

    watchdog = ComplianceWatchdog(target_gei, sim_gei, tolerance=0.5)
    prob, msg = watchdog.evaluate_breach_probability()

    assert prob == 0.8
    assert "PREEMPTIVE HEDGE RECOMMENDED" in msg
