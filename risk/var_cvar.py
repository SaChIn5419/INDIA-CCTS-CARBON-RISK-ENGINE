import numpy as np
from utils.logger import setup_logger

logger = setup_logger('var_cvar', 'logs/var_cvar.log')

def compute_var_cvar(simulated_terminal_prices, deficit_tonnes, penalty_multiplier=2.0, confidence_levels=[0.95, 0.99]):
    """
    Computes Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall).

    Parameters:
    - simulated_terminal_prices: 1D array of the final prices from Monte Carlo paths.
    - deficit_tonnes: The compliance gap (shortfall) in tCO2e.
    - penalty_multiplier: Domestic penalty rule (e.g., 2x market price).
    - confidence_levels: List of confidence intervals for risk metrics.

    Returns:
    - Dictionary with VaR and CVaR at specified confidence levels.
    """
    # Calculate potential financial losses across all simulated scenarios
    losses = deficit_tonnes * simulated_terminal_prices * penalty_multiplier

    risk_metrics = {}

    for conf in confidence_levels:
        percentile_val = conf * 100

        # VaR answers: "What is the worst loss we can expect with X% confidence?"
        # It's the Xth percentile of the loss distribution
        var = np.percentile(losses, percentile_val)

        # CVaR answers: "If things go worse than VaR, how much should we expect to lose?"
        # It's the average of all losses that exceed the VaR
        tail_losses = losses[losses >= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

        conf_label = f"{int(percentile_val)}"
        risk_metrics[f"VaR_{conf_label}"] = var
        risk_metrics[f"CVaR_{conf_label}"] = cvar

        logger.info(f"Computed {conf_label}% VaR: {var:,.2f} INR, CVaR: {cvar:,.2f} INR")

    return risk_metrics

class ComplianceWatchdog:
    def __init__(self, target_gei, simulated_gei_distribution=None, tolerance=0.05):
        """
        Calculates breach probabilities and triggers preemptive hedge alerts.
        """
        self.target_gei = target_gei
        self.tolerance = tolerance

        # Simulated GEI distribution can come from stochastic production/emission factors
        # If not provided, we just monitor based on single point estimates.
        self.simulated_gei = simulated_gei_distribution

    def evaluate_breach_probability(self):
        if self.simulated_gei is None or len(self.simulated_gei) == 0:
            return 0.0, "No distribution provided"

        breach_count = np.sum(self.simulated_gei > self.target_gei)
        p_breach = breach_count / len(self.simulated_gei)

        alert_msg = "SAFE"
        if p_breach > self.tolerance:
            alert_msg = "PREEMPTIVE HEDGE RECOMMENDED"
            logger.warning(f"WATCHDOG ALERT: {p_breach*100:.1f}% probability of breach. ({alert_msg})")

        return p_breach, alert_msg
