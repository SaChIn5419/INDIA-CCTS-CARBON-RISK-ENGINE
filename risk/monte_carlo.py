import numpy as np
from numba import njit
from utils.logger import setup_logger

logger = setup_logger('monte_carlo', 'logs/monte_carlo.log')

@njit
def simulate_gbm_jump_diffusion(S0, mu, sigma, lambda_jump, mu_jump, sigma_jump, T, dt, num_paths):
    """
    Simulates Geometric Brownian Motion with Jump Diffusion.
    Numba-optimized for extreme speed (e.g., 10,000 paths).

    Parameters:
    - S0: Initial Price
    - mu: Drift (from Transformer model)
    - sigma: Conditional Volatility (from EGARCH model)
    - lambda_jump: Intensity of Poisson process (expected jumps per year)
    - mu_jump: Mean jump size (log normal)
    - sigma_jump: Standard deviation of jump size
    - T: Time horizon (years)
    - dt: Time step (years, e.g., 1/252)
    - num_paths: Number of Monte Carlo paths

    Returns:
    - paths: Array of shape (num_paths, num_steps) containing simulated price paths.
    """
    num_steps = int(T / dt)
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0

    # Pre-calculate constants for speed
    drift_term = (mu - 0.5 * sigma**2) * dt
    vol_term = sigma * np.sqrt(dt)

    for i in range(num_paths):
        for t in range(1, num_steps + 1):
            # Brownian motion standard normal
            Z = np.random.standard_normal()

            # Poisson jump component
            # Number of jumps in dt interval
            N = np.random.poisson(lambda_jump * dt)

            jump_multiplier = 0.0
            if N > 0:
                # Sum the log-jumps
                for _ in range(N):
                    J = np.random.normal(mu_jump, sigma_jump)
                    jump_multiplier += J

            # Calculate next price using logarithmic Euler-Maruyama
            log_S_next = np.log(paths[i, t-1]) + drift_term + vol_term * Z + jump_multiplier
            paths[i, t] = np.exp(log_S_next)

    return paths

class MonteCarloSimulator:
    def __init__(self, S0, mu, sigma, n_simulations=10000, n_steps=252):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps

        # Jump-diffusion defaults (calibrated from typical emerging carbon markets)
        self.lambda_jump = 2.0      # Expect ~2 policy/regulatory jumps per year
        self.mu_jump = -0.05        # Regulatory tightening often drops/spikes price, let's assume slight downward bias on jumps for conservatism or adapt it
        self.sigma_jump = 0.15      # 15% jump volatility

    def run(self):
        logger.info(f"Running Monte Carlo: {self.n_simulations} paths, {self.n_steps} steps.")
        logger.info(f"Params: S0={self.S0:.2f}, Drift={self.mu:.4f}, Vol={self.sigma:.4f}")

        dt = 1.0 / self.n_steps  # Assuming n_steps is trading days in a year
        T = 1.0                  # 1 Year horizon

        # Run Numba-compiled function
        paths = simulate_gbm_jump_diffusion(
            self.S0,
            self.mu,
            self.sigma,
            self.lambda_jump,
            self.mu_jump,
            self.sigma_jump,
            T,
            dt,
            self.n_simulations
        )

        logger.info("Monte Carlo simulation complete.")
        return paths
