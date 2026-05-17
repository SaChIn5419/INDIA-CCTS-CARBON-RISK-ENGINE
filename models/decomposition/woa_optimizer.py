import numpy as np
from numba import njit
from models.decomposition.vmd import compute_fitness
from utils.logger import setup_logger

logger = setup_logger('woa_optimizer', 'logs/woa.log')

@njit
def optimize_step_numba(positions, leader_pos, a, a2, pop_size, dim):
    """
    Numba-optimized inner loop for updating positions.
    """
    # Create a copy so we can modify it
    new_positions = positions.copy()
    for i in range(pop_size):
        r1 = np.random.rand()
        r2 = np.random.rand()

        A = 2 * a * r1 - a
        C = 2 * r2

        b = 1.0  # constant for spiral shape
        l = (a2 - 1) * np.random.rand() + 1

        p = np.random.rand()  # random probability

        for j in range(dim):
            if p < 0.5:
                if abs(A) >= 1:
                    # Search for prey (exploration)
                    rand_leader_index = np.random.randint(0, pop_size)
                    x_rand = positions[rand_leader_index]
                    D_X_rand = abs(C * x_rand[j] - positions[i, j])
                    new_positions[i, j] = x_rand[j] - A * D_X_rand
                else:
                    # Shrinking encircling mechanism (exploitation)
                    D_Leader = abs(C * leader_pos[j] - positions[i, j])
                    new_positions[i, j] = leader_pos[j] - A * D_Leader
            else:
                # Spiral updating position (exploitation)
                distance_to_leader = abs(leader_pos[j] - positions[i, j])
                new_positions[i, j] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos[j]
    return new_positions

class WhaleOptimizationAlgorithm:
    def __init__(self, signal, pop_size=10, max_iter=20, bounds=None):
        """
        Whale Optimization Algorithm for finding optimal VMD parameters.

        Parameters:
        - signal: 1D time series array to decompose.
        - pop_size: Number of whale agents. (Reduced from 30 to 10 for speed during POC)
        - max_iter: Optimization iterations. (Reduced from 50 to 20 for speed during POC)
        - bounds: Tuple of (min_bounds, max_bounds).
                  e.g., ([K_min, alpha_min], [K_max, alpha_max])
        """
        self.signal = signal
        self.pop_size = pop_size
        self.max_iter = max_iter

        # Default bounds: K in [3, 10], alpha in [100, 5000]
        if bounds is None:
            self.lb = np.array([3, 100])
            self.ub = np.array([10, 5000])
        else:
            self.lb = np.array(bounds[0])
            self.ub = np.array(bounds[1])

        self.dim = len(self.lb)

    def _clip_to_bounds(self, position):
        return np.clip(position, self.lb, self.ub)

    def optimize(self):
        logger.info(f"Starting WOA Optimization. Pop Size: {self.pop_size}, Iterations: {self.max_iter}")

        # 1. Initialize population positions randomly within bounds
        positions = np.random.uniform(0, 1, (self.pop_size, self.dim)) * (self.ub - self.lb) + self.lb

        # Leader (best solution found so far)
        leader_pos = np.zeros(self.dim)
        leader_score = np.inf  # We are minimizing entropy

        # Track convergence
        convergence_curve = np.zeros(self.max_iter)

        # Log interval
        log_interval = max(1, self.max_iter // 5)

        for t in range(self.max_iter):
            # Evaluate fitness of population
            for i in range(self.pop_size):
                # Ensure K is integer
                pos = positions[i].copy()
                pos[0] = np.round(pos[0])

                # Boundary check
                pos = self._clip_to_bounds(pos)
                positions[i] = pos

                # Calculate fitness
                score = compute_fitness(self.signal, K=pos[0], alpha=pos[1])

                # Update leader
                if score < leader_score:
                    leader_score = score
                    leader_pos = pos.copy()

            # Update exploration parameter 'a' linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)

            # Update parameter a2 linearly from -1 to -2
            # a2 is used to calculate t in spiral equation
            a2 = -1 + t * (-1 / self.max_iter)

            # Call Numba-optimized update step
            positions = optimize_step_numba(positions, leader_pos, a, a2, self.pop_size, self.dim)

            convergence_curve[t] = leader_score

            if t % log_interval == 0:
                logger.info(f"Iteration {t}: Best K={int(leader_pos[0])}, alpha={leader_pos[1]:.2f}, Score={leader_score:.4f}")

        logger.info(f"WOA Complete. Optimal Parameters - K: {int(leader_pos[0])}, alpha: {leader_pos[1]:.2f}")
        return leader_pos, convergence_curve
