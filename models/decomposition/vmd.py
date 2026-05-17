import numpy as np
from vmdpy import VMD
from utils.logger import setup_logger

logger = setup_logger('vmd_module', 'logs/vmd.log')

def apply_vmd(signal, K, alpha, tau=0.0, tol=1e-7, max_iter=500):
    """
    Applies Variational Mode Decomposition to a 1D signal.

    Parameters:
    - signal (np.ndarray): 1D array of the time series signal.
    - K (int): Number of modes to decompose into.
    - alpha (float): Data-fidelity tolerance parameter.
    - tau (float): Time-step of the dual ascent (0 for noise-slack).
    - tol (float): Tolerance of convergence criterion.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - u (np.ndarray): Decomposed modes (IMFs).
    - u_hat (np.ndarray): Spectra of the modes.
    - omega (np.ndarray): Estimated mode center-frequencies.
    """
    try:
        # vmdpy expects specific integer/float formats
        K = int(np.round(K))
        alpha = float(alpha)
        DC = 0   # No DC part imposed
        init = 1 # 1 = all omegas start uniformly distributed

        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        return u
    except Exception as e:
        logger.error(f"VMD Error with K={K}, alpha={alpha}: {str(e)}")
        # Return empty or original signal as fallback
        return np.array([signal])

def compute_envelope_entropy(signal):
    """
    Computes the envelope entropy of a signal.
    Used as the fitness function for WOA to optimize VMD.
    Lower entropy means the signal has clearer, distinct features (less noise).
    """
    from scipy.signal import hilbert

    # 1. Analytic signal using Hilbert transform
    analytic_signal = hilbert(signal)

    # 2. Envelope extraction
    envelope = np.abs(analytic_signal)

    # 3. Normalize envelope to form a probability distribution
    envelope_sum = np.sum(envelope)
    if envelope_sum == 0:
        return np.inf

    p = envelope / envelope_sum

    # 4. Compute entropy (add epsilon to avoid log(0))
    epsilon = 1e-12
    entropy = -np.sum(p * np.log(p + epsilon))

    return entropy

def compute_fitness(signal, K, alpha):
    """
    Objective function for WOA.
    We decompose the signal using VMD with given K and alpha,
    then we calculate the mean envelope entropy of all IMFs.
    The goal is to MINIMIZE this value.
    """
    # Prevent invalid parameters
    if K < 2 or alpha < 10:
        return np.inf

    imfs = apply_vmd(signal, K, alpha)

    # If VMD failed, return high penalty
    if len(imfs) == 1:
        return np.inf

    # Calculate envelope entropy for each IMF
    entropies = []
    for imf in imfs:
        ent = compute_envelope_entropy(imf)
        entropies.append(ent)

    # Return the mean envelope entropy as the fitness score
    return np.mean(entropies)
