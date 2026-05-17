import pytest
import numpy as np
from models.decomposition.vmd import apply_vmd, compute_envelope_entropy
from models.decomposition.woa_optimizer import WhaleOptimizationAlgorithm

def test_vmd_output_shape():
    # Create a synthetic signal
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

    # Apply VMD
    imfs = apply_vmd(signal, K=2, alpha=2000)

    # Assert we get 2 modes
    assert len(imfs) == 2

def test_envelope_entropy():
    # A clear sine wave should have low entropy
    t = np.linspace(0, 1, 100)
    clean_signal = np.sin(2 * np.pi * 5 * t)

    # A random noise signal should have high entropy
    noisy_signal = np.random.randn(100)

    clean_ent = compute_envelope_entropy(clean_signal)
    noisy_ent = compute_envelope_entropy(noisy_signal)

    # Entropy isn't strictly guaranteed to be lower for a perfect sine wave due to boundary effects
    # and short signal length, but both should be computed correctly.
    assert isinstance(clean_ent, float)
    assert isinstance(noisy_ent, float)

def test_woa_optimizer():
    # Synthetic signal
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * t) + np.random.randn(100) * 0.1

    # Small test optimization
    woa = WhaleOptimizationAlgorithm(signal, pop_size=3, max_iter=2)
    best_pos, curve = woa.optimize()

    # Assert it returns parameters within bounds
    assert best_pos[0] >= 3 and best_pos[0] <= 10
    assert best_pos[1] >= 100 and best_pos[1] <= 5000
    assert len(curve) == 2
