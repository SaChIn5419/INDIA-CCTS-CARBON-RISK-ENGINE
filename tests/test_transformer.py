import torch
import pytest
from models.forecasting.transformer import TransformerEncoderModule
from models.forecasting.hybrid_model import HybridForecaster

def test_transformer_encoder():
    batch_size = 8
    seq_len = 10
    d_model = 64

    model = TransformerEncoderModule(d_model=d_model, nhead=4, num_layers=2)
    x = torch.randn(batch_size, seq_len, d_model)

    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)

def test_hybrid_forecaster():
    batch_size = 4
    seq_len = 30
    K_modes = 3
    d_model = 32

    model = HybridForecaster(K_modes=K_modes, seq_len=seq_len, d_model=d_model, slstm_layers=1, nhead=4, transformer_layers=1)

    # Input has K features (the K IMFs) at each timestep
    x = torch.randn(batch_size, seq_len, K_modes)

    out = model(x)

    # We predict the t+1 value for all K IMFs
    assert out.shape == (batch_size, K_modes)

    # Verify we can sum them to get a single price prediction
    final_price = out.sum(dim=1)
    assert final_price.shape == (batch_size,)
