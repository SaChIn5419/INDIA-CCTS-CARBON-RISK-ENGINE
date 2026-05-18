import torch
import pytest
from models.forecasting.slstm import sLSTMCell, sLSTM

def test_slstm_cell():
    batch_size = 4
    input_size = 3
    hidden_size = 8

    cell = sLSTMCell(input_size, hidden_size)

    x = torch.randn(batch_size, input_size)
    h = torch.randn(batch_size, hidden_size)
    c = torch.randn(batch_size, hidden_size)
    n = torch.ones(batch_size, hidden_size)

    h_next, (h_n, c_n, n_n) = cell(x, (h, c, n))

    # Assert shapes
    assert h_next.shape == (batch_size, hidden_size)
    assert h_n.shape == (batch_size, hidden_size)
    assert c_n.shape == (batch_size, hidden_size)
    assert n_n.shape == (batch_size, hidden_size)

    # Assert memory normalizer increases or stays positive
    assert torch.all(n_n > 0)

def test_slstm_network():
    batch_size = 4
    seq_len = 10
    input_size = 5
    hidden_size = 16
    num_layers = 2

    model = sLSTM(input_size, hidden_size, num_layers=num_layers)

    x = torch.randn(batch_size, seq_len, input_size)

    outputs, final_states = model(x)

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert len(final_states) == num_layers
    assert final_states[-1][0].shape == (batch_size, hidden_size)
