import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTMCell(nn.Module):
    """
    Scalar LSTM (sLSTM) Cell with exponential gating and memory normalizer.
    This architecture helps mitigate vanishing gradients over long sequences
    better than a standard LSTM.
    """
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combine weight matrices for faster computation
        # W_i, W_f, W_z, W_o
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        """
        x: (batch_size, input_size)
        state: tuple of (h_prev, c_prev, n_prev)
               h_prev: (batch_size, hidden_size)
               c_prev: (batch_size, hidden_size)
               n_prev: (batch_size, hidden_size) - memory normalizer
        """
        h_prev, c_prev, n_prev = state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)

        # Compute gates
        gates = self.W(combined)
        i_gate, f_gate, z_gate, o_gate = gates.chunk(4, 1)

        # Exponential Gating
        # We use exp() instead of sigmoid() to allow larger gradient flow
        # as described in recent extended memory architectures.
        i = torch.exp(i_gate)
        f = torch.exp(f_gate)

        z = torch.tanh(z_gate)
        o = torch.sigmoid(o_gate) # Output gate usually remains sigmoid

        # Memory update with normalizer
        # c_t = f_t * c_{t-1} + i_t * z_t
        # n_t = f_t * n_{t-1} + i_t
        c = f * c_prev + i * z
        n = f * n_prev + i

        # Normalized hidden state
        # h_t = o_t * tanh(c_t / n_t)
        h = o * torch.tanh(c / (n + 1e-6)) # Add epsilon to prevent division by zero

        return h, (h, c, n)

class sLSTM(nn.Module):
    """
    A multi-layer sLSTM network.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            sLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            ) for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, init_states=None):
        """
        x: (batch_size, seq_len, input_size)
        init_states: List of tuples (h_0, c_0, n_0) for each layer
        """
        batch_size, seq_len, _ = x.size()

        if init_states is None:
            init_states = [
                (torch.zeros(batch_size, self.hidden_size, device=x.device),
                 torch.zeros(batch_size, self.hidden_size, device=x.device),
                 torch.ones(batch_size, self.hidden_size, device=x.device)) # Initialize normalizer to 1
                for _ in range(self.num_layers)
            ]

        states = init_states
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            for i, cell in enumerate(self.cells):
                h_t, states[i] = cell(x_t, states[i])
                x_t = h_t
                if i < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, states
