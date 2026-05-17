import torch
import torch.nn as nn
from models.forecasting.slstm import sLSTM
from models.forecasting.transformer import TransformerEncoderModule

class HybridForecaster(nn.Module):
    """
    Hybrid WOA-VMD-sLSTM-Transformer pipeline model.
    Takes `K` decomposed IMFs as input.
    """
    def __init__(self, K_modes, seq_len, d_model=128, slstm_layers=2, nhead=8, transformer_layers=3):
        super(HybridForecaster, self).__init__()
        self.K_modes = K_modes

        # 1. Feature Extractor: sLSTM for each IMF mode separately
        # We project the 1D IMF into d_model dimensional space
        self.imf_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model // 2),
                sLSTM(input_size=d_model // 2, hidden_size=d_model, num_layers=slstm_layers)
            ) for _ in range(K_modes)
        ])

        # 2. Attention: Transformer Encoder on combined features
        self.transformer = TransformerEncoderModule(
            d_model=d_model,
            nhead=nhead,
            num_layers=transformer_layers
        )

        # 3. Output Heads: Predict the t+1 value for each IMF
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(K_modes)
        ])

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, K_modes)
        returns: (batch_size, K_modes) predictions for t+1
        """
        batch_size, seq_len, K = x.size()

        imf_features = []
        for i in range(K):
            # Extract IMF
            imf = x[:, :, i].unsqueeze(-1) # (batch, seq, 1)

            # Pass through its dedicated sLSTM
            linear_proj = self.imf_extractors[i][0]
            slstm_model = self.imf_extractors[i][1]

            proj = linear_proj(imf) # (batch, seq, d_model/2)
            out, _ = slstm_model(proj) # (batch, seq, d_model)

            # We only want the last timestep representation for each IMF to feed to transformer
            # Or we can feed the sequence. The blueprint implies attending across IMFs.
            # We will take the last timestep representation
            imf_features.append(out[:, -1, :].unsqueeze(1)) # (batch, 1, d_model)

        # Combine extracted representations
        # Shape: (batch, K_modes, d_model)
        combined = torch.cat(imf_features, dim=1)

        # Apply Transformer Encoder to learn relationships between the K different IMFs
        attended = self.transformer(combined) # (batch, K_modes, d_model)

        # Predict the next step for each IMF independently from attended features
        predictions = []
        for i in range(K):
            pred = self.heads[i](attended[:, i, :]) # (batch, 1)
            predictions.append(pred)

        # Final output shape: (batch, K_modes)
        out = torch.cat(predictions, dim=1)

        return out
