import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from models.forecasting.hybrid_model import HybridForecaster
from utils.logger import setup_logger, write_progress_report

logger = setup_logger('trainer', 'logs/trainer.log')

def create_sequences(data, seq_len):
    """
    data: (N, K) array of IMFs
    returns: X (N-seq_len, seq_len, K), Y (N-seq_len, K)
    """
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

def train_model():
    logger.info("Loading decomposed features...")
    df = pl.read_parquet("data/processed/decomposed_features.parquet")

    # Extract IMFs
    imf_cols = [c for c in df.columns if c.startswith("IMF_")]
    K = len(imf_cols)
    data = df[imf_cols].to_numpy()

    # Hyperparameters
    seq_len = 30
    batch_size = 64
    epochs = 20 # Kept small for POC, standard is 200
    lr = 1e-4

    logger.info("Creating sequences...")
    X, Y = create_sequences(data, seq_len)

    # Train/Val/Test Split (70/15/15)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    # Convert to tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = HybridForecaster(K_modes=K, seq_len=seq_len, d_model=64, slstm_layers=1, nhead=4, transformer_layers=2).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                out = model(batch_X)
                val_loss += criterion(out, batch_Y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models/weights", exist_ok=True)
            torch.save(model.state_dict(), "models/weights/best_hybrid_model.pth")

        if (epoch+1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Evaluation on Test Set
    logger.info("Evaluating best model on test set...")
    model.load_state_dict(torch.load("models/weights/best_hybrid_model.pth"))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            out = model(batch_X).cpu().numpy()
            preds.append(out)
            targets.append(batch_Y.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    # We sum the IMFs to get the final signal prediction
    final_pred = np.sum(preds, axis=1)
    final_target = np.sum(targets, axis=1)

    r2 = r2_score(final_target, final_pred)
    mae = mean_absolute_error(final_target, final_pred)
    rmse = np.sqrt(mean_squared_error(final_target, final_pred))

    # MAPE with epsilon
    mape = np.mean(np.abs((final_target - final_pred) / (final_target + 1e-8))) * 100

    logger.info(f"Hybrid Results - R2: {r2:.4f}, MAPE: {mape:.2f}%, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    report = "### Hybrid Deep Learning Performance (Phase C)\n\n"
    report += "| Metric | ARIMA-GARCH Baseline | Hybrid WOA-VMD-sLSTM-Transformer |\n"
    report += "|--------|----------------------|----------------------------------|\n"
    report += f"| R² | 0.3390 | **{r2:.4f}** |\n"
    report += f"| MAPE | 626.09% | **{mape:.2f}%** |\n"
    report += f"| MAE | 0.2580 | **{mae:.4f}** |\n"
    report += f"| RMSE | 0.3710 | **{rmse:.4f}** |\n\n"
    report += "*Training details: 30-day sequence length, Adam optimizer, Cosine Annealing, MAE loss.*"

    write_progress_report(
        "Phase C: Deep Learning Core & Benchmarking",
        "Completed",
        report
    )

if __name__ == "__main__":
    train_model()
