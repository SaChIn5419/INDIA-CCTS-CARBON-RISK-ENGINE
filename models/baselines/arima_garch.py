import polars as pl
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from utils.logger import setup_logger, write_progress_report
import warnings

warnings.filterwarnings("ignore")

logger = setup_logger('baseline_model', 'logs/baseline_model.log')

def load_data(filepath="data/processed/master_dataset.parquet"):
    df = pl.read_parquet(filepath)
    # We will use the normalized log returns of KEUA as our target
    target = df["keua_close_log_ret_norm"].to_numpy()

    # Exogenous variables for ARIMAX
    exog_cols = [
        "krbn_close_log_ret_norm",
        "brent_close_log_ret_norm",
        "nifty_close_log_ret_norm",
        "usdinr_close_log_ret_norm",
        "coal_india_close_log_ret_norm"
    ]
    exog = df[exog_cols].to_numpy()

    return target, exog, df["date"].to_list()

def train_arima_garch():
    logger.info("Loading preprocessed dataset...")
    target, exog, dates = load_data()

    # Train/Test Split (85% train, 15% test as per Phase A, ignoring validation for baseline simplicty)
    split_idx = int(len(target) * 0.85)

    train_target, test_target = target[:split_idx], target[split_idx:]
    train_exog, test_exog = exog[:split_idx], exog[split_idx:]

    logger.info(f"Training ARIMA model on {len(train_target)} samples...")
    # Step 1: ARIMAX Model for the conditional mean
    # We use a simple (1,0,1) order for baseline speed
    arima_model = ARIMA(train_target, exog=train_exog, order=(1, 0, 1))
    arima_result = arima_model.fit()

    logger.info("Forecasting mean with ARIMA...")
    # Forecast mean
    arima_forecast = arima_result.forecast(steps=len(test_target), exog=test_exog)

    logger.info("Extracting residuals to train GARCH...")
    # Step 2: GARCH Model for conditional variance using ARIMA residuals
    residuals = arima_result.resid

    garch_model = arch_model(residuals, vol='GARCH', p=1, q=1, dist='Normal')
    garch_result = garch_model.fit(disp='off')

    logger.info("Forecasting variance with GARCH...")
    garch_forecast = garch_result.forecast(horizon=len(test_target))
    # We get the variance forecast
    variance_forecast = garch_forecast.variance.values[-1, :]

    # Combine predictions (for point forecast, we just use the ARIMA mean)
    # GARCH gives us the confidence intervals/risk which is Phase D, but we evaluate point accuracy now

    logger.info("Computing metrics...")

    # Note: MAPE can be tricky with normalized returns (which cluster around 0)
    # We compute it anyway, but MAE/RMSE/R2 are more robust here.

    # Adding epsilon to avoid zero division in MAPE
    epsilon = 1e-8
    mape = np.mean(np.abs((test_target - arima_forecast) / (test_target + epsilon))) * 100

    r2 = r2_score(test_target, arima_forecast)
    mae = mean_absolute_error(test_target, arima_forecast)
    rmse = np.sqrt(mean_squared_error(test_target, arima_forecast))

    logger.info(f"Baseline Results - R2: {r2:.4f}, MAPE: {mape:.2f}%, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return {
        "R2": r2,
        "MAPE": mape,
        "MAE": mae,
        "RMSE": rmse
    }

if __name__ == "__main__":
    logger.info("Starting baseline training...")
    metrics = train_arima_garch()

    report = "### ARIMA-GARCH Baseline Performance\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| R² | {metrics['R2']:.4f} |\n"
    report += f"| MAPE | {metrics['MAPE']:.2f}% |\n"
    report += f"| MAE | {metrics['MAE']:.4f} |\n"
    report += f"| RMSE | {metrics['RMSE']:.4f} |\n\n"
    report += "*Note: Baseline target is normalized log returns.*"

    write_progress_report(
        "Phase A: Baseline Model (ARIMA-GARCH)",
        "Completed",
        report
    )
    logger.info("Baseline training complete.")
