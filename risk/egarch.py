import numpy as np
import polars as pl
from arch import arch_model
from utils.logger import setup_logger

logger = setup_logger('egarch_volatility', 'logs/egarch.log')

def fit_egarch(returns_data, p=1, q=1):
    """
    Fits an EGARCH model to extract conditional volatility.
    EGARCH models asymmetric shocks (e.g., negative news hitting harder).

    Parameters:
    - returns_data: numpy array of historical log returns
    - p: GARCH lag order
    - q: ARCH lag order

    Returns:
    - Conditional volatility for the last day
    - Standard deviation of the residuals
    """
    logger.info("Fitting EGARCH model...")
    # Scale returns by 100 for better optimization stability in arch
    scaled_returns = returns_data * 100

    try:
        # Volatility = EGARCH, distribution = studentst (heavy tails)
        model = arch_model(scaled_returns, mean='ARX', vol='EGARCH', p=p, q=q, dist='studentst')
        result = model.fit(disp='off')

        # We need the forecasted conditional volatility for the next time step
        forecast = result.forecast(horizon=1)
        # Rescale back
        next_volatility = np.sqrt(forecast.variance.values[-1, :][0]) / 100.0

        # Estimate jump standard deviation from residuals
        residuals = result.resid / 100.0
        # Calculate standard deviation of residuals for the jump diffusion parameter estimation
        resid_std = np.nanstd(residuals)

        logger.info(f"EGARCH Fit Complete. Next Vol: {next_volatility:.6f}, Resid Std: {resid_std:.6f}")
        return {
            "current_volatility": next_volatility,
            "residual_std": resid_std,
            "model_fit": result
        }
    except Exception as e:
        logger.error(f"Failed to fit EGARCH: {str(e)}")
        # Fallback to standard deviation if optimization fails
        std_vol = np.std(returns_data)
        return {
            "current_volatility": std_vol,
            "residual_std": std_vol,
            "model_fit": None
        }

if __name__ == "__main__":
    # Test on preprocessed data
    df = pl.read_parquet("data/processed/master_dataset.parquet")
    # Take the normalized log returns
    signal = df["keua_close_log_ret_norm"].to_numpy()

    res = fit_egarch(signal)
    print("EGARCH Result:", res["current_volatility"])
