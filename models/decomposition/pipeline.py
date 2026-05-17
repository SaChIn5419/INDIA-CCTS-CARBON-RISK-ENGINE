import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from models.decomposition.vmd import apply_vmd
from models.decomposition.woa_optimizer import WhaleOptimizationAlgorithm
from utils.logger import setup_logger, write_progress_report

logger = setup_logger('decomposition_pipeline', 'logs/decomposition_pipeline.log')

def run_decomposition_pipeline():
    logger.info("Loading preprocessed dataset for decomposition...")
    try:
        df = pl.read_parquet("data/processed/master_dataset.parquet")
    except Exception as e:
        logger.error(f"Could not load dataset: {str(e)}")
        return

    # We will decompose the normalized log returns of KEUA (Primary Carbon Price Signal)
    signal = df["keua_close_log_ret_norm"].to_numpy()
    dates = df["date"].to_list()

    # For testing/POC speed, we'll only optimize on the last 200 days
    # (In full production, we'd use train set or rolling window)
    window_size = min(200, len(signal))
    signal_sample = signal[-window_size:]

    logger.info(f"Running WOA Optimizer on a {window_size}-day sample...")

    # Initialize WOA (bounds: K=3..10, alpha=100..5000)
    # Using small pop_size and max_iter for execution speed in this environment.
    woa = WhaleOptimizationAlgorithm(
        signal=signal_sample,
        pop_size=5,
        max_iter=10,
        bounds=([3, 100], [8, 3000])
    )

    best_params, convergence = woa.optimize()

    best_K = int(best_params[0])
    best_alpha = best_params[1]

    logger.info(f"Applying VMD to full signal with optimal K={best_K}, alpha={best_alpha:.2f}...")

    # Apply VMD to the FULL signal using the discovered optimal parameters
    imfs = apply_vmd(signal, K=best_K, alpha=best_alpha)

    logger.info(f"Successfully extracted {len(imfs)} IMFs.")

    # Save the decomposed IMFs to parquet for Phase C
    # Note: vmdpy sometimes returns arrays that are shorter or longer by 1 due to padding/parity
    # We will slice them to exactly match the length of our signal
    imf_dict = {"date": dates, "original_signal": signal}
    for i, imf in enumerate(imfs):
        # Truncate or pad to match the original signal length to prevent ShapeError in Polars
        if len(imf) > len(signal):
            imf = imf[:len(signal)]
        elif len(imf) < len(signal):
            imf = np.pad(imf, (0, len(signal) - len(imf)), 'edge')

        imf_dict[f"IMF_{i+1}"] = imf

    imf_df = pl.DataFrame(imf_dict)

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "decomposed_features.parquet")
    imf_df.write_parquet(out_path)

    logger.info(f"Saved Decomposed IMFs to {out_path}")

    # Generate progress report details
    report = "### WOA-VMD Signal Decomposition\n\n"
    report += "**Whale Optimization Results:**\n"
    report += f"- **Optimal K (Modes)**: {best_K}\n"
    report += f"- **Optimal Alpha (Penalty)**: {best_alpha:.2f}\n"
    report += f"- **Final Minimum Envelope Entropy**: {convergence[-1]:.4f}\n\n"

    report += "**VMD Output:**\n"
    report += f"- Extracted **{len(imfs)} Intrinsic Mode Functions (IMFs)** from the primary KEUA signal.\n"
    report += f"- Decomposed data shape: {imf_df.shape}\n"
    report += f"- File saved to `data/processed/decomposed_features.parquet`\n"

    write_progress_report(
        "Phase B: Signal Decomposition",
        "Completed",
        report
    )

    return imfs

if __name__ == "__main__":
    run_decomposition_pipeline()
