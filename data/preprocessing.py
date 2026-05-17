import polars as pl
import duckdb
import os
import numpy as np
from utils.logger import setup_logger, write_progress_report

logger = setup_logger('data_preprocessing', 'logs/data_preprocessing.log')

def preprocess_data(cache_dir="data/cache", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    # We will use DuckDB to join the parquet files efficiently.
    con = duckdb.connect(database=':memory:')

    try:
        # Load tables (yfinance multiindex flattening produced columns like Close_KEUA)
        # Note: Depending on yfinance version and single vs multi download, columns might just be "Close"
        # We use DuckDB's robust COLUMNS regex or fallback to handle both cases securely.
        con.execute(f"CREATE TABLE keua AS SELECT date, COLUMNS('Close.*') as keua_close FROM read_parquet('{cache_dir}/KEUA.parquet')")
        con.execute(f"CREATE TABLE krbn AS SELECT date, COLUMNS('Close.*') as krbn_close FROM read_parquet('{cache_dir}/KRBN.parquet')")
        con.execute(f"CREATE TABLE brent AS SELECT date, COLUMNS('Close.*') as brent_close FROM read_parquet('{cache_dir}/BRENT.parquet')")
        con.execute(f"CREATE TABLE nifty AS SELECT date, COLUMNS('Close.*') as nifty_close FROM read_parquet('{cache_dir}/NIFTY.parquet')")
        con.execute(f"CREATE TABLE usd_inr AS SELECT date, COLUMNS('Close.*') as usdinr_close FROM read_parquet('{cache_dir}/USD_INR.parquet')")
        con.execute(f"CREATE TABLE coal_india AS SELECT date, COLUMNS('Close.*') as coal_india_close FROM read_parquet('{cache_dir}/COAL_INDIA.parquet')")

        # We use KEUA as the main driver, outer join to keep dates. DuckDB is fast.
        query = """
        SELECT
            k.date,
            k.keua_close,
            kr.krbn_close,
            b.brent_close,
            n.nifty_close,
            u.usdinr_close,
            c.coal_india_close
        FROM keua k
        LEFT JOIN krbn kr ON k.date = kr.date
        LEFT JOIN brent b ON k.date = b.date
        LEFT JOIN nifty n ON k.date = n.date
        LEFT JOIN usd_inr u ON k.date = u.date
        LEFT JOIN coal_india c ON k.date = c.date
        ORDER BY k.date
        """

        logger.info("Executing DuckDB join query...")
        joined_df = con.execute(query).pl()
        logger.info(f"Joined dataframe shape: {joined_df.shape}")

        # Impute missing values (forward fill then backward fill)
        joined_df = joined_df.fill_null(strategy="forward").fill_null(strategy="backward")

        # We need log returns for financial modeling (specifically for GARCH and VMD)
        # Log returns: log(P_t) - log(P_{t-1})

        # To compute log returns in Polars
        numeric_cols = [col for col in joined_df.columns if col != "date"]

        exprs = []
        for col in numeric_cols:
            # Add small epsilon to prevent log(0) just in case
            expr = (pl.col(col) / pl.col(col).shift(1)).log().alias(f"{col}_log_ret")
            exprs.append(expr)

        joined_df = joined_df.with_columns(exprs)

        # Drop the first row which will have null log returns
        joined_df = joined_df.drop_nulls()

        # Normalization (Z-score)
        norm_exprs = []
        for col in numeric_cols:
            log_ret_col = f"{col}_log_ret"
            mean_val = joined_df[log_ret_col].mean()
            std_val = joined_df[log_ret_col].std()

            expr = ((pl.col(log_ret_col) - mean_val) / std_val).alias(f"{col}_log_ret_norm")
            norm_exprs.append(expr)

        joined_df = joined_df.with_columns(norm_exprs)

        output_path = os.path.join(output_dir, "master_dataset.parquet")
        joined_df.write_parquet(output_path)
        logger.info(f"Successfully saved processed dataset to {output_path}")

        return joined_df

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
    finally:
        con.close()

if __name__ == "__main__":
    logger.info("Starting data preprocessing...")
    df = preprocess_data()

    stats_str = "### Preprocessing Stats\n\n"
    stats_str += f"- **Final Shape**: {df.shape}\n"
    stats_str += f"- **Start Date**: {df['date'].min()}\n"
    stats_str += f"- **End Date**: {df['date'].max()}\n\n"

    stats_str += "| Column | Mean | Std |\n"
    stats_str += "|--------|------|-----|\n"
    for col in df.columns:
        if "norm" in col:
            stats_str += f"| {col} | {df[col].mean():.4f} | {df[col].std():.4f} |\n"

    write_progress_report(
        "Phase A: Data Preprocessing with Polars & DuckDB",
        "Completed",
        stats_str
    )
    logger.info("Data preprocessing complete.")
