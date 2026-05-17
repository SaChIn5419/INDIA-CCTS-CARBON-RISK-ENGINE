import yfinance as yf
import polars as pl
import os
from utils.logger import setup_logger, write_progress_report

logger = setup_logger('data_fetcher', 'logs/data_fetcher.log')

# Define the assets and their yfinance tickers
TICKERS = {
    "KEUA": "KEUA",
    "KRBN": "KRBN",
    "BRENT": "BZ=F",
    "NIFTY": "^NSEI",
    "USD_INR": "USDINR=X",
    "COAL_INDIA": "COALINDIA.NS"
}

def fetch_yfinance_data(ticker_symbol: str, start_date="2018-01-01") -> pl.DataFrame:
    """Fetches data from Yahoo Finance and returns a Polars DataFrame."""
    logger.info(f"Fetching data for {ticker_symbol} from {start_date}...")
    try:
        data = yf.download(ticker_symbol, start=start_date, progress=False)
        if data.empty:
            logger.warning(f"No data found for {ticker_symbol}.")
            return pl.DataFrame()

        # Reset index to make Date a column
        data.reset_index(inplace=True)

        # In newer yfinance versions, columns might be MultiIndex if we requested multiple tickers,
        # but here we request one by one. We should flatten if it's MultiIndex.
        if isinstance(data.columns, list) or hasattr(data.columns, "levels"):
            # Flatten columns to string
            data.columns = ['_'.join(col).strip('_') if type(col) is tuple else str(col) for col in data.columns]

        # Convert pandas DataFrame to Polars DataFrame
        pl_df = pl.from_pandas(data)

        # Rename 'Date' or 'Datetime' column to lowercase 'date' for consistency
        if "Date" in pl_df.columns:
            pl_df = pl_df.rename({"Date": "date"})
        elif "Datetime" in pl_df.columns:
            pl_df = pl_df.rename({"Datetime": "date"})

        logger.info(f"Successfully fetched {len(pl_df)} rows for {ticker_symbol}.")
        return pl_df

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker_symbol}: {str(e)}")
        return pl.DataFrame()

def fetch_and_save_all(output_dir="data/cache"):
    """Fetches all configured assets and saves them as Parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, ticker in TICKERS.items():
        df = fetch_yfinance_data(ticker)
        if not df.is_empty():
            out_path = os.path.join(output_dir, f"{name}.parquet")
            df.write_parquet(out_path)
            logger.info(f"Saved {name} to {out_path}")
            results[name] = len(df)
        else:
            results[name] = 0

    return results

if __name__ == "__main__":
    logger.info("Starting data fetching process...")
    results = fetch_and_save_all()

    report_details = "### Data Fetcher Results\n\n| Asset | Rows |\n|-------|------|\n"
    for name, rows in results.items():
        report_details += f"| {name} | {rows} |\n"

    write_progress_report(
        "Phase A: High-Performance Data Fetchers",
        "Completed",
        report_details
    )
    logger.info("Data fetching process complete.")
