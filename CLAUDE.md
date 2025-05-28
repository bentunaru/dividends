# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dividend analysis project that analyzes the top-performing dividend aristocrat stocks over the last 10 years using machine learning techniques. The project uses yfinance to fetch financial data from Yahoo Finance and applies various ML algorithms (XGBoost, Random Forest, etc.) to predict future performance.

## Environment Setup

### Virtual Environment
```bash
# Activate the virtual environment
source dividends/bin/activate

# Install dependencies (if needed)
pip install yfinance pandas numpy matplotlib seaborn scikit-learn xgboost optuna
```

### Jupyter Notebook
```bash
# Start Jupyter Lab/Notebook
jupyter lab stocks.ipynb
# or
jupyter notebook stocks.ipynb
```

## Project Architecture

### Main Components

1. **YFinanceClient**: Central data fetching class that handles:
   - Historical stock data retrieval with caching
   - Company fundamental information
   - Dividend data integration
   - Error handling for failed API calls

2. **Data Collection Pipeline**:
   - Predefined list of dividend aristocrats (15 stocks: JNJ, PG, KO, etc.)
   - 10+ years of historical data (2014-present)
   - Automatic dividend amount calculation and integration

3. **Analysis Framework**:
   - Time series analysis with proper date handling
   - Machine learning pipeline using scikit-learn
   - Feature engineering for financial metrics
   - Cross-validation using TimeSeriesSplit for temporal data

### Key Configuration

- **Stock Universe**: `DIVIDEND_ARISTOCRATS` list contains 15 dividend-paying stocks
- **Time Period**: `START_DATE = "2014-01-01"` to present
- **Data Source**: Yahoo Finance via yfinance (no API key required)

### Data Structure

The `YFinanceClient.get_stock_data()` returns DataFrames with columns:
- Standard OHLCV: `open`, `high`, `low`, `close`, `adj close`, `volume`
- Dividend data: `dividends`, `dividend_amount`, `stock splits`
- Index: DatetimeIndex with timezone-aware timestamps

## Development Workflow

### Working with the Notebook

1. Run cells sequentially - the notebook has dependencies between cells
2. The `YFinanceClient` uses caching to avoid re-downloading data
3. Test with smaller stock lists first (see cell 11 for example)

### Adding New Stocks

Update the `DIVIDEND_ARISTOCRATS` list with new ticker symbols. Ensure they are valid Yahoo Finance tickers.

### Extending Analysis

The project is structured for ML pipeline extension:
- Feature engineering can be added after data collection
- New models can be integrated with the existing sklearn pipeline
- Optuna is included for hyperparameter optimization

## Data Handling Notes

- All data is cached in the `YFinanceClient` instance to avoid re-downloading
- Dividend amounts are properly aligned with trading dates
- Missing data is handled gracefully with informative error messages
- Time series data maintains proper temporal ordering for ML applications