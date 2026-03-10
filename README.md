# Basket Trading with Bayesian Optimization

A momentum-based basket trading strategy for S&P 500 stocks, with hyperparameter tuning via Bayesian optimization.

## Strategy

- Ranks stocks by momentum (price change over a lookback window)
- Selects top N performers to form an equal-weighted basket
- Rebalances daily

## Optimization

Uses Bayesian optimization with Gaussian Process regression to find optimal:
- **Momentum window**: Lookback period (63–252 days)
- **Basket size**: Number of top stocks to hold (5–40)

Objective: Maximize Sharpe ratio on training data (2019–2021), then validate on test data (2022+).

## Files

- `bayesian_optimization.py` — Bayesian optimizer with Expected Improvement acquisition
- `pipeline.ipynb` — Full pipeline: data download, optimization, backtesting, visualization
- `research.ipynb` — Exploratory research and alternative optimization with Optuna

## Usage

```bash
pip install numpy pandas yfinance scikit-learn scipy matplotlib
```

Run `pipeline.ipynb` to execute the full strategy pipeline.

## Results

Compares strategy equity curve and drawdowns against SPY benchmark.