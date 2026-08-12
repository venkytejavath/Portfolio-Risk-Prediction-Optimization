# Portfolio Risk Classification & Optimization

An end-to-end pipeline combining Modern Portfolio Theory with machine learning
for portfolio volatility prediction, risk classification, and Sharpe-optimal
allocation — implementing and extending the methodology of Rathi et al.
(ICAART 2024).

Given a set of asset weights, the pipeline predicts portfolio risk (via
classification and regression), validates those predictions against real
held-out market data, and separately computes the Sharpe-optimal allocation
using the Efficient Frontier — then cross-checks the optimizer's output
against the trained models.

## Dataset

- **Assets (7 tickers):** VTSAX, VTIAX, VBTLX, VDE, VGSLX, OPGSX, BTC-USD
  (six mutual funds spanning equities/bonds/real estate, plus one
  cryptocurrency)
- **Price history:** daily adjusted close, 2015-01-01 to 2025-12-30, via the
  `yfinance` API
- **Simulated portfolios:** 10,000 random weight allocations (Dirichlet
  sampling with realistic constraints, e.g. BTC-USD ≤ 10%) used to build the
  training set for volatility/risk-category prediction

## Pipeline

| Notebook | Purpose |
|---|---|
| `data_collection_and_portfolio_construction.ipynb` | Downloads price history, computes returns, simulates 10,000 random portfolios |
| `EDA.ipynb` | Exploratory analysis of returns, correlations, and volatility distributions |
| `Portfolio_Metrics_Preprocessing.ipynb` | Computes expected returns (CAGR), covariance matrix, analytical portfolio volatility, and risk-category labels (quartile-based) |
| `Classification_models.ipynb` | Random Forest & XGBoost classifiers (Optuna-tuned) predicting risk category from weights; SHAP interpretability |
| `Regression_models.ipynb` | Random Forest & XGBoost regressors (Optuna-tuned) predicting volatility from weights |
| `Portfolio_Optimization.ipynb` | Efficient Frontier / Sharpe-ratio optimization via `scipy.optimize`, with user-defined allocation constraints; cross-checks optimized portfolios against the trained models |
| `Model_Validation.ipynb` | Out-of-sample validation: tests trained models against **real, never-seen 2026 market data** |

## Key results

**Classification (risk category: Low / Medium / Moderate / High)**

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest | 92.7% | 0.9273 |
| XGBoost | 95.95% | 0.9597 |

**Regression (annualized volatility)**

| Model | RMSE | R² |
|---|---|---|
| Random Forest | 0.00199 | 0.9956 |
| XGBoost | 0.00917 | 0.9072 |

**Out-of-sample validation (real 2026 H1 data, unseen during training)**

| Model | Mean Absolute Error |
|---|---|
| Random Forest | 3.06% |
| XGBoost | 2.95% |

Training data ends 2025-12-30; validation data spans 2026-01-01 to
2026-06-30, so this is a genuine test of generalization to a period the
models never saw — not just a re-test on the training distribution. For
comparison, the original paper (Rathi et al., 2024) reports a mean absolute
error of 3.57% on their own out-of-sample validation, so these results are
consistent with the published benchmark.

## Assumptions & limitations

**Assumptions:**
- Annualized volatility (σ = daily std × √252) is treated as a sufficient
  proxy for portfolio risk.
- The risk-free rate is held constant (6%) rather than time-varying,
  following standard Sharpe Ratio practice (Sharpe, 1998), and is used for
  reproducibility rather than pulled live.
- Portfolio weight bounds (BTC-USD ≤ 10%, OPGSX ≤ 20%, VGSLX ≤ 20%) reflect
  common retail allocation guidance rather than a specific investor's
  constraints.
- Random Forest / XGBoost are assumed suitable given their robustness to the
  moderate-dimensional, tabular nature of the weight vectors.

**Limitations:**
- The Efficient Frontier / Sharpe optimization assumes returns are
  approximately normally distributed; real markets exhibit fat tails, so
  extreme-event risk may be understated.
- Models are trained on daily closing prices only; intraday volatility
  dynamics are not captured.
- Because volatility is a deterministic function of weights given a fixed
  covariance matrix, in-sample classification/regression accuracy reflects
  how well the model approximates that function under a *fixed* covariance
  estimate — the out-of-sample validation above is what actually tests
  generalization to a new period.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in the order listed in the Pipeline table above — each
downstream notebook loads `.pkl` artifacts (`data.pkl`, `Returns.pkl`,
`Expected_Returns.pkl`, `Covariance.pkl`, `portfolio_data.pkl`, trained
models) saved by the earlier ones.

## Reference

This project's methodology (ticker selection, weight-generation scheme,
volatility target construction, and RF-based risk classification/regression)
follows and extends:

> Rathi, V., Kshirsagar, M., & Ryan, C. (2024). Enhancing Portfolio
> Performance: A Random Forest Approach to Volatility Prediction and
> Optimization. *Proceedings of the 16th International Conference on Agents
> and Artificial Intelligence (ICAART 2024)*, 1278–1285.
> https://doi.org/10.5220/0012464600003636



Extensions made in this project: XGBoost as a second model family alongside
Random Forest, Optuna-based hyperparameter tuning (vs. grid search in the
original), SHAP-based interpretability analysis, and out-of-sample
validation against real market data from a period after training cutoff.