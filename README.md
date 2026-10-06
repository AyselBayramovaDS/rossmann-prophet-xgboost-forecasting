# Rossmann Demand Forecasting: Prophet + XGBoost Hybrid

Forecasting the total daily sales of the Rossmann store chain with a hybrid model: **Prophet** captures trend and seasonality, **XGBoost** learns Prophet's errors from promotions, holidays, open stores and store attributes. The hybrid is compared with Prophet alone and with an end-to-end XGBoost to check whether the added complexity is justified.

## Result

Validation: last 60 days (2015-06-02 to 2015-07-31).

| Model | MAE | MAPE (%) | RMSPE (%) |
|---|---|---|---|
| Prophet | 1,180,093 | 24.82 | 36.37 |
| End-to-end XGBoost | **419,422** | **5.77** | **7.59** |
| Hybrid (Prophet + XGBoost) | 509,997 | 16.99 | 32.11 |

The hybrid beats Prophet alone on all three metrics but is clearly worse than end-to-end XGBoost, so in this setup its added complexity is not justified. Most of the gap comes from Sundays, when only about 27 stores are open (Sunday MAPE: Prophet 80.0%, hybrid 81.9%, end-to-end XGBoost 5.2%). See [`note.md`](note.md) for the full write-up.

## Repository contents

| File | Description |
|---|---|
| `rossmannss.ipynb` | The analysis: data cleaning, EDA, series setup, the three models, metrics and error analysis |
| `note.md` | Written report: decisions, findings, results and limitations |

## Approach

1. Clean `train.csv` and `store.csv` and merge them on `Store`.
2. Explore the data (trend, weekday, month, promo, holidays, store type).
3. Aggregate the 1115 stores into one daily series of total sales (942 days).
4. Split chronologically: 882 days for training, 60 days for validation.
5. Fit Prophet, then train XGBoost on Prophet's residuals (hybrid), and train an end-to-end XGBoost on `Sales`.
6. Compare the models with MAE, MAPE and RMSPE and analyse the errors.

## How to run

1. Download `train.csv` and `store.csv` from the [Rossmann Store Sales](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales) dataset on Kaggle and place them next to the notebook. The data files are not included in this repository.
2. Install the dependencies:
   ```
   pip install pandas numpy matplotlib scikit-learn prophet xgboost jupyter
   ```
3. Open `rossmannss.ipynb` and run all cells.

**Troubleshooting:** on older CPUs without AVX2 the kernel may crash right after Prophet fits, because of the `polars` package that Prophet loads. Installing `polars-lts-cpu` instead of `polars` resolves this.

## Data

Rossmann Store Sales (Kaggle): `train.csv` (daily sales history, 1,017,209 rows, 1115 stores) and `store.csv` (one row of fixed attributes per store). `test.csv` is not used, because its real sales are hidden.
