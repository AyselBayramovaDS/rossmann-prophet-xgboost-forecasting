# Demand Forecasting for Rossmann: Prophet, XGBoost and a Hybrid of Both

## 1. Objective

Forecast the daily sales of the Rossmann store chain with a hybrid model and test, with numbers, whether the hybrid is better than simpler alternatives.

- **Prophet** captures the time structure: trend, weekly and yearly seasonality.
- **XGBoost** learns Prophet's errors (residuals) from external factors: open stores, promotions, holidays and store attributes.
- **Hybrid forecast = Prophet forecast + XGBoost residual forecast.**

The hybrid is compared with Prophet alone and with an end-to-end XGBoost (no Prophet, `Sales` is the direct target) using MAE, MAPE and RMSPE. The question is whether the added complexity is justified by the numbers.

## 2. Workflow

| Step | What was done | Why |
|---|---|---|
| 1 | Read and cleaned `train.csv` and `store.csv` separately | Find problems in the file they belong to |
| 2 | Merged them on `Store` | XGBoost needs daily factors and store attributes together |
| 3 | Explored the data with 7 charts | Decide what Prophet and what XGBoost should learn |
| 4 | Aggregated 1115 stores into one daily series | Prophet needs a single series |
| 5 | Split the series chronologically (882 / 60 days) | Evaluate on days the models have not seen |
| 6 | Built Prophet, hybrid and end-to-end XGBoost | The three models to compare |
| 7 | Compared them on MAE, MAPE, RMSPE and analysed the errors | Answer the objective |

## 3. Data

| File | Content |
|---|---|
| `train.csv` | Daily sales history: `Store`, `Date`, `Sales`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday` (1,017,209 rows, 1115 stores) |
| `store.csv` | One row per store: `StoreType`, `Assortment`, `CompetitionDistance`, `Promo2` |

`test.csv` is not used: it covers the future period of the Kaggle competition and its real sales are hidden.

## 4. Data preparation

- **`Date`** was read as datetime (`parse_dates`), which Prophet and the calendar features require. `low_memory=False` avoids type mismatches in mixed-type columns such as `StateHoliday`.
- **`StateHoliday`** (`0` none, `a` public holiday, `b` Easter, `c` Christmas) was one-hot encoded into `NoHoliday`, `PublicHoliday`, `EasterHoliday`, `Christmas`, because XGBoost cannot handle text.
- **Sorting:** the data came in descending date order and was reversed to chronological order, so that the model learns from the past to predict the future.
- **`Customers`** was dropped. It is not known for future days, so keeping it would leak information and give falsely optimistic scores.
- **`store.csv`:** five columns not needed for the task were dropped (`Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval`, `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`). The `Promo2Since*` and `PromoInterval` blanks (544) belong to stores with `Promo2 = 0`, which are not in the programme, so they were not missing data.
- **`CompetitionDistance`** was missing for 3 stores (291, 622, 879) and was filled with the median, which is not affected by extreme values. Only 3 of 1115 stores are affected.
- **`StoreType` and `Assortment`** were one-hot encoded. `Assortment` columns were named after the data description: `basic`, `extra`, `extended`.
- **Merge:** `train.merge(store, on="Store", how="left", validate="many_to_one")`. `how="left"` keeps every row of `train`; `validate` raises an error if `store` had duplicate stores, which would silently multiply rows. The result has 1,017,209 rows and 20 columns, with no missing values or duplicates.
- **Zero sales:** 172,871 rows have `Sales = 0`; 172,817 rows are closed stores (`Open = 0`) and 54 are open stores with no recorded sales. Zero sales therefore almost entirely means a closed store. These 54 rows were not examined individually.

## 5. Exploratory analysis

| # | Chart | Finding | Relevant to |
|---|---|---|---|
| 1 | Weekly average daily sales | Fluctuates between about 5 and 7 million with no clear upward or downward trend; a peak every December and a dip in the second half of 2014 | Prophet (trend, yearly seasonality) |
| 2 | Average sales by weekday, open stores only | Describes the sales of one open store. Sunday looks high only because the few stores open on Sundays are strong ones | Store behaviour |
| 3 | Average sales by month, open stores only | December is highest (about 8,600), other months fairly flat. January-July include three years, August-December two | Prophet |
| 4 | Average sales by `Promo` | About 8,200 on promo days vs 5,900 without, roughly 38% higher | XGBoost |
| 5 | Sales by holiday type and store type | Open stores sell more on holidays than on normal days (holiday groups contain few rows because most stores are closed); store type `b` is clearly higher than the others | XGBoost |
| 6 | Total daily sales by weekday (aggregated series) | Monday about 8.43 million, Tuesday 7.56, Wednesday 7.07, Thursday 6.75, Friday 7.26, Saturday 6.32, **Sunday about 0.22 million** (about 27 stores open on average) | Prophet (weekly seasonality) |
| 7 | Total daily sales by month (aggregated series) | December highest (about 7.0 million), August-October lowest (about 5.7 million) | Prophet (yearly seasonality) |

Charts 1, 6 and 7 show time structure that Prophet can capture. Charts 4 and 5 show external factors that Prophet does not see and XGBoost can learn. This is the reason for a two-part hybrid.

## 6. Series setup

The 1115 stores are branches of one chain, so the daily records were summed into **one series of the chain's total daily sales** (942 days, 2013-01-01 to 2015-07-31). Prophet requires a single series, and a separate Prophet per store would mean 1115 models.

| Column | Aggregation | Meaning |
|---|---|---|
| `Sales` | sum | Total chain sales that day |
| `Open` | sum | Number of stores open that day |
| `Promo`, `SchoolHoliday`, `PublicHoliday`, `EasterHoliday`, `Christmas` | mean | Share of stores where the flag is active |
| `CompetitionDistance`, `Promo2`, `StoreType_*`, `Assortment_*` | mean over **open stores only** | Composition of the stores that were actually trading |

Result: 942 rows, 17 columns, no missing values or duplicates.

**Drawback:** differences between individual stores are lost, and store attributes change little from day to day, so they are a weak signal in the aggregated series.

## 7. Train / validation split

The split is chronological, not random, because letting future days into training would leak information. The last **60 days** (2015-06-02 to 2015-07-31) are used for validation and the earlier **882 days** (2013-01-01 to 2015-06-01) for training. All three models are trained and evaluated on the same split.

## 8. Models

**Prophet baseline.** `Prophet(yearly_seasonality=True, weekly_seasonality=True)`, fitted on date and total sales only. It is the number the hybrid has to beat.

**Hybrid (Prophet + XGBoost).**
1. Prophet predicts the training days; residual = actual sales − Prophet prediction.
2. XGBoost (`n_estimators=100`, `learning_rate=0.05`, `max_depth=5`, `random_state=42`) is trained on these residuals using 15 features: `Open`, `Promo`, `SchoolHoliday`, the holiday shares, `CompetitionDistance`, `Promo2`, `StoreType_*` and `Assortment_*`.
3. On validation: final forecast = Prophet prediction + XGBoost residual prediction.

**End-to-end XGBoost.** Same hyperparameters, with `Sales` as the direct target, using the same 15 features plus calendar features (`DayOfWeek`, `Month`, `Day`), 18 in total.

## 9. Metrics

- **MAE:** mean absolute error, in sales units.
- **MAPE:** mean absolute percentage error.
- **RMSPE:** `sqrt(mean(((y − ŷ) / y)²))`. It measures relative error and punishes large percentage errors heavily; it is the official Kaggle metric for this competition.
- Rows with `y = 0` are excluded from RMSPE to avoid division by zero. The aggregated series has no zero-sales days, but the rule is kept.

## 10. Results

| Model | MAE | MAPE (%) | RMSPE (%) |
|---|---|---|---|
| Prophet | 1,180,093 | 24.82 | 36.37 |
| End-to-end XGBoost | **419,422** | **5.77** | **7.59** |
| Hybrid (Prophet + XGBoost) | 509,997 | 16.99 | 32.11 |

| Comparison | MAE | MAPE | RMSPE |
|---|---|---|---|
| Hybrid vs Prophet | error reduced by 56.8% | error reduced by 31.5% | error reduced by 11.7% |
| Hybrid vs end-to-end XGBoost | error 1.2 times larger | error 2.9 times larger | error 4.2 times larger |

The notebook prints the RMSPE comparison with end-to-end XGBoost as −323%. The formula is (7.59 − 32.11) / 7.59; the negative sign means the hybrid is worse, and 32.11 / 7.59 ≈ 4.2 gives the same fact as a ratio.

**Prophet on validation.** The chart shows that Prophet follows the weekly rhythm but misses the Monday peaks (real sales reach 10-12 million, the forecast is about 8 million). It has no information about promotions, holidays or the number of open stores.

**Error analysis.** MAPE on Sundays vs all other days of the validation window (8 of the 60 days are Sundays):

| Model | Sunday MAPE (%) | Other days MAPE (%) |
|---|---|---|
| Prophet | 80.0 | 16.3 |
| Hybrid | 81.9 | 7.0 |
| End-to-end XGBoost | 5.2 | 5.9 |

- On the other 52 days the hybrid cuts Prophet's error from 16.3% to 7.0% and ends up close to end-to-end XGBoost (5.9%).
- On Sundays the hybrid is no better than Prophet (81.9% vs 80.0%), while end-to-end XGBoost is at 5.2%. Sales on Sundays are only about 0.22 million, so even a small absolute miss is a large percentage error.
- RMSPE squares percentage errors, so these 8 Sundays dominate it. This is why the hybrid's RMSPE (32.11%) stays close to Prophet's (36.37%) although its MAPE on other days is much lower.

## 11. Conclusion

The hybrid is better than Prophet alone on all three metrics, but it is clearly worse than end-to-end XGBoost on all three. In this setup **the added complexity of the hybrid is not justified by the numbers**: the simpler end-to-end XGBoost is both more accurate and easier to build.

The analysis shows where the gap comes from: almost all of it is on Sundays. It does not show why the hybrid's XGBoost fails to correct Sundays although it receives `Open`; this was not investigated.

## 12. Limitations

- **In-sample residuals:** XGBoost learns Prophet's errors on training days that Prophet has already seen, so these errors are smaller than the errors on new days.
- **Different feature sets:** only the end-to-end model receives calendar features (`DayOfWeek`, `Month`, `Day`), as the task description specifies.
- **`Open` as a feature:** the number of open stores is strongly tied to sales and is assumed to be known in advance for the forecast day, for example from an opening schedule.
- **One validation window:** the results are based on a single 60-day period (June-July 2015); the ranking could differ in another season, such as December.
- **Fixed hyperparameters:** both XGBoost models use the same settings, and no tuning was done.
- **Aggregation:** the result applies to the chain's total sales, not to individual stores.