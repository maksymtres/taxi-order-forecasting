# taxi-order-forecasting
Time series forecasting model to predict the number of taxi orders for the next hour (RMSE-focused) using lag/rolling features and LightGBM.

## Project overview
The company **“Chyotenkoe Taxi”** collected historical data on taxi orders at airports.  
To attract more drivers during peak periods, we need to **forecast the number of taxi orders for the next hour**.

**Goal:** build a model that predicts next-hour demand with **RMSE ≤ 48** on the test set.

## Tech stack
- **Python** (Jupyter Notebook)
- **pandas**, **numpy**
- **scikit-learn** (Pipeline, GridSearchCV, TimeSeriesSplit, metrics)
- **lightgbm** (LGBMRegressor)
- **matplotlib**, **seaborn**

## Dataset
The data comes from a single file:

- `taxi.csv` — time-indexed series of taxi orders (`num_orders`)

> **Note:** The dataset is **not included** in this repository.  
> To run the notebook, place the file into `datasets/taxi.csv` or update the file path inside the notebook.

### Target
- `num_orders` — number of taxi orders per hour (the model uses past values to forecast the next hour).

## Approach
1. Load data, sort by time, **resample to 1-hour intervals** (sum).
2. EDA: identify trend, daily and weekly seasonality, peaks.
3. Feature engineering (leakage-safe):
   - time features: `hour`, `dayofweek`, `month`
   - lags: `lag_1 ... lag_24`
   - rolling means with shift(1): windows `3/6/12/24`
   - weekly features: `lag_168`, `roll_mean_168`
4. Train/validation:
   - hold-out test: **last 10%** of the time series (chronological split)
   - CV: **TimeSeriesSplit**
   - tuning via GridSearchCV
5. Evaluate using **RMSE** and compare with a Dummy baseline.

## Results
**Best model:** `LGBMRegressor` (LightGBM)

- **CV RMSE:** ~ **23.9**
- **Test RMSE:** **34.88** ✅ (meets requirement RMSE ≤ 48)
- **Baseline (DummyRegressor) RMSE:** **84.45**

The pipeline uses correct time splitting and generates train/test features separately, preventing data leakage.

## How to run

### System requirements
- Python **3.9+**
- Jupyter Notebook / JupyterLab

### Installation
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn jupyter
```
