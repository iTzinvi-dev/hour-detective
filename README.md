# hour-detective

A Baseline model for haggle to predict `hours_spent` on projects

## What it Does :
Uses `HistGradientBoostingRegressor` trained on `log1p(hours_spent)` to match RMSLE. Text columns are converted categories (aligned with train data), and final predictions are converted back with `expm1` and clipped at 0. 

it also had three activity ratio fetures (`commits_per_active_day`, `active_day_ratio`, `lines_per_active_day`) and tuned the model worh `max_iter=`, and `learning_rate=`.
## How to Run :
Put `train.csv` and `test.csv` in the project folder and run :

```bash
python predict.py
```

this creates `predictions.csv`.