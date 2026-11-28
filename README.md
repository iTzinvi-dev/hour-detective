# hour-detective

A Baseline model for haggle to predict `hours_spent` on projects

## What it Does :
Uses `CatBoostRegressor` trained on `log1p(hours_spent)` to match RMSLE. Text columns are filled with "missing" and passed via cat feature, and final predictions are converted back with `expm1` and clipped to 0.1 - 140.7 

Dropped the HGB blend and switched to 5-fold StratifiedKFold on `hours_bucket` with early stopping `l2_leaf_reg=`, `learning_rate=`

it also had some activity ratio fetures (`commits_per_active_day`, `active_day_ratio`, `lines_per_active_day`, `lines_per_commit`, `total_churn_lines`, `effort_score`, `readme_density` and a few more ratio features) and tuned the model with `learning_rate=` `iterations=`, `depth=`.
## How to Run :
Put `train.csv` and `test.csv` in the project folder and run :

```bash
python predict.py
```

this creates `predictions.csv`.