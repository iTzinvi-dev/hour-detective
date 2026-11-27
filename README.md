# hour-detective

A Baseline model for haggle to predict `hours_spent` on projects

## What it Does :
Uses `CatBoostRegressor` trained on `log1p(hours_spent)` to match RMSLE. Text columns are filled with "missing" and passed via cat feature, and final predictions are converted back with `expm1` and clipped at 0. 

it also had some activity ratio fetures (`commits_per_active_day`, `active_day_ratio`, `lines_per_active_day`, `lines_per_commit`, `total_churn_lines`, `effort_score`, `readme_density` ) and tuned the model with `learning_rate=` `iterations=`, `depth=`.
## How to Run :
Put `train.csv` and `test.csv` in the project folder and run :

```bash
python predict.py
```

this creates `predictions.csv`.