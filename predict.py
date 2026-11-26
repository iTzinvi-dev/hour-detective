import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

drop_cols = ["project_id", "hours_spent",
"log_hours", "hours_bucket"]
X = train.drop(columns=drop_cols)
y = np.log1p(train["hours_spent"])
X_test = test.drop(columns=["project_id"])
for df in [X, X_test]:
    df["commits_per_active_day"] = df["num_commits"] / (df["active_commit_days"] + 1)
    df["active_day_ratio"] = df["active_commit_days"] / (df["project_span_days"] + 1)
    df["lines_per_active_day"] = df["lines_of_code"] / (df["active_commit_days"] + 1)

cat_cols = list(X.select_dtypes(include=["object", "string"]).columns)
X[cat_cols] = X[cat_cols].fillna("missing").astype(str)
X_test[cat_cols] = X_test[cat_cols].fillna("missing").astype(str)

model = CatBoostRegressor(
    iteration=700,
    learning_rate=0.04,
    depth=6,
    cat_features=cat_cols,
    random_seed=42
)
model.fit(X, y)
preds = np.expm1(model.predict(X_test))
preds = np.clip(preds, 0, None)
sub = pd.DataFrame({"project_id":
test["project_id"], "hours_spent": preds})
sub.to_csv("predictions.csv", index=False)