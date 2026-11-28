import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

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
    df["lines_per_commit"] = df["lines_of_code"] / (df["num_commits"] + 1)
    df["total_churn_lines"] = df["lines_added_total"] + df["lines_deleted_total"]
    df["effort_score"] = (1 - df["pct_code_in_first_commit"].fillna(0)) * (1 - df["pct_generic_commit_msgs"].fillna(0))
    df["readme_density"] = df["readme_word_count"] / (df["readme_num_headings"] + 1)
    df["real_commits"] = df["num_commits"] * (1 - df["pct_generic_commit_msgs"].fillna(0))
    df["non_first_commit_loc"] = df["lines_of_code"] * (1 - df["pct_code_in_first_commit"].fillna(0))
    df["deleted_per_active_day"] = df["lines_deleted_total"] / (df["active_commit_days"] + 1)
    df["added_per_active_day"] = df["lines_added_total"] / (df["active_commit_days"] + 1)
    df["max_commit_share"] = df["max_lines_in_one_commit"] / (df["lines_added_total"] + 1)
    df["total_assets"] = df["num_image_files"] + df["num_audio_files"] + df["num_3d_model_files"] + df["num_font_files"]
    df["deleted_to_loc"] = df["lines_deleted_total"] / (df["lines_of_code"] + 1)
    df["added_to_loc"] = df["lines_added_total"] / (df["lines_of_code"] + 1)

cat_cols = list(X.select_dtypes(include=["object", "string"]).columns)
X[cat_cols] = X[cat_cols].fillna("missing").astype(str)
X_test[cat_cols] = X_test[cat_cols].fillna("missing").astype(str)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
preds_log = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(skf.split(X,train["hours_bucket"])):
    model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    cat_features=cat_cols,
    random_seed=42 + fold,
    verbose=0
)
    model.fit(
    X.iloc[tr], y.iloc[tr],
    eval_set=(X.iloc[va], y.iloc[va]),
    early_stopping_rounds=100,
    verbose=False
)
    preds_log += model.predict(X_test) / 5

preds = np.expm1(preds_log)
preds = np.clip(preds, 0.1, 140.7)

sub = pd.DataFrame({"project_id":
test["project_id"], "hours_spent": preds})
sub.to_csv("predictions.csv", index=False)