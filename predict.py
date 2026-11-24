import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

drop_cols = ["project_id", "hours_spent",
"log_hours", "hours_bucket"]
X = train.drop(columns=drop_cols)
y = np.log1p(train["hours_spent"])
X_test = test.drop(columns=["project_id"])

cat_cols = X.select_dtypes(include=["object", "string"]).columns
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = pd.Categorical(X_test[col],
    categories=X[col].cat.categories)
model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)
model.fit(X, y)
preds = np.expm1(model.predict(X_test))
preds = np.clip(preds, 0, None)
sub = pd.DataFrame({"project_id":
test["project_id"], "hours_spent": preds})
sub.to_csv("predictions.csv", index=False)