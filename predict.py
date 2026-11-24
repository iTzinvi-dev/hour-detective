import pandas as pd
import numpy as np

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
    catagories=X[col].cat.catagories)
print(X.shape, X_test.shape, len(cat_cols))