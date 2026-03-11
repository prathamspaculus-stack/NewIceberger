import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle

df = pd.read_csv("ALL_model.csv")

train = df[df["date"] < "2025-07-01"]
test = df[df["date"] >= '2025-07-01']

features = [
    "lag_1","lag_7","lag_21","lag_28","roll_mean_7","roll_mean_14","dayofweek","is_us_holiday","roll_std_7"
]

target = "call_count"

from xgboost import XGBRegressor

model = XGBRegressor(
    objective="count:poisson",
    n_estimators=385,
    learning_rate=0.06,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(train[features], train[target])

preds = model.predict(test[features])

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test[target], preds)
print("MAE:", mae)


# with open("call_volume_ALL_model.pkl", "wb") as file:
#     pickle.dump(model, file)