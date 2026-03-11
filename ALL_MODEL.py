import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit

df = pd.read_csv("ALL_model.csv")

train = df[df["date"] < "2025-07-01"]
test = df[df["date"] >= '2025-07-01']

features = [
    "lag_1","lag_7","lag_14","lag_21","lag_28","roll_mean_7","roll_mean_14","dayofweek","is_us_holiday","roll_std_7"
]

target = "call_count"

model = LGBMRegressor(
    objective="poisson",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(train[features], train[target])

preds = model.predict(test[features])

mae = mean_absolute_error(test[target], preds)

print("MAE:", mae)

avg_calls = test[target].mean()
print("avg_calls:",avg_calls)

error_percent = (mae / avg_calls) * 100
print("error: ",error_percent)



with open("call_volume_ALL_model.pkl", "wb") as file:
    pickle.dump(model, file)

