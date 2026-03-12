import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle
from lightgbm import LGBMRegressor

df = pd.read_csv("main.csv")

features = [
    "lag_1","lag_7","lag_21","lag_28","roll_mean_7","roll_mean_14","dayofweek","roll_std_7"
]

target = "call_count"

train = df[df['date'] < "2025-12-31"]
test = df[df['date'] >= '2025-12-31']

# model = XGBRegressor(
#     objective="count:poisson",
#     n_estimators=325,
#     learning_rate=0.06,
#     max_depth=2,
#     subsample=0.5,
#     colsample_bytree=0.9,
#     random_state=42
# )

model = LGBMRegressor(
    objective="poisson",
    n_estimators=480,
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
print(avg_calls)

error_percent = (mae / avg_calls) * 100
print("error: ",error_percent)

# with open("call_volume_model.pkl", "wb") as file:
#     pickle.dump(model, file)

print(preds)