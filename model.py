import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle

df = pd.read_csv("main.csv")

features = [
    "lag_1","lag_7","lag_21","lag_28","roll_mean_7","roll_mean_14","dayofweek","roll_std_7"
]

target = "call_count"

train = df[df['date'] < "2025-07-01"]
test = df[df['date'] >= '2025-07-01']

model = XGBRegressor(
    objective="count:poisson",
    n_estimators=357,
    learning_rate=0.06,
    max_depth=2,
    subsample=0.5,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(train[features], train[target])

preds = model.predict(test[features])

mae = mean_absolute_error(test[target], preds)
print("MAE:", mae)

avg_calls = test[target].mean()
print(avg_calls)


# with open("call_volume_model.pkl", "wb") as file:
#     pickle.dump(model, file)