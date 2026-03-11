from fastapi import FastAPI,HTTPException
import pandas as pd
import joblib
from datetime import date as DateType
from pydantic import BaseModel, Field
import holidays

model = joblib.load("call_volume_ALL_model.pkl")

app = FastAPI(title = "Call Volume Forecast API")

us_holidays = holidays.US(years=range(2023, 2031))

class ForecastRequest(BaseModel):
    date: DateType = Field(
        ...,
        description="Forecast date in YYYY-MM-DD format (example: 2025-12-01)"
    )

    model_config = {
        "json_schema_extra": {
            "example" : {
                "date": "2025-12-01"
            }
        }
    }

def build_features(input_date: DateType, history_df: pd.DataFrame):

    df = history_df.copy()

    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 28:
        raise ValueError("Need at least 28 days history")
    
    input_date = pd.to_datetime(input_date)
    last_date = df["date"].iloc[-1]

    is_us_holiday = 1 if input_date.date() in us_holidays else 0

    if input_date in df["date"].values:

        idx = df.index[df["date"] == input_date][0]

        if idx < 28:
            raise ValueError("Not enough history before this date")
        
        features = {
            "lag_1": df["call_count"].iloc[idx - 1],
            "lag_7": df["call_count"].iloc[idx - 7],
            "lag_14":df["call_count"].iloc[idx - 14],
            "lag_21": df["call_count"].iloc[idx - 21],
            "lag_28" : df["call_count"].iloc[idx - 28],
            "roll_mean_7": df["call_count"].iloc[idx-7:idx].mean(),
            "roll_mean_14": df["call_count"].iloc[idx-14:idx].mean(),
            "dayofweek": input_date.dayofweek,
            "is_us_holiday": is_us_holiday,
            "roll_std_7" : df["call_count"].iloc[idx-7:idx].std(),
        }

        return pd.DataFrame([features])
    
    temp_df = df.copy()
    current_date = last_date

    while current_date < input_date:

        next_date = current_date + pd.Timedelta(days=1)

        feat = {
    "lag_1": temp_df["call_count"].iloc[-1],
    "lag_7": temp_df["call_count"].iloc[-7],
    "lag_14": temp_df["call_count"].iloc[-7],
    "lag_21": temp_df["call_count"].iloc[-21],
    "lag_28": temp_df["call_count"].iloc[-28],
    "roll_mean_7": temp_df["call_count"].iloc[-7:].mean(),
    "roll_mean_14": temp_df["call_count"].iloc[-14:].mean(),
    "dayofweek": next_date.dayofweek,
    "is_us_holiday": 1 if next_date.date() in us_holidays else 0,
    "roll_std_7": temp_df["call_count"].iloc[-7:].std()
}

        X_step = pd.DataFrame([feat])

        y_hat = model.predict(X_step)[0]

        new_row = pd.DataFrame({
            "date": [next_date],
            "call_count": [y_hat]
        })

        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

        current_date = next_date

    return X_step

history_df = pd.read_csv("2023 to 2026(all).csv")

@app.post("/forecast")
def forecast_calls(request: ForecastRequest):

    try:

        X_pred = build_features(request.date, history_df)

        pred = model.predict(X_pred)[0]

        return {
            "date" : request.date,
            "prediction_call_volume" : round(float(pred), 2)
        }
    
    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/debug-features")
def debug_features(request: ForecastRequest):

    try:
        X_pred = build_features(request.date, history_df)

        return{
            "date": request.date,
            "generated_features": X_pred.to_dict(orient= "records")[0]
        }
    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))