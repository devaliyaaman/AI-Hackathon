"""
Demand Forecasting Model using Azure AI / GenAI Lab MaaS - DeepSeek-R1
=======================================================================
Handles:
  - Weekly seasonality
  - External regressors: promotion, holiday
  - 14-day ahead forecast
  - Forecast DataFrame output
  - Forecast plot
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from openai import AzureOpenAI  # DeepSeek-R1 on Azure uses OpenAI-compatible SDK

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/")
AZURE_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY",  "<your-api-key>")
DEPLOYMENT     = os.getenv("AZURE_DEPLOYMENT_NAME", "azure_ai/genailab-maas-DeepSeek-R1")
API_VERSION    = "2024-05-01-preview"

FORECAST_HORIZON = 14   # days to predict
CONTEXT_WINDOW   = 60   # historical days sent to the model as context


# ─────────────────────────────────────────────
# 2. DATA GENERATION (synthetic retail dataset)
# ─────────────────────────────────────────────

def generate_retail_dataset(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic retail dataset with realistic patterns."""
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    # Weekly seasonality: higher sales on weekends
    day_of_week   = dates.dayofweek
    weekly_effect = np.where(day_of_week >= 5, 1.35, 1.0)

    # Base trend + noise
    trend = np.linspace(100, 140, n_days)
    noise = np.random.normal(0, 8, n_days)

    # Promotions: ~15% of days
    promotion = np.random.binomial(1, 0.15, n_days)
    promo_lift = promotion * np.random.uniform(20, 40, n_days)

    # Holidays: fixed dates
    holiday_dates = pd.to_datetime([
        "2023-01-01", "2023-01-26", "2023-03-08", "2023-08-15",
        "2023-10-02", "2023-10-24", "2023-11-01", "2023-12-25",
    ])
    holiday = dates.isin(holiday_dates).astype(int)
    holiday_lift = holiday * np.random.uniform(30, 60, n_days)

    sales = (trend * weekly_effect + promo_lift + holiday_lift + noise).clip(min=0).round(2)

    return pd.DataFrame({
        "date":      dates,
        "sales":     sales,
        "promotion": promotion,
        "holiday":   holiday,
    })


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for seasonality and trend."""
    df = df.copy()
    df["day_of_week"]  = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    # Fourier terms for weekly seasonality
    df["sin_week"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ─────────────────────────────────────────────
# 4. AZURE AI CLIENT
# ─────────────────────────────────────────────

def build_client() -> AzureOpenAI:
    """Initialise Azure OpenAI client pointing at DeepSeek-R1."""
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=API_VERSION,
    )


# ─────────────────────────────────────────────
# 5. PROMPT CONSTRUCTION
# ─────────────────────────────────────────────

def build_system_prompt() -> str:
    return (
        "You are an expert time-series forecasting engine. "
        "Given historical retail sales data with features (day_of_week, promotion, holiday, "
        "is_weekend, sin_week, cos_week), you will predict future daily sales values. "
        "You MUST respond ONLY with a valid JSON array of numbers — one per day — "
        "with NO extra text, explanation, or markdown. "
        "Example format: [120.5, 135.2, 98.7]"
    )


def build_user_prompt(history_df: pd.DataFrame, future_df: pd.DataFrame) -> str:
    history_records = history_df[
        ["date", "sales", "day_of_week", "is_weekend",
         "promotion", "holiday", "sin_week", "cos_week"]
    ].copy()
    history_records["date"] = history_records["date"].dt.strftime("%Y-%m-%d")

    future_records = future_df[
        ["date", "day_of_week", "is_weekend",
         "promotion", "holiday", "sin_week", "cos_week"]
    ].copy()
    future_records["date"] = future_records["date"].dt.strftime("%Y-%m-%d")

    return (
        f"### Historical Sales (last {len(history_records)} days)\n"
        f"{history_records.to_json(orient='records', indent=2)}\n\n"
        f"### Future Features for the next {len(future_records)} days\n"
        f"{future_records.to_json(orient='records', indent=2)}\n\n"
        f"Predict daily sales for the {len(future_records)} future days. "
        f"Return ONLY a JSON array of {len(future_records)} numbers."
    )


# ─────────────────────────────────────────────
# 6. MODEL INFERENCE
# ─────────────────────────────────────────────

def call_deepseek(client: AzureOpenAI, system_prompt: str, user_prompt: str) -> list[float]:
    """Send prompt to DeepSeek-R1 and parse the forecast array."""
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,   # low temperature for deterministic forecasting
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        forecast = json.loads(raw)
        return [float(v) for v in forecast]
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek returned non-JSON output:\n{raw}") from exc


# ─────────────────────────────────────────────
# 7. FORECASTING PIPELINE
# ─────────────────────────────────────────────

def build_future_features(
    last_date: pd.Timestamp,
    horizon: int,
    promotion_schedule: list[int] | None = None,
    holiday_schedule:   list[int] | None = None,
) -> pd.DataFrame:
    """Build the feature DataFrame for the forecast horizon."""
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({"date": future_dates})

    # Default: no promotions or holidays unless provided
    future_df["promotion"] = promotion_schedule if promotion_schedule else [0] * horizon
    future_df["holiday"]   = holiday_schedule   if holiday_schedule   else [0] * horizon

    return engineer_features(future_df)


def forecast(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    context_window: int = CONTEXT_WINDOW,
    promotion_schedule: list[int] | None = None,
    holiday_schedule:   list[int] | None = None,
) -> pd.DataFrame:
    """
    Full forecasting pipeline.
    Returns a DataFrame with columns: date, forecasted_sales.
    """
    df = engineer_features(df.copy())

    # Use the most recent `context_window` days as history
    history_df  = df.tail(context_window).reset_index(drop=True)
    last_date   = df["date"].max()

    future_df = build_future_features(last_date, horizon, promotion_schedule, holiday_schedule)

    client        = build_client()
    system_prompt = build_system_prompt()
    user_prompt   = build_user_prompt(history_df, future_df)

    print("⏳  Calling DeepSeek-R1 on Azure AI …")
    predictions = call_deepseek(client, system_prompt, user_prompt)

    if len(predictions) != horizon:
        raise ValueError(
            f"Expected {horizon} predictions, got {len(predictions)}."
        )

    forecast_df = pd.DataFrame({
        "date":             future_df["date"].values,
        "forecasted_sales": [round(p, 2) for p in predictions],
        "promotion":        future_df["promotion"].values,
        "holiday":          future_df["holiday"].values,
    })
    return forecast_df


# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────

def plot_forecast(
    history_df:  pd.DataFrame,
    forecast_df: pd.DataFrame,
    context_days: int = 60,
    save_path: str | None = "forecast_plot.png",
) -> None:
    """Plot historical sales alongside the 14-day forecast."""
    hist = history_df.tail(context_days).copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # Historical line
    ax.plot(hist["date"], hist["sales"],
            color="#2C6FEB", linewidth=1.8, label="Historical Sales", zorder=3)

    # Forecast line + shaded CI (±10% naive band)
    fc_vals = forecast_df["forecasted_sales"].values
    lo = fc_vals * 0.90
    hi = fc_vals * 1.10
    ax.plot(forecast_df["date"], fc_vals,
            color="#E84C4C", linewidth=2.2, linestyle="--",
            marker="o", markersize=4, label="Forecast (14-day)", zorder=4)
    ax.fill_between(forecast_df["date"], lo, hi,
                    color="#E84C4C", alpha=0.12, label="±10% Confidence Band")

    # Divider
    ax.axvline(x=hist["date"].iloc[-1], color="#888", linewidth=1.2,
               linestyle=":", label="Forecast Start")

    # Annotate promotions / holidays in forecast window
    for _, row in forecast_df.iterrows():
        if row["promotion"]:
            ax.axvspan(row["date"] - timedelta(hours=12),
                       row["date"] + timedelta(hours=12),
                       color="#F4A636", alpha=0.25)
        if row["holiday"]:
            ax.axvspan(row["date"] - timedelta(hours=12),
                       row["date"] + timedelta(hours=12),
                       color="#A855F7", alpha=0.20)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=30, ha="right")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sales", fontsize=11)
    ax.set_title("Retail Demand Forecast — DeepSeek-R1 (Azure AI MaaS)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊  Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────

def main() -> pd.DataFrame:
    # ── Step 1: Load / generate data ──────────────────────────────
    print("📦  Generating retail dataset …")
    df = generate_retail_dataset(n_days=365)
    print(df.tail(5).to_string(index=False))

    # ── Step 2: (Optional) define upcoming promotions / holidays ──
    # Index 0 = day+1 from last date, etc.  1 = active, 0 = not.
    promotion_schedule = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # 14 values
    holiday_schedule   = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # 14 values

    # ── Step 3: Run forecast ───────────────────────────────────────
    forecast_df = forecast(
        df,
        horizon=FORECAST_HORIZON,
        context_window=CONTEXT_WINDOW,
        promotion_schedule=promotion_schedule,
        holiday_schedule=holiday_schedule,
    )

    print("\n✅  Forecast DataFrame:")
    print(forecast_df.to_string(index=False))

    # ── Step 4: Plot ───────────────────────────────────────────────
    plot_forecast(df, forecast_df, context_days=60, save_path="forecast_plot.png")

    return forecast_df


if __name__ == "__main__":
    result = main()
