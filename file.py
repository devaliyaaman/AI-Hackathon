import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'
NUM_PRODUCTS = 10
NUM_STORES = 5

# Generate date range
dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
num_days = len(dates)

# Create all combinations of date, product, and store
records = []

for store_id in range(1, NUM_STORES + 1):
    for product_id in range(1, NUM_PRODUCTS + 1):
        
        # Base sales level varies by product and store
        base_sales = np.random.randint(50, 150)
        
        # Trend: slight upward or downward over time
        trend_direction = np.random.choice([-1, 1])
        trend_slope = np.random.uniform(0.05, 0.15)
        trend = trend_direction * trend_slope * np.arange(num_days)
        
        for i, date in enumerate(dates):
            # Weekly seasonality: higher sales on weekends (Sat=5, Sun=6)
            day_of_week = date.dayofweek
            if day_of_week == 5:  # Saturday
                weekly_factor = 1.4
            elif day_of_week == 6:  # Sunday
                weekly_factor = 1.3
            elif day_of_week == 4:  # Friday
                weekly_factor = 1.15
            else:
                weekly_factor = 1.0
            
            # Holiday flag: major US holidays and random local events
            is_holiday = 0
            if (date.month == 1 and date.day == 1) or \
               (date.month == 2 and date.day == 14) or \
               (date.month == 5 and date.day >= 25 and day_of_week == 0) or \
               (date.month == 7 and date.day == 4) or \
               np.random.random() < 0.02:  # 2% chance of random local holiday
                is_holiday = 1
            
            holiday_boost = 1.5 if is_holiday else 1.0
            
            # Promotion: ~15% of days have promotions, clustered
            is_promotion = 0
            if np.random.random() < 0.15:
                is_promotion = 1
            
            promotion_boost = np.random.uniform(1.3, 1.8) if is_promotion else 1.0
            
            # Calculate sales with all factors
            sales = base_sales * weekly_factor * holiday_boost * promotion_boost
            sales += trend[i]
            
            # Add random noise (Gaussian)
            noise = np.random.normal(0, base_sales * 0.1)
            sales += noise
            
            # Ensure sales are non-negative integers
            sales = max(0, int(round(sales)))
            
            records.append({
                'date': date,
                'store_id': store_id,
                'product_id': product_id,
                'sales': sales,
                'promotion': is_promotion,
                'holiday': is_holiday
            })

# Create DataFrame
df = pd.DataFrame(records)

# Sort by date, store, product
df = df.sort_values(['date', 'store_id', 'product_id']).reset_index(drop=True)

# Verify no missing values
assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes)
print("\nFirst 10 Rows:")
print(df.head(10))
print("\nBasic Statistics:")
print(df.describe())
print("\nDate Range:", df['date'].min(), "to", df['date'].max())
print("Unique Products:", df['product_id'].nunique())
print("Unique Stores:", df['store_id'].nunique())
print("Promotion Days (%):", df['promotion'].mean() * 100)
print("Holiday Days (%):", df['holiday'].mean() * 100)

# Save to CSV
OUTPUT_FILE = 'retail_demand_forecasting_data.csv'
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDataset saved to '{OUTPUT_FILE}'")
