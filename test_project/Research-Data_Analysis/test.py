import pandas as pd

# Load data correctly - skip the header row
df = pd.read_csv('EURUSD.raw_M15_202106170000_202507021515.csv', 
                 sep='\t',
                 skiprows=1,  # Skip the header row
                 header=None,
                 names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD'])

# Convert price columns to numeric
price_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
df[price_cols] = df[price_cols].astype(float)

# Create datetime column (this was missing!)
df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])

print("Fixed data:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")

# Now timezone analysis will work
df['hour'] = df['datetime'].dt.hour
df['range'] = df['HIGH'] - df['LOW']

hourly_stats = df.groupby('hour').agg({
    'range': 'mean',
    'TICKVOL': 'mean'
}).round(5)

print("\nHourly Activity:")
print(hourly_stats.sort_values('range', ascending=False).head(8))