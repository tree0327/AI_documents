import pandas as pd

try:
    df = pd.read_csv("./data/clock_all.csv")
    target_brands = ['samsung', 'apple', 'xiaomi']
    df = df[df['brand'].isin(target_brands)]
    
    df['event_time'] = pd.to_datetime(df['event_time'])
    print(f"Filtered Min Date: {df['event_time'].min()}")
    print(f"Filtered Max Date: {df['event_time'].max()}")
    
    # Calculate 80% split point
    sorted_dates = df['event_time'].sort_values()
    split_idx = int(len(sorted_dates) * 0.8)
    split_point = sorted_dates.iloc[split_idx]
    print(f"Suggested 80% Split Date: {split_point}")
except Exception as e:
    print(e)
