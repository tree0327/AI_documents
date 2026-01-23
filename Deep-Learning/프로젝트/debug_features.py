import pandas as pd
import numpy as np
import warnings
from datetime import timedelta
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def create_ml_features(df, target_brand='samsung'):
    df = df.copy()
    
    # Session ID 생성 (30분 단위)
    if 'user_session' not in df.columns:
        print("Creating user sessions...")
        df['prev_time'] = df.groupby('user_id')['event_time'].shift(1)
        df['time_diff'] = (df['event_time'] - df['prev_time']).dt.total_seconds() / 60
        df['is_new_session'] = (df['time_diff'] > 30).fillna(1).astype(int)
        df['user_session'] = df.groupby('user_id')['is_new_session'].cumsum()

    snapshot_date = df['event_time'].max()
    print(f"Snapshot date: {snapshot_date}")
    
    # 기본 집계 프레임
    user_feats = pd.DataFrame(index=df['user_id'].unique())
    fs_30d = df[df['event_time'] >= (snapshot_date - timedelta(days=30))].copy()
    
    # --- 1. 활동성 & 빈도 (Frequency) ---
    user_feats['cnt_target_events_30d'] = fs_30d[(fs_30d['brand'] == target_brand) & (fs_30d['event_type'] != 'purchase')].groupby('user_id').size()
    user_feats['cnt_target_cart_30d'] = fs_30d[(fs_30d['brand'] == target_brand) & (fs_30d['event_type'] == 'cart')].groupby('user_id').size()
    user_feats['n_target_sessions_30d'] = fs_30d[fs_30d['brand'] == target_brand].groupby('user_id')['user_session'].nunique()

    # --- 2. 심리/고민 (Hesitation) ---
    target_remove = fs_30d[(fs_30d['brand'] == target_brand) & (fs_30d['event_type'] == 'remove_from_cart')].groupby('user_id').size()
    user_feats['remove_per_cart_target'] = target_remove / (user_feats['cnt_target_cart_30d'] + 1) # 분모 0 방지
    
    # --- 3. 최신성 (Recency) ---
    last_target = df[df['brand'] == target_brand].groupby('user_id')['event_time'].max()
    # Check what last_target looks like
    print(f"Last target sample: {last_target.head() if not last_target.empty else 'Empty'}")
    
    user_feats['days_since_last_target_event'] = (snapshot_date - last_target).dt.days.fillna(365)
    
    # --- 4. 경쟁사 관심 (Competitor) ---
    user_feats['cnt_competitor_events_30d'] = fs_30d[fs_30d['brand'] != target_brand].groupby('user_id').size()
    
    # --- 5. 몰입도 (Duration) ---
    session_dur = df.groupby(['user_id', 'user_session'])['event_time'].agg(lambda x: (x.max() - x.min()).total_seconds())
    user_feats['avg_session_duration'] = session_dur.groupby(level=0).mean()
    
    # --- 6. 구매력/선호 (Preference) ---
    target_fs = fs_30d[fs_30d['brand'] == target_brand]
    if not target_fs.empty:
        max_price = target_fs['price'].max()
        bins = [0, 100, 300, 600, float('inf')] 
        if max_price < 2.0: 
            bins = [0, 0.25, 0.5, 0.75, float('inf')]
            
        target_fs['price_bucket'] = pd.cut(target_fs['price'].fillna(0), bins=bins, labels=[0, 1, 2, 3], include_lowest=True)
        mode_price = target_fs.groupby('user_id')['price_bucket'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 0)
        user_feats['price_bucket_mode_target'] = mode_price
    else:
        user_feats['price_bucket_mode_target'] = 0
            
    # 결측치 채우기
    user_feats = user_feats.fillna(0)
    
    # 타입 보정
    if 'price_bucket_mode_target' in user_feats.columns:
        user_feats['price_bucket_mode_target'] = user_feats['price_bucket_mode_target'].astype(int)

    return user_feats

def main():
    try:
        data_path = "./data/clock_all.csv"
        print(f"Loading data from {data_path}...")
        # Load a sample if file is large, but need enough to get brands
        df = pd.read_csv(data_path, nrows=50000)
        
        target_brands = ['samsung', 'apple', 'xiaomi']
        df = df[df['brand'].isin(target_brands)]
        
        if not pd.api.types.is_datetime64_any_dtype(df['event_time']):
            df['event_time'] = pd.to_datetime(df['event_time'])
            
        df = df.sort_values(['user_id', 'event_time']).reset_index(drop=True)
        print(f"Data loaded: {df.shape}")
        
        print("Running feature engineering...")
        feats = create_ml_features(df, target_brand='samsung')
        
        print("\nFeature Dtypes:")
        print(feats.dtypes)
        
        neg_cols = ['cnt_target_events_30d', 'n_target_sessions_30d', 'cnt_target_cart_30d', 'days_since_last_target_event', 'avg_session_duration']
        print("\nChecking for datetime columns in neg_cols...")
        
        for col in neg_cols:
            if col in feats.columns:
                print(f"{col}: {feats[col].dtype}")
                if pd.api.types.is_datetime64_any_dtype(feats[col]):
                    print(f"!!! {col} IS DATETIME !!!")
                elif pd.api.types.is_numeric_dtype(feats[col]):
                    print(f"{col} is numeric.")
                else:
                    print(f"{col} is {feats[col].dtype}")
            else:
                print(f"{col} NOT FOUND")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
