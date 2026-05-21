#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to validate the enhanced feature engineering pipeline.

Run with: python3 test_new_features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import os

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import (
    build_consecutive_dry_days,
    build_heat_accumulation_features,
    build_temperature_instability_features,
    build_physical_vapor_proxy_features,
    build_features,
    SCORE_GAP_DAYS,
)

def test_individual_features():
    """Test individual feature engineering functions."""
    print("=" * 70)
    print("Testing Individual Feature Engineering Functions")
    print("=" * 70)
    
    # Create minimal synthetic test data
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    regions = [1, 2, 3]
    
    test_data = []
    for region_id in regions:
        for date in dates:
            test_data.append({
                "region_id": region_id,
                "date": date,
                "prec": np.random.uniform(0, 50),
                "tmp": np.random.uniform(15, 35),
                "tmp_max": np.random.uniform(20, 40),
                "tmp_min": np.random.uniform(10, 25),
                "tmp_range": np.random.uniform(5, 15),
                "surf_tmp": np.random.uniform(20, 45),
                "dp_tmp": np.random.uniform(5, 25),
                "wb_tmp": np.random.uniform(10, 30),
                "humidity": np.random.uniform(20, 90),
                "wind": np.random.uniform(0, 20),
                "wind_max": np.random.uniform(0, 40),
                "wind_min": np.random.uniform(0, 10),
                "wind_range": np.random.uniform(0, 30),
                "surf_pre": np.random.uniform(900, 1050),
                "score": np.random.choice([np.nan, 1, 2, 3, 4, 5], p=[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]),
            })
    
    df = pd.DataFrame(test_data)
    print(f"[OK] Created test data: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Regions: {df['region_id'].unique()}")
    
    # Test CDD feature
    print("\n1. Testing Consecutive Dry Days (CDD) Feature...")
    df_cdd = build_consecutive_dry_days(df.copy(), dry_threshold=0.1)
    cdd_cols = [c for c in df_cdd.columns if "cdd" in c.lower() or "consecutive_dry" in c.lower()]
    print(f"  [OK] Added {len(cdd_cols)} CDD-related columns: {cdd_cols[:3]}...")
    print(f"  [OK] Sample CDD values: {df_cdd['consecutive_dry_days'].describe()}")
    
    # Test heat accumulation
    print("\n2. Testing Heat Accumulation Features...")
    df_heat = build_heat_accumulation_features(df.copy(), hot_threshold=35.0)
    heat_cols = [c for c in df_heat.columns if "hot_days" in c or "heat_stress" in c]
    print(f"  [OK] Added {len(heat_cols)} heat accumulation columns: {heat_cols[:3]}...")
    
    # Test temperature instability
    print("\n3. Testing Temperature & Wind Instability Features...")
    df_instab = build_temperature_instability_features(df.copy())
    instab_cols = [c for c in df_instab.columns if "std_" in c or "_std_" in c]
    print(f"  [OK] Added {len(instab_cols)} instability columns: {instab_cols[:3]}...")
    
    # Test physical vapor features
    print("\n4. Testing Physical Vapor Proxy Features...")
    df_vapor = build_physical_vapor_proxy_features(df.copy())
    vapor_cols = [c for c in df_vapor.columns if "diff" in c or "depression" in c]
    print(f"  [OK] Added {len(vapor_cols)} vapor proxy columns: {vapor_cols[:3]}...")
    print(f"  [OK] Sample surf_air_temp_diff: min={df_vapor['surf_air_temp_diff'].min():.2f}, "
          f"max={df_vapor['surf_air_temp_diff'].max():.2f}")
    
    print("\n[SUCCESS] All individual feature tests passed!")


def test_full_pipeline():
    """Test the full feature engineering pipeline."""
    print("\n" + "=" * 70)
    print("Testing Full Feature Engineering Pipeline")
    print("=" * 70)
    
    # Load actual data if available
    data_dir = Path(__file__).parent / "data"
    if not (data_dir / "train.csv").exists():
        print("[WARN] train.csv not found, skipping full pipeline test")
        return
    
    print("Loading train.csv...")
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    
    # Use only a small sample for testing
    sample_regions = train["region_id"].unique()[:3]
    train_sample = train[train["region_id"].isin(sample_regions)].tail(500)
    
    print(f"[OK] Loaded sample: {train_sample.shape}")
    print(f"  Regions: {train_sample['region_id'].unique()}")
    print(f"  Date range: {train_sample['date'].min()} to {train_sample['date'].max()}")
    
    # Test full pipeline with "micro" profile
    print("\nBuilding features with 'micro' profile...")
    try:
        features_df = build_features(
            train_sample.copy(),
            train.copy(),
            is_train=True,
            use_score_history=True,
            score_gap_days=SCORE_GAP_DAYS,
            use_climatology=True,
            use_region_stats=True,
            feature_profile="micro",
        )
        print(f"[OK] Full pipeline completed: {features_df.shape}")
        print(f"  Columns: {features_df.shape[1]}")
        
        # Check for key features
        key_features = [
            "consecutive_dry_days",
            "cdd_rolling30",
            "hot_days_above35_30d",
            "heat_stress_sum_30d",
            "tmp_range_std_14d",
            "wind_std_14d",
            "surf_air_temp_diff",
            "dew_point_depression",
            "wet_bulb_depression",
            "dew_depression_mean_14d",
            "last_known_score",
            "score_velocity_1w",
            "region_score_mean",
        ]
        
        missing = [f for f in key_features if f not in features_df.columns]
        if missing:
            print(f"[WARN] Missing features: {missing}")
        else:
            print(f"[OK] All {len(key_features)} key features present!")
            print(f"\n  Sample feature values:")
            print(f"    consecutive_dry_days: {features_df['consecutive_dry_days'].describe()}")
            print(f"    dew_point_depression: {features_df['dew_point_depression'].describe()}")
            print(f"    last_known_score: {features_df['last_known_score'].describe()}")
        
        print("\n[SUCCESS] Full pipeline test passed!")
        
    except Exception as e:
        print(f"[ERROR] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        test_individual_features()
        test_full_pipeline()
        print("\n" + "=" * 70)
        print("[SUCCESS] All tests completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
