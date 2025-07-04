"""
Trading Automation Configuration
ADHD-friendly: All settings in one place
"""

import os

# CANDLE CLASSIFICATION THRESHOLDS
CANDLE_THRESHOLDS = {
    'base_max_ratio': 0.50,      # Base: body-to-range ratio ≤ 50%
    'decisive_max_ratio': 0.80,   # Decisive: 50% < ratio ≤ 80%
    'explosive_min_ratio': 0.80   # Explosive: ratio > 80%
}

# ZONE DETECTION PARAMETERS
ZONE_PARAMETERS = {
    'max_base_candles': 6,        # Maximum base candles (1-6 range)
    'ideal_base_candles': 3,      # Sweet spot (1-3 highest probability)
    'min_leg_out_ratio': 1.5,     # Leg-out must be 2x base size
    'min_leg_strength': 2         # Minimum decisive/explosive candles in leg
}

# DATA SETTINGS - ADAPTED FOR YOUR FILE STRUCTURE
DATA_SETTINGS = {
    'primary_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'CADJPY'],
    'timeframes': ['Daily', 'Weekly', 'H12', 'H4'],
    'required_history': 500,      # Minimum candles needed
    'validation_threshold': 0.95, # 95% accuracy requirement
    'data_source': 'MetaTrader'   # Your data source format
}

# FILE PATHS - ADAPTED FOR YOUR STRUCTURE
BASE_DIR = r'C:\Users\sim\Desktop\Quant\OTC Strat\trading-strategy-automation\S&D'

PATHS = {
    'raw_data': os.path.join(BASE_DIR, 'Data', 'raw'),
    'processed_data': os.path.join(BASE_DIR, 'Data', 'processed'),
    'results': os.path.join(BASE_DIR, 'results'),
    'charts': os.path.join(BASE_DIR, 'results', 'charts'),
    'validation': os.path.join(BASE_DIR, 'tests', 'manual_validation')
}

# DATA FILE PATTERNS - FOR YOUR METATRADER FORMAT
FILE_PATTERNS = {
    'daily': '{pair}.raw_Daily_*.csv',
    'weekly': '{pair}.raw_Weekly_*.csv',
    'h12': '{pair}.raw_H12_*.csv',
    'h8': '{pair}.raw_H8_*.csv',
    'h4': '{pair}.raw_H4_*.csv'
}

# COLUMN MAPPING - FOR YOUR CSV FORMAT
COLUMN_MAPPING = {
    'datetime': '<DATE>',
    'open': '<OPEN>',
    'high': '<HIGH>',
    'low': '<LOW>',
    'close': '<CLOSE>',
    'volume': '<TICKVOL>'
}

# VALIDATION SETTINGS
VALIDATION = {
    'min_candles_for_test': 100,
    'accuracy_threshold': 0.95,
    'sample_size_for_manual_check': 50
}

# ZONE DETECTION SETTINGS - MODULE 2 
ZONE_CONFIG = {
    'min_base_candles': 1,
    'max_base_candles': 6,
    'optimal_base_candles': 3,
    'min_legout_ratio': 1.5,
    'min_leg_strength': 0.5,
    'max_base_retracement': 0.3,
    'min_pattern_pips': 10,
    'pip_value': 0.0001,
    'pattern_types': ['D-B-D', 'R-B-R'],
    'focus_patterns': ['D-B-D', 'R-B-R']  # Momentum patterns priority
}

# TESTING SETTINGS - MODULE 2 
TEST_CONFIG = {
    'test_data_size': 100,
    'accuracy_threshold': 0.95,
    'sample_size': 50,
    'validation_pairs': ['EURUSD', 'GBPUSD', 'CADJPY'],
    'debug_mode': True
}

print("✅ Zone Detection configuration added!")
print("✅ Configuration loaded successfully!")
print(f"📁 Data path: {PATHS['raw_data']}")