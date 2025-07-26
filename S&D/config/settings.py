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

# DATA SETTINGS - ADAPTED FOR YOUR FILE STRUCTURE
DATA_SETTINGS = {
    'primary_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'CADJPY'],
    'timeframes': ['Daily', 'Weekly', 'H12', 'H4'],
    'required_history': 500,      # Minimum candles needed
    'validation_threshold': 0.95, # 95% accuracy requirement
    'data_source': 'MetaTrader'   # Your data source format
}

# FILE PATHS - CROSS-PLATFORM COMPATIBLE
# Auto-detect project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_ROOT  

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

# DATA VALIDATION SETTINGS
VALIDATION = {
    'min_candles_for_test': 100,
    'accuracy_threshold': 0.95,
    'sample_size_for_manual_check': 50,
    'max_price_deviation_pct': 1.0,    # Allow 1% price deviation for real-world data
    'min_data_completeness': 0.95      # Require 95% complete data
}

# ZONE DETECTION CONFIGURATION - UNIFIED
ZONE_CONFIG = {
    'min_base_candles': 1,
    'max_base_candles': 6,
    'optimal_base_candles': 3,
    'min_legout_ratio': 0.5,      # Basic formation quality
    'min_leg_strength': 1,
    'max_base_retracement': 0.3,
    'min_pattern_pips': 10,
    'pip_value': 0.0001,
    'momentum_patterns': ['D-B-D', 'R-B-R'],
    'focus_patterns': ['D-B-D', 'R-B-R'],
    'reversal_patterns': ['D-B-R', 'R-B-D']
}

# TREND CLASSIFICATION SETTINGS - SIMPLIFIED
TREND_CONFIG = {
    'fast_ema': 50,               # EMA50 for trend detection
    'slow_ema': 200,              # EMA200 for trend detection
    'method': 'dual_ema',         # Dual EMA crossover method
    'trend_classifications': [
        'bullish',                # EMA50 > EMA200
        'bearish'                 # EMA50 < EMA200
    ]
}

# RISK MANAGEMENT CONFIGURATION - MODULE 4
RISK_CONFIG = {
    'account_settings': {
        'starting_balance': 10000,      # $10,000 account
        'currency': 'USD',
        'broker_leverage': 30,          # EU regulation limit
        'decimal_precision': 5          # Precision for forex pairs
    },
    'risk_limits': {
        'max_risk_per_trade': 5.0     # 5% max per trade
    },
    'position_sizing': {
        'method': 'fixed_risk_percent', # Fixed 5% risk per trade
        'min_lot_size': 0.01,          # Micro lots
        'max_lot_size': 5.0,           # 5 standard lots max
        'lot_size_increment': 0.01     # Micro lot increments
    },
    'stop_loss_rules': {
        'method': 'zone_boundary_plus_buffer',
        'buffer_percent': 0.33,        # 33% buffer beyond zone boundary
        'round_to_level': False        # Round to psychological levels
    },
    'take_profit_rules': {
        'risk_reward_ratio': 2.5,      # Minimum 1:25 RR
        'scale_out_enabled': False,     # Take partial profits
        'scale_levels': [1.0, 2.0, 3.0], # 1R, 2R, 3R exits
        'scale_percentages': [33, 33, 34] # % of position to close
    }
}

# SIGNAL GENERATION CONFIGURATION - MODULE 5
SIGNAL_CONFIG = {
    'zone_timeframes': ['H4', 'H12', 'Daily', 'Weekly'],
    'trend_timeframe': 'Daily',  # Always Daily for consistency
    'signal_types': ['zone_entry', 'zone_retest'],
    'quality_thresholds': {
        'min_zone_score': 50,      # Minimum zone quality score
        'min_trend_strength': 0.3,  # Minimum EMA separation
        'min_signal_score': 50,    # Minimum overall signal score
        'trend_alignment': True    # Must align with trend
    },
    'risk_management': {
        'position_sizing': 'fixed_risk',
        'stop_loss_method': 'zone_boundary'
    },
    'entry_methods': {
        'market_entry_threshold': 70,   # Zone score for market entry
        'limit_entry_threshold': 55,    # Zone score for limit entry
        'wait_retest_threshold': 40     # Zone score for waiting
    }
}

# Validate configuration on load
def validate_paths():
    """Ensure all required directories exist"""
    for path_name, path_value in PATHS.items():
        os.makedirs(path_value, exist_ok=True)

# Auto-create directories on import
validate_paths()

print("✅ Configuration loaded successfully!")
print(f"📁 Data path: {PATHS['raw_data']}")
print(f"💾 Results path: {PATHS['results']}")