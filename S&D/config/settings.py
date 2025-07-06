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

# TREND CLASSIFICATION SETTINGS - MODULE 3 (UPDATED WITH FILTER)
TREND_CONFIG = {
    'ema_fast': 50,
    'ema_medium': 100,
    'ema_slow': 200,
    'min_separation': 0.3,        # Minimum EMA separation (0-1 scale)
    'separation_lookback': 5,     # Smooth separation over N periods
    'ranging_threshold': 0.3,     # Below this = ranging market
    'trending_threshold': 0.6,    # Above this = strong trending market
    'trend_classifications': [
        'strong_bullish',
        'medium_bullish',
        'weak_bullish',
        'ranging',               # New classification
        'strong_bearish',
        'medium_bearish',
        'weak_bearish'
    ]
}

# RISK MANAGEMENT CONFIGURATION - MODULE 4
RISK_CONFIG = {
    'account_settings': {
        'starting_balance': 10000,      # $100,000 account
        'currency': 'USD',
        'broker_leverage': 30,          # EU regulation limit
      # 'min_free_margin': 10000        # Keep $10,000 minimum margin
    },
    'risk_limits': {
        'max_risk_per_trade': 5.0,     # 5% max per trade
        #'max_daily_risk': 6.0,         # 6% max daily exposure  
        #'max_portfolio_risk': 20.0,    # 20% total exposure limit
        #'max_correlated_risk': 4.0     # 4% max on correlated pairs
    },
    'position_sizing': {
        'method': 'fixed_risk_percent', # Fixed 5% risk per trade
        'min_lot_size': 0.01,          # Micro lots
        'max_lot_size': 5.0,           # 5 standard lots max
        'lot_size_increment': 0.01     # Micro lot increments
    },
    'stop_loss_rules': {
        'method': 'zone_boundary_plus_buffer',
        'buffer_pips': 5,              # 5 pip buffer beyond zone
        #'max_stop_distance': 80,       # Max 80 pip stop loss
        #'min_stop_distance': 15,       # Min 15 pip stop loss
        'round_to_level': False         # Round to psychological levels
    },
    'take_profit_rules': {
        'risk_reward_ratio': 2.0,      # Minimum 1:2 RR
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
        'min_zone_score': 60,      # Minimum zone quality score
        'min_trend_strength': 0.3,  # Minimum EMA separation
        'min_signal_score': 65,    # Minimum overall signal score
        'trend_alignment': True    # Must align with trend
    },
    'risk_management': {
        #'max_signals_per_day': 3,
        'position_sizing': 'fixed_risk',
        'stop_loss_method': 'zone_boundary'
    },
    'entry_methods': {
        'market_entry_threshold': 80,   # Zone score for market entry
        'limit_entry_threshold': 65,    # Zone score for limit entry
        'wait_retest_threshold': 50     # Zone score for waiting
    }
}

print("✅ Signal Generation configuration added!")
print("✅ Risk Management configuration added!")
print("✅ Triple EMA Trend Classification configured!")
print("✅ Zone Detection configuration added!")
print("✅ Configuration loaded successfully!")
print("✅ Trend Classification configuration added!")
print(f"📁 Data path: {PATHS['raw_data']}")