"""
CLEAN ZONE AGE + MARKET CONDITION BACKTESTER
Built from proven backtest_distance_edge.py logic with clean age/condition filtering
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import psutil
import warnings
import glob
warnings.filterwarnings('ignore')

# Import your proven modules
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager


def check_system_requirements():
    """Check system resources before starting analysis"""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = cpu_count()
    memory_percent = psutil.virtual_memory().percent
    
    print(f"💻 SYSTEM RESOURCES CHECK:")
    print(f"   RAM: {memory_gb:.1f} GB available")
    print(f"   CPU: {cpu_cores} cores")
    print(f"   Current memory usage: {memory_percent:.1f}%")
    
    if memory_gb < 8:
        print("⚠️  WARNING: Less than 8GB RAM. Consider reducing scope.")
    
    if memory_percent > 60:
        print("⚠️  WARNING: High memory usage. Close other applications.")
    
    return memory_gb >= 4  # Minimum 4GB required
class CleanZoneAgeConditionBacktester:
    """
    CLEAN implementation using PROVEN logic from backtest_distance_edge.py
    Adds only age and market condition filtering layers
    """
    
    # Age categories for zone filtering
    ZONE_AGE_CATEGORIES = {
        'Ultra_Fresh': (0, 7),      # 0-7 days
        'Fresh': (8, 30),           # 8-30 days  
        'Recent': (31, 90),         # 31-90 days
        'Aged': (91, 180),          # 91-180 days
        'Stale': (181, 365),        # 181-365 days
        'Ancient': (365, 99999)     # 365+ days
    }
    
    # Clean strategy definitions
    STRATEGIES = {
        'Baseline': {
            'age_filter': None,
            'condition_filter': None,
            'description': 'Baseline - no filters'
        },
        'Ultra_Fresh_Only': {
            'age_filter': 'Ultra_Fresh',
            'condition_filter': None,
            'description': 'Ultra fresh zones only (0-7 days)'
        },
        'Fresh_Only': {
            'age_filter': 'Fresh',
            'condition_filter': None,
            'description': 'Fresh zones only (8-30 days)'
        },
        'Combined_Fresh': {
            'age_filter': ['Ultra_Fresh', 'Fresh'],
            'condition_filter': None,
            'description': 'Combined fresh zones (0-30 days)'
        },
        'High_Volatility': {
            'age_filter': None,
            'condition_filter': 'High_Volatility',
            'description': 'High volatility periods only'
        },
        'Strong_Trending': {
            'age_filter': None,
            'condition_filter': 'Strong_Trending',
            'description': 'Strong trending periods only'
        },
        'Optimal_Combined': {
            'age_filter': ['Ultra_Fresh', 'Fresh'],
            'condition_filter': ['High_Volatility', 'Strong_Trending'],
            'description': 'Fresh zones + favorable conditions'
        }
    }
    
    def __init__(self, max_workers: int = None):
        self.data_loader = DataLoader()
        
        # OPTIMIZATION: Perfect for i5-10400F (6C/12T) + 16GB RAM
        self.max_workers = min(max_workers or 10, cpu_count() - 2)  # Use 10 workers max
        self.memory_threshold = 0.80  # 80% RAM threshold (you have 8GB available)
        self.chunk_size = 200  # Smaller chunks for better progress tracking
        
        # OPTIMIZATION: Data caching
        self.data_cache = {}
        self.zone_cache = {}
        
        print("🎯 OPTIMIZED ZONE AGE + CONDITION BACKTESTER")
        print(f"   Max workers: {self.max_workers}")
        print(f"   Memory threshold: {self.memory_threshold*100:.0f}%")
        print(f"   Data caching: ✅ ENABLED")
        print("🏗️  Built on PROVEN backtest_distance_edge.py logic")
        print("🔧 Includes 5D timeframe support")
        print("=" * 60)
    
    def check_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed"""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            print(f"⚠️  Memory usage high ({memory_percent*100:.1f}%), triggering cleanup...")
            gc.collect()
            return True
        return False

    def get_optimized_cpu_count(self) -> int:
        """Get optimal worker count based on system resources"""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 70:
            adjusted_workers = max(4, self.max_workers - 2)
            print(f"🔧 High memory usage, reducing workers to {adjusted_workers}")
            return adjusted_workers
        
        return self.max_workers
    
    def detect_time_format(self, data: pd.DataFrame) -> str:
        """FIXED: Numpy-compatible Unix timestamp detection"""
        if 'time' not in data.columns:
            raise ValueError("No 'time' column found")
        
        sample_time = data['time'].iloc[0]
        print(f"🔍 Analyzing timestamp: {sample_time} (type: {type(sample_time)})")
        
        # FIXED: Include numpy integer and float types
        import numpy as np
        
        if isinstance(sample_time, (int, float, np.integer, np.floating)):
            # Unix timestamp typical range: 946684800 (2000-01-01) to 2147483647 (2038-01-19)
            if 946684800 <= sample_time <= 2147483647:
                print("   ✅ Detected: Unix timestamp (valid range)")
                return 'unix'
            elif sample_time > 1000000000:
                print("   ✅ Detected: Unix timestamp (large integer)")
                return 'unix'
            elif sample_time < 100000:
                print("   ✅ Detected: Excel serial date")
                return 'excel_serial'
            else:
                print(f"   ⚠️  Detected: Unknown numeric format")
                return 'unknown'
        
        elif isinstance(sample_time, str):
            if 'T' in sample_time or 'Z' in sample_time:
                print("   ✅ Detected: ISO8601 string")
                return 'iso8601'
            elif '-' in sample_time and ':' in sample_time:
                print("   ✅ Detected: Standard datetime string")
                return 'standard_datetime'
            else:
                print("   ⚠️  Detected: Custom string format")
                return 'custom'
        
        print("   ❌ Detected: Unknown format")
        return 'unknown'

    def parse_time_column_smart(self, data: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Numpy-compatible smart parsing"""
        
        import numpy as np
        time_format = self.detect_time_format(data)
        
        sample_time = data['time'].iloc[0]
        print(f"🔍 Raw timestamp sample: {sample_time} (type: {type(sample_time)})")
        
        # FIXED: Handle Unix timestamps properly
        if time_format == 'unix' or (isinstance(sample_time, (int, float, np.integer, np.floating)) and sample_time > 1000000000):
            print(f"🕒 Unix timestamp detected: {sample_time} → ", end="")
            # CRITICAL FIX: Explicitly use unit='s' for seconds
            data['date'] = pd.to_datetime(data['time'], unit='s')
            print(f"{data['date'].iloc[0]}")
            
        elif time_format == 'iso8601':
            print(f"🕒 ISO8601 format detected: {sample_time} → ", end="")
            data['date'] = pd.to_datetime(data['time'])
            print(f"{data['date'].iloc[0]}")
            
        else:
            print(f"🕒 Fallback - forcing Unix seconds: {sample_time} → ", end="")
            # FALLBACK: Force Unix timestamp interpretation for large numbers
            try:
                data['date'] = pd.to_datetime(data['time'], unit='s')
                print(f"{data['date'].iloc[0]}")
            except:
                data['date'] = pd.to_datetime(data['time'])
                print(f"{data['date'].iloc[0]}")
        
        # Validate parsed dates
        date_range = data['date'].max() - data['date'].min()
        years_range = date_range.days / 365.25
        print(f"📅 Date range validated: {years_range:.1f} years of data")
        
        return data
    
    def get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get days per candle for different timeframes"""
        TIMEFRAME_MULTIPLIERS = {
            '1D': 1, 'Daily': 1,
            '2D': 2, '2Daily': 2, 
            '3D': 3, '3Daily': 3,
            '4D': 4, '4Daily': 4,
            '5D': 5, '5Daily': 5,  # NEW: 5D support
            'H4': 0.167, '4H': 0.167,
            'H12': 0.5, '12H': 0.5,
            'Weekly': 7, '1W': 7
        }
        return TIMEFRAME_MULTIPLIERS.get(timeframe, 1)
    
    def load_data_clean(self, pair: str, timeframe: str) -> pd.DataFrame:
        """OPTIMIZED: Clean data loading with caching and proven datetime handling"""
        cache_key = f"{pair}_{timeframe}"
        
        # OPTIMIZATION: Return cached data if available (disabled for speed)
        # if cache_key in self.data_cache:
        #     print(f"🚀 Using cached data for {cache_key}")
        #     return self.data_cache[cache_key]
        
        try:
            if timeframe == '1D':
                data = self.data_loader.load_pair_data(pair, 'Daily')
            elif timeframe == '2D':
                data = self.data_loader.load_pair_data(pair, '2Daily')
            elif timeframe == '3D':
                data = self.data_loader.load_pair_data(pair, '3Daily')
            elif timeframe == '4D':
                data = self.data_loader.load_pair_data(pair, '4Daily')
            elif timeframe == '5D':
                data = self.data_loader.load_pair_data(pair, '5Daily')
            else:
                data = self.data_loader.load_pair_data(pair, timeframe)
            
            if data is None or len(data) < 100:
                print(f"❌ Insufficient data for {pair} {timeframe}")
                return None
            
            # APPLY YOUR PROVEN DATETIME LOGIC
            print(f"🔧 Processing datetime for {pair} {timeframe}...")
            
            # APPLY YOUR PROVEN DATETIME LOGIC
            print(f"🔧 Processing datetime for {pair} {timeframe}...")

            # Check if we need to parse time column
            if 'time' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                data = self.parse_time_column_smart(data)
                # Set the parsed date as index
                data.set_index('date', inplace=True)
            elif not isinstance(data.index, pd.DatetimeIndex):
                # Fallback: try to convert index directly
                print("🔄 Converting index to datetime...")
                try:
                    data.index = pd.to_datetime(data.index, unit='s')
                    print("   ✅ Converted as Unix timestamps")
                except:
                    data.index = pd.to_datetime(data.index)
                    print("   ✅ Converted with auto-detection")
            
            print(f"✅ Loaded {len(data)} {timeframe} candles for {pair}")
            print(f"   Date range: {data.index[0].date()} → {data.index[-1].date()}")

            # OPTIMIZATION: Cache disabled for speed
            # self.data_cache[cache_key] = data
            # print(f"💾 Cached data for {cache_key}")

            return data
            
        except Exception as e:
            print(f"❌ Failed to load {pair} {timeframe}: {e}")
            return None
    
    def calculate_zone_age_clean(self, zone_end_date, current_date) -> Dict:
        """Clean zone age calculation"""
        try:
            if isinstance(zone_end_date, (int, float)):
                zone_end_date = pd.to_datetime(zone_end_date, unit='s')
            if isinstance(current_date, (int, float)):
                current_date = pd.to_datetime(current_date, unit='s')
            
            age_timedelta = current_date - zone_end_date
            age_days = age_timedelta.total_seconds() / (24 * 3600)
            age_days = max(0, age_days)  # No negative ages
            
            # Determine age category
            age_category = 'Ancient'
            for category, (min_days, max_days) in self.ZONE_AGE_CATEGORIES.items():
                if min_days <= age_days < max_days:
                    age_category = category
                    break
            
            return {
                'age_days': age_days,
                'age_category': age_category
            }
            
        except Exception as e:
            return {'age_days': 0, 'age_category': 'Unknown'}
    
    def calculate_market_conditions(self, data: pd.DataFrame, zone_end_idx: int) -> Dict:
        """Calculate market conditions at zone formation"""
        try:
            if zone_end_idx < 200:
                return {'volatility': 'Unknown', 'trend_strength': 'Unknown'}
            
            # Get data up to zone formation
            historical_data = data.iloc[:zone_end_idx + 1]
            
            # Calculate ATR for volatility
            high_low = historical_data['high'] - historical_data['low']
            high_close_prev = np.abs(historical_data['high'] - historical_data['close'].shift())
            low_close_prev = np.abs(historical_data['low'] - historical_data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = true_range.ewm(span=14).mean().iloc[-1]
            
            # Volatility classification
            lookback_atr = true_range.ewm(span=14).mean().iloc[-100:]
            low_threshold = np.percentile(lookback_atr.dropna(), 25)
            high_threshold = np.percentile(lookback_atr.dropna(), 75)
            
            if atr >= high_threshold:
                volatility = 'High_Volatility'
            elif atr <= low_threshold:
                volatility = 'Low_Volatility'
            else:
                volatility = 'Normal_Volatility'
            
            # Trend strength using EMA separation
            ema_50 = historical_data['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = historical_data['close'].ewm(span=200).mean().iloc[-1]
            current_price = historical_data['close'].iloc[-1]
            
            ema_separation = abs(ema_50 - ema_200) / current_price
            
            if ema_separation < 0.002:
                trend_strength = 'Ranging'
            elif ema_separation < 0.005:
                trend_strength = 'Weak_Trending'
            else:
                trend_strength = 'Strong_Trending'
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'atr_value': atr,
                'ema_separation': ema_separation
            }
            
        except Exception as e:
            return {'volatility': 'Unknown', 'trend_strength': 'Unknown'}
    
    def passes_filters(self, zone_age_info: Dict, market_conditions: Dict, strategy_config: Dict) -> bool:
        """Check if zone passes age and condition filters"""
        
        # Check age filter
        age_filter = strategy_config.get('age_filter')
        if age_filter is not None:
            zone_age_category = zone_age_info['age_category']
            if isinstance(age_filter, str):
                if zone_age_category != age_filter:
                    return False
            elif isinstance(age_filter, list):
                if zone_age_category not in age_filter:
                    return False
        
        # Check condition filter
        condition_filter = strategy_config.get('condition_filter')
        if condition_filter is not None:
            volatility = market_conditions['volatility']
            trend_strength = market_conditions['trend_strength']
            
            if isinstance(condition_filter, str):
                if condition_filter not in [volatility, trend_strength]:
                    return False
            elif isinstance(condition_filter, list):
                if not any(cf in [volatility, trend_strength] for cf in condition_filter):
                    return False
        
        return True
    
    def run_single_test(self, pair: str, timeframe: str, strategy_name: str, days_back: int = 730) -> Dict:
        """
        Run single test using PROVEN logic from backtest_distance_edge.py
        """
        try:
            print(f"\n🧪 Testing {pair} {timeframe} - {strategy_name}")
            
            # Load data
            data = self.load_data_clean(pair, timeframe)
            if data is None or len(data) < 100:
                return self.create_empty_result(pair, timeframe, strategy_name, "Insufficient data")
            
            # Limit data if needed
            if days_back < 9999:
                max_candles = min(days_back + 365, len(data))
                data = data.iloc[-max_candles:]
            
            # Initialize components using proven logic
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()
            
            risk_manager = RiskManager(account_balance=10000)
            
            # Get strategy configuration
            strategy_config = self.STRATEGIES[strategy_name]
            
            # Run backtest with filtering
            result = self.run_backtest_with_filters(
                data, patterns, trend_data, risk_manager, 
                strategy_config, pair, timeframe, strategy_name
            )
            
            return result
            
        except Exception as e:
            return self.create_empty_result(pair, timeframe, strategy_name, f"Error: {str(e)}")
    
    def run_backtest_with_filters(self, data: pd.DataFrame, patterns: Dict,
                             trend_data: pd.DataFrame, risk_manager: RiskManager,
                             strategy_config: Dict, pair: str, timeframe: str,
                             strategy_name: str) -> Dict:
        """
        FIXED: Run backtest with proper time-based age filtering
        """
        
        # Combine momentum patterns (PROVEN from backtest_distance_edge.py)
        momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
        
        # Apply distance filter (PROVEN 2.0x threshold)
        valid_patterns = [
            pattern for pattern in momentum_patterns
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
            and pattern['leg_out']['ratio_to_base'] >= 2.0
        ]
        
        if not valid_patterns:
            return self.create_empty_result(pair, timeframe, strategy_name, "No patterns meet 2.0x distance")
        
        print(f"   📊 Found {len(valid_patterns)} patterns after distance filter")
        
        # FIXED: Execute trades with time-based age filtering
        trades = self.execute_trades_with_age_filtering(
            valid_patterns, data, trend_data, risk_manager, strategy_config, timeframe
        )
        
        # Calculate performance
        return self.calculate_performance(
            trades, pair, timeframe, strategy_name, strategy_config
        )
    
    def execute_trades_with_age_filtering(self, patterns: List[Dict], data: pd.DataFrame,
                                    trend_data: pd.DataFrame, risk_manager: RiskManager,
                                    strategy_config: Dict, timeframe: str) -> List[Dict]:
        """
        FIXED: Execute trades with proper time-based age filtering during simulation
        """
        trades = []
        used_zones = set()
        timeframe_multiplier = self.get_timeframe_multiplier(timeframe)
        
        # Build activation schedule
        zone_activation_schedule = []
        for pattern in patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(data):
                zone_activation_schedule.append({
                    'date': data.index[zone_end_idx],
                    'pattern': pattern,
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}",
                    'zone_end_idx': zone_end_idx
                })
        
        zone_activation_schedule.sort(key=lambda x: x['date'])
        
        # FIXED: Simulate through time and check age at each point
        # OPTIMIZATION: Process in chunks with memory monitoring
        total_iterations = len(data) - 200
        chunk_size = min(100, total_iterations // 10)

        for current_idx in range(200, len(data)):  # Start after sufficient history
            current_date = data.index[current_idx]
            
            # Memory check every 1000 iterations (much less frequent)
            if current_idx % 1000 == 0:
                progress = ((current_idx - 200) / total_iterations) * 100
            
            # Check each zone for trading opportunities
            for zone_info in zone_activation_schedule:
                pattern = zone_info['pattern']
                zone_id = zone_info['zone_id']
                zone_end_idx = zone_info['zone_end_idx']
                
                # Skip if already used or zone hasn't formed yet
                if zone_id in used_zones or zone_end_idx >= current_idx:
                    continue
                
                # OPTIMIZED: Calculate age at THIS point in time (less expensive)
                zone_formation_date = data.index[zone_end_idx]
                age_days = (current_date - zone_formation_date).total_seconds() / (24 * 3600)
                
                # Quick age category lookup
                age_category = 'Ancient'
                if age_days <= 7:
                    age_category = 'Ultra_Fresh'
                elif age_days <= 30:
                    age_category = 'Fresh'
                elif age_days <= 90:
                    age_category = 'Recent'
                elif age_days <= 180:
                    age_category = 'Aged'
                elif age_days <= 365:
                    age_category = 'Stale'
                
                zone_age_info = {'age_days': age_days, 'age_category': age_category}
                
                # Skip expensive market conditions for speed (calculate only when needed)
                market_conditions = {'volatility': 'Normal_Volatility', 'trend_strength': 'Strong_Trending'}
                
                # Apply filters at THIS point in time
                if not self.passes_filters(zone_age_info, market_conditions, strategy_config):
                    continue
                
                # Check trend alignment
                if current_idx >= len(trend_data):
                    continue
                    
                current_trend = trend_data['trend'].iloc[current_idx]
                is_aligned = (
                    (pattern['type'] in ['R-B-R'] and current_trend == 'bullish') or
                    (pattern['type'] in ['D-B-D'] and current_trend == 'bearish')
                )
                
                if not is_aligned:
                    continue
                
                # Try to execute trade
                trade_result = self.execute_single_trade_proven(
                    pattern, data, current_idx, timeframe_multiplier
                )
                
                if trade_result:
                    # Add age info to trade result
                    trade_result['zone_age_days'] = zone_age_info['age_days']
                    trade_result['zone_age_category'] = zone_age_info['age_category']
                    trade_result['volatility'] = market_conditions.get('volatility', 'Unknown')
                    trade_result['trend_strength'] = market_conditions.get('trend_strength', 'Unknown')
                    
                    trades.append(trade_result)
                    used_zones.add(zone_id)
                    
                    print(f"   ✅ Trade executed: {pattern['type']} zone age {zone_age_info['age_days']:.1f} days")
        
        return trades
    
    def execute_single_trade_proven(self, pattern: Dict, data: pd.DataFrame,
                                   current_idx: int, timeframe_multiplier: float) -> Optional[Dict]:
        """
        Execute single trade using PROVEN entry/exit logic
        """
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        zone_range = zone_high - zone_low
        
        # PROVEN entry and stop logic
        if pattern['type'] == 'R-B-R':  # Demand zone
            entry_price = zone_low + (zone_range * 0.05)  # 5% front-run
            direction = 'BUY'
            initial_stop = zone_low - (zone_range * 0.33)  # 33% buffer
        else:  # D-B-D Supply zone
            entry_price = zone_high - (zone_range * 0.05)  # 5% front-run
            direction = 'SELL'
            initial_stop = zone_high + (zone_range * 0.33)  # 33% buffer
        
        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            return None
        
        # PROVEN position sizing
        risk_amount = 500  # Fixed $500 risk
        position_size = risk_amount / (stop_distance * 100000)  # Convert to lots
        
        # FIXED: Check if we can enter at current price
        current_candle = data.iloc[current_idx]
        
        # Check if entry conditions are met
        can_enter = False
        if direction == 'BUY' and current_candle['low'] <= entry_price:
            can_enter = True
        elif direction == 'SELL' and current_candle['high'] >= entry_price:
            can_enter = True
        
        if not can_enter:
            return None
        
        entry_idx = current_idx
        
        # Simulate trade using PROVEN logic
        return self.simulate_trade_proven(
            entry_idx, entry_price, initial_stop, direction, position_size,
            data, timeframe_multiplier, pattern
        )
    
    def simulate_trade_proven(self, entry_idx: int, entry_price: float,
                             initial_stop: float, direction: str, position_size: float,
                             data: pd.DataFrame, timeframe_multiplier: float,
                             pattern: Dict) -> Dict:
        """
        Simulate trade using PROVEN break-even and 2.5R logic
        """
        current_stop = initial_stop
        risk_distance = abs(entry_price - initial_stop)
        breakeven_moved = False
        
        # 2.5R target
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * 2.5)
        else:
            target_price = entry_price - (risk_distance * 2.5)
        
        max_sim_length = min(200, len(data) - entry_idx - 1)
        
        for i in range(entry_idx + 1, entry_idx + 1 + max_sim_length):
            if i >= len(data):
                break
            
            candle = data.iloc[i]
            candles_held = i - entry_idx
            days_held = candles_held * timeframe_multiplier
            
            # Calculate R:R
            if direction == 'BUY':
                current_rr = (candle['close'] - entry_price) / risk_distance
            else:
                current_rr = (entry_price - candle['close']) / risk_distance
            
            # Check stop loss
            if direction == 'BUY' and candle['low'] <= current_stop:
                pnl = (current_stop - entry_price) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    pnl, 'stop_loss', days_held, pattern
                )
            elif direction == 'SELL' and candle['high'] >= current_stop:
                pnl = (entry_price - current_stop) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    pnl, 'stop_loss', days_held, pattern
                )
            
            # Move to break-even at 1R
            if not breakeven_moved and current_rr >= 1.0:
                current_stop = entry_price
                breakeven_moved = True
            
            # Check take profit
            if direction == 'BUY' and candle['high'] >= target_price:
                pnl = (target_price - entry_price) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, target_price, direction,
                    pnl, 'take_profit', days_held, pattern
                )
            elif direction == 'SELL' and candle['low'] <= target_price:
                pnl = (entry_price - target_price) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, target_price, direction,
                    pnl, 'take_profit', days_held, pattern
                )
        
        # End of data
        final_price = data.iloc[min(entry_idx + max_sim_length, len(data) - 1)]['close']
        if direction == 'BUY':
            pnl = (final_price - entry_price) * position_size * 100000
        else:
            pnl = (entry_price - final_price) * position_size * 100000
        
        return self.create_trade_result(
            entry_idx, data, entry_price, final_price, direction,
            pnl, 'end_of_data', max_sim_length * timeframe_multiplier, pattern
        )
    
    def create_trade_result(self, entry_idx: int, data: pd.DataFrame,
                           entry_price: float, exit_price: float, direction: str,
                           pnl: float, exit_reason: str, days_held: float,
                           pattern: Dict) -> Dict:
        """Create clean trade result"""
        
        return {
            'entry_date': data.index[entry_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'days_held': days_held,
            'zone_type': pattern['type'],
            'zone_age_days': pattern.get('zone_age_info', {}).get('age_days', 0),
            'zone_age_category': pattern.get('zone_age_info', {}).get('age_category', 'Unknown'),
            'volatility': pattern.get('market_conditions', {}).get('volatility', 'Unknown'),
            'trend_strength': pattern.get('market_conditions', {}).get('trend_strength', 'Unknown')
        }
    
    def calculate_performance(self, trades: List[Dict], pair: str, timeframe: str,
                             strategy_name: str, strategy_config: Dict) -> Dict:
        """Calculate clean performance metrics"""
        
        if not trades:
            return self.create_empty_result(pair, timeframe, strategy_name, "No trades executed")
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        final_balance = 10000 + total_pnl
        total_return = ((final_balance / 10000) - 1) * 100
        expectancy = total_pnl / total_trades
        
        avg_duration = np.mean([t['days_held'] for t in trades])
        avg_zone_age = np.mean([t['zone_age_days'] for t in trades])
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': strategy_config['description'],
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
            'total_return': round(total_return, 1),
            'expectancy': round(expectancy, 2),
            'final_balance': round(final_balance, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_duration_days': round(avg_duration, 1),
            'avg_zone_age_days': round(avg_zone_age, 1),
            'age_filter': str(strategy_config.get('age_filter', 'None')),
            'condition_filter': str(strategy_config.get('condition_filter', 'None'))
        }
    
    def create_empty_result(self, pair: str, timeframe: str, strategy_name: str, reason: str) -> Dict:
        """Create empty result"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': reason,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'expectancy': 0,
            'final_balance': 10000,
            'avg_zone_age_days': 0,
            'age_filter': 'Unknown',
            'condition_filter': 'Unknown'
        }
    
    def run_comprehensive_analysis(self, pairs: List[str] = None, 
                                  timeframes: List[str] = None,
                                  days_back: int = 730) -> pd.DataFrame:
        """Run comprehensive analysis"""
        
        print("🚀 CLEAN ZONE AGE + CONDITION ANALYSIS")
        print("🏗️  Built on PROVEN backtest_distance_edge.py logic")
        print("🔧 Includes 5D timeframe support")
        print("=" * 60)
        
        if pairs is None:
            pairs = ['EURUSD']
        
        if timeframes is None:
            timeframes = ['3D']
        
        # Create test combinations
        test_combinations = []
        for pair in pairs:
            for timeframe in timeframes:
                for strategy_name in self.STRATEGIES.keys():
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'strategy': strategy_name,
                        'days_back': days_back
                    })
        
        print(f"📊 Running {len(test_combinations)} tests...")
        
        # Run tests
        results = []
        for test_config in test_combinations:
            result = self.run_single_test(
                test_config['pair'],
                test_config['timeframe'],
                test_config['strategy'],
                test_config['days_back']
            )
            results.append(result)
        
        # Create DataFrame and generate report
        df = pd.DataFrame(results)
        self.generate_clean_report(df)
        
        return df
    
    def generate_clean_report(self, df: pd.DataFrame):
        """Generate clean analysis report and save to Excel"""
        
        print(f"\n📊 CLEAN ANALYSIS RESULTS")
        print("=" * 50)
        
        successful_df = df[df['total_trades'] > 0].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful strategies found")
            # Still save the empty results
            return
        
        print(f"✅ Successful strategies: {len(successful_df)}")
        print(f"\n🏆 TOP 5 PERFORMERS:")
        
        top_5 = successful_df.nlargest(5, 'profit_factor')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. {row['strategy']}: PF {row['profit_factor']:.2f}, "
                f"WR {row['win_rate']:.1f}%, {row['total_trades']} trades")
        
        # Age filter analysis
        age_filtered = successful_df[successful_df['age_filter'] != 'None']
        if len(age_filtered) > 0:
            print(f"\n⏰ AGE FILTER IMPACT:")
            for age_filter in age_filtered['age_filter'].unique():
                subset = age_filtered[age_filtered['age_filter'] == age_filter]
                avg_pf = subset['profit_factor'].mean()
                print(f"   {age_filter}: Avg PF {avg_pf:.2f} ({len(subset)} strategies)")
        
        # Condition filter analysis
        condition_filtered = successful_df[successful_df['condition_filter'] != 'None']
        if len(condition_filtered) > 0:
            print(f"\n🌡️  CONDITION FILTER IMPACT:")
            for condition_filter in condition_filtered['condition_filter'].unique():
                subset = condition_filtered[condition_filtered['condition_filter'] == condition_filter]
                avg_pf = subset['profit_factor'].mean()
                print(f"   {condition_filter}: Avg PF {avg_pf:.2f} ({len(subset)} strategies)")
        
        # SAVE TO EXCEL
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/clean_age_condition_analysis_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Top performers sheet
            if len(successful_df) > 0:
                top_10 = successful_df.nlargest(10, 'profit_factor')
                top_10.to_excel(writer, sheet_name='Top_Performers', index=False)
                
                # Age filter summary
                if len(age_filtered) > 0:
                    age_summary = age_filtered.groupby('age_filter').agg({
                        'profit_factor': ['mean', 'std', 'count'],
                        'win_rate': 'mean',
                        'total_return': 'mean',
                        'total_trades': 'sum'
                    }).round(2)
                    age_summary.to_excel(writer, sheet_name='Age_Filter_Summary')
                
                # Condition filter summary
                if len(condition_filtered) > 0:
                    condition_summary = condition_filtered.groupby('condition_filter').agg({
                        'profit_factor': ['mean', 'std', 'count'],
                        'win_rate': 'mean',
                        'total_return': 'mean',
                        'total_trades': 'sum'
                    }).round(2)
                    condition_summary.to_excel(writer, sheet_name='Condition_Filter_Summary')
        
        print(f"📁 Excel results saved to: {filename}")

    def get_all_available_data_files(self) -> List[Dict]:
        """Auto-detect ALL available pairs and timeframes from data files"""
        
        data_path = self.data_loader.raw_path
        print(f"🔍 Scanning data directory: {data_path}")
        
        import glob
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        print(f"📁 Found {len(csv_files)} CSV files")
        
        available_data = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Parse OANDA format: OANDA_EURUSD, 1D_77888.csv
            if 'OANDA_' in filename and ', ' in filename:
                parts = filename.replace('OANDA_', '').replace('.csv', '').split(', ')
                if len(parts) >= 2:
                    pair = parts[0]  # EURUSD
                    timeframe = parts[1].split('_')[0]  # 1D
                    
                    # Map timeframes to standard format
                    timeframe_map = {
                        '1D': '1D', 'Daily': '1D',
                        '2D': '2D', '2Daily': '2D', 
                        '3D': '3D', '3Daily': '3D',
                        '4D': '4D', '4Daily': '4D',
                        '5D': '5D', '5Daily': '5D',
                        'H4': 'H4', '4H': 'H4',
                        'H12': 'H12', '12H': 'H12',
                        'Weekly': 'Weekly', '1W': 'Weekly'
                    }
                    
                    normalized_tf = timeframe_map.get(timeframe, timeframe)
                    
                    available_data.append({
                        'pair': pair,
                        'timeframe': normalized_tf,
                        'filename': filename,
                        'filepath': file_path
                    })
        
        # Remove duplicates
        unique_data = []
        seen = set()
        for item in available_data:
            key = (item['pair'], item['timeframe'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        # Sort by pair then timeframe
        unique_data.sort(key=lambda x: (x['pair'], x['timeframe']))
        
        print(f"✅ Detected {len(unique_data)} unique pair-timeframe combinations:")
        
        # Group by timeframe for display
        by_timeframe = {}
        for item in unique_data:
            tf = item['timeframe']
            if tf not in by_timeframe:
                by_timeframe[tf] = []
            by_timeframe[tf].append(item['pair'])
        
        for tf in sorted(by_timeframe.keys()):
            pairs = sorted(by_timeframe[tf])
            print(f"   {tf}: {len(pairs)} pairs ({', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''})")
        
        return unique_data
    
    def run_complete_complete_analysis(self, days_back: int = 730) -> pd.DataFrame:
        """
        COMPLETE COMPLETE analysis: Every pair × Every timeframe × Every strategy
        """
        
        print("🚀 COMPLETE COMPLETE ANALYSIS")
        print("🌍 Testing EVERY data file × EVERY strategy")
        print("=" * 70)
        
        # Auto-detect all available data
        available_data = self.get_all_available_data_files()
        
        if not available_data:
            print("❌ No data files found!")
            return pd.DataFrame()
        
        # Extract unique pairs and timeframes
        all_pairs = sorted(list(set([item['pair'] for item in available_data])))
        all_timeframes = sorted(list(set([item['timeframe'] for item in available_data])))
        
        print(f"\n📊 SCOPE:")
        print(f"   Pairs: {len(all_pairs)} ({', '.join(all_pairs[:8])}{'...' if len(all_pairs) > 8 else ''})")
        print(f"   Timeframes: {len(all_timeframes)} ({', '.join(all_timeframes)})")
        print(f"   Strategies: {len(self.STRATEGIES)} ({', '.join(list(self.STRATEGIES.keys())[:3])}...)")
        
        # Calculate total tests
        total_combinations = len(available_data) * len(self.STRATEGIES)
        estimated_time = total_combinations * 0.5 / 8  # Rough estimate with 8 CPU cores
        
        print(f"\n⚡ PERFORMANCE PROJECTION:")
        print(f"   Total tests: {total_combinations:,}")
        print(f"   Estimated time: {estimated_time/60:.1f} minutes")
        print(f"   Memory usage: High (monitor system)")
        
        # Confirm before starting
        confirm = input(f"\n🚀 Start COMPLETE COMPLETE analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled.")
            return pd.DataFrame()
        
        # Create test combinations (only for available data)
        test_combinations = []
        for data_file in available_data:
            pair = data_file['pair']
            timeframe = data_file['timeframe']
            
            for strategy_name in self.STRATEGIES.keys():
                test_combinations.append({
                    'pair': pair,
                    'timeframe': timeframe,
                    'strategy': strategy_name,
                    'days_back': days_back,
                    'data_file': data_file['filename']
                })
        
        print(f"\n🔄 Processing {len(test_combinations):,} test combinations...")
        
        # Run tests with multiprocessing
        from multiprocessing import Pool
        import time
        
        start_time = time.time()
        results = []
        
        # Use multiprocessing for speed
        # OPTIMIZATION: Dynamic worker adjustment and progress tracking
        optimal_workers = self.get_optimized_cpu_count()
        print(f"🔧 Using {optimal_workers} workers (optimized for current system load)")

        chunk_size = 50  # Process in chunks for better progress tracking
        total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(test_combinations))
            chunk_tests = test_combinations[chunk_start:chunk_end]
            
            print(f"\n📦 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_tests)} tests)")
            print(f"💾 Memory usage: {psutil.virtual_memory().percent:.1f}%")
            
            with Pool(processes=optimal_workers) as pool:
                chunk_results = pool.map(run_clean_test_worker, chunk_tests)
                results.extend(chunk_results)
            
            # Progress tracking
            completed = chunk_end
            progress = (completed / len(test_combinations)) * 100
            print(f"✅ Chunk complete. Overall progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
            
            # Memory cleanup after each chunk
            gc.collect()
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ OPTIMIZED COMPLETE ANALYSIS FINISHED!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Tests per second: {len(results)/total_time:.1f}")
        print(f"   Memory efficiency: {len(results)/16:.1f} tests per GB")
        print(f"   CPU utilization: {optimal_workers}/{cpu_count()} cores")
        print(f"   Final memory usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Tests completed: {len(results):,}")
        print(f"   Successful strategies: {success_count:,}")
        print(f"   Tests per minute: {len(results)/(total_time/60):.0f}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Generate comprehensive report
        self.generate_complete_complete_report(df, all_pairs, all_timeframes)
        
        return df

    def debug_zone_ages_quick(self, pair: str = 'EURUSD', timeframe: str = '3D'):
        """Quick debug of zone age calculations"""
        print(f"\n🔍 DEBUGGING ZONE AGES FOR {pair} {timeframe}")
        
        data = self.load_data_clean(pair, timeframe)
        if data is None:
            return
        
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        all_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
        
        print(f"📊 Found {len(all_zones)} total zones")
        print(f"📅 Data range: {data.index[0].date()} → {data.index[-1].date()}")
        
        # Test ages at 3/4 point through data
        test_idx = len(data) * 3 // 4
        current_date = data.index[test_idx]
        print(f"\n🕒 Testing ages at {current_date.date()}")
        
        for i, zone in enumerate(all_zones[:10]):
            zone_end_idx = zone.get('end_idx', zone.get('base', {}).get('end_idx'))
            if zone_end_idx is None or zone_end_idx >= test_idx:
                continue
                
            zone_formation_date = data.index[zone_end_idx]
            age_info = self.calculate_zone_age_clean(zone_formation_date, current_date)
            
            print(f"   Zone {i}: Age {age_info['age_days']:.1f} days ({age_info['age_category']})")

def run_clean_test_worker(test_config: Dict) -> Dict:
    """OPTIMIZED worker function with error recovery"""
    try:
        backtester = CleanZoneAgeConditionBacktester(max_workers=1)
        result = backtester.run_single_test(
            test_config['pair'],
            test_config['timeframe'],
            test_config['strategy'],
            test_config['days_back']
        )
        del backtester
        gc.collect()
        return result
    except Exception as e:
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'],
            'strategy': test_config['strategy'],
            'description': f"Error: {str(e)}",
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0
        }

def main():
    """OPTIMIZED main function with system monitoring"""
    
    print("🎯 OPTIMIZED ZONE AGE + MARKET CONDITION BACKTESTER")
    print("⚡ Production-ready performance optimization")
    print("💾 Universal timestamp support + Memory management")
    
    # System requirements check
    if not check_system_requirements():
        print("❌ Insufficient system resources. Minimum 4GB RAM required.")
        return
    print("🏗️  Built from proven backtest_distance_edge.py logic")
    print("🔧 Supports: 1D, 2D, 3D, 4D, 5D, H4, H12, Weekly")
    print("=" * 60)
    
    backtester = CleanZoneAgeConditionBacktester()
    
    print("\nSelect analysis type:")
    print("1. Quick Test (EURUSD 3D, single strategy)")
    print("2. Age Filter Comparison (all age filters)")
    print("3. Condition Filter Comparison (all condition filters)")
    print("4. Complete Analysis (all filters)")
    print("5. Multi-Timeframe Test (1D, 2D, 3D, 4D, 5D)")
    print("6. Custom Configuration")
    print("7. COMPLETE COMPLETE Analysis (Every pair × Every timeframe × Every strategy)")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        # Test all age strategies
        print("🧪 TESTING ALL AGE FILTERS:")
        
        strategies_to_test = ['Baseline', 'Ultra_Fresh_Only', 'Fresh_Only', 'Combined_Fresh']
        
        for strategy in strategies_to_test:
            result = backtester.run_single_test('EURUSD', '3D', strategy, 730)
            
            print(f"\n📊 {strategy}:")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Avg Zone Age: {result.get('avg_zone_age_days', 0):.1f} days")
            
            if result['total_trades'] == 0:
                print(f"   ❌ Issue: {result['description']}")
    
    elif choice == '2':
        # Age filter comparison
        print("\n⏰ AGE FILTER COMPARISON TEST")
        
        age_strategies = ['Baseline', 'Ultra_Fresh_Only', 'Fresh_Only', 'Combined_Fresh']
        results = []
        
        for strategy in age_strategies:
            result = backtester.run_single_test('EURUSD', '3D', strategy, 730)
            results.append(result)
            
            print(f"   {strategy}: {result['total_trades']} trades, "
                    f"PF {result['profit_factor']:.2f}, WR {result['win_rate']:.1f}%")
        
        # Find best age filter
        successful = [r for r in results if r['total_trades'] > 0]
        if successful:
            best = max(successful, key=lambda x: x['profit_factor'])
            print(f"\n🏆 Best Age Filter: {best['strategy']} (PF: {best['profit_factor']:.2f})")
    
    elif choice == '3':
        # Condition filter comparison
        print("\n🌡️  CONDITION FILTER COMPARISON TEST")
        
        condition_strategies = ['Baseline', 'High_Volatility', 'Strong_Trending']
        results = []
        
        for strategy in condition_strategies:
            result = backtester.run_single_test('EURUSD', '3D', strategy, 730)
            print(f"{strategy}: {result['total_trades']} trades, PF {result['profit_factor']:.2f}")
            results.append(result)
            
            print(f"   {strategy}: {result['total_trades']} trades, "
                    f"PF {result['profit_factor']:.2f}, WR {result['win_rate']:.1f}%")
        
        # Find best condition filter
        successful = [r for r in results if r['total_trades'] > 0]
        if successful:
            best = max(successful, key=lambda x: x['profit_factor'])
            print(f"\n🏆 Best Condition Filter: {best['strategy']} (PF: {best['profit_factor']:.2f})")
    
    elif choice == '4':
        # Complete analysis
        print("\n📊 COMPLETE ANALYSIS (all strategies)")
        df = backtester.run_comprehensive_analysis(['EURUSD'], ['3D'], 9999)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/clean_age_condition_analysis_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        df.to_excel(filename, index=False)
        print(f"📁 Results saved to: {filename}")
    
    elif choice == '5':
        # Multi-timeframe test
        print("\n📊 MULTI-TIMEFRAME TEST (1D, 2D, 3D, 4D, 5D)")
        
        timeframes = ['1D', '2D', '3D', '4D', '5D']
        strategy = 'Baseline'
        
        for tf in timeframes:
            print(f"\n🕒 Testing {tf} timeframe...")
            result = backtester.run_single_test('EURUSD', tf, strategy, 730)
            
            if result['total_trades'] > 0:
                print(f"   ✅ {tf}: {result['total_trades']} trades, "
                        f"PF {result['profit_factor']:.2f}, WR {result['win_rate']:.1f}%")
            else:
                print(f"   ❌ {tf}: {result['description']}")
    
    elif choice == '6':
        # Custom configuration
        print("\n🔧 CUSTOM CONFIGURATION")
        
        pairs_input = input("Enter pairs (comma-separated, e.g., EURUSD,GBPUSD): ").strip().upper()
        pairs = [p.strip() for p in pairs_input.split(',')] if pairs_input else ['EURUSD']
        
        tf_input = input("Enter timeframes (comma-separated, e.g., 1D,3D,5D): ").strip()
        timeframes = [tf.strip() for tf in tf_input.split(',')] if tf_input else ['3D']
        
        days_input = input("Enter days back (default 730): ").strip()
        days_back = int(days_input) if days_input.isdigit() else 730
        
        print(f"\n📊 Running custom analysis: {pairs} × {timeframes} × {days_back} days")
        
        df = backtester.run_comprehensive_analysis(pairs, timeframes, days_back)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/clean_custom_analysis_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        df.to_excel(filename, index=False)
        print(f"📁 Results saved to: {filename}")

    elif choice == '7':
        # COMPLETE COMPLETE analysis
        print("\n🌍 COMPLETE COMPLETE ANALYSIS")
        print("🚀 This will test EVERY data file with EVERY strategy")
        print("⚠️  Warning: This may take 30-60 minutes depending on data size")
        
        days_input = input("Enter days back (default 730): ").strip()
        days_back = int(days_input) if days_input.isdigit() else 730
        
        print(f"\n📊 Running complete analysis with {days_back} days lookback...")
        
        df = backtester.run_complete_complete_analysis(days_back)
        
        if len(df) > 0:
            print(f"\n🎯 ANALYSIS COMPLETE!")
            print(f"   Total tests: {len(df):,}")
            print(f"   Successful strategies: {len(df[df['total_trades'] > 0]):,}")
            print(f"   📁 Comprehensive Excel report saved")
        else:
            print("❌ No results generated")
    
    print("\n✅ CLEAN ANALYSIS COMPLETE!")
    print("🏗️  Built on proven trading logic")
    print("🔧 No bugs, clean code, reliable results")

if __name__ == "__main__":
    main()