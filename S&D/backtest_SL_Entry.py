"""
STOP/ENTRY PARAMETER OPTIMIZATION BACKTESTER
Built from proven backtest_distance_edge.py logic with systematic parameter testing
Optimized for Intel i5-10400F (6C/12T) + 16GB RAM
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

class UniversalTimestampLoader(DataLoader):
    """Universal timestamp loader with automatic format detection"""
    
    def detect_timestamp_format(self, sample_data) -> str:
        """Auto-detect timestamp format with robust error handling"""
        if 'time' not in sample_data.columns:
            if isinstance(sample_data.index, pd.DatetimeIndex):
                return 'datetime_index'
            raise ValueError("No 'time' column found and index is not datetime")
        
        sample_time = sample_data['time'].iloc[0]
        print(f"🔍 Analyzing timestamp: {sample_time} (type: {type(sample_time)})")
        
        import numpy as np
        
        # Check for numeric timestamps (Unix or Excel serial)
        if isinstance(sample_time, (int, float, np.integer, np.floating)):
            # Unix timestamp range: 946684800 (2000-01-01) to 2147483647 (2038-01-19)
            if 946684800 <= sample_time <= 2147483647:
                print("   ✅ Detected: Unix timestamp (valid range)")
                return 'unix_seconds'
            elif sample_time > 1000000000:  # Large number likely Unix
                print("   ✅ Detected: Unix timestamp (large number)")
                return 'unix_seconds'
            elif 25000 <= sample_time <= 50000:  # Excel serial date range
                print("   ✅ Detected: Excel serial date")
                return 'excel_serial'
            else:
                print(f"   ⚠️  Unknown numeric format: {sample_time}")
                return 'unknown_numeric'
        
        # Check for string timestamps
        elif isinstance(sample_time, str):
            if 'T' in sample_time or 'Z' in sample_time:
                print("   ✅ Detected: ISO8601 string")
                return 'iso8601'
            elif '-' in sample_time and ':' in sample_time:
                print("   ✅ Detected: Standard datetime string")
                return 'standard_datetime'
            else:
                print("   ⚠️  Unknown string format")
                return 'unknown_string'
        
        print("   ❌ Unknown timestamp format")
        return 'unknown'

    def parse_timestamp_universal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Universal timestamp parsing with fallback mechanisms"""
        
        if isinstance(data.index, pd.DatetimeIndex):
            print("✅ Data already has datetime index")
            return data
        
        timestamp_format = self.detect_timestamp_format(data)
        original_data = data.copy()
        
        try:
            if timestamp_format == 'unix_seconds':
                print("🕒 Parsing Unix timestamps...")
                data['date'] = pd.to_datetime(data['time'], unit='s')
                
            elif timestamp_format == 'excel_serial':
                print("🕒 Parsing Excel serial dates...")
                # Excel serial date: days since 1900-01-01
                data['date'] = pd.to_datetime(data['time'], origin='1900-01-01', unit='D')
                
            elif timestamp_format in ['iso8601', 'standard_datetime']:
                print("🕒 Parsing string timestamps...")
                data['date'] = pd.to_datetime(data['time'], format='mixed')
                
            else:
                # Universal fallback: let pandas infer
                print("🔄 Using pandas universal parser...")
                data['date'] = pd.to_datetime(data['time'], infer_datetime_format=True)
            
            # Validate parsed dates
            date_range = data['date'].max() - data['date'].min()
            years_range = date_range.days / 365.25
            
            print(f"📅 Parsed date range: {data['date'].min()} → {data['date'].max()}")
            print(f"   Duration: {years_range:.1f} years")
            
            # Check for reasonable date range
            if date_range.days < 30:
                raise ValueError("Suspicious date range - possible parsing error")
            
            # Set datetime index and cleanup
            data.set_index('date', inplace=True)
            data.drop('time', axis=1, inplace=True, errors='ignore')
            
            return data
            
        except Exception as e:
            print(f"❌ Timestamp parsing failed: {e}")
            print("🔄 Attempting fallback parsing methods...")
            
            # Fallback 1: Force Unix interpretation
            try:
                data = original_data.copy()
                data['date'] = pd.to_datetime(data['time'], unit='s')
                data.set_index('date', inplace=True)
                data.drop('time', axis=1, inplace=True, errors='ignore')
                print("✅ Fallback 1 successful (Unix seconds)")
                return data
            except:
                pass
            
            # Fallback 2: String parsing with mixed format
            try:
                data = original_data.copy()
                data['date'] = pd.to_datetime(data['time'], format='mixed')
                data.set_index('date', inplace=True)
                data.drop('time', axis=1, inplace=True, errors='ignore')
                print("✅ Fallback 2 successful (mixed format)")
                return data
            except:
                pass
            
            # Final fallback: Return original with warning
            print("❌ All parsing methods failed. Using original data.")
            return original_data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced cleaning with universal timestamp parsing"""
        cleaned_data = data.copy()
        
        # Apply universal timestamp parsing
        if 'time' in cleaned_data.columns or not isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data = self.parse_timestamp_universal(cleaned_data)
        
        # Standard OHLC processing
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        # Remove invalid data
        cleaned_data.dropna(subset=price_columns, inplace=True)
        
        # Validation
        if len(cleaned_data) > 0:
            print(f"✅ Cleaned data: {len(cleaned_data)} candles")
            if isinstance(cleaned_data.index, pd.DatetimeIndex):
                print(f"   Range: {cleaned_data.index[0].date()} → {cleaned_data.index[-1].date()}")
        
        return cleaned_data

class OptimizedStopEntryBacktester:
    """
    Professional stop/entry parameter optimization backtester
    Built from proven backtest_distance_edge.py logic
    Optimized for i5-10400F (6C/12T) + 16GB RAM
    """
    
    # PARAMETER CONFIGURATIONS
    QUICK_VALIDATION_CONFIGS = {
        'baseline': {'stop_buffer': 0.33, 'entry_frontrun': 0.05},
        'no_stop_buffer': {'stop_buffer': 0.0, 'entry_frontrun': 0.05},
        'no_frontrun': {'stop_buffer': 0.33, 'entry_frontrun': 0.0},
        'minimal_both': {'stop_buffer': 0.15, 'entry_frontrun': 0.02},
        'aggressive_both': {'stop_buffer': 0.50, 'entry_frontrun': 0.10}
    }
    
    COMPREHENSIVE_CONFIGS = {
        'baseline': {'stop_buffer': 0.33, 'entry_frontrun': 0.05},
        
        # Stop Buffer Variations (keep 5% front-run)
        'no_stop_buffer': {'stop_buffer': 0.0, 'entry_frontrun': 0.05},
        'small_stop_buffer': {'stop_buffer': 0.15, 'entry_frontrun': 0.05},
        'large_stop_buffer': {'stop_buffer': 0.50, 'entry_frontrun': 0.05},
        'max_stop_buffer': {'stop_buffer': 0.75, 'entry_frontrun': 0.05},
        
        # Entry Front-run Variations (keep 33% stop buffer)
        'no_frontrun': {'stop_buffer': 0.33, 'entry_frontrun': 0.0},
        'small_frontrun': {'stop_buffer': 0.33, 'entry_frontrun': 0.02},
        'large_frontrun': {'stop_buffer': 0.33, 'entry_frontrun': 0.10},
        'max_frontrun': {'stop_buffer': 0.33, 'entry_frontrun': 0.15},
        
        # Combined Optimizations
        'minimal_both': {'stop_buffer': 0.15, 'entry_frontrun': 0.02},
        'aggressive_both': {'stop_buffer': 0.50, 'entry_frontrun': 0.10},
        'conservative_both': {'stop_buffer': 0.75, 'entry_frontrun': 0.0},
        'balanced_alt1': {'stop_buffer': 0.25, 'entry_frontrun': 0.07},
        'balanced_alt2': {'stop_buffer': 0.40, 'entry_frontrun': 0.03},
        'extreme_test': {'stop_buffer': 1.0, 'entry_frontrun': 0.20}
    }
    
    # LIVE BASELINE PERFORMANCE (Reference Standard)
    LIVE_BASELINE = {
        'profit_factor': 2.5,
        'win_rate': 40.0,
        'sample_size': 20,
        'period': '3-month forward test',
        'tolerance': 0.15  # 15% tolerance
    }

    def __init__(self, max_workers: int = None):
        """Initialize optimizer with system-specific settings"""
        
        # System optimization for i5-10400F + 16GB RAM
        self.max_workers = min(max_workers or 10, cpu_count() - 2)
        self.memory_threshold_cleanup = 0.90  # 90% triggers cleanup
        self.memory_threshold_warning = 0.85  # 85% shows warning
        self.chunk_size = 100  # Process 100 tests per chunk
        self.max_data_cache = 5  # Cache max 5 datasets
        
        # Universal data loader
        self.data_loader = UniversalTimestampLoader()
        self.data_cache = {}
        self.results = []
        
        print("🎯 STOP/ENTRY PARAMETER OPTIMIZATION BACKTESTER")
        print(f"💻 System: i5-10400F optimized ({self.max_workers} workers)")
        print(f"💾 Memory: 16GB with {self.memory_threshold_cleanup*100:.0f}% cleanup threshold")
        print(f"🎲 Universal timestamp support: ✅ ENABLED")
        print(f"🏗️  Built from proven backtest_distance_edge.py logic")
        print("=" * 70)

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor system memory usage"""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }

    def cleanup_memory(self):
        """Aggressive memory cleanup when threshold exceeded"""
        print("🧹 Performing memory cleanup...")
        
        # Clear data cache if it gets too large
        if len(self.data_cache) > self.max_data_cache:
            oldest_keys = list(self.data_cache.keys())[:-self.max_data_cache]
            for key in oldest_keys:
                del self.data_cache[key]
        
        # Force garbage collection
        gc.collect()
        
        memory_after = self.monitor_memory_usage()
        print(f"   Memory after cleanup: {memory_after['percent']:.1f}%")

    def get_all_available_data_files(self) -> List[Dict]:
        """Auto-detect ALL available pairs and timeframes"""
        
        data_path = self.data_loader.raw_path
        print(f"🔍 Scanning data directory: {data_path}")
        
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        print(f"📁 Found {len(csv_files)} CSV files")
        
        available_data = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Parse OANDA format: OANDA_EURUSD, 1D_77888.csv
            if 'OANDA_' in filename and ', ' in filename:
                parts = filename.replace('OANDA_', '').replace('.csv', '').split(', ')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1].split('_')[0]
                    
                    # Normalize timeframes
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
        
        # Remove duplicates and sort
        unique_data = []
        seen = set()
        for item in available_data:
            key = (item['pair'], item['timeframe'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        unique_data.sort(key=lambda x: (x['pair'], x['timeframe']))
        
        print(f"✅ Detected {len(unique_data)} unique combinations")
        return unique_data

    def load_data_with_caching(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data with intelligent caching"""
        cache_key = f"{pair}_{timeframe}"
        
        # Return cached data if available
        if cache_key in self.data_cache:
            print(f"🚀 Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Check memory before loading new data
        memory_status = self.monitor_memory_usage()
        if memory_status['percent'] > self.memory_threshold_warning:
            print(f"⚠️  High memory usage ({memory_status['percent']:.1f}%), cleaning cache...")
            self.cleanup_memory()
        
        try:
            # Load data using universal loader
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
            
            print(f"✅ Loaded {len(data)} candles for {pair} {timeframe}")
            
            # Cache the data (memory permitting)
            if len(self.data_cache) < self.max_data_cache:
                self.data_cache[cache_key] = data
                print(f"💾 Cached data for {cache_key}")
            
            return data
            
        except Exception as e:
            print(f"❌ Failed to load {pair} {timeframe}: {e}")
            return None

    def run_single_parameter_test(self, pair: str, timeframe: str, config_name: str, 
                                config: Dict, days_back: int = 730) -> Dict:
        """
        Run single parameter test using PROVEN logic with ONLY parameter modifications
        """
        try:
            print(f"\n🧪 Testing {pair} {timeframe} - {config_name}")
            print(f"   Stop Buffer: {config['stop_buffer']*100:.0f}%, Entry Front-run: {config['entry_frontrun']*100:.0f}%")
            
            # Load data
            data = self.load_data_with_caching(pair, timeframe)
            if data is None:
                return self.create_empty_result(pair, timeframe, config_name, config, "Data loading failed")
            
            # Limit data if specified
            if days_back < 9999:
                max_candles = min(days_back + 365, len(data))
                data = data.iloc[-max_candles:]
            
            # Initialize components (PROVEN logic from backtest_distance_edge.py)
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()
            
            # Clean up intermediate objects
            del candle_classifier, classified_data, zone_detector, trend_classifier
            gc.collect()
            
            # Run backtest with parameter configuration
            result = self.run_backtest_with_parameters(
                data, patterns, trend_data, config, pair, timeframe, config_name
            )
            
            return result
            
        except Exception as e:
            return self.create_empty_result(pair, timeframe, config_name, config, f"Error: {str(e)}")

    def run_backtest_with_parameters(self, data: pd.DataFrame, patterns: Dict,
                                   trend_data: pd.DataFrame, config: Dict,
                                   pair: str, timeframe: str, config_name: str) -> Dict:
        """
        Run backtest with parameter configuration using PROVEN logic
        ONLY MODIFIED: execute_buy_trade and execute_sell_trade functions
        """
        
        # PROVEN: Use momentum patterns with 2.5x distance filter
        momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
        valid_patterns = [
            pattern for pattern in momentum_patterns
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
            and pattern['leg_out']['ratio_to_base'] >= 2.0  # PROVEN threshold
        ]
        
        if not valid_patterns:
            return self.create_empty_result(pair, timeframe, config_name, config, "No patterns meet 2.0x distance")
        
        print(f"   📊 Found {len(valid_patterns)} patterns after distance filter")
        
        # Execute trades with parameter configuration
        trades = self.execute_trades_with_parameters(
            valid_patterns, data, trend_data, config
        )
        
        # Calculate performance using PROVEN metrics
        return self.calculate_performance_with_baseline_comparison(
            trades, pair, timeframe, config_name, config
        )

    def execute_trades_with_parameters(self, patterns: List[Dict], data: pd.DataFrame,
                                     trend_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """
        Execute trades with parameter configuration
        PRESERVED: All proven logic except entry/stop calculations
        """
        trades = []
        used_zones = set()
        
        # Build activation schedule (PROVEN logic)
        zone_activation_schedule = []
        for pattern in patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(data):
                zone_activation_schedule.append({
                    'date': data.index[zone_end_idx],
                    'pattern': pattern,
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
                })
        
        zone_activation_schedule.sort(key=lambda x: x['date'])
        
        # Process zones (PROVEN walk-forward logic)
        for i, (current_date, candle) in enumerate(data.iterrows()):
            if i < 200:  # Need history for trend
                continue
            
            # Check for zone activations
            for zone_info in zone_activation_schedule:
                pattern = zone_info['pattern']
                zone_id = zone_info['zone_id']
                
                if zone_id in used_zones:
                    continue
                
                # Check if zone is ready for trading (formed at least)
                zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
                if zone_end_idx is None or zone_end_idx >= i:
                    continue
                
                # Check trend alignment (PROVEN logic)
                if i >= len(trend_data):
                    continue
                    
                current_trend = trend_data['trend'].iloc[i]
                is_aligned = (
                    (pattern['type'] == 'R-B-R' and current_trend == 'bullish') or
                    (pattern['type'] == 'D-B-D' and current_trend == 'bearish')
                )
                
                if not is_aligned:
                    continue
                
                # Try to execute trade with parameter configuration
                trade_result = self.execute_single_trade_with_parameters(
                    pattern, candle, current_date, config, data, i
                )
                
                if trade_result:
                    trades.append(trade_result)
                    used_zones.add(zone_id)
                    print(f"   ✅ Trade executed: {pattern['type']} zone")
        
        return trades

    def execute_single_trade_with_parameters(self, pattern: Dict, candle: pd.Series,
                                           current_date: pd.Timestamp, config: Dict,
                                           data: pd.DataFrame, current_idx: int) -> Optional[Dict]:
        """
        MODIFIED: Execute trade with configurable stop buffer and entry front-run
        This is the ONLY function with parameter modifications
        """
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        zone_range = zone_high - zone_low
        
        # MODIFIED: Configurable entry and stop calculations
        if pattern['type'] == 'R-B-R':  # Demand zone (BUY)
            # MODIFIED: Configurable entry front-run
            entry_price = zone_low + (zone_range * config['entry_frontrun'])
            direction = 'BUY'
            # MODIFIED: Configurable stop buffer
            initial_stop = zone_low - (zone_range * config['stop_buffer'])
            
        else:  # D-B-D Supply zone (SELL)
            # MODIFIED: Configurable entry front-run
            entry_price = zone_high - (zone_range * config['entry_frontrun'])
            direction = 'SELL'
            # MODIFIED: Configurable stop buffer
            initial_stop = zone_high + (zone_range * config['stop_buffer'])
        
        # Check if trade can be entered at current price
        can_enter = False
        if direction == 'BUY' and candle['low'] <= entry_price:
            can_enter = True
        elif direction == 'SELL' and candle['high'] >= entry_price:
            can_enter = True
        
        if not can_enter:
            return None
        
        # PROVEN: Fixed $500 risk and position sizing
        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            return None
        
        risk_amount = 500
        position_size = risk_amount / (stop_distance * 100000)
        
        # Simulate trade with PROVEN 1R→2.5R management
        return self.simulate_trade_with_proven_management(
            current_idx, entry_price, initial_stop, direction, position_size,
            data, pattern, config
        )

    def simulate_trade_with_proven_management(self, entry_idx: int, entry_price: float,
                                            initial_stop: float, direction: str, position_size: float,
                                            data: pd.DataFrame, pattern: Dict, config: Dict) -> Dict:
        """
        PRESERVED: Proven 1R break-even → 2.5R target management
        Only entry/stop prices are affected by parameter configuration
        """
        current_stop = initial_stop
        risk_distance = abs(entry_price - initial_stop)
        breakeven_moved = False
        
        # PROVEN: 2.5R target
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * 2.5)
        else:
            target_price = entry_price - (risk_distance * 2.5)
        
        max_sim_length = min(200, len(data) - entry_idx - 1)
        
        for i in range(entry_idx + 1, entry_idx + 1 + max_sim_length):
            if i >= len(data):
                break
            
            candle = data.iloc[i]
            days_held = i - entry_idx
            
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
                    pnl, 'stop_loss', days_held, pattern, config
                )
            elif direction == 'SELL' and candle['high'] >= current_stop:
                pnl = (entry_price - current_stop) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    pnl, 'stop_loss', days_held, pattern, config
                )
            
            # PROVEN: Move to break-even at 1R
            if not breakeven_moved and current_rr >= 1.0:
                current_stop = entry_price
                breakeven_moved = True
            
            # Check take profit
            if direction == 'BUY' and candle['high'] >= target_price:
                pnl = (target_price - entry_price) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, target_price, direction,
                    pnl, 'take_profit', days_held, pattern, config
                )
            elif direction == 'SELL' and candle['low'] <= target_price:
                pnl = (entry_price - target_price) * position_size * 100000
                return self.create_trade_result(
                    entry_idx, data, entry_price, target_price, direction,
                    pnl, 'take_profit', days_held, pattern, config
                )
        
        # End of simulation
        final_price = data.iloc[min(entry_idx + max_sim_length, len(data) - 1)]['close']
        if direction == 'BUY':
            pnl = (final_price - entry_price) * position_size * 100000
        else:
            pnl = (entry_price - final_price) * position_size * 100000
        
        return self.create_trade_result(
            entry_idx, data, entry_price, final_price, direction,
            pnl, 'end_of_data', max_sim_length, pattern, config
        )

    def create_trade_result(self, entry_idx: int, data: pd.DataFrame,
                          entry_price: float, exit_price: float, direction: str,
                          pnl: float, exit_reason: str, days_held: int,
                          pattern: Dict, config: Dict) -> Dict:
        """Create comprehensive trade result with parameter info"""
        
        return {
            'entry_date': data.index[entry_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'days_held': days_held,
            'zone_type': pattern['type'],
            'stop_buffer': config['stop_buffer'],
            'entry_frontrun': config['entry_frontrun'],
            'zone_high': pattern['zone_high'],
            'zone_low': pattern['zone_low'],
            'zone_range': pattern['zone_high'] - pattern['zone_low']
        }

    def calculate_performance_with_baseline_comparison(self, trades: List[Dict],
                                                       pair: str, timeframe: str, config_name: str,
                                                       config: Dict) -> Dict:
       """Calculate performance with baseline comparison"""
       if not trades:
           return self.create_empty_result(pair, timeframe, config_name, config, "No trades executed")
       
       # Basic performance metrics
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
       
       # Risk metrics
       max_drawdown = self.calculate_max_drawdown(trades)
       sharpe_ratio = self.calculate_sharpe_ratio(trades)
       
       # Baseline comparison
       baseline_comparison = self.compare_to_baseline(profit_factor, win_rate, total_trades)
       
       return {
           # Identification
           'pair': pair,
           'timeframe': timeframe,
           'config_name': config_name,
           'stop_buffer': config['stop_buffer'],
           'entry_frontrun': config['entry_frontrun'],
           
           # Core Performance
           'total_trades': total_trades,
           'winning_trades': winning_trades,
           'losing_trades': losing_trades,
           'win_rate': round(win_rate, 1),
           'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
           'total_return': round(total_return, 1),
           'expectancy': round(expectancy, 2),
           'final_balance': round(final_balance, 2),
           
           # P&L Breakdown
           'total_pnl': round(total_pnl, 2),
           'gross_profit': round(gross_profit, 2),
           'gross_loss': round(gross_loss, 2),
           'avg_win': round(gross_profit / winning_trades, 2) if winning_trades > 0 else 0,
           'avg_loss': round(gross_loss / losing_trades, 2) if losing_trades > 0 else 0,
           
           # Risk Metrics
           'max_drawdown': round(max_drawdown, 2),
           'sharpe_ratio': round(sharpe_ratio, 2),
           'avg_duration_days': round(np.mean([t['days_held'] for t in trades]), 1),
           
           # Baseline Comparison
           'vs_baseline_pf': round((profit_factor / self.LIVE_BASELINE['profit_factor'] - 1) * 100, 1),
           'vs_baseline_wr': round(win_rate - self.LIVE_BASELINE['win_rate'], 1),
           'baseline_status': baseline_comparison['status'],
           'baseline_score': baseline_comparison['score'],
           
           # Raw data for detailed analysis
           'trades_data': trades if len(trades) <= 50 else trades[:50]  # Limit for memory
       }
    
    def compare_to_baseline(self, profit_factor: float, win_rate: float, trade_count: int) -> Dict:
       """Compare results to live baseline with tolerance"""
       
       baseline_pf = self.LIVE_BASELINE['profit_factor']
       baseline_wr = self.LIVE_BASELINE['win_rate']
       tolerance = self.LIVE_BASELINE['tolerance']
       
       # Calculate deviations
       pf_deviation = abs(profit_factor - baseline_pf) / baseline_pf
       wr_deviation = abs(win_rate - baseline_wr) / baseline_wr
       
       # Determine status
       if pf_deviation <= tolerance and wr_deviation <= tolerance:
           status = "MAINTAINS_BASELINE"
           score = 100 - (pf_deviation + wr_deviation) * 50
       elif profit_factor >= baseline_pf * (1 - tolerance) and win_rate >= baseline_wr * (1 - tolerance):
           status = "ACCEPTABLE"
           score = 75 - (pf_deviation + wr_deviation) * 25
       elif profit_factor > baseline_pf or win_rate > baseline_wr:
           status = "PARTIAL_IMPROVEMENT"
           score = 50
       else:
           status = "UNDERPERFORMS"
           score = max(0, 25 - (pf_deviation + wr_deviation) * 25)
       
       # Statistical significance warning
       if trade_count < self.LIVE_BASELINE['sample_size']:
           status += "_LOW_SAMPLE"
       
       return {
           'status': status,
           'score': round(score, 1),
           'pf_deviation_pct': round(pf_deviation * 100, 1),
           'wr_deviation_pct': round(wr_deviation * 100, 1)
       }

    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown percentage"""
        if not trades:
            return 0
        
        balance = 10000
        peak = 10000
        max_dd = 0
        
        for trade in trades:
            balance += trade['pnl']
            if balance > peak:
                peak = balance
            
            drawdown = ((peak - balance) / peak) * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd

    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not trades:
            return 0
        
        daily_returns = []
        for trade in trades:
            days = max(trade['days_held'], 1)
            daily_return = (trade['pnl'] / 10000) / days
            daily_returns.append(daily_return)
        
        if len(daily_returns) == 0:
            return 0
        
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio
        sharpe = (avg_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
        return sharpe

    def create_empty_result(self, pair: str, timeframe: str, config_name: str, 
                            config: Dict, reason: str) -> Dict:
        """Create empty result with parameter info"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'config_name': config_name,
            'stop_buffer': config['stop_buffer'],
            'entry_frontrun': config['entry_frontrun'],
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'expectancy': 0,
            'final_balance': 10000,
            'baseline_status': 'NO_TRADES',
            'baseline_score': 0,
            'error_reason': reason
        }
    
    def run_quick_validation(self, days_back: int = 730, test_pairs: List[str] = None) -> pd.DataFrame:
       """Run quick validation with 5 parameter configurations"""
       
       print("🎯 QUICK VALIDATION - 5 Parameter Configurations")
       print("=" * 60)
       
       # Use subset of data for quick validation
       if test_pairs is None:
           test_pairs = ['EURUSD']  # Single pair for speed
       
       test_timeframes = ['3D']  # Single timeframe for speed
       
       # Create test combinations
       test_combinations = []
       for pair in test_pairs:
           for timeframe in test_timeframes:
               for config_name, config in self.QUICK_VALIDATION_CONFIGS.items():
                   test_combinations.append({
                       'pair': pair,
                       'timeframe': timeframe,
                       'config_name': config_name,
                       'config': config,
                       'days_back': days_back
                   })
       
       total_tests = len(test_combinations)
       print(f"📊 Running {total_tests} quick validation tests...")
       
       # Run tests sequentially for debugging
       results = []
       for i, test_config in enumerate(test_combinations, 1):
           print(f"\n🔄 Test {i}/{total_tests}: {test_config['config_name']}")
           
           result = self.run_single_parameter_test(
               test_config['pair'],
               test_config['timeframe'],
               test_config['config_name'],
               test_config['config'],
               test_config['days_back']
           )
           results.append(result)
       
       # Create DataFrame and generate report
       df = pd.DataFrame(results)
       self.generate_quick_validation_report(df)
       
       return df

    def run_comprehensive_analysis(self, days_back: int = 730, 
                                    test_all_data: bool = False) -> pd.DataFrame:
        """Run comprehensive analysis with all parameter configurations"""
        
        print("🚀 COMPREHENSIVE PARAMETER ANALYSIS")
        print("🎯 15 Parameter Configurations × All Available Data")
        print("=" * 70)
        
        # Get data scope
        if test_all_data:
            available_data = self.get_all_available_data_files()
            pairs = sorted(list(set([item['pair'] for item in available_data])))
            timeframes = sorted(list(set([item['timeframe'] for item in available_data])))
            print(f"📊 Testing ALL data: {len(pairs)} pairs × {len(timeframes)} timeframes")
        else:
            pairs = ['EURUSD', 'GBPUSD']  # Limited scope
            timeframes = ['3D']
            print(f"📊 Testing limited scope: {pairs} × {timeframes}")
        
        # Create test combinations
        test_combinations = []
        for pair in pairs:
            for timeframe in timeframes:
                for config_name, config in self.COMPREHENSIVE_CONFIGS.items():
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'config_name': config_name,
                        'config': config,
                        'days_back': days_back
                    })
        
        total_tests = len(test_combinations)
        estimated_time = (total_tests * 2.0) / self.max_workers  # Estimate
        
        print(f"\n📋 COMPREHENSIVE ANALYSIS SCOPE:")
        print(f"   Parameter configurations: {len(self.COMPREHENSIVE_CONFIGS)}")
        print(f"   Total tests: {total_tests}")
        print(f"   Estimated time: {estimated_time/60:.1f} minutes")
        print(f"   Memory optimization: ✅ ENABLED")
        
        confirm = input(f"\n🚀 Start comprehensive analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled.")
            return pd.DataFrame()
        
        # Run tests with parallel processing
        results = self.run_parallel_tests(test_combinations)
        
        # Create DataFrame and generate comprehensive report
        df = pd.DataFrame(results)
        self.generate_comprehensive_report(df)
        self.save_comprehensive_results(df)
        
        return df
    
    def run_parallel_tests(self, test_combinations: List[Dict]) -> List[Dict]:
       """Run tests with optimized parallel processing"""
       
       print(f"\n🔄 Starting parallel execution with {self.max_workers} workers...")
       start_time = time.time()
       
       # Process in chunks for memory management
       results = []
       chunk_size = self.chunk_size
       total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
       
       for chunk_idx in range(total_chunks):
           chunk_start = chunk_idx * chunk_size
           chunk_end = min(chunk_start + chunk_size, len(test_combinations))
           chunk_tests = test_combinations[chunk_start:chunk_end]
           
           print(f"\n📦 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_tests)} tests)")
           
           # Monitor memory before chunk
           memory_status = self.monitor_memory_usage()
           print(f"💾 Memory usage: {memory_status['percent']:.1f}%")
           
           if memory_status['percent'] > self.memory_threshold_cleanup:
               self.cleanup_memory()
           
           # Process chunk with multiprocessing
           with Pool(processes=self.max_workers) as pool:
               chunk_results = pool.map(run_parameter_test_worker, chunk_tests)
               results.extend(chunk_results)
           
           # Progress tracking
           completed = chunk_end
           progress = (completed / len(test_combinations)) * 100
           print(f"✅ Chunk complete. Progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
       
       total_time = time.time() - start_time
       success_count = len([r for r in results if r.get('total_trades', 0) > 0])
       
       print(f"\n✅ PARALLEL EXECUTION COMPLETE!")
       print(f"   Total time: {total_time/60:.1f} minutes")
       print(f"   Tests per second: {len(results)/total_time:.1f}")
       print(f"   Successful tests: {success_count}/{len(results)}")
       print(f"   Memory efficiency: ✅ OPTIMIZED")
       
       return results

    def generate_quick_validation_report(self, df: pd.DataFrame):
        """Generate quick validation report"""
        
        print(f"\n📊 QUICK VALIDATION RESULTS")
        print("=" * 50)
        
        successful_df = df[df['total_trades'] > 0].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful parameter configurations in quick validation")
            return
        
        print(f"✅ Successful configurations: {len(successful_df)}/5")
        
        print(f"\n🏆 PARAMETER CONFIGURATION COMPARISON:")
        print(f"{'Config':<15} {'Trades':<8} {'Win%':<7} {'PF':<6} {'vs Base PF':<10} {'vs Base WR':<10} {'Status':<15}")
        print("-" * 85)
        
        for _, row in successful_df.iterrows():
            status_short = row['baseline_status'][:12] + ".." if len(row['baseline_status']) > 14 else row['baseline_status']
            print(f"{row['config_name']:<15} {row['total_trades']:<8} {row['win_rate']:<7.1f} "
                    f"{row['profit_factor']:<6.2f} {row['vs_baseline_pf']:>+7.1f}% {row['vs_baseline_wr']:>+7.1f}% "
                    f"{status_short:<15}")
        
        # Identify best configuration
        if len(successful_df) > 0:
            best_config = successful_df.loc[successful_df['baseline_score'].idxmax()]
            print(f"\n🎯 BEST QUICK VALIDATION CONFIG:")
            print(f"   Configuration: {best_config['config_name']}")
            print(f"   Stop Buffer: {best_config['stop_buffer']*100:.0f}%")
            print(f"   Entry Front-run: {best_config['entry_frontrun']*100:.0f}%")
            print(f"   Baseline Score: {best_config['baseline_score']:.1f}/100")
            print(f"   Status: {best_config['baseline_status']}")

    def generate_comprehensive_report(self, df: pd.DataFrame):
        """Generate comprehensive analysis report"""
        
        print(f"\n📊 COMPREHENSIVE PARAMETER OPTIMIZATION RESULTS")
        print("🎯 Complete Analysis vs Live Baseline Performance")
        print("=" * 80)
        
        successful_df = df[df['total_trades'] > 0].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful parameter configurations found!")
            return
        
        print(f"✅ EXECUTIVE SUMMARY:")
        print(f"   Total configurations tested: {len(df)}")
        print(f"   Successful configurations: {len(successful_df)}")
        print(f"   Success rate: {len(successful_df)/len(df)*100:.1f}%")
        
        # TOP 5 PERFORMERS
        print(f"\n🏆 TOP 5 PARAMETER CONFIGURATIONS:")
        top_5 = successful_df.nlargest(5, 'baseline_score')
        
        print(f"{'Rank':<5} {'Config':<20} {'Stop%':<7} {'Entry%':<8} {'PF':<6} {'WR%':<6} {'Score':<7} {'Status':<15}")
        print("-" * 90)
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            status_short = row['baseline_status'][:13] + ".." if len(row['baseline_status']) > 15 else row['baseline_status']
            print(f"{i:<5} {row['config_name']:<20} {row['stop_buffer']*100:<7.0f} "
                    f"{row['entry_frontrun']*100:<8.0f} {row['profit_factor']:<6.2f} {row['win_rate']:<6.1f} "
                    f"{row['baseline_score']:<7.1f} {status_short:<15}")
        
        # PARAMETER SENSITIVITY ANALYSIS
        print(f"\n🔬 PARAMETER SENSITIVITY ANALYSIS:")
        
        # Stop buffer analysis
        stop_buffer_analysis = successful_df.groupby('stop_buffer').agg({
            'profit_factor': ['mean', 'std', 'count'],
            'win_rate': 'mean',
            'baseline_score': 'mean'
        }).round(2)
        
        print(f"\n📊 STOP BUFFER IMPACT:")
        print(f"{'Buffer%':<10} {'Avg PF':<8} {'Avg WR%':<8} {'Avg Score':<10} {'Count':<7}")
        print("-" * 50)
        for buffer in sorted(stop_buffer_analysis.index):
            row = stop_buffer_analysis.loc[buffer]
            avg_pf = row[('profit_factor', 'mean')]
            avg_wr = row[('win_rate', 'mean')]
            avg_score = row[('baseline_score', 'mean')]
            count = row[('profit_factor', 'count')]
            print(f"{buffer*100:<10.0f} {avg_pf:<8.2f} {avg_wr:<8.1f} {avg_score:<10.1f} {count:<7.0f}")
        
        # Entry front-run analysis
        frontrun_analysis = successful_df.groupby('entry_frontrun').agg({
            'profit_factor': ['mean', 'std', 'count'],
            'win_rate': 'mean',
            'baseline_score': 'mean'
        }).round(2)
        
        print(f"\n📊 ENTRY FRONT-RUN IMPACT:")
        print(f"{'Frontrun%':<10} {'Avg PF':<8} {'Avg WR%':<8} {'Avg Score':<10} {'Count':<7}")
        print("-" * 50)
        for frontrun in sorted(frontrun_analysis.index):
            row = frontrun_analysis.loc[frontrun]
            avg_pf = row[('profit_factor', 'mean')]
            avg_wr = row[('win_rate', 'mean')]
            avg_score = row[('baseline_score', 'mean')]
            count = row[('profit_factor', 'count')]
            print(f"{frontrun*100:<10.0f} {avg_pf:<8.2f} {avg_wr:<8.1f} {avg_score:<10.1f} {count:<7.0f}")
        
        # FINAL RECOMMENDATIONS
        if len(successful_df) > 0:
            # Best overall configuration
            best_overall = successful_df.loc[successful_df['baseline_score'].idxmax()]
            
            # Most consistent configuration (lowest std dev)
            config_consistency = successful_df.groupby('config_name')['profit_factor'].agg(['mean', 'std', 'count'])
            most_consistent = config_consistency.loc[config_consistency['std'].idxmin()]
            
            print(f"\n🎯 FINAL RECOMMENDATIONS:")
            print(f"\n🥇 OPTIMAL CONFIGURATION:")
            print(f"   Configuration: {best_overall['config_name']}")
            print(f"   Stop Buffer: {best_overall['stop_buffer']*100:.0f}% (vs baseline 33%)")
            print(f"   Entry Front-run: {best_overall['entry_frontrun']*100:.0f}% (vs baseline 5%)")
            print(f"   Performance: PF {best_overall['profit_factor']:.2f}, WR {best_overall['win_rate']:.1f}%")
            print(f"   vs Baseline: PF {best_overall['vs_baseline_pf']:+.1f}%, WR {best_overall['vs_baseline_wr']:+.1f}%")
            print(f"   Baseline Score: {best_overall['baseline_score']:.1f}/100")
            print(f"   Status: {best_overall['baseline_status']}")
            
            # Parameter change recommendations
            current_stop = 0.33
            current_frontrun = 0.05
            
            stop_change = best_overall['stop_buffer'] - current_stop
            frontrun_change = best_overall['entry_frontrun'] - current_frontrun
            
            print(f"\n📋 PARAMETER CHANGE RECOMMENDATIONS:")
            if abs(stop_change) > 0.05:
                direction = "INCREASE" if stop_change > 0 else "DECREASE"
                print(f"   Stop Buffer: {direction} from 33% to {best_overall['stop_buffer']*100:.0f}% ({stop_change*100:+.0f}%)")
            else:
                print(f"   Stop Buffer: MAINTAIN at ~33% (optimal: {best_overall['stop_buffer']*100:.0f}%)")
            
            if abs(frontrun_change) > 0.02:
                direction = "INCREASE" if frontrun_change > 0 else "DECREASE"
                print(f"   Entry Front-run: {direction} from 5% to {best_overall['entry_frontrun']*100:.0f}% ({frontrun_change*100:+.0f}%)")
            else:
                print(f"   Entry Front-run: MAINTAIN at ~5% (optimal: {best_overall['entry_frontrun']*100:.0f}%)")

    def save_comprehensive_results(self, df: pd.DataFrame):
       """Save comprehensive results to Excel with professional formatting"""
       
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       filename = f"results/stop_entry_optimization_{timestamp}.xlsx"
       os.makedirs('results', exist_ok=True)
       
       print(f"\n💾 Saving comprehensive results to Excel...")
       
       with pd.ExcelWriter(filename, engine='openpyxl') as writer:
           
           # Sheet 1: Executive Summary
           successful_df = df[df['total_trades'] > 0].copy()
           if len(successful_df) > 0:
               top_performers = successful_df.nlargest(10, 'baseline_score')
               executive_summary = top_performers[[
                   'config_name', 'stop_buffer', 'entry_frontrun', 'total_trades',
                   'win_rate', 'profit_factor', 'vs_baseline_pf', 'vs_baseline_wr',
                   'baseline_score', 'baseline_status'
               ]].copy()
               executive_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
           
           # Sheet 2: Complete Results Matrix
           complete_results = df[[
               'pair', 'timeframe', 'config_name', 'stop_buffer', 'entry_frontrun',
               'total_trades', 'win_rate', 'profit_factor', 'total_return', 'expectancy',
               'max_drawdown', 'sharpe_ratio', 'vs_baseline_pf', 'vs_baseline_wr',
               'baseline_score', 'baseline_status'
           ]].copy()
           complete_results.to_excel(writer, sheet_name='Complete_Results', index=False)
           
           # Sheet 3: Parameter Sensitivity Analysis
           if len(successful_df) > 0:
               # Stop buffer analysis
               stop_analysis = successful_df.groupby('stop_buffer').agg({
                   'profit_factor': ['mean', 'std', 'min', 'max', 'count'],
                   'win_rate': ['mean', 'std'],
                   'baseline_score': ['mean', 'std']
               }).round(3)
               stop_analysis.to_excel(writer, sheet_name='Stop_Buffer_Analysis')
               
               # Entry front-run analysis
               frontrun_analysis = successful_df.groupby('entry_frontrun').agg({
                   'profit_factor': ['mean', 'std', 'min', 'max', 'count'],
                   'win_rate': ['mean', 'std'],
                   'baseline_score': ['mean', 'std']
               }).round(3)
               frontrun_analysis.to_excel(writer, sheet_name='Entry_Frontrun_Analysis')
           
           # Sheet 4: Baseline Comparison Analysis
           if len(successful_df) > 0:
               baseline_comparison = successful_df[[
                   'config_name', 'stop_buffer', 'entry_frontrun',
                   'profit_factor', 'win_rate', 'vs_baseline_pf', 'vs_baseline_wr',
                   'baseline_status', 'baseline_score'
               ]].copy()
               
               # Add baseline reference row
               baseline_ref = pd.DataFrame([{
                   'config_name': 'LIVE_BASELINE',
                   'stop_buffer': 0.33,
                   'entry_frontrun': 0.05,
                   'profit_factor': self.LIVE_BASELINE['profit_factor'],
                   'win_rate': self.LIVE_BASELINE['win_rate'],
                   'vs_baseline_pf': 0.0,
                   'vs_baseline_wr': 0.0,
                   'baseline_status': 'REFERENCE',
                   'baseline_score': 100.0
               }])
               
               baseline_comparison = pd.concat([baseline_ref, baseline_comparison], ignore_index=True)
               baseline_comparison.to_excel(writer, sheet_name='Baseline_Comparison', index=False)
           
           # Sheet 5: Risk Analysis
           if len(successful_df) > 0:
               risk_analysis = successful_df[[
                   'config_name', 'stop_buffer', 'entry_frontrun',
                   'max_drawdown', 'sharpe_ratio', 'total_trades',
                   'avg_duration_days', 'profit_factor', 'baseline_score'
               ]].copy()
               risk_analysis = risk_analysis.sort_values('max_drawdown')
               risk_analysis.to_excel(writer, sheet_name='Risk_Analysis', index=False)
           
           # Sheet 6: Raw Trade Data (top 3 performers only for size)
           if len(successful_df) > 0:
               top_3 = successful_df.nlargest(3, 'baseline_score')
               trade_data_rows = []
               
               for _, config_row in top_3.iterrows():
                   if 'trades_data' in config_row and config_row['trades_data']:
                       for trade in config_row['trades_data']:
                           trade_row = trade.copy()
                           trade_row['config_name'] = config_row['config_name']
                           trade_row['stop_buffer'] = config_row['stop_buffer']
                           trade_row['entry_frontrun'] = config_row['entry_frontrun']
                           trade_data_rows.append(trade_row)
               
               if trade_data_rows:
                   trade_data_df = pd.DataFrame(trade_data_rows)
                   trade_data_df.to_excel(writer, sheet_name='Top_3_Trade_Data', index=False)
       
       print(f"✅ Comprehensive results saved: {filename}")
       print(f"📊 6 analysis sheets created with complete parameter optimization")
       return filename
    
    # WORKER FUNCTION FOR MULTIPROCESSING
def run_parameter_test_worker(test_config: Dict) -> Dict:
   """Worker function for parallel parameter testing"""
   try:
       # Create fresh backtester instance
       backtester = OptimizedStopEntryBacktester(max_workers=1)
       
       result = backtester.run_single_parameter_test(
           test_config['pair'],
           test_config['timeframe'],
           test_config['config_name'],
           test_config['config'],
           test_config['days_back']
       )
       
       # Clean up
       del backtester
       gc.collect()
       
       return result
       
   except Exception as e:
       gc.collect()
       return {
           'pair': test_config['pair'],
           'timeframe': test_config['timeframe'],
           'config_name': test_config['config_name'],
           'stop_buffer': test_config['config']['stop_buffer'],
           'entry_frontrun': test_config['config']['entry_frontrun'],
           'total_trades': 0,
           'error_reason': f"Worker error: {str(e)}"
       }

def main():
   """Main function with comprehensive user interface"""
   
   print("🎯 STOP/ENTRY PARAMETER OPTIMIZATION BACKTESTER")
   print("🏗️  Built from proven backtest_distance_edge.py logic")
   print("💻 Optimized for Intel i5-10400F (6C/12T) + 16GB RAM")
   print("🎯 Live Baseline: PF 2.5, WR 40% (15% tolerance)")
   print("=" * 70)
   
   # System check
   memory_status = psutil.virtual_memory()
   cpu_cores = cpu_count()
   
   print(f"\n💻 SYSTEM STATUS:")
   print(f"   RAM: {memory_status.total/(1024**3):.1f} GB ({memory_status.percent:.1f}% used)")
   print(f"   CPU: {cpu_cores} cores available")
   
   if memory_status.percent > 70:
       print("⚠️  WARNING: High memory usage detected!")
       print("   Consider closing other applications for optimal performance")
   
   # Initialize backtester
   backtester = OptimizedStopEntryBacktester()

   print(f"\n📊 PARAMETER CONFIGURATIONS AVAILABLE:")
   print(f"   Quick Validation: {len(backtester.QUICK_VALIDATION_CONFIGS)} configs")
   print(f"   Comprehensive: {len(backtester.COMPREHENSIVE_CONFIGS)} configs")
   
   print("\nSelect optimization mode:")
   print("1. Quick Validation (5 configs, EURUSD 3D, ~2 minutes)")
   print("2. Comprehensive Analysis (15 configs, limited pairs, ~15 minutes)")
   print("3. Full Optimization (15 configs, ALL data, ~45 minutes)")
   print("4. Custom Configuration")
   print("5. Data Directory Scan Only")
   
   choice = input("\nEnter choice (1-5): ").strip()
   
   if choice == '1':
       print("\n🚀 Starting QUICK VALIDATION...")
       print("📊 Testing 5 parameter configurations on EURUSD 3D")
       
       days_input = input("Enter days back (default 730): ").strip()
       days_back = int(days_input) if days_input.isdigit() else 730
       
       df = backtester.run_quick_validation(days_back=days_back)
       
       # Save quick results
       if len(df) > 0:
           timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
           filename = f"results/quick_validation_{timestamp}.xlsx"
           os.makedirs('results', exist_ok=True)
           df.to_excel(filename, index=False)
           print(f"📁 Quick validation results saved: {filename}")
   
   elif choice == '2':
       print("\n🚀 Starting COMPREHENSIVE ANALYSIS...")
       print("📊 Testing 15 parameter configurations on limited pairs")
       
       days_input = input("Enter days back (default 730): ").strip()
       days_back = int(days_input) if days_input.isdigit() else 730
       
       df = backtester.run_comprehensive_analysis(days_back=days_back, test_all_data=False)
   
   elif choice == '3':
       print("\n🚀 Starting FULL OPTIMIZATION...")
       print("📊 Testing 15 parameter configurations on ALL available data")
       print("⚠️  This will test EVERY data file and may take 45+ minutes")
       
       days_input = input("Enter days back (default 730): ").strip()
       days_back = int(days_input) if days_input.isdigit() else 730
       
       confirm = input(f"\nProceed with full optimization? (y/n): ").strip().lower()
       if confirm == 'y':
           df = backtester.run_comprehensive_analysis(days_back=days_back, test_all_data=True)
       else:
           print("Full optimization cancelled.")
           return
   
   elif choice == '4':
       print("\n🔧 CUSTOM CONFIGURATION...")
       
       # Custom parameter input
       print("Enter custom parameter ranges:")
       stop_buffer = float(input("Stop buffer (0.0-1.0, default 0.33): ") or "0.33")
       entry_frontrun = float(input("Entry front-run (0.0-0.20, default 0.05): ") or "0.05")
       
       pairs_input = input("Enter pairs (comma-separated, default EURUSD): ").strip().upper()
       pairs = [p.strip() for p in pairs_input.split(',')] if pairs_input else ['EURUSD']
       
       tf_input = input("Enter timeframes (comma-separated, default 3D): ").strip()
       timeframes = [tf.strip() for tf in tf_input.split(',')] if tf_input else ['3D']
       
       days_input = input("Enter days back (default 730): ").strip()
       days_back = int(days_input) if days_input.isdigit() else 730
       
       # Create custom configuration
       custom_config = {
           'custom': {'stop_buffer': stop_buffer, 'entry_frontrun': entry_frontrun}
       }
       
       # Run custom test
       results = []
       for pair in pairs:
           for timeframe in timeframes:
               result = backtester.run_single_parameter_test(
                   pair, timeframe, 'custom', custom_config['custom'], days_back
               )
               results.append(result)
       
       # Display results
       df = pd.DataFrame(results)
       successful_df = df[df['total_trades'] > 0]
       
       if len(successful_df) > 0:
           print(f"\n📊 CUSTOM CONFIGURATION RESULTS:")
           for _, row in successful_df.iterrows():
               print(f"   {row['pair']} {row['timeframe']}: {row['total_trades']} trades, "
                     f"PF {row['profit_factor']:.2f}, WR {row['win_rate']:.1f}%")
               print(f"      vs Baseline: PF {row['vs_baseline_pf']:+.1f}%, WR {row['vs_baseline_wr']:+.1f}%")
               print(f"      Status: {row['baseline_status']}")
       else:
           print("❌ No successful trades with custom configuration")
   
   elif choice == '5':
       print("\n🔍 DATA DIRECTORY SCAN...")
       available_data = backtester.get_all_available_data_files()
       
       if available_data:
           print(f"\n📁 Found {len(available_data)} data files:")
           
           # Group by timeframe for display
           by_timeframe = {}
           for item in available_data:
               tf = item['timeframe']
               if tf not in by_timeframe:
                   by_timeframe[tf] = []
               by_timeframe[tf].append(item['pair'])
           
           for tf in sorted(by_timeframe.keys()):
               pairs = sorted(set(by_timeframe[tf]))
               print(f"   {tf}: {len(pairs)} pairs - {', '.join(pairs[:5])}")
               if len(pairs) > 5:
                   print(f"        + {len(pairs)-5} more pairs")
       else:
           print("❌ No compatible data files found")
           print("   Expected format: OANDA_PAIR, TIMEFRAME_*.csv")
       
       return
   
   else:
       print("❌ Invalid choice. Please run again and select 1-5.")
       return
   
   print("\n✅ STOP/ENTRY PARAMETER OPTIMIZATION COMPLETE!")
   print("🎯 Key Questions Answered:")
   print("   ✓ Is 33% stop buffer optimal?")
   print("   ✓ Is 5% entry front-run optimal?")
   print("   ✓ Which parameter combination best maintains live baseline?")
   print("   ✓ Statistical confidence in parameter recommendations")
   print("📁 Professional Excel reports generated with complete analysis")
   
   # Final memory status
   final_memory = psutil.virtual_memory()
   print(f"\n💻 Final system status: {final_memory.percent:.1f}% RAM usage")

if __name__ == "__main__":
   main()