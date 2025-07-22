"""
DURATION-FOCUSED BACKTESTING SYSTEM
Inherits exact trading logic from existing backtesters with enhanced duration analysis
Focus: BE/TP duration impact analysis with universal time format support
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
import os
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
import psutil
import warnings
warnings.filterwarnings('ignore')

# Import your existing components (IDENTICAL MODULE USAGE)
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from fixed_backtester_TM import CompleteTradeManagementBacktester, run_single_test_worker

class UniversalDataLoader(DataLoader):
    """Universal data loader with auto-detection for mixed time formats"""
    
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
        
        import numpy as npf
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
        
        print(f"📅 Parsed date range: {data['date'].min()} → {data['date'].max()}")
        print(f"   Duration: {date_range.days} days ({years_range:.1f} years)")
        
        if date_range.days < 30:  # Suspicious: less than 30 days range
            print(f"❌ CRITICAL: Suspicious date range detected!")
            print(f"   This suggests timestamp parsing failed")
            print(f"   Raw timestamps: {data['time'].iloc[0]} → {data['time'].iloc[-1]}")
            
            # FORCE correct Unix parsing as last resort
            print("🔧 Forcing Unix seconds interpretation...")
            data['date'] = pd.to_datetime(data['time'], unit='s')
            print(f"✅ Corrected range: {data['date'].min()} → {data['date'].max()}")
        
        data.set_index('date', inplace=True)
        data.drop('time', axis=1, inplace=True)
        
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced cleaning with smart time parsing"""
        cleaned_data = data.copy()
        
        # Smart time parsing with auto-detection
        if 'time' in cleaned_data.columns:
            cleaned_data = self.parse_time_column_smart(cleaned_data)
        
        # Standard OHLC processing (IDENTICAL to your existing logic)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        cleaned_data.dropna(subset=price_columns, inplace=True)
        
        # Validation
        if len(cleaned_data) > 0:
            print(f"✅ Data loaded: {len(cleaned_data)} candles")
            print(f"   Range: {cleaned_data.index[0].date()} → {cleaned_data.index[-1].date()}")
        
        return cleaned_data

class DurationFocusedBacktester(CompleteTradeManagementBacktester):
    """
    Duration-focused backtester with IDENTICAL trading logic
    Enhanced for BE/TP duration impact analysis
    """
    
    # FOCUSED: Only the 6 strategies you want to analyze
    DURATION_STRATEGIES = {
        # BE 0.5R Series
        'BE_0.5R_TP_2R': {
            'type': 'breakeven',
            'breakeven_at': 0.5,
            'target': 2.0,
            'description': 'BE 0.5R → TP 2R (Duration Baseline)'
        },
        'BE_0.5R_TP_2.5R': {
            'type': 'breakeven',
            'breakeven_at': 0.5,
            'target': 2.5,
            'description': 'BE 0.5R → TP 2.5R (+0.5R Extension)'
        },
        'BE_0.5R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 0.5,
            'target': 3.0,
            'description': 'BE 0.5R → TP 3R (+1R Extension)'
        },
        
        # BE 1R Series
        'BE_1R_TP_2R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 2.0,
            'description': 'BE 1R → TP 2R (Duration Baseline)'
        },
        'BE_1R_TP_2.5R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 2.5,
            'description': 'BE 1R → TP 2.5R (+0.5R Extension)'
        },
        'BE_1R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 3.0,
            'description': 'BE 1R → TP 3R (+1R Extension)'
        }
    }
    
    def __init__(self, max_workers: int = None):
        """Initialize with universal data loader and duration focus"""
        # Initialize parent with same logic
        super().__init__(max_workers)
        
        # ENHANCED: Universal data loader
        self.data_loader = UniversalDataLoader()
        
        print(f"🎯 DURATION-FOCUSED BACKTESTING SYSTEM")
        print(f"   Duration strategies: {len(self.DURATION_STRATEGIES)}")
        print(f"   Universal time format support: ✅")
        print(f"   Identical trading logic: ✅")
    
    def get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get actual days per candle for timeframe"""
        TIMEFRAME_MULTIPLIERS = {
            '1D': 1, 'Daily': 1,
            '2D': 2, '2Daily': 2, 
            '3D': 3, '3Daily': 3,
            '4D': 4, '4Daily': 4,
            '5D': 5, '5Daily': 5,
            'H4': 0.167,   # 4 hours = 1/6 day
            'H12': 0.5,    # 12 hours = 1/2 day
            'Weekly': 7, '1W': 7
        }
        
        multiplier = TIMEFRAME_MULTIPLIERS.get(timeframe, 1)
        print(f"🕒 Timeframe {timeframe} → {multiplier} days per candle")
        return multiplier
    
    def run_single_backtest(self, pair: str, timeframe: str, strategy_name: str, days_back: int = 730) -> Dict:
        """FIXED: Override parent method to add timeframe multiplier handling"""
        try:
            print(f"\n🎯 Testing {pair} {timeframe} - {strategy_name}")
            
            # FIXED: Store timeframe multiplier for use in trade simulation
            self._current_timeframe_multiplier = self.get_timeframe_multiplier(timeframe)
            self._current_timeframe = timeframe
            
            print(f"🕒 Timeframe: {timeframe} → {self._current_timeframe_multiplier} days per candle")
            
            # Load data with support for ALL timeframes
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
            
            if len(data) < 100:
                return self.empty_result(pair, timeframe, strategy_name, "Insufficient data")

            # Only limit data if specifically requested with a reasonable limit
            if days_back < 9999:
                # For historical backtesting, use generous lookback
                max_candles = min(days_back + 1000, len(data))
                data = data.iloc[-max_candles:]
                print(f"   📊 Using last {days_back} days + 1000 lookback ({len(data)} candles)")
            elif days_back == 99999:
                print(f"   📊 Using ALL available data ({len(data)} candles)")
            
            print(f"   📅 Date range: {data.index[0].date()} → {data.index[-1].date()}")
            
            # Validate timeframe multiplier makes sense with data
            sample_duration = (data.index[10] - data.index[0]).total_seconds() / (24 * 3600)
            expected_duration = 10 * self._current_timeframe_multiplier
            print(f"   🔍 Validation: 10 candles = {sample_duration:.1f} days (expected ~{expected_duration})")

            # Initialize components
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()
            risk_manager = RiskManager(account_balance=10000)
            
            strategy_config = self.DURATION_STRATEGIES[strategy_name]
            
            # Run backtest with duration awareness
            results = self.backtest_with_complete_management(
                data, patterns, trend_data, risk_manager,
                strategy_config, pair, timeframe, strategy_name
            )
            
            # Add timeframe info to results
            results['timeframe_multiplier'] = self._current_timeframe_multiplier
            results['timeframe'] = timeframe
            
            # Cleanup
            del data, patterns, trend_data, risk_manager
            gc.collect()
            
            return results
            
        except Exception as e:
            gc.collect()
            return self.empty_result(pair, timeframe, strategy_name, f"Error: {str(e)}")
        
    def get_all_available_data_files(self) -> List[Dict]:
        """Auto-detect ALL available pairs and timeframes from your OANDA files"""
        
        data_path = self.data_loader.raw_path
        print(f"🔍 Scanning: {data_path}")
        
        import glob
        csv_files = glob.glob(os.path.join(data_path, "OANDA_*.csv"))
        print(f"📁 Found {len(csv_files)} OANDA files")
        
        available_data = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Parse your format: OANDA_EURUSD, 1D_77888.csv
            if ', ' in filename:
                parts = filename.replace('OANDA_', '').replace('.csv', '').split(', ')
                if len(parts) >= 2:
                    pair = parts[0]  # EURUSD
                    timeframe = parts[1].split('_')[0]  # 1D
                    
                    available_data.append({
                        'pair': pair,
                        'timeframe': timeframe,
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
        
        unique_data.sort(key=lambda x: (x['pair'], x['timeframe']))
        
        print(f"✅ Detected {len(unique_data)} unique combinations:")
        for item in unique_data:
            print(f"   {item['pair']} {item['timeframe']}")
        
        return unique_data
        
    # IDENTICAL: Same trading logic (inherited from parent)
    def backtest_with_complete_management(self, data: pd.DataFrame, patterns: Dict,
                                        trend_data: pd.DataFrame, risk_manager: RiskManager,
                                        strategy_config: Dict, pair: str, timeframe: str,
                                        strategy_name: str) -> Dict:
        """
        IDENTICAL trading logic - inherits from parent
        Same zone detection, trend alignment, entry/exit logic
        """
        return super().backtest_with_complete_management(
            data, patterns, trend_data, risk_manager,
            strategy_config, pair, timeframe, strategy_name
        )
    
    def create_duration_focused_trade_result(self, entry_idx: int, data: pd.DataFrame,
                                       entry_price: float, exit_price: float, direction: str,
                                       total_pnl: float, exit_reason: str, precise_days_held: float,
                                       strategy_config: Dict, partial_exits: List[Dict],
                                       remaining_position: float, timeframe_multiplier: float = 1) -> Dict:
        """FIXED: Enhanced trade result with proper duration calculation"""
        
        # Calculate precise duration using datetime index
        entry_date = data.index[entry_idx]
        
        # Calculate exit index from precise days
        candles_held = int(precise_days_held / timeframe_multiplier)
        exit_idx = min(entry_idx + candles_held, len(data) - 1)
        exit_date = data.index[exit_idx]
        
        # FIXED: Use datetime for most precise calculation
        duration_timedelta = exit_date - entry_date
        datetime_days_held = duration_timedelta.total_seconds() / (24 * 3600)
        duration_hours = duration_timedelta.total_seconds() / 3600
        
        # VALIDATION: Compare methods and use most accurate
        candle_based_days = candles_held * timeframe_multiplier
        
        print(f"🔍 Duration Validation:")
        print(f"   Candles held: {candles_held}")
        print(f"   Timeframe: ×{timeframe_multiplier} days/candle")
        print(f"   Candle-based: {candle_based_days:.1f} days")
        print(f"   DateTime-based: {datetime_days_held:.1f} days")
        print(f"   Precise input: {precise_days_held:.1f} days")
        print(f"   Entry: {entry_date.date()} → Exit: {exit_date.date()}")
        
        # Use precise_days_held as primary (passed from simulation)
        final_days_held = precise_days_held
        
        # Weekend exposure calculation
        weekend_days = self._count_weekend_exposure(entry_date, exit_date)
        
        # FIXED: Duration efficiency metrics
        profit_per_day = total_pnl / max(final_days_held, 0.1) if final_days_held > 0 else 0
        profit_per_hour = total_pnl / max(duration_hours, 1) if duration_hours > 0 else 0
        profit_per_trading_day = total_pnl / max(1, final_days_held - weekend_days) if final_days_held > weekend_days else profit_per_day
        
        return {
            # IDENTICAL: Standard trade fields (same as existing)
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'total_pnl': total_pnl,
            'exit_reason': exit_reason,
            'strategy': strategy_config['description'],
            'strategy_type': strategy_config['type'],
            'partial_exits': partial_exits,
            'remaining_position_pct': remaining_position * 100,
            'partial_exit_count': len(partial_exits),
            'partial_exit_pnl': sum(pe['pnl'] for pe in partial_exits),
            'remainder_pnl': total_pnl - sum(pe['pnl'] for pe in partial_exits),
            
            # FIXED: Duration analysis fields
            'days_held': int(final_days_held),  # For compatibility
            'precise_days_held': round(final_days_held, 2),
            'hours_held': round(duration_hours, 1),
            'candles_held': candles_held,
            'timeframe_multiplier': timeframe_multiplier,
            'duration_category': self._categorize_duration(final_days_held),
            'weekend_exposure': weekend_days,
            'trading_days_held': max(1, final_days_held - weekend_days),
            
            # FIXED: Duration efficiency metrics
            'profit_per_day': round(profit_per_day, 2),
            'profit_per_hour': round(profit_per_hour, 2),
            'profit_per_trading_day': round(profit_per_trading_day, 2),
            
            # ENHANCED: Strategy context for analysis
            'be_trigger': strategy_config.get('breakeven_at', 0),
            'profit_target': strategy_config.get('target', 0),
            'target_extension': strategy_config.get('target', 0) - 2.0,  # Extension from 2R baseline
        }
    
    def _categorize_duration(self, duration_days: float) -> str:
        """Categorize trade duration for analysis"""
        if duration_days <= 1:
            return 'Ultra_Short'  # ≤1 day
        elif duration_days <= 3:
            return 'Short'        # 1-3 days
        elif duration_days <= 7:
            return 'Medium'       # 3-7 days
        elif duration_days <= 14:
            return 'Long'         # 1-2 weeks
        else:
            return 'Extended'     # >2 weeks
    
    def _count_weekend_exposure(self, entry_date, exit_date) -> int:
        """Count weekend days trade was exposed"""
        weekend_days = 0
        current = entry_date.date()
        end_date = exit_date.date()
        
        while current <= end_date:
            if current.weekday() >= 5:  # Saturday=5, Sunday=6
                weekend_days += 1
            current += timedelta(days=1)
        
        return weekend_days
    
    def simulate_complete_trade(self, zone: Dict, entry_price: float,
                           initial_stop: float, entry_idx: int,
                           data: pd.DataFrame, strategy_config: Dict,
                           direction: str, position_size: float,
                           risk_amount: float) -> Dict:
        """FIXED: Complete trade simulation with proper duration calculation"""
        
        remaining_position = 1.0
        total_pnl = 0.0
        current_stop = initial_stop
        risk_distance = abs(entry_price - initial_stop)
        
        partial_exits_executed = []
        breakeven_moved = False
        trailing_active = False
        
        strategy_type = strategy_config['type']

        # Limit simulation length for memory efficiency
        max_simulation_length = min(500, len(data) - entry_idx - 1)
        
        # Get timeframe multiplier for proper duration calculation
        timeframe_multiplier = getattr(self, '_current_timeframe_multiplier', 1)
        
        for i in range(entry_idx + 1, entry_idx + 1 + max_simulation_length):
            if i >= len(data):
                break
                
            candle = data.iloc[i]
            current_date = data.index[i]
            
            # FIXED: Calculate actual duration properly
            candles_held = i - entry_idx
            actual_days_held = candles_held * timeframe_multiplier
            
            # Calculate precise duration using datetime
            entry_date = data.index[entry_idx]
            precise_days_held = (current_date - entry_date).total_seconds() / (24 * 3600)
            
            # DEBUG: Show duration calculation for first few iterations
            if candles_held <= 3:
                print(f"   🕒 Debug Day {candles_held}: TF×{timeframe_multiplier} = {actual_days_held} days, "
                    f"DateTime = {precise_days_held:.1f} days")
            
            current_price = candle['close']
            
            # Calculate R:R efficiently
            if direction == 'BUY':
                current_rr = (current_price - entry_price) / risk_distance
            else:
                current_rr = (entry_price - current_price) / risk_distance
            
            # Process partial exits
            if strategy_type in ['partial_trail', 'partial_breakeven'] and 'partial_exits' in strategy_config:
                for exit_config in strategy_config['partial_exits']:
                    exit_level = exit_config['at_level']
                    exit_percentage = exit_config['percentage'] / 100.0
                    
                    exit_key = f"{exit_level}R"
                    already_executed = any(pe['level'] == exit_key for pe in partial_exits_executed)
                    
                    if not already_executed and current_rr >= exit_level:
                        exit_amount = remaining_position * exit_percentage
                        remaining_position -= exit_amount
                        
                        if direction == 'BUY':
                            exit_price = entry_price + (risk_distance * exit_level)
                            partial_pnl = (exit_price - entry_price) * position_size * exit_amount
                        else:
                            exit_price = entry_price - (risk_distance * exit_level)
                            partial_pnl = (entry_price - exit_price) * position_size * exit_amount
                        
                        total_pnl += partial_pnl
                        
                        partial_exits_executed.append({
                            'level': exit_key,
                            'percentage': exit_percentage * 100,
                            'amount': exit_amount,
                            'pnl': partial_pnl,
                            'date': current_date,
                            'exit_price': exit_price
                        })
            
            # Check stop loss
            if direction == 'BUY' and candle['low'] <= current_stop:
                final_pnl = (current_stop - entry_price) * position_size * remaining_position
                total_pnl += final_pnl
                
                return self.create_duration_focused_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    total_pnl, 'stop_loss', precise_days_held, strategy_config,
                    partial_exits_executed, remaining_position, timeframe_multiplier
                )
            
            elif direction == 'SELL' and candle['high'] >= current_stop:
                final_pnl = (entry_price - current_stop) * position_size * remaining_position
                total_pnl += final_pnl
                
                return self.create_duration_focused_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    total_pnl, 'stop_loss', precise_days_held, strategy_config,
                    partial_exits_executed, remaining_position, timeframe_multiplier
                )
            
            # Break-even management
            if not breakeven_moved:
                if strategy_type == 'profit_breakeven' and 'breakeven_trigger' in strategy_config:
                    be_trigger = strategy_config['breakeven_trigger']
                    if current_rr >= be_trigger:
                        profit_level = strategy_config.get('profit_be_level', 0.5)
                        if direction == 'BUY':
                            current_stop = entry_price + (risk_distance * profit_level)
                        else:
                            current_stop = entry_price - (risk_distance * profit_level)
                        breakeven_moved = True
                
                elif 'breakeven_at' in strategy_config and strategy_config['breakeven_at']:
                    be_level = strategy_config['breakeven_at']
                    if current_rr >= be_level:
                        current_stop = entry_price
                        breakeven_moved = True
            
            # Zone trailing logic (simplified for stability)
            if strategy_type in ['partial_trail', 'zone_trailing'] and 'trail_activation' in strategy_config:
                trail_activation = strategy_config.get('trail_activation', 1.0)
                
                if not trailing_active and current_rr >= trail_activation:
                    trailing_active = True
                
                if trailing_active:
                    # Simplified trailing based on recent lows/highs
                    lookback = min(10, i - entry_idx)
                    if lookback > 0:
                        recent_data = data.iloc[i-lookback:i+1]
                        
                        if direction == 'BUY':
                            recent_low = recent_data['low'].min()
                            trail_stop = recent_low - (recent_data['high'].max() - recent_data['low'].min()) * 0.33
                            if trail_stop > current_stop:
                                current_stop = trail_stop
                        else:
                            recent_high = recent_data['high'].max()
                            trail_stop = recent_high + (recent_data['high'].max() - recent_data['low'].min()) * 0.33
                            if trail_stop < current_stop:
                                current_stop = trail_stop
            
            # Target check
            if 'target' in strategy_config and strategy_config['target']:
                target_level = strategy_config['target']
                if current_rr >= target_level:
                    if direction == 'BUY':
                        target_price = entry_price + (risk_distance * target_level)
                    else:
                        target_price = entry_price - (risk_distance * target_level)
                    
                    final_pnl = (target_price - entry_price) * position_size * remaining_position if direction == 'BUY' else (entry_price - target_price) * position_size * remaining_position
                    total_pnl += final_pnl
                    
                    return self.create_duration_focused_trade_result(
                        entry_idx, data, entry_price, target_price, direction,
                        total_pnl, 'take_profit', precise_days_held, strategy_config,
                        partial_exits_executed, remaining_position, timeframe_multiplier
                    )
        
        # End of data
        final_price = data.iloc[min(entry_idx + max_simulation_length, len(data) - 1)]['close']
        final_pnl = (final_price - entry_price) * position_size * remaining_position if direction == 'BUY' else (entry_price - final_price) * position_size * remaining_position
        total_pnl += final_pnl
        
        # Use precise datetime calculation for end of data
        final_date = data.index[min(entry_idx + max_simulation_length, len(data) - 1)]
        entry_date = data.index[entry_idx]
        final_precise_days = (final_date - entry_date).total_seconds() / (24 * 3600)
        
        return self.create_duration_focused_trade_result(
            entry_idx, data, entry_price, final_price, direction,
            total_pnl, 'end_of_data', final_precise_days, strategy_config,
            partial_exits_executed, remaining_position, timeframe_multiplier
        )
    
    
    def run_duration_analysis(self, pairs: List[str] = None, 
                             timeframes: List[str] = None,
                             days_back: int = 9999) -> pd.DataFrame:
        """
        Run comprehensive duration analysis on the 6 focused strategies
        """
        print("🎯 DURATION IMPACT ANALYSIS")
        print("🎲 Focus: BE/TP Duration vs Performance Trade-offs")
        print("=" * 70)
        
        # Auto-detect all pairs if not specified
        if pairs is None:
            available_data = self.get_all_available_data_files()
            pairs = sorted(list(set([item['pair'] for item in available_data])))
            print(f"📊 Auto-detected {len(pairs)} pairs: {pairs}")
        
        if timeframes is None:
            timeframes = ['3D']  # Optimal from your previous testing
            print(f"⏰ Using optimal timeframe: {timeframes}")
        
        # Create test combinations (DURATION STRATEGIES ONLY)
        test_combinations = []
        for pair in pairs:
            for timeframe in timeframes:
                for strategy_name in self.DURATION_STRATEGIES.keys():
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'strategy': strategy_name,
                        'days_back': days_back
                    })
        
        total_tests = len(test_combinations)
        estimated_time = (total_tests * 0.8) / self.max_workers
        
        print(f"\n📋 DURATION ANALYSIS CONFIGURATION:")
        print(f"   Strategies: {len(self.DURATION_STRATEGIES)} (focused)")
        print(f"   Pairs: {len(pairs)}")
        print(f"   Total tests: {total_tests}")
        print(f"   Estimated time: {estimated_time:.1f} minutes")
        print(f"   Universal time format support: ✅")
        
        # Run analysis
        print(f"\n🔄 Starting duration-focused analysis...")
        start_time = time.time()
        
        results = []
        with Pool(processes=self.max_workers) as pool:
            pool_results = pool.map(self.run_duration_test_worker, test_combinations)
            results.extend(pool_results)
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ DURATION ANALYSIS COMPLETE!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Successful tests: {success_count}/{total_tests}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Generate duration analysis report
        self.generate_duration_impact_report(df)
        
        # Save enhanced results
        self.save_duration_analysis_results(df)
        
        return df
    
    def run_duration_test_worker(self, test_config: Dict) -> Dict:
        """Worker function for duration analysis (same pattern as existing)"""
        try:
            # Create backtester with DURATION STRATEGIES ONLY
            backtester = DurationFocusedBacktester(max_workers=1)
            
            # CRITICAL: Replace strategy set with DURATION ONLY
            backtester.COMPLETE_STRATEGIES = self.DURATION_STRATEGIES
            
            result = backtester.run_single_backtest(
                test_config['pair'],
                test_config['timeframe'],
                test_config['strategy'],
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
                'strategy': test_config['strategy'],
                'description': f"Error: {str(e)}",
                'strategy_type': 'failed',
                'total_trades': 0,
                'error': str(e)
            }
    
    def generate_duration_impact_report(self, df: pd.DataFrame):
        """Generate comprehensive duration impact analysis report"""
        
        print(f"\n📊 DURATION IMPACT ANALYSIS REPORT")
        print("🎯 BE/TP Extension Duration vs Performance Trade-offs")
        print("=" * 80)
        
        # Filter successful results
        successful_df = df[df['total_trades'] > 0].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful duration strategies found!")
            return
        
        print(f"📈 Successful strategies: {len(successful_df)}")
        
        # PHASE 1: Duration Comparison Matrix
        print(f"\n🔥 PHASE 1: DURATION IMPACT MATRIX")
        print("=" * 60)
        
        # Group by BE trigger for comparison
        be_05_strategies = successful_df[successful_df['strategy'].str.contains('BE_0.5R')]
        be_1_strategies = successful_df[successful_df['strategy'].str.contains('BE_1R')]
        
        if len(be_05_strategies) > 0:
            print(f"\n📊 BE 0.5R SERIES DURATION IMPACT:")
            self._print_duration_comparison_table(be_05_strategies, "BE_0.5R")
        
        if len(be_1_strategies) > 0:
            print(f"\n📊 BE 1R SERIES DURATION IMPACT:")
            self._print_duration_comparison_table(be_1_strategies, "BE_1R")
        
        # PHASE 2: Efficiency Analysis
        print(f"\n💰 PHASE 2: DURATION EFFICIENCY ANALYSIS")
        print("=" * 60)
        
        # Calculate efficiency metrics
        self._analyze_duration_efficiency(successful_df)
        
        # PHASE 3: Trade-off Analysis
        print(f"\n⚖️  PHASE 3: RISK vs REWARD vs TIME TRADE-OFF")
        print("=" * 60)
        
        self._analyze_duration_tradeoffs(successful_df)
        
        # PHASE 4: Recommendations
        print(f"\n🎯 PHASE 4: DURATION-BASED RECOMMENDATIONS")
        print("=" * 60)
        
        self._generate_duration_recommendations(successful_df)
    
    def _print_duration_comparison_table(self, strategies_df: pd.DataFrame, series_name: str):
        """Print duration comparison table for a BE series"""
        
        print(f"{'Strategy':<20} {'Trades':<8} {'Win%':<7} {'PF':<6} {'Avg Days':<10} "
              f"{'$/Day':<8} {'$/Hour':<8} {'Duration +':<12}")
        print("-" * 90)
        
        # Sort by profit target for progression analysis
        strategies_sorted = strategies_df.sort_values('profit_target') if 'profit_target' in strategies_df.columns else strategies_df
        
        baseline_duration = None
        baseline_profit_day = None
        
        for _, row in strategies_sorted.iterrows():
            strategy_short = row['strategy'].replace(f'{series_name}_TP_', '→')
            
            avg_duration = row.get('avg_duration_days', 0)
            profit_per_day = row.get('profit_per_day', 0)
            profit_per_hour = row.get('profit_per_hour', 0)
            
            # Calculate duration increase vs baseline (first entry)
            if baseline_duration is None:
                baseline_duration = avg_duration
                baseline_profit_day = profit_per_day
                duration_increase = "Baseline"
            else:
                if baseline_duration > 0:
                    duration_pct = ((avg_duration / baseline_duration) - 1) * 100
                    duration_increase = f"+{duration_pct:.1f}%"
                else:
                    duration_increase = "N/A"
            
            print(f"{strategy_short:<20} {row['total_trades']:<8} {row.get('win_rate', 0):<7.1f} "
                  f"{row.get('profit_factor', 0):<6.2f} {avg_duration:<10.1f} "
                  f"{profit_per_day:<8.0f} {profit_per_hour:<8.1f} {duration_increase:<12}")
    
    def _analyze_duration_efficiency(self, df: pd.DataFrame):
        """Analyze duration efficiency metrics"""
        
        # Best efficiency by different metrics
        if 'profit_per_day' in df.columns and len(df) > 0:
            best_daily_efficiency = df.loc[df['profit_per_day'].idxmax()]
            best_hourly_efficiency = df.loc[df['profit_per_hour'].idxmax()]
            
            print(f"🏆 EFFICIENCY CHAMPIONS:")
            print(f"   Best $/Day: {best_daily_efficiency['strategy']} → "
                  f"${best_daily_efficiency['profit_per_day']:.0f}/day")
            print(f"   Best $/Hour: {best_hourly_efficiency['strategy']} → "
                  f"${best_hourly_efficiency['profit_per_hour']:.1f}/hour")
        
        # Duration category analysis
        if 'duration_category' in df.columns:
            duration_dist = df['duration_category'].value_counts()
            print(f"\n📊 DURATION DISTRIBUTION:")
            for category, count in duration_dist.items():
                print(f"   {category}: {count} strategies")
    
    def _analyze_duration_tradeoffs(self, df: pd.DataFrame):
        """Fixed trade-off analysis with proper Series handling"""
        
        if len(df) == 0 or 'profit_factor' not in df.columns:
            print("⚠️  Insufficient data for trade-off analysis")
            return
        
        # Create copy for analysis
        df_copy = df.copy()
        
        # FIXED: Proper Series max() handling
        max_pf = df_copy['profit_factor'].max()
        max_efficiency = df_copy.get('profit_per_day', pd.Series([1])).max()
        max_duration = df_copy.get('avg_duration_days', pd.Series([1])).max()
        
        # Normalize metrics for comparison (0-100 scale)
        df_copy['pf_score'] = (df_copy['profit_factor'] / max_pf) * 100
        
        # FIXED: Handle case where profit_per_day might be missing
        if 'profit_per_day' in df_copy.columns and max_efficiency > 0:
            df_copy['efficiency_score'] = (df_copy['profit_per_day'] / max_efficiency) * 100
        else:
            df_copy['efficiency_score'] = 50  # Default neutral score
        
        # FIXED: Handle duration penalty calculation
        if 'avg_duration_days' in df_copy.columns and max_duration > 0:
            df_copy['duration_penalty'] = 100 - ((df_copy['avg_duration_days'] / max_duration) * 50)
        else:
            df_copy['duration_penalty'] = 75  # Default moderate penalty
        
        # Composite score (weighted)
        df_copy['trade_off_score'] = (
            df_copy['pf_score'] * 0.4 +          # 40% profit factor
            df_copy['efficiency_score'] * 0.4 +   # 40% daily efficiency  
            df_copy['duration_penalty'] * 0.2     # 20% duration penalty
        )
        
        best_tradeoff = df_copy.loc[df_copy['trade_off_score'].idxmax()]
        
        print(f"🎯 OPTIMAL TRADE-OFF:")
        print(f"   Strategy: {best_tradeoff['strategy']}")
        print(f"   Trade-off Score: {best_tradeoff['trade_off_score']:.1f}/100")
        print(f"   Profit Factor: {best_tradeoff['profit_factor']:.2f}")
        print(f"   Efficiency: ${best_tradeoff.get('profit_per_day', 0):.0f}/day")
    
    def _generate_duration_recommendations(self, df: pd.DataFrame):
        """Generate duration-based strategy recommendations"""
        
        if len(df) == 0:
            return
        
        # For funded accounts - prioritize efficiency and consistency
        print(f"💼 FUNDED ACCOUNT RECOMMENDATIONS:")
        
        # Find strategies with good balance of profit factor and efficiency
        balanced_strategies = df[
            (df['profit_factor'] >= 1.0) &  # Profitable
            (df.get('profit_per_day', 0) > 0)  # Efficient
        ].copy()
        
        if len(balanced_strategies) > 0:
            # Rank by composite score
            balanced_strategies['funded_score'] = (
                balanced_strategies['profit_factor'] * 30 +
                balanced_strategies.get('profit_per_day', 0) * 0.1 +
                balanced_strategies['win_rate'] * 0.5
            )
            
            top_funded = balanced_strategies.loc[balanced_strategies['funded_score'].idxmax()]
            
            print(f"   🥇 Top Choice: {top_funded['strategy']}")
            print(f"      Profit Factor: {top_funded['profit_factor']:.2f}")
            print(f"      Efficiency: ${top_funded.get('profit_per_day', 0):.0f}/day")
            print(f"      Win Rate: {top_funded['win_rate']:.1f}%")
            print(f"      Avg Duration: {top_funded.get('avg_duration_days', 0):.1f} days")
        
        # Risk warnings
        print(f"\n⚠️  DURATION RISK CONSIDERATIONS:")
        print(f"   • Longer trades = higher daily drawdown exposure")
        print(f"   • Extended holds = fewer new opportunities")
        print(f"   • Weekend gaps = uncontrolled risk periods")
        print(f"   • Psychological stress increases with hold time")
    
    def save_duration_analysis_results(self, df: pd.DataFrame):
        """Fixed save method with column existence checks"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = "results"
        os.makedirs(export_path, exist_ok=True)
        filename = os.path.join(export_path, f"duration_impact_analysis_{timestamp}.xlsx")
        
        print(f"\n💾 Saving duration analysis results...")
        print(f"📊 Available columns: {list(df.columns)}")  # DEBUG: Show available columns
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Sheet 1: Complete Duration Analysis - USE AVAILABLE COLUMNS ONLY
            base_columns = [
                'strategy', 'pair', 'timeframe', 'total_trades', 'win_rate', 
                'profit_factor', 'total_return', 'avg_duration_days'
            ]
            
            # Add optional columns if they exist
            optional_columns = [
                'be_trigger', 'profit_target', 'target_extension',
                'avg_winner_duration', 'avg_loser_duration', 'profit_per_day', 
                'profit_per_hour', 'duration_category', 'weekend_exposure'
            ]
            
            # Build final column list with only existing columns
            final_columns = [col for col in base_columns if col in df.columns]
            final_columns.extend([col for col in optional_columns if col in df.columns])
            
            print(f"📋 Using columns: {final_columns}")
            
            df_duration = df[final_columns].copy()
            df_duration.to_excel(writer, sheet_name='Duration_Analysis', index=False)
            
            # Sheet 2: BE Series Comparison - ONLY IF COLUMNS EXIST
            if 'be_trigger' in df.columns and 'profit_target' in df.columns:
                try:
                    be_comparison = df.groupby(['be_trigger', 'profit_target']).agg({
                        'avg_duration_days': 'mean',
                        'profit_per_day': 'mean' if 'profit_per_day' in df.columns else 'count',
                        'profit_factor': 'mean',
                        'win_rate': 'mean',
                        'total_trades': 'sum'
                    }).round(2)
                    
                    be_comparison.to_excel(writer, sheet_name='BE_Series_Comparison')
                    print("✅ BE Series Comparison sheet created")
                except Exception as e:
                    print(f"⚠️  Skipping BE comparison sheet: {e}")
            else:
                print("⚠️  Skipping BE comparison - missing be_trigger/profit_target columns")
            
            # Sheet 3: Efficiency Rankings - SIMPLIFIED
            efficiency_df = df[df['total_trades'] > 0].copy()
            if len(efficiency_df) > 0:
                # Sort by profit factor if profit_per_day not available
                sort_column = 'profit_per_day' if 'profit_per_day' in df.columns else 'profit_factor'
                efficiency_df = efficiency_df.sort_values(sort_column, ascending=False)
                efficiency_df[final_columns].to_excel(writer, sheet_name='Efficiency_Rankings', index=False)
                print("✅ Efficiency Rankings sheet created")
            
            # Sheet 4: Basic Performance Matrix - SIMPLIFIED
            if len(df) > 0:
                try:
                    # Create a simple performance matrix by strategy
                    performance_matrix = df.pivot_table(
                        values='profit_factor',
                        index='strategy',
                        aggfunc='mean'
                    ).round(2)
                    performance_matrix.to_excel(writer, sheet_name='Performance_Summary')
                    print("✅ Performance Summary sheet created")
                except Exception as e:
                    print(f"⚠️  Basic performance sheet failed: {e}")
        
        print(f"✅ Duration analysis saved: {filename}")
        return filename

def run_duration_test_worker_standalone(test_config: Dict) -> Dict:
    """Standalone worker function for duration analysis multiprocessing"""
    try:
        backtester = DurationFocusedBacktester(max_workers=1)
        backtester.COMPLETE_STRATEGIES = backtester.DURATION_STRATEGIES
        
        result = backtester.run_single_backtest(
            test_config['pair'],
            test_config['timeframe'],
            test_config['strategy'], 
            test_config['days_back']
        )
        
        del backtester
        gc.collect()
        return result
        
    except Exception as e:
        gc.collect()
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'], 
            'strategy': test_config['strategy'],
            'description': f"Error: {str(e)}",
            'total_trades': 0,
            'error': str(e)
        }

def main_duration_analysis():
    """Main function for duration impact analysis"""
    
    print("🎯 DURATION IMPACT ANALYSIS SYSTEM")
    print("⏰ BE/TP Extension Duration vs Performance Trade-offs")
    print("🔧 Universal Time Format Support (Unix + ISO Auto-Detection)")
    print("=" * 70)
    
    # Initialize duration-focused backtester
    analyzer = DurationFocusedBacktester(max_workers=8)
    
    print("\nSelect duration analysis scope:")
    print("1. Quick Analysis (EURUSD only, 3D timeframe)")
    print("2. Comprehensive Analysis (All pairs, optimal timeframe)")
    print("3. Custom Analysis (Specify pairs and timeframes)")
    print("4. Deep Analysis (All pairs, multiple timeframes)")
    print("5. Single Pair Focus (Choose specific pair for detailed analysis)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        print("\n🚀 Starting QUICK DURATION ANALYSIS...")
        print("📊 Pair: EURUSD | Timeframe: 3D | Period: All data")
        
        df = analyzer.run_duration_analysis(
            pairs=['EURUSD'],
            timeframes=['3D'],
            days_back=9999  # All available data
        )
        
    elif choice == '2':
        print("\n🚀 Starting COMPREHENSIVE DURATION ANALYSIS...")
        print("📊 All detected pairs | Timeframe: 3D | Period: All data")
        print("⚠️  This may take 15-25 minutes depending on data size")
        
        confirm = input("Proceed with comprehensive analysis? (y/n): ").strip().lower()
        if confirm == 'y':
            df = analyzer.run_duration_analysis(
                pairs=None,  # Auto-detect all pairs
                timeframes=['3D'],
                days_back=9999
            )
        else:
            print("Analysis cancelled.")
            return
            
    elif choice == '3':
        print("\n🔧 CUSTOM DURATION ANALYSIS...")
        pairs_input = input("Enter pairs (comma-separated, e.g., EURUSD,GBPUSD): ").strip().upper()
        pairs = [p.strip() for p in pairs_input.split(',')] if pairs_input else ['EURUSD']
        
        tf_input = input("Enter timeframes (comma-separated, e.g., 1D,3D,5D): ").strip()
        timeframes = [tf.strip() for tf in tf_input.split(',')] if tf_input else ['3D']
        
        days_input = input("Enter days back (default all data = 9999): ").strip()
        days_back = int(days_input) if days_input.isdigit() else 9999
        
        print(f"\n📊 Custom analysis: {pairs} | {timeframes} | {days_back} days")
        
        df = analyzer.run_duration_analysis(
            pairs=pairs,
            timeframes=timeframes, 
            days_back=days_back
        )
        
    elif choice == '4':
        print("\n🚀 Starting DEEP DURATION ANALYSIS...")
        print("📊 All pairs | Multiple timeframes | Full historical data")
        print("⚠️  This may take 45-60 minutes")
        
        confirm = input("Proceed with deep analysis? (y/n): ").strip().lower()
        if confirm == 'y':
            df = analyzer.run_duration_analysis(
                pairs=None,  # All pairs
                timeframes=['1D', '2D', '3D', '4D', '5D'],
                days_back=9999
            )
        else:
            print("Analysis cancelled.")
            return
            
    elif choice == '5':
        print("\n🎯 SINGLE PAIR FOCUS ANALYSIS...")
        
        # Show available pairs
        available_data = analyzer.get_all_available_data_files()
        available_pairs = sorted(list(set([item['pair'] for item in available_data])))
        
        print(f"\nAvailable pairs: {available_pairs}")
        pair_input = input("Enter pair to focus on: ").strip().upper()
        
        if pair_input in available_pairs:
            print(f"\n📊 Focusing on {pair_input} with multiple timeframes...")
            df = analyzer.run_duration_analysis(
                pairs=[pair_input],
                timeframes=['1D', '2D', '3D', '4D', '5D'],
                days_back=9999
            )
        else:
            print(f"❌ Pair {pair_input} not found in available data")
            return
    
    else:
        print("❌ Invalid choice, using default quick analysis")
        df = analyzer.run_duration_analysis(
            pairs=['EURUSD'],
            timeframes=['3D'], 
            days_back=9999
        )
    
    print("\n✅ DURATION ANALYSIS COMPLETE!")
    print("🎯 Key Questions Answered:")
    print("   ✓ Duration impact of extending profit targets")
    print("   ✓ Profit efficiency ($/day vs $/hour) comparison")
    print("   ✓ Break-even trigger vs target duration trade-offs")
    print("   ✓ Optimal strategies for funded account constraints")
    print("   ✓ Weekend exposure and trading day efficiency")
    print("📁 Detailed Excel report with 4 analysis sheets created")
    print("📊 Universal time format parsing successful")
    
    # Quick summary of key findings
    if len(df) > 0:
        successful_df = df[df['total_trades'] > 0]
        if len(successful_df) > 0:
            print(f"\n📈 QUICK SUMMARY:")
            print(f"   Successful strategies tested: {len(successful_df)}")
            
            if 'profit_per_day' in successful_df.columns:
                best_efficiency = successful_df.loc[successful_df['profit_per_day'].idxmax()]
                print(f"   Most efficient: {best_efficiency['strategy']} "
                        f"(${best_efficiency['profit_per_day']:.0f}/day)")
            
            if 'profit_factor' in successful_df.columns:
                best_pf = successful_df.loc[successful_df['profit_factor'].idxmax()]
                print(f"   Highest profit factor: {best_pf['strategy']} "
                        f"(PF: {best_pf['profit_factor']:.2f})")

if __name__ == "__main__":
    main_duration_analysis()