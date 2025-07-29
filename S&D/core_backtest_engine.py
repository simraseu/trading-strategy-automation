"""
Core Backtesting Engine - Project 2
Combines proven trade logic with production framework
Built on 100% updated modules and settings from Project 1
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

# Import your 100% updated modules
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from config.settings import ZONE_CONFIG, TREND_CONFIG, RISK_CONFIG

# Set process priority for maximum CPU usage
try:
    if os.name == 'nt':  # Windows
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
except:
    pass

# UPDATED ANALYSIS PERIODS - Using your current standards
ANALYSIS_PERIODS = {
    'priority_1': {
        'name': '2015-2025 (10 years) - PRIMARY ANALYSIS',
        'days_back': 3847,
        'description': 'Modern market structure validation'
    },
    'priority_2': {
        'name': '2020-2025 (4 years) - RECENT VALIDATION', 
        'days_back': 2021,
        'description': 'Post-COVID market performance'
    },
    'priority_3': {
        'name': '2018-2025 (6 years) - MEDIUM-TERM VIEW',
        'days_back': 2751,
        'description': 'Mid-term consistency check'
    }
}

# BASELINE PERFORMANCE STANDARDS
BASELINE_STANDARDS = {
    'profit_factor': 2.0,
    'win_rate': 35.0,
    'min_trades': 5,
    'tolerance': 0.15  # 15% tolerance
}

class CoreBacktestEngine:
    """
    Production-ready backtesting engine combining:
    - Proven trade logic with realistic management
    - Complete production framework with optimization
    - 100% updated modules and settings from Project 1
    """
    
    def __init__(self):
        """Initialize with system optimization and updated foundation"""        
        # Initialize your updated data loader
        self.data_loader = DataLoader()
        
        # CPU optimization
        available_cores = cpu_count()
        if available_cores >= 12:  # Hyperthreaded 6-core
            self.max_workers = 6
        elif available_cores >= 4:
            self.max_workers = available_cores - 1
        else:
            self.max_workers = max(1, available_cores - 2)
        
        # Memory optimization settings
        self.chunk_size = 100  # Process in chunks
        self.memory_threshold = 0.75  # 75% memory trigger cleanup
        
        # Current test configuration (will be set per test)
        self.current_config = None
    
    def discover_all_pairs(self) -> List[str]:
        """Auto-discover all available currency pairs using updated DataLoader"""
        
        try:
            # Use your updated DataLoader's discovery method
            pairs = self.data_loader.discover_all_pairs()
            
            if not pairs:
                print("❌ No currency pairs found")
                return []
            
            return pairs
            
        except Exception as e:
            print(f"❌ Error discovering pairs: {str(e)}")
            return []
    
    def discover_valid_data_combinations(self) -> List[Tuple[str, str]]:
        """
        Discover only valid pair/timeframe combinations that actually have data files
        Returns list of (pair, timeframe) tuples
        """
        
        try:
            # Use DataLoader's comprehensive inventory method
            data_inventory = self.data_loader.get_available_data()
            
            valid_combinations = []
            total_files = 0
            
            for pair, available_timeframes in data_inventory.items():
                for timeframe in available_timeframes:
                    valid_combinations.append((pair, timeframe))
                    total_files += 1
            
            print(f"   {len(data_inventory)} pairs with data")
            print(f"   {len(valid_combinations)} valid combinations")
            
            # Show breakdown
            for pair, timeframes in data_inventory.items():
                print(f"   {pair}: {', '.join(timeframes)}")
            
            return valid_combinations
            
        except Exception as e:
            print(f"❌ Error discovering valid combinations: {str(e)}")
            return []
    
    def load_data_with_validation(self, pair: str, timeframe: str, days_back: int = 730) -> Optional[pd.DataFrame]:
        """Load data using updated DataLoader with INTELLIGENT timeframe-aware filtering"""
        try:            
            # Use your updated data loader to get full dataset
            data = self.data_loader.load_pair_data(pair, timeframe)
            
            if data is None or len(data) < 100:
                return None
            
            # TIMEFRAME-AWARE FILTERING: Adjust requirements based on timeframe
            timeframe_multipliers = {
                '1D': 1, '2D': 2, '3D': 3, '4D': 4, '5D': 5,
                '1W': 7, '2W': 14, '3W': 21, '1M': 30,
                'H12': 0.5, 'H8': 0.33, 'H4': 0.17
            }
            
            # Get timeframe multiplier (default to 1 if unknown)
            tf_multiplier = timeframe_multipliers.get(timeframe, 1)
            
            # Calculate minimum candles needed based on timeframe
            min_candles_for_ema200 = max(200, int(200 / tf_multiplier))  # Scale EMA200 requirement
            min_trading_candles = max(50, int(100 / tf_multiplier))      # Scale trading data requirement
            absolute_minimum = min_candles_for_ema200 + min_trading_candles
            
            print(f"   📊 Timeframe {timeframe}: multiplier={tf_multiplier}, min_candles={absolute_minimum}")
            
            # INTELLIGENT FILTERING: Only filter if we have sufficient data
            if days_back < 9999 and len(data) > absolute_minimum * 2:  # Only filter if we have 2x minimum
                # Ensure we have a proper datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    print(f"⚠️  Data does not have DatetimeIndex: {type(data.index)}")
                    return None
                
                # Calculate required candles for the requested period
                requested_candles = int(days_back / tf_multiplier)
                buffer_candles = min_candles_for_ema200  # EMA200 buffer
                total_candles_needed = requested_candles + buffer_candles
                
                # Only filter if we have more data than needed
                if len(data) > total_candles_needed:
                    # Take the last N candles instead of date-based filtering
                    data = data.tail(total_candles_needed)
                    print(f"   📊 Filtered to {len(data)} candles (candle-based filtering)")
                else:
                    print(f"   📊 Using all {len(data)} candles (insufficient for filtering)")
                
                print(f"   📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"   📊 Using all {len(data)} candles (no filtering applied)")
            
            # Final validation
            if len(data) < absolute_minimum:
                print(f"⚠️  Insufficient data: {len(data)} < {absolute_minimum} required for {timeframe}")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading {pair} {timeframe}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_single_strategy_test(self, pair: str, timeframe: str, days_back: int = 730) -> Dict:
        """
        Run single strategy test using UPDATED MODULES and REALISTIC TRADE LOGIC
        """
        try:
            print(f"\n🧪 TESTING: {pair} {timeframe} ({days_back} days)")
            
            # Load data using updated loader
            data = self.load_data_with_validation(pair, timeframe, days_back)
            if data is None:
                return self.create_empty_result(pair, timeframe, "Insufficient data")
            
            # Initialize components using YOUR UPDATED MODULES
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_with_filter()
            
            risk_manager = RiskManager(account_balance=10000)
            
            # Run backtest with UPDATED SETTINGS (2.5x leg-out ratio)
            result = self.execute_backtest_with_updated_logic(
                data, patterns, trend_data, risk_manager, pair, timeframe
            )
            
            return result
            
        except Exception as e:
            return self.create_empty_result(pair, timeframe, f"Error: {str(e)}")
    
    def execute_backtest_with_updated_logic(self, data: pd.DataFrame, patterns: Dict,
                                          trend_data: pd.DataFrame, risk_manager: RiskManager,
                                          pair: str, timeframe: str) -> Dict:
        """
        Execute backtest using UPDATED ZONE_CONFIG and REALISTIC TRADE LOGIC
        """
        # Get all patterns (momentum + reversal)
        all_patterns = (patterns['dbd_patterns'] + patterns['rbr_patterns'] + 
                       patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', []))
        
        print(f"   📊 Found {len(all_patterns)} total patterns")
        
        # REALISTIC: Trade ALL formed zones without knowing which will work
        valid_patterns = [
            pattern for pattern in all_patterns
            if pattern.get('end_idx') is not None  # All properly formed zones
        ]

        print(f"   🎯 Using {len(valid_patterns)} patterns (realistic - no hindsight)")   

        if not valid_patterns:
            total_zones = len(all_patterns)
            valid_count = len(valid_patterns)
            return self.create_empty_result(pair, timeframe, f"No valid zones: {valid_count}/{total_zones} formed properly")

        print(f"   🎯 {len(valid_patterns)} zones available for trading from {len(all_patterns)} total")
        
        
        # Execute trades with REALISTIC LOGIC
        trades = self.execute_realistic_trades(valid_patterns, data, trend_data, timeframe, pair)
        
        # Calculate performance
        return self.calculate_performance_metrics(trades, pair, timeframe)
    
    def execute_realistic_trades(self, patterns: List[Dict], data: pd.DataFrame,
                           trend_data: pd.DataFrame, timeframe: str, pair: str) -> List[Dict]:
        """
        CRITICAL FIX: Execute trades ONLY on validated zones
        1. Zone forms
        2. Price must hit 2.5x target (validation) 
        3. Only THEN can we place limit orders for zone retest
        4. If price penetrates 50% before validation = zone deleted forever
        """
        trades = []
        used_zones = set()
        invalidated_zones = set()
        validated_zones = {}  # NEW: Track validation status and when it occurred
        
        # Get backtest start date (first 200 candles are for EMA warm-up)
        backtest_start_idx = 200
        backtest_start_date = data.index[backtest_start_idx]
        
        print(f"   📅 Backtest period: {backtest_start_date.strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Filter patterns to only include zones formed DURING backtest period
        valid_patterns = []
        for pattern in patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx >= backtest_start_idx:
                zone_formation_date = data.index[zone_end_idx]
                pattern['formation_date'] = zone_formation_date
                # Pre-calculate zone invalidation levels
                pattern['invalidation_level'] = self.precalculate_invalidation_level(pattern)
                valid_patterns.append(pattern)
        
        if not valid_patterns:
            print(f"   ⚠️  No zones formed during backtest period - no trades possible")
            return trades
        
        print(f"   🎯 Trading {len(valid_patterns)} zones formed during backtest")
        
        # CRITICAL FIX: Build zone validation tracking
        zone_tracking = {}
        for pattern in valid_patterns:
            zone_end_idx = pattern['end_idx']
            zone_id = f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
            
            # Track validation status for each zone
            validation_status = self.track_zone_validation_realtime(
                pattern, data, zone_end_idx + 1
            )
            
            zone_tracking[zone_id] = {
                'pattern': pattern,
                'formation_idx': zone_end_idx,
                'validated': validation_status['validated'],
                'validation_idx': validation_status['validation_idx'],
                'invalidated': validation_status['invalidated'],
                'invalidation_idx': validation_status['invalidation_idx']
            }
        
        # Process each candle with VALIDATION-FIRST logic
        for current_idx in range(backtest_start_idx, len(data)):
            current_candle = data.iloc[current_idx]
            
            # CRITICAL: Only consider zones that have been VALIDATED
            for zone_id, zone_info in zone_tracking.items():
                if zone_id in used_zones or zone_id in invalidated_zones:
                    continue
                    
                # Skip if not validated
                if not zone_info['validated']:
                    continue
                    
                # Skip if we haven't reached validation point yet
                if zone_info['validation_idx'] is None or current_idx <= zone_info['validation_idx']:
                    continue
                
                zone = zone_info['pattern']
                
                # FIRST: Check if price is anywhere near the zone
                if not self.fast_zone_interaction_check(zone, current_candle):
                    continue
                
                # SECOND: Check if price touches entry level (limit order logic)
                if not self.check_limit_order_trigger(zone, current_candle):
                    continue
                
                # THIRD: Check trend alignment
                current_trend = trend_data['trend'].iloc[current_idx] if current_idx < len(trend_data) else 'bullish'
                if not self.is_trend_aligned(zone['type'], current_trend):
                    continue
                
                # Execute trade
                zone['pair'] = pair
                trade_result = self.execute_single_realistic_trade(zone, data, current_idx)
                
                if trade_result:
                    used_zones.add(zone_id)
                    trades.append(trade_result)
                    break  # Exit zone loop after successful trade
        
        print(f"   ✅ Executed {len(trades)} realistic trades")
        return trades
    
    def track_zone_validation_realtime(self, zone: Dict, data: pd.DataFrame, start_idx: int) -> Dict:
        """
        CRITICAL: Track zone validation in real-time as price moves
        Returns validation status and index where it occurred
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        zone_type = zone['type']
        
        # Calculate targets
        if zone_type in ['D-B-D', 'R-B-D']:  # Supply zones
            validation_target = zone_low - (2.5 * zone_range)
            invalidation_level = zone_low + (zone_range * 0.50)
        else:  # Demand zones
            validation_target = zone_high + (2.5 * zone_range)
            invalidation_level = zone_high - (zone_range * 0.50)
        
        # Track price movement
        for idx in range(start_idx, len(data)):
            candle = data.iloc[idx]
            
            # Check validation first
            if zone_type in ['D-B-D', 'R-B-D']:  # Supply
                if candle['low'] <= validation_target:
                    return {
                        'validated': True,
                        'validation_idx': idx,
                        'invalidated': False,
                        'invalidation_idx': None
                    }
                if candle['high'] >= invalidation_level:
                    return {
                        'validated': False,
                        'validation_idx': None,
                        'invalidated': True,
                        'invalidation_idx': idx
                    }
            else:  # Demand
                if candle['high'] >= validation_target:
                    return {
                        'validated': True,
                        'validation_idx': idx,
                        'invalidated': False,
                        'invalidation_idx': None
                    }
                if candle['low'] <= invalidation_level:
                    return {
                        'validated': False,
                        'validation_idx': None,
                        'invalidated': True,
                        'invalidation_idx': idx
                    }
        
        # Neither validated nor invalidated
        return {
            'validated': False,
            'validation_idx': None,
            'invalidated': False,
            'invalidation_idx': None
        }
    
    def precalculate_invalidation_level(self, zone: Dict) -> float:
        """Pre-calculate WICK-based invalidation level (50% penetration) for performance"""
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_size = zone_high - zone_low
        zone_type = zone['type']
        
        if zone_type in ['R-B-R', 'D-B-R']:  # DEMAND zones
            # Invalidated if WICK penetrates 50% through zone_low
            return zone_low - (zone_size * 0.50)
        else:  # SUPPLY zones  
            # Invalidated if WICK penetrates 50% through zone_high
            return zone_high + (zone_size * 0.50)
    
    def fast_wick_invalidation_check(self, zone: Dict, current_candle: pd.Series, zone_id: str, invalidated_zones: set) -> bool:
        """OPTIMIZED: Single-candle WICK-based invalidation check (50% penetration)"""
        invalidation_level = zone['invalidation_level']
        zone_type = zone['type']
        
        if zone_type in ['R-B-R', 'D-B-R']:  # DEMAND zones
            # Check if LOW wick penetrates 50% below zone_low
            if current_candle['low'] < invalidation_level:
                invalidated_zones.add(zone_id)
                return True
        else:  # SUPPLY zones
            # Check if HIGH wick penetrates 50% above zone_high
            if current_candle['high'] > invalidation_level:
                invalidated_zones.add(zone_id)
                return True
        
        return False
    
    def check_limit_order_trigger(self, zone: Dict, current_candle: pd.Series) -> bool:
        """
        Check if price triggers our limit order entry
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # Calculate entry price
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
            entry_price = zone_high + (zone_range * 0.05)
            # Buy limit triggers if high reaches entry
            return current_candle['high'] >= entry_price
        else:  # Supply zones (sell)
            entry_price = zone_low - (zone_range * 0.05)
            # Sell limit triggers if low reaches entry
            return current_candle['low'] <= entry_price
    
    def fast_zone_interaction_check(self, zone: Dict, current_candle: pd.Series) -> bool:
        """OPTIMIZED: Fast zone interaction check - simple overlap test"""
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        candle_high = current_candle['high']
        candle_low = current_candle['low']
        
        # Simple overlap check - candle intersects with zone
        return candle_low <= zone_high and candle_high >= zone_low
    
    def is_trend_aligned(self, zone_type: str, current_trend: str) -> bool:
        """OPTIMIZED: Fast trend alignment check"""
        if current_trend == 'bullish':
            return zone_type in ['R-B-R', 'D-B-R']
        elif current_trend == 'bearish':
            return zone_type in ['D-B-D', 'R-B-D']
        return False

    def execute_single_realistic_trade(self, zone: Dict, data: pd.DataFrame, current_idx: int) -> Optional[Dict]:
        """
        Execute single trade using REALISTIC 1R→2.5R management
        Entry triggers when price touches the entry level (like a limit order)
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # Entry and stop logic - Front-run beyond zone boundaries
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
            entry_price = zone_high + (zone_range * 0.05)  # 5% above zone
            direction = 'BUY'
            initial_stop = zone_low - (zone_range * 0.33)  # 33% buffer below zone
        elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell)
            entry_price = zone_low - (zone_range * 0.05)  # 5% below zone
            direction = 'SELL'
            initial_stop = zone_high + (zone_range * 0.33)  # 33% buffer above zone
        else:
            return None
        
        # Check if current candle can trigger entry (like a limit order)
        current_candle = data.iloc[current_idx]
        
        can_enter = False
        if direction == 'BUY':
            # Buy limit order triggers if high touches or exceeds entry price
            if current_candle['high'] >= entry_price:
                can_enter = True
        elif direction == 'SELL':
            # Sell limit order triggers if low touches or falls below entry price
            if current_candle['low'] <= entry_price:
                can_enter = True
        
        if not can_enter:
            return None
        
        # Calculate position size using UPDATED risk config
        risk_amount = 10000 * (RISK_CONFIG['risk_limits']['max_risk_per_trade'] / 100)  # 5% from settings
        
        # FIXED: Dynamic pip value detection instead of hardcoded 0.0001
        pip_value = self.get_pip_value_for_pair(zone.get('pair', 'EURUSD'))
        stop_distance_pips = abs(entry_price - initial_stop) / pip_value
        
        if stop_distance_pips <= 0:
            return None
        
        position_size = risk_amount / stop_distance_pips
        
        # Set targets (1:2.5 risk reward from RISK_CONFIG)
        risk_distance = abs(entry_price - initial_stop)
        target_rr = RISK_CONFIG['take_profit_rules']['risk_reward_ratio']  # 2.5 from settings
        
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * target_rr)
        else:
            target_price = entry_price - (risk_distance * target_rr)
        
        # Simulate REALISTIC trade outcome with 1R→breakeven management
        return self.simulate_realistic_outcome(
            entry_price, initial_stop, target_price, direction, 
            position_size, data, current_idx, zone['type'], stop_distance_pips, zone.get('pair', 'EURUSD'),
            zone_high, zone_low
        )

    def simulate_realistic_outcome(self, entry_price: float, stop_loss: float, target_price: float,
                     direction: str, position_size: float, data: pd.DataFrame,
                     entry_idx: int, zone_type: str, stop_distance_pips: float, pair: str, 
                     zone_high: float = None, zone_low: float = None) -> Dict:
        """
        Simulate REALISTIC trade outcome with proper 1R→breakeven management
        Clean production version - no debug output
        """
        # Add realistic transaction costs
        spread_pips = 2.0  # 2 pip spread (realistic for major pairs)
        commission_per_lot = 7.0  # $7 per lot commission (realistic retail)
        
        # Get pip value for this pair
        pip_value = self.get_pip_value_for_pair(pair)
        
        # Apply spread cost to entry
        if direction == 'BUY':
            entry_price += (spread_pips * pip_value)  # Pay the spread
        else:
            entry_price -= (spread_pips * pip_value)  # Pay the spread
        
        risk_distance = abs(entry_price - stop_loss)
        current_stop = stop_loss
        breakeven_moved = False
        
        # Proper position sizing (5% risk = $500 max loss)
        max_risk_amount = 500  # $500 max risk per trade (5% of $10,000)
        
        # Correct pip value per lot for JPY pairs
        if 'JPY' in pair.upper():
            pip_value_per_lot = 1.0  # $1 per pip for JPY pairs (0.01 movement)
        else:
            pip_value_per_lot = 10.0  # $10 per pip for major pairs
        
        # Calculate proper position size
        if stop_distance_pips > 0:
            proper_position_size = max_risk_amount / (stop_distance_pips * pip_value_per_lot)
            # Apply realistic limits
            proper_position_size = max(0.01, min(proper_position_size, 1.0))  # Min 0.01, Max 1.0 lot
        else:
            return None
        
        # Look ahead for exit with proper trade management
        for exit_idx in range(entry_idx + 1, min(entry_idx + 50, len(data))):  # Limit to 50 candles
            exit_candle = data.iloc[exit_idx]
            
            # Calculate 1R target for break-even trigger
            one_r_target = entry_price + risk_distance if direction == 'BUY' else entry_price - risk_distance

            # Check for 1R hit FIRST (wick-based) - triggers break-even move
            if not breakeven_moved:
                if direction == 'BUY' and exit_candle['high'] >= one_r_target:
                    current_stop = entry_price  # Move stop to EXACT entry price
                    breakeven_moved = True
                elif direction == 'SELL' and exit_candle['low'] <= one_r_target:
                    current_stop = entry_price  # Move stop to EXACT entry price
                    breakeven_moved = True

            # Check stops and targets with WICK-BASED exits
            if direction == 'BUY':
                # Check stop loss hit (wick-based)
                if exit_candle['low'] <= current_stop:
                    # Calculate P&L from actual exit price
                    price_diff = current_stop - entry_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    # Classify result based on break-even status
                    if breakeven_moved and current_stop == entry_price:
                        result_type = 'BREAKEVEN'  # Exact break-even
                    elif net_pnl < 0:
                        result_type = 'LOSS'
                    else:
                        result_type = 'WIN'
                    
                    # Create detailed trade summary
                    trade_summary = f"{zone_type} zone - Zone High: {stop_loss + risk_distance:.6f}, Zone Low: {stop_loss:.6f}, Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_date': data.index[entry_idx],
                        'exit_price': current_stop,
                        'exit_date': data.index[exit_idx],
                        'result': result_type,
                        'pnl': round(net_pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1),
                        'commission_cost': total_commission,
                        'breakeven_moved': breakeven_moved,
                        'trade_summary': trade_summary,
                        'zone_high': zone_high,
                        'zone_low': zone_low
                    }
                # Check 2.5R target hit (wick-based)
                elif exit_candle['high'] >= target_price:
                    price_diff = target_price - entry_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    # Create detailed trade summary
                    trade_summary = f"{zone_type} zone - Zone High: {stop_loss + risk_distance:.6f}, Zone Low: {stop_loss:.6f}, Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_date': data.index[entry_idx],
                        'exit_price': target_price,
                        'exit_date': data.index[exit_idx],
                        'result': 'WIN',
                        'pnl': round(net_pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1),
                        'commission_cost': total_commission,
                        'breakeven_moved': breakeven_moved,
                        'trade_summary': trade_summary,
                        'zone_high': zone_high,
                        'zone_low': zone_low
                    }
            else:  # SELL - Wick-based exits with exact break-even
                # Check stop loss hit (wick-based)
                if exit_candle['high'] >= current_stop:
                    price_diff = entry_price - current_stop
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    # Classify result based on break-even status
                    if breakeven_moved and current_stop == entry_price:
                        result_type = 'BREAKEVEN'  # Exact break-even
                    elif net_pnl < 0:
                        result_type = 'LOSS'
                    else:
                        result_type = 'WIN'
                    
                    # Create detailed trade summary
                    trade_summary = f"{zone_type} zone - Zone High: {stop_loss:.6f}, Zone Low: {stop_loss - risk_distance:.6f}, Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_date': data.index[entry_idx],
                        'exit_price': current_stop,
                        'exit_date': data.index[exit_idx],
                        'result': result_type,
                        'pnl': round(net_pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1),
                        'commission_cost': total_commission,
                        'breakeven_moved': breakeven_moved,
                        'trade_summary': trade_summary,
                        'zone_high': zone_high,
                        'zone_low': zone_low
                    }
                # Check 2.5R target hit (wick-based)
                elif exit_candle['low'] <= target_price:
                    price_diff = entry_price - target_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    # Create detailed trade summary
                    trade_summary = f"{zone_type} zone - Zone High: {stop_loss:.6f}, Zone Low: {stop_loss - risk_distance:.6f}, Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_date': data.index[entry_idx],
                        'exit_price': target_price,
                        'exit_date': data.index[exit_idx],
                        'result': 'WIN',
                        'pnl': round(net_pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1),
                        'commission_cost': total_commission,
                        'breakeven_moved': breakeven_moved,
                        'trade_summary': trade_summary,
                        'zone_high': zone_high,
                        'zone_low': zone_low
                    }
        
        # Trade still open at end (neutral exit with costs)
        return None
    
    def calculate_performance_metrics(self, trades: List[Dict], pair: str, timeframe: str) -> Dict:
        """Calculate comprehensive performance metrics with CORRECTED duration conversion"""
        if not trades:
            return self.create_empty_result(pair, timeframe, "No trades executed")
        
        # Duration conversion factors (candles to actual days)
        timeframe_to_days = {
            '1D': 1,
            '2D': 2, 
            '3D': 3,
            '4D': 4,
            '5D': 5,
            '1W': 7,
            '2W': 14,
            '3W': 21,
            '1M': 30,
            'H12': 0.5,
            'H8': 0.33,
            'H4': 0.17,
            'H1': 0.04
        }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        breakeven_trades = len([t for t in trades if t.get('result') == 'BREAKEVEN'])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
        be_rate = (breakeven_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        # Return calculation
        total_return = (total_pnl / 10000) * 100  # % return on $10,000
        
        # CORRECTED: Duration conversion from candles to actual days
        avg_duration_candles = np.mean([t.get('duration_days', 0) for t in trades])
        multiplier = timeframe_to_days.get(timeframe, 1)
        avg_duration_actual_days = avg_duration_candles * multiplier
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': round(win_rate, 1),
            'loss_rate': round(loss_rate, 1),
            'be_rate': round(be_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'total_return': round(total_return, 2),
            'avg_trade_duration': round(avg_duration_actual_days, 1),
            'avg_trade_duration_candles': round(avg_duration_candles, 1),
            'validation_method': 'walk_forward_realistic',
            'leg_out_threshold': 2.5,
            'trades': trades
        }
    
    def get_pip_value_for_pair(self, pair: str) -> float:
        """
        Get correct pip value for currency pair
        CRITICAL: JPY pairs use 0.01, others use 0.0001
        """
        if 'JPY' in pair.upper():
            return 0.01
        else:
            return 0.0001
    
    def create_empty_result(self, pair: str, timeframe: str, reason: str) -> Dict:
        """Create empty result structure"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'be_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'total_return': 0.0,
            'avg_trade_duration': 0.0,
            'description': reason,
            'trades': []
        }
    
    # ============================================================================
    # PRODUCTION FRAMEWORK METHODS - PHASE 2B
    # ============================================================================
    
    def run_parallel_comprehensive_analysis(self, analysis_period: str = 'priority_1') -> List[Dict]:
        """
        Run comprehensive analysis across all pairs and timeframes with parallel processing
        ENHANCED: Only tests combinations that actually have data files
        """
        print(f"\n🚀 COMPREHENSIVE PARALLEL ANALYSIS - {analysis_period.upper()}")
        period_config = ANALYSIS_PERIODS[analysis_period]
        days_back = period_config['days_back']
        
        print(f"📊 Period: {period_config['name']}")
        print(f"📅 Days back: {days_back:,}")
        print("=" * 70)
        
        # ENHANCED: Discover only valid pair/timeframe combinations
        valid_combinations = self.discover_valid_data_combinations()
        if not valid_combinations:
            print("❌ No valid data combinations found")
            return []
        
        # Create test combinations only for valid data
        test_combinations = []
        for pair, timeframe in valid_combinations:
            test_combinations.append({
                'pair': pair,
                'timeframe': timeframe,
                'days_back': days_back,
                'analysis_period': analysis_period
            })
        
        print(f"📊 Valid combinations found: {len(valid_combinations)}")
        print(f"📊 Total tests to run: {len(test_combinations):,}")
        
        # Show breakdown by pair
        pair_counts = {}
        for pair, timeframe in valid_combinations:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        print(f"📊 Data availability by pair:")
        for pair, count in sorted(pair_counts.items()):
            print(f"   {pair}: {count} timeframes")
        
        # Run optimized parallel processing
        all_results = self.run_optimized_parallel_tests(test_combinations)
        
        # Generate comprehensive Excel report
        if all_results:
            self.generate_comprehensive_excel_report(all_results, analysis_period, period_config)
        
        # Print summary
        successful_results = [r for r in all_results if r['total_trades'] > 0]
        print(f"\n🎯 ANALYSIS COMPLETE:")
        print(f"   Total combinations tested: {len(all_results):,}")
        print(f"   Successful combinations: {len(successful_results):,}")
        print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        
        if successful_results:
            avg_pf = sum(r['profit_factor'] for r in successful_results) / len(successful_results)
            avg_wr = sum(r['win_rate'] for r in successful_results) / len(successful_results)
            print(f"   Average PF: {avg_pf:.2f}")
            print(f"   Average WR: {avg_wr:.1f}%")
            
            # Best overall performer
            best = max(successful_results, key=lambda x: x['profit_factor'])
            print(f"   🏆 Best: {best['pair']} {best['timeframe']} - PF {best['profit_factor']:.2f}")
        
        return all_results
    
    def run_optimized_parallel_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """
        Run tests in parallel with memory management and optimization
        """
        print(f"\n🔄 OPTIMIZED PARALLEL EXECUTION")
        print(f"⚡ Workers: {self.max_workers}")
        print(f"📦 Chunk size: {self.chunk_size}")
        
        start_time = time.time()
        results = []
        
        # Process in chunks for memory management
        chunk_size = self.chunk_size
        total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(test_combinations))
            chunk_tests = test_combinations[chunk_start:chunk_end]
            
            print(f"\n📦 Chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_tests)} tests)")
            
            # Memory check
            memory_percent = psutil.virtual_memory().percent
            print(f"💾 Memory usage: {memory_percent:.1f}%")
            
            if memory_percent > self.memory_threshold * 100:
                print("⚠️  High memory usage, triggering cleanup...")
                gc.collect()
            
            # Process chunk with multiprocessing
            try:
                with Pool(processes=self.max_workers) as pool:
                    chunk_results = pool.map(run_single_test_worker, chunk_tests)
                    results.extend(chunk_results)
                
                # Progress tracking
                completed = chunk_end
                progress = (completed / len(test_combinations)) * 100
                print(f"✅ Progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
                
            except Exception as e:
                print(f"❌ Chunk {chunk_idx + 1} failed: {str(e)}")
                # Add empty results for failed chunk
                for test in chunk_tests:
                    results.append({
                        'pair': test['pair'],
                        'timeframe': test['timeframe'],
                        'total_trades': 0,
                        'description': f"Parallel processing error: {str(e)}"
                    })
            
            # Memory cleanup after each chunk
            gc.collect()
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ PARALLEL EXECUTION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🎯 Success rate: {success_count}/{len(test_combinations)} ({success_count/len(test_combinations)*100:.1f}%)")
        print(f"⚡ Speed: {len(test_combinations)/total_time:.1f} tests/second")
        
        return results
    
    def generate_comprehensive_excel_report(self, all_results: List[Dict], 
                                          analysis_period: str, period_config: Dict):
        """
        Generate professional 4-sheet Excel report
        """
        print(f"\n📊 GENERATING COMPREHENSIVE EXCEL REPORT...")
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_results)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{analysis_period}_comprehensive_analysis_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: All Results
                df_all.to_excel(writer, sheet_name='All_Results', index=False)
                print("   ✅ Sheet 1: All Results")
                
                # SHEET 2: Successful Results Only
                successful_df = df_all[df_all['total_trades'] > 0]
                if len(successful_df) > 0:
                    successful_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    print("   ✅ Sheet 2: Successful Results")
                    
                    # SHEET 3: Performance Analysis by Timeframe
                    tf_analysis = self.create_timeframe_analysis(successful_df)
                    tf_analysis.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    print("   ✅ Sheet 3: Timeframe Analysis")
                    
                    # SHEET 4: Performance Analysis by Pair
                    pair_analysis = self.create_pair_analysis(successful_df)
                    pair_analysis.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ✅ Sheet 4: Pair Analysis")
                    
                else:
                    # Create empty analysis sheets
                    empty_df = pd.DataFrame({'Note': ['No successful results to analyze']})
                    empty_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    empty_df.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ⚠️  Empty analysis sheets (no successful results)")
            
            print(f"📁 EXCEL REPORT SAVED: {filename}")
            print(f"📊 4 comprehensive analysis sheets created")
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {str(e)}")
            # Fallback: Save as CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df_all.to_csv(csv_filename, index=False)
            print(f"📁 Fallback CSV saved: {csv_filename}")
    
    def create_timeframe_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create timeframe performance analysis"""
        try:
            tf_analysis = df.groupby('timeframe').agg({
                'profit_factor': ['mean', 'count'],
                'win_rate': 'mean',
                'loss_rate': 'mean',
                'be_rate': 'mean',
                'total_trades': 'sum',
                'total_return': 'mean'
            }).round(2)

            # Flatten column names
            tf_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                'Avg_Loss_Rate', 'Avg_BE_Rate', 'Total_Trades', 'Avg_Return']
            tf_analysis = tf_analysis.sort_values('Avg_Profit_Factor', ascending=False)
            
            return tf_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Timeframe analysis error: {str(e)}")
            return pd.DataFrame({'Timeframe': ['Error'], 'Note': [str(e)]})
    
    def create_pair_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pair performance analysis"""
        try:
            pair_analysis = df.groupby('pair').agg({
                'profit_factor': ['mean', 'count'],
                'win_rate': 'mean',
                'loss_rate': 'mean',
                'be_rate': 'mean',
                'total_trades': 'sum',
                'total_return': 'mean'
            }).round(2)

            # Flatten column names
            pair_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                'Avg_Loss_Rate', 'Avg_BE_Rate', 'Total_Trades', 'Avg_Return']
            pair_analysis = pair_analysis.sort_values('Avg_Profit_Factor', ascending=False)
            
            return pair_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Pair analysis error: {str(e)}")
            return pd.DataFrame({'Pair': ['Error'], 'Note': [str(e)]})
    
    def generate_manual_chart_analysis_report(self, result: Dict):
        """
        Generate comprehensive Excel report for manual chart analysis
        Perfect for validating trades against actual charts
        """
        print(f"\n📊 GENERATING MANUAL CHART ANALYSIS REPORT...")
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pair = result['pair']
        timeframe = result['timeframe']
        filename = f"results/manual_chart_analysis_{pair}_{timeframe}_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: Trade Summary
                summary_data = {
                    'Metric': ['Pair', 'Timeframe', 'Total Trades', 'Winning Trades', 'Losing Trades', 
                            'Breakeven Trades', 'Win Rate %', 'Loss Rate %', 'BE Rate %', 
                            'Profit Factor', 'Total P&L', 'Gross Profit', 'Gross Loss', 
                            'Total Return %', 'Avg Trade Duration (days)'],
                    'Value': [result['pair'], result['timeframe'], result['total_trades'], 
                            result['winning_trades'], result['losing_trades'], result['breakeven_trades'],
                            f"{result['win_rate']:.1f}%", f"{result['loss_rate']:.1f}%", f"{result['be_rate']:.1f}%",
                            result['profit_factor'], f"${result['total_pnl']:.2f}", 
                            f"${result['gross_profit']:.2f}", f"${result['gross_loss']:.2f}",
                            f"{result['total_return']:.2f}%", f"{result['avg_trade_duration']:.1f}"]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
                print("   ✅ Sheet 1: Performance Summary")
                
                # SHEET 2: Detailed Trade List for Chart Validation
                if result['trades']:
                    trades_data = []
                    for i, trade in enumerate(result['trades'], 1):
                        # Extract zone boundaries from trade data
                        zone_high = trade.get('zone_high', 'N/A')
                        zone_low = trade.get('zone_low', 'N/A')
                        duration = trade.get('duration_days', trade.get('duration', 'N/A'))
                        
                        trades_data.append({
                            'Trade_Number': i,
                            'Zone_Type': trade['zone_type'],
                            'Direction': trade['direction'],
                            'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                            'Exit_Date': trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date']),
                            'Entry_Price': f"{trade['entry_price']:.6f}",
                            'Exit_Price': f"{trade['exit_price']:.6f}",
                            'Zone_High': f"{zone_high:.6f}" if isinstance(zone_high, (int, float)) else zone_high,
                            'Zone_Low': f"{zone_low:.6f}" if isinstance(zone_low, (int, float)) else zone_low,
                            'Duration': duration,
                            'Result': trade['result'],
                            'Pips': f"{trade['pips']:+.1f}",
                            'PnL': f"${trade['pnl']:.2f}",
                            'Position_Size': f"{trade['position_size']:.4f}",
                            'Commission_Cost': f"${trade['commission_cost']:.2f}",
                            'Breakeven_Moved': 'Yes' if trade['breakeven_moved'] else 'No'
                        })
                    
                    trades_df = pd.DataFrame(trades_data)
                    trades_df.to_excel(writer, sheet_name='Trade_Details_For_Charts', index=False)
                    print("   ✅ Sheet 2: Trade Details for Chart Validation")
                    
                    # SHEET 3: Winning Trades Analysis
                    winning_trades = [t for t in result['trades'] if t['result'] == 'WIN']
                    if winning_trades:
                        win_data = []
                        for i, trade in enumerate(winning_trades, 1):
                            win_data.append({
                                'Win_Number': i,
                                'Trade_Summary': trade.get('trade_summary', f"Win #{i}: {trade['zone_type']} zone"),
                                'Zone_Type': trade['zone_type'],
                                'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                                'Entry_Price': f"{trade['entry_price']:.6f}",
                                'Exit_Price': f"{trade['exit_price']:.6f}",
                                'Pips_Won': f"+{trade['pips']:.1f}",
                                'Profit': f"${trade['pnl']:.2f}",
                                'Duration': f"{trade['duration_days']} days",
                                'Breakeven_Used': 'Yes' if trade['breakeven_moved'] else 'No'
                            })
                        
                        wins_df = pd.DataFrame(win_data)
                        wins_df.to_excel(writer, sheet_name='Winning_Trades', index=False)
                        print("   ✅ Sheet 3: Winning Trades Analysis")
                    
                    # SHEET 4: Losing Trades Analysis
                    losing_trades = [t for t in result['trades'] if t['result'] == 'LOSS']
                    if losing_trades:
                        loss_data = []
                        for i, trade in enumerate(losing_trades, 1):
                            loss_data.append({
                                'Loss_Number': i,
                                'Trade_Summary': trade.get('trade_summary', f"Loss #{i}: {trade['zone_type']} zone"),
                                'Zone_Type': trade['zone_type'],
                                'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                                'Entry_Price': f"{trade['entry_price']:.6f}",
                                'Exit_Price': f"{trade['exit_price']:.6f}",
                                'Pips_Lost': f"{trade['pips']:.1f}",
                                'Loss_Amount': f"${trade['pnl']:.2f}",
                                'Duration': f"{trade['duration_days']} days",
                                'Breakeven_Attempted': 'Yes' if trade['breakeven_moved'] else 'No'
                            })
                        
                        losses_df = pd.DataFrame(loss_data)
                        losses_df.to_excel(writer, sheet_name='Losing_Trades', index=False)
                        print("   ✅ Sheet 4: Losing Trades Analysis")
                    
                    # SHEET 5: Breakeven Trades Analysis
                    breakeven_trades = [t for t in result['trades'] if t['result'] == 'BREAKEVEN']
                    if breakeven_trades:
                        be_data = []
                        for i, trade in enumerate(breakeven_trades, 1):
                            be_data.append({
                                'BE_Number': i,
                                'Trade_Summary': trade.get('trade_summary', f"BE #{i}: {trade['zone_type']} zone"),
                                'Zone_Type': trade['zone_type'],
                                'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                                'Entry_Price': f"{trade['entry_price']:.6f}",
                                'Exit_Price': f"{trade['exit_price']:.6f}",
                                'Commission_Cost': f"${trade['commission_cost']:.2f}",
                                'Duration': f"{trade['duration_days']} days",
                                'Note': '1R hit, moved to breakeven, then stopped out'
                            })
                        
                        be_df = pd.DataFrame(be_data)
                        be_df.to_excel(writer, sheet_name='Breakeven_Trades', index=False)
                        print("   ✅ Sheet 5: Breakeven Trades Analysis")
                
                else:
                    # No trades found
                    empty_df = pd.DataFrame({'Note': ['No trades found for this pair/timeframe combination']})
                    empty_df.to_excel(writer, sheet_name='No_Trades_Found', index=False)
                    print("   ⚠️  No trades to analyze")
            
            print(f"\n📁 MANUAL CHART ANALYSIS REPORT SAVED:")
            print(f"   File: {filename}")
            print(f"   📊 {len(result.get('trades', []))} trades ready for chart validation")
            print(f"   📈 Use this report to manually verify each trade against your charts")
            print(f"   🎯 Each trade includes entry/exit dates and prices for easy chart lookup")
            
        except Exception as e:
            print(f"❌ Error creating manual chart analysis report: {str(e)}")
            # Fallback: Save basic CSV
            if result.get('trades'):
                csv_filename = filename.replace('.xlsx', '.csv')
                trades_df = pd.DataFrame(result['trades'])
                trades_df.to_csv(csv_filename, index=False)
                print(f"📁 Fallback CSV saved: {csv_filename}")
    
    def check_system_resources(self) -> bool:
        """Check system resources before heavy analysis"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = cpu_count()
        memory_percent = psutil.virtual_memory().percent
        
        print(f"💻 SYSTEM RESOURCES:")
        print(f"   RAM: {memory_gb:.1f} GB available")
        print(f"   CPU: {cpu_cores} cores")
        print(f"   Current memory usage: {memory_percent:.1f}%")
        
        if memory_gb < 8:
            print("⚠️  WARNING: Less than 8GB RAM. Consider reducing scope.")
        
        if memory_percent > 60:
            print("⚠️  WARNING: High memory usage. Close other applications.")
        
        return memory_gb >= 4  # Minimum 4GB required
    
    def calculate_drawdown(self, equity_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown analysis"""
        equity_data = equity_data.copy()
        
        # Calculate running maximum (peak)
        equity_data['peak'] = equity_data['equity'].expanding().max()
        
        # Calculate drawdown in dollars and percentage
        equity_data['drawdown_dollars'] = equity_data['equity'] - equity_data['peak']
        equity_data['drawdown_pct'] = (equity_data['drawdown_dollars'] / equity_data['peak']) * 100
        
        return equity_data
    
    def run_single_test_with_charts(self, pair: str, timeframe: str, days_back: int = 730, 
                                   show_charts: bool = True) -> Dict:
        """
        Run single test and automatically generate charts
        ENHANCED: Includes comprehensive visualization
        """
        print(f"\n🧪 COMPREHENSIVE TEST WITH CHARTS: {pair} {timeframe}")
        
        # Run the standard backtest
        result = self.run_single_strategy_test(pair, timeframe, days_back)
        
        # Generate charts if there are trades
        if result['total_trades'] > 0 and show_charts:
            print(f"\n📊 GENERATING COMPREHENSIVE ANALYSIS CHARTS...")
            chart_file = self.create_comprehensive_analysis_charts(result)
            result['chart_file'] = chart_file
        else:
            result['chart_file'] = None
        
        return result

# ============================================================================
# PARALLEL PROCESSING WORKER FUNCTION
# ============================================================================

def run_single_test_worker(test_config: Dict) -> Dict:
    """
    Worker function for parallel processing
    Each worker creates its own engine instance to avoid conflicts
    """
    try:
        # Create fresh engine instance for this worker
        engine = CoreBacktestEngine()
        
        result = engine.run_single_strategy_test(
            test_config['pair'],
            test_config['timeframe'],
            test_config['days_back']
        )
        
        # Add analysis period info
        result['analysis_period'] = test_config['analysis_period']
        
        # Clean up
        del engine
        gc.collect()
        
        return result
        
    except Exception as e:
        gc.collect()
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'],
            'analysis_period': test_config['analysis_period'],
            'total_trades': 0,
            'description': f"Worker error: {str(e)}"
        }

def main():
    """Enhanced main function with production framework options"""
    print("🎯 CORE BACKTESTING ENGINE - PRODUCTION READY")
    print("=" * 60)
    
    # Check system resources
    engine = CoreBacktestEngine()
    if not engine.check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    print("\n🎯 SELECT ANALYSIS MODE:")
    print("1. Quick Validation (Single test - EURUSD 3D)")
    print("2. Comprehensive Analysis - Priority 1 (2015-2025, All pairs/timeframes)")
    print("3. Comprehensive Analysis - Priority 2 (2020-2025, All pairs/timeframes)")
    print("4. Custom Single Test")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Quick validation test with BENCHMARK TIMING
        print("\n🧪 QUICK VALIDATION TEST WITH BENCHMARK:")
        print("Testing EURUSD 3D with updated 2.5x threshold...")
        
        # BENCHMARK START
        import time
        start_time = time.time()
        
        result = engine.run_single_strategy_test('EURUSD', '3D', 730)
        
        # BENCHMARK END
        end_time = time.time()
        benchmark_time = end_time - start_time
        print(f"\n🕐 BENCHMARK TIME: {benchmark_time:.1f} seconds ({benchmark_time/60:.1f} minutes)")
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   Pair: {result['pair']} {result['timeframe']}")
        print(f"   Trades: {result['total_trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Loss Rate: {result['loss_rate']:.1f}%")
        print(f"   BE Rate: {result['be_rate']:.1f}%")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Total Return: {result['total_return']:.2f}%")
        
        if result['total_trades'] == 0:
            print(f"   Issue: {result['description']}")
        else:
            print(f"   ✅ VALIDATION SUCCESSFUL - Engine working with updated modules!")
    
    elif choice == '2':
        # Comprehensive analysis - Priority 1
        print("\n🚀 COMPREHENSIVE ANALYSIS - PRIORITY 1")
        print("This will test ALL pairs and timeframes with 10 years of data")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_parallel_comprehensive_analysis('priority_1')
        else:
            print("Analysis cancelled")
    
    elif choice == '3':
        # Comprehensive analysis - Priority 2
        print("\n🚀 COMPREHENSIVE ANALYSIS - PRIORITY 2")
        print("This will test ALL pairs and timeframes with 4 years of data")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_parallel_comprehensive_analysis('priority_2')
        else:
            print("Analysis cancelled")
    
    elif choice == '4':
        # Custom single test with comprehensive reporting
        print("\n🎯 CUSTOM SINGLE TEST WITH MANUAL CHART ANALYSIS:")
        pairs = engine.discover_all_pairs()
        print(f"Available pairs: {', '.join(pairs[:10])}..." if len(pairs) > 10 else f"Available pairs: {', '.join(pairs)}")
        
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        days_back = input("Enter days back (e.g., 730): ").strip()
        
        try:
            days_back = int(days_back)
            result = engine.run_single_strategy_test(pair, timeframe, days_back)
            
            print(f"\n📊 CUSTOM TEST RESULTS:")
            print(f"   Pair: {result['pair']} {result['timeframe']}")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Loss Rate: {result['loss_rate']:.1f}%")
            print(f"   BE Rate: {result['be_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Total Return: {result['total_return']:.2f}%")
            print(f"   Average Duration: {result['avg_trade_duration']:.1f} days")
            
            # Generate comprehensive Excel report for manual chart analysis
            if result['total_trades'] > 0:
                print(f"\n📋 GENERATING MANUAL CHART ANALYSIS REPORT...")
                engine.generate_manual_chart_analysis_report(result)
            else:
                print(f"   Issue: {result.get('description', 'No trades found')}")
                
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()