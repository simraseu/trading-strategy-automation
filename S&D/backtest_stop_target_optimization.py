"""
Stop/Target Optimization Engine - Parameter Sensitivity Analysis  
Extends CoreBacktestEngine to test ALL combinations of stop buffers and take profit targets
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
import itertools
warnings.filterwarnings('ignore')

# Import the base engine
from core_backtest_engine import CoreBacktestEngine, run_single_test_worker

# Set process priority for maximum CPU usage
try:
    if os.name == 'nt':  # Windows
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
except:
    pass

class StopTargetOptimizationEngine(CoreBacktestEngine):
    """
    EXTENDS CoreBacktestEngine to test ALL stop buffer and take profit target combinations
    
    CRITICAL: Uses IDENTICAL trades - same zones, entries, timing, break-even triggers
    ONLY modifies stop loss buffer and take profit target calculations
    
    Test Matrix: 5 stop buffers × 6 take profit targets = 30 combinations
    Stop Buffers: 25%, 33% (baseline), 40%, 50%, 60%
    Take Profit Targets: 1.0R, 1.5R, 2.0R, 2.5R (baseline), 3.0R, 4.0R
    """
    
    def __init__(self):
        """Initialize with base engine and optimization parameters"""
        super().__init__()
        
        # Define all combinations to test
        self.stop_variations = {
            '25pct': 0.25,
            '33pct': 0.33,  # Baseline
            '40pct': 0.40,
            '50pct': 0.50,
            '60pct': 0.60
        }
        
        self.target_variations = {
            '1R': 1.0,
            '1_5R': 1.5,
            '2R': 2.0,
            '2_5R': 2.5,  # Baseline
            '3R': 3.0,
            '4R': 4.0
        }
        
        # Create all 30 combinations
        self.combinations = []
        for stop_key, stop_val in self.stop_variations.items():
            for target_key, target_val in self.target_variations.items():
                combo_name = f"Stop_{stop_key}_Target_{target_key}"
                self.combinations.append({
                    'name': combo_name,
                    'stop_buffer': stop_val,
                    'target_multiplier': target_val,
                    'stop_key': stop_key,
                    'target_key': target_key
                })
        
        # Current combination being tested (for method overrides)
        self.current_combination = None
        
        print(f"🎯 STOP/TARGET OPTIMIZATION ENGINE INITIALIZED:")
        print(f"   Stop Buffer Variations: {len(self.stop_variations)}")
        print(f"   Take Profit Variations: {len(self.target_variations)}")
        print(f"   Total Combinations: {len(self.combinations)}")
        print(f"   Baseline: Stop_33pct_Target_2_5R")
    
    def calculate_stop_loss_with_buffer(self, zone: Dict, buffer_percent: float) -> float:
        """
        Calculate stop loss with CUSTOM buffer percentage
        OVERRIDES the standard 33% buffer from base engine
        """
        if zone['type'] == 'R-B-R':  # Bullish demand zone
            # Stop goes buffer% below zone_low
            zone_boundary = zone['zone_low']
            zone_size = zone['zone_high'] - zone['zone_low']
            buffer_distance = zone_size * buffer_percent
            stop_loss_price = zone_boundary - buffer_distance
            
        else:  # D-B-D bearish supply zone
            # Stop goes buffer% above zone_high
            zone_boundary = zone['zone_high']
            zone_size = zone['zone_high'] - zone['zone_low']
            buffer_distance = zone_size * buffer_percent
            stop_loss_price = zone_boundary + buffer_distance
        
        return stop_loss_price
    
    
    def simulate_realistic_outcome_optimized(self, entry_price: float, stop_loss: float, 
                                           target_price: float, direction: str, position_size: float, 
                                           data: pd.DataFrame, entry_idx: int, zone_type: str, 
                                           stop_distance_pips: float, pair: str, 
                                           zone_high: float = None, zone_low: float = None, 
                                           zone: Dict = None) -> Dict:
        """
        OVERRIDES base simulate_realistic_outcome with CUSTOM stop/target parameters
        CRITICAL: Maintains IDENTICAL break-even logic (always at 1R)
        """
        # Add realistic transaction costs (same as base)
        spread_pips = 2.0
        commission_per_lot = 7.0
        
        # Get pip value for this pair
        pip_value = self.get_pip_value_for_pair(pair)
        
        # Apply spread cost to entry (same as base)
        if direction == 'BUY':
            entry_price += (spread_pips * pip_value)
        else:
            entry_price -= (spread_pips * pip_value)
        
        risk_distance = abs(entry_price - stop_loss)
        current_stop = stop_loss
        breakeven_moved = False
        
        # Position sizing (same as base - 5% max risk)
        max_risk_amount = 500
        
        if 'JPY' in pair.upper():
            pip_value_per_lot = 1.0
        else:
            pip_value_per_lot = 10.0
        
        if stop_distance_pips > 0:
            proper_position_size = max_risk_amount / (stop_distance_pips * pip_value_per_lot)
            proper_position_size = max(0.01, min(proper_position_size, 1.0))
        else:
            return None
        
        # Look ahead for exit with CUSTOM target but SAME break-even logic
        for exit_idx in range(entry_idx + 1, min(entry_idx + 50, len(data))):
            exit_candle = data.iloc[exit_idx]
            
            # Calculate 1R target for break-even trigger (ALWAYS 1R regardless of final target)
            one_r_target = entry_price + risk_distance if direction == 'BUY' else entry_price - risk_distance

            # Check for 1R hit FIRST - triggers break-even move (IDENTICAL to base)
            if not breakeven_moved:
                if direction == 'BUY' and exit_candle['high'] >= one_r_target:
                    current_stop = entry_price
                    breakeven_moved = True
                elif direction == 'SELL' and exit_candle['low'] <= one_r_target:
                    current_stop = entry_price
                    breakeven_moved = True

            # Check stops and CUSTOM targets with wick-based exits
            if direction == 'BUY':
                # Check stop loss hit (wick-based)
                if exit_candle['low'] <= current_stop:
                    price_diff = current_stop - entry_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    if breakeven_moved and current_stop == entry_price:
                        result_type = 'BREAKEVEN'
                    elif net_pnl < 0:
                        result_type = 'LOSS'
                    else:
                        result_type = 'WIN'
                    
                    # Add combination info to trade summary
                    combination_name = self.current_combination['name'] if self.current_combination else 'Unknown'
                    trade_summary = f"{combination_name} - {zone_type} zone - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Target: {target_price:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'combination': combination_name,
                        'stop_buffer': self.current_combination['stop_buffer'] if self.current_combination else 0.33,
                        'target_multiplier': self.current_combination['target_multiplier'] if self.current_combination else 2.5,
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
                        'zone_low': zone_low,
                        'target_price': target_price,
                        'initial_stop': stop_loss
                    }
                # Check CUSTOM target hit (wick-based)
                elif exit_candle['high'] >= target_price:
                    price_diff = target_price - entry_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    combination_name = self.current_combination['name'] if self.current_combination else 'Unknown'
                    trade_summary = f"{combination_name} - {zone_type} zone - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Target: {target_price:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'combination': combination_name,
                        'stop_buffer': self.current_combination['stop_buffer'] if self.current_combination else 0.33,
                        'target_multiplier': self.current_combination['target_multiplier'] if self.current_combination else 2.5,
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
                        'zone_low': zone_low,
                        'target_price': target_price,
                        'initial_stop': stop_loss
                    }
            else:  # SELL - CUSTOM target with exact break-even
                # Check stop loss hit (wick-based)
                if exit_candle['high'] >= current_stop:
                    price_diff = entry_price - current_stop
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    if breakeven_moved and current_stop == entry_price:
                        result_type = 'BREAKEVEN'
                    elif net_pnl < 0:
                        result_type = 'LOSS'
                    else:
                        result_type = 'WIN'
                    
                    combination_name = self.current_combination['name'] if self.current_combination else 'Unknown'
                    trade_summary = f"{combination_name} - {zone_type} zone - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Target: {target_price:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'combination': combination_name,
                        'stop_buffer': self.current_combination['stop_buffer'] if self.current_combination else 0.33,
                        'target_multiplier': self.current_combination['target_multiplier'] if self.current_combination else 2.5,
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
                        'zone_low': zone_low,
                        'target_price': target_price,
                        'initial_stop': stop_loss
                    }
                # Check CUSTOM target hit (wick-based)
                elif exit_candle['low'] <= target_price:
                    price_diff = entry_price - target_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    combination_name = self.current_combination['name'] if self.current_combination else 'Unknown'
                    trade_summary = f"{combination_name} - {zone_type} zone - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Target: {target_price:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
                    return {
                        'combination': combination_name,
                        'stop_buffer': self.current_combination['stop_buffer'] if self.current_combination else 0.33,
                        'target_multiplier': self.current_combination['target_multiplier'] if self.current_combination else 2.5,
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
                        'zone_low': zone_low,
                        'target_price': target_price,
                        'initial_stop': stop_loss
                    }
        
        # Trade still open at end
        return None
    
    def execute_single_realistic_trade_optimized(self, zone: Dict, data: pd.DataFrame, 
                                           current_idx: int) -> Optional[Dict]:
        """
        OVERRIDES base execute_single_realistic_trade with CUSTOM stop/target calculations
        CRITICAL: Same entry logic, same break-even timing, ONLY stop/target differences
        FIXED: Properly recalculates risk distance when stop buffer changes
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # IDENTICAL entry calculation to base engine
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
            entry_price = self.calculate_deep_retracement_entry(zone, data, 'demand', True)
            direction = 'BUY'
            # CUSTOM stop loss with current combination buffer
            initial_stop = self.calculate_stop_loss_with_buffer(zone, self.current_combination['stop_buffer'])
        elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell)
            entry_price = self.calculate_deep_retracement_entry(zone, data, 'supply', True)
            direction = 'SELL'
            # CUSTOM stop loss with current combination buffer
            initial_stop = self.calculate_stop_loss_with_buffer(zone, self.current_combination['stop_buffer'])
        else:
            return None
        
        # IDENTICAL entry trigger logic to base engine
        current_candle = data.iloc[current_idx]
        
        can_enter = False
        if direction == 'BUY':
            if current_candle['high'] >= entry_price:
                can_enter = True
        elif direction == 'SELL':
            if current_candle['low'] <= entry_price:
                can_enter = True
        
        if not can_enter:
            return None
        
        # CRITICAL FIX: Calculate NEW risk distance based on ADJUSTED stop loss
        new_risk_distance = abs(entry_price - initial_stop)
        
        # CUSTOM target calculation using NEW risk distance and target multiplier
        if direction == 'BUY':
            target_price = entry_price + (new_risk_distance * self.current_combination['target_multiplier'])
        else:  # SELL
            target_price = entry_price - (new_risk_distance * self.current_combination['target_multiplier'])
        
        # Position sizing based on NEW risk distance
        risk_amount = 10000 * 0.05  # 5% risk
        pip_value = self.get_pip_value_for_pair(zone.get('pair', 'EURUSD'))
        stop_distance_pips = new_risk_distance / pip_value
        
        if stop_distance_pips <= 0:
            return None
        
        position_size = risk_amount / stop_distance_pips
        
        # Simulate with PROPERLY CALCULATED parameters
        return self.simulate_realistic_outcome_optimized(
            entry_price, initial_stop, target_price, direction, 
            position_size, data, current_idx, zone['type'], stop_distance_pips, 
            zone.get('pair', 'EURUSD'), zone_high, zone_low, zone
        )
    
    def execute_realistic_trades_with_combination(self, patterns: List[Dict], data: pd.DataFrame,
                                                timeframe: str, pair: str, combination: Dict) -> List[Dict]:
        """
        Execute trades with SPECIFIC stop/target combination
        IDENTICAL trade timing and selection to base engine
        """
        # Set current combination for method overrides
        self.current_combination = combination
        
        trades = []
        used_zones = set()
        invalidated_zones = set()
        validated_zones = {}
        
        # IDENTICAL backtest period logic to base engine
        backtest_start_idx = 200
        backtest_start_date = data.index[backtest_start_idx]
        
        # IDENTICAL pattern filtering to base engine
        valid_patterns = []
        for pattern in patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx >= backtest_start_idx:
                zone_formation_date = data.index[zone_end_idx]
                pattern['formation_date'] = zone_formation_date
                pattern['invalidation_level'] = self.precalculate_invalidation_level(pattern)
                valid_patterns.append(pattern)
        
        if not valid_patterns:
            return trades
        
        # IDENTICAL zone validation tracking to base engine
        zone_tracking = {}
        for pattern in valid_patterns:
            zone_end_idx = pattern['end_idx']
            zone_id = f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
            
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
        
        # IDENTICAL candle processing loop to base engine
        for current_idx in range(backtest_start_idx, len(data)):
            current_candle = data.iloc[current_idx]
            
            for zone_id, zone_info in zone_tracking.items():
                if zone_id in used_zones or zone_id in invalidated_zones:
                    continue
                    
                if not zone_info['validated']:
                    continue
                    
                if zone_info['validation_idx'] is None or current_idx <= zone_info['validation_idx']:
                    continue
                
                zone = zone_info['pattern']
                
                # IDENTICAL interaction checks to base engine
                if not self.fast_zone_interaction_check(zone, current_candle):
                    continue
                
                if not self.check_limit_order_trigger(zone, current_candle):
                    continue
                
                used_zones.add(zone_id)
                
                # Execute trade with CUSTOM stop/target parameters
                zone['pair'] = pair
                trade_result = self.execute_single_realistic_trade_optimized(zone, data, current_idx)
                
                if trade_result:
                    trades.append(trade_result)
                    break
        
        return trades
    
    def run_combination_optimization_test(self, pair: str, timeframe: str, days_back: int = 730) -> List[Dict]:
        """
        Run ALL 30 stop/target combinations on IDENTICAL trades for given pair/timeframe
        """
        print(f"\n🎯 COMBINATION OPTIMIZATION: {pair} {timeframe}")
        print("=" * 60)
        
        # Load data ONCE (same for all combinations)
        data = self.load_data_with_validation(pair, timeframe, days_back)
        if data is None:
            return []
        
        # Detect zones ONCE (same for all combinations) 
        from modules.candle_classifier import CandleClassifier
        from modules.zone_detector import ZoneDetector
        
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        # Get all patterns (IDENTICAL for all combinations)
        all_patterns = (patterns['dbd_patterns'] + patterns['rbr_patterns'] + 
                       patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', []))
        
        valid_patterns = [
            pattern for pattern in all_patterns
            if pattern.get('end_idx') is not None
        ]
        
        if not valid_patterns:
            print(f"   ⚠️  No valid patterns found for {pair} {timeframe}")
            return []
        
        print(f"   📊 Found {len(valid_patterns)} patterns to test on")
        print(f"   🔄 Testing {len(self.combinations)} combinations...")
        
        all_results = []
        
        # Test each combination on IDENTICAL trades
        for i, combination in enumerate(self.combinations, 1):
            print(f"   🧪 [{i:2d}/30] {combination['name']}...")
            
            # Execute trades with this specific combination
            trades = self.execute_realistic_trades_with_combination(
                valid_patterns, data, timeframe, pair, combination
            )
            
            # Calculate performance for this combination
            result = self.calculate_performance_metrics_optimized(
                trades, pair, timeframe, combination
            )
            
            all_results.append(result)
            
            # Progress update
            if result['total_trades'] > 0:
                print(f"       ✅ {result['total_trades']} trades, PF: {result['profit_factor']:.2f}, WR: {result['win_rate']:.1f}%")
            else:
                print(f"       ❌ No trades")
        
        # Summary
        successful_combinations = [r for r in all_results if r['total_trades'] > 0]
        print(f"\n   📊 COMBINATION RESULTS:")
        print(f"   Successful combinations: {len(successful_combinations)}/30")
        
        if successful_combinations:
            # Best by profit factor
            best_pf = max(successful_combinations, key=lambda x: x['profit_factor'])
            print(f"   🏆 Best PF: {best_pf['combination']} - {best_pf['profit_factor']:.2f}")
            
            # Best by win rate
            best_wr = max(successful_combinations, key=lambda x: x['win_rate'])
            print(f"   🎯 Best WR: {best_wr['combination']} - {best_wr['win_rate']:.1f}%")
        
        return all_results
    
    def calculate_performance_metrics_optimized(self, trades: List[Dict], pair: str, 
                                              timeframe: str, combination: Dict) -> Dict:
        """
        EXTENDS base performance calculation with combination-specific metrics
        """
        if not trades:
            return {
                'pair': pair,
                'timeframe': timeframe,
                'combination': combination['name'],
                'stop_buffer': combination['stop_buffer'],
                'target_multiplier': combination['target_multiplier'],
                'stop_key': combination['stop_key'],
                'target_key': combination['target_key'],
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
                'avg_win_pips': 0.0,
                'avg_loss_pips': 0.0,
                'risk_reward_ratio': combination['target_multiplier'],
                'description': 'No trades executed',
                'trades': []
            }
        
        # Calculate base metrics (same as parent)
        timeframe_to_days = {
            '1D': 1, '2D': 2, '3D': 3, '4D': 4, '5D': 5,
            '1W': 7, '2W': 14, '3W': 21, '1M': 30,
            'H12': 0.5, 'H8': 0.33, 'H4': 0.17, 'H1': 0.04
        }
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('result') == 'WIN'])
        breakeven_trades = len([t for t in trades if t.get('result') == 'BREAKEVEN'])
        losing_trades = len([t for t in trades if t.get('result') == 'LOSS'])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
        be_rate = (breakeven_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        total_return = (total_pnl / 10000) * 100
        
        avg_duration_candles = np.mean([t.get('duration_days', 0) for t in trades])
        multiplier = timeframe_to_days.get(timeframe, 1)
        avg_duration_actual_days = avg_duration_candles * multiplier
        
        # ADDITIONAL optimization-specific metrics
        winning_pips = [t['pips'] for t in trades if t.get('result') == 'WIN']
        losing_pips = [t['pips'] for t in trades if t.get('result') == 'LOSS']
        
        avg_win_pips = np.mean(winning_pips) if winning_pips else 0
        avg_loss_pips = np.mean(losing_pips) if losing_pips else 0
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'combination': combination['name'],
           'stop_buffer': combination['stop_buffer'],
           'target_multiplier': combination['target_multiplier'],
           'stop_key': combination['stop_key'],
           'target_key': combination['target_key'],
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
           'avg_win_pips': round(avg_win_pips, 1),
           'avg_loss_pips': round(avg_loss_pips, 1),
           'risk_reward_ratio': combination['target_multiplier'],
           'validation_method': 'stop_target_optimization',
           'trades': trades
       }
   
    def run_parallel_combination_optimization(self, analysis_period: str = 'priority_1') -> List[Dict]:
        """
        Run combination optimization across ALL pairs/timeframes with parallel processing
        Tests all 30 combinations on each valid pair/timeframe combination
        """
        print(f"\n🚀 PARALLEL COMBINATION OPTIMIZATION - {analysis_period.upper()}")
        from core_backtest_engine import ANALYSIS_PERIODS
        period_config = ANALYSIS_PERIODS[analysis_period]
        days_back = period_config['days_back']
        
        print(f"📊 Period: {period_config['name']}")
        print(f"📅 Days back: {days_back:,}")
        print(f"🔄 Combinations per pair/timeframe: 30")
        print("=" * 70)
        
        # Discover valid data combinations
        valid_combinations = self.discover_valid_data_combinations()
        if not valid_combinations:
            print("❌ No valid data combinations found")
            return []
        
        # Create test combinations for parallel processing
        test_combinations = []
        for pair, timeframe in valid_combinations:
            test_combinations.append({
                'pair': pair,
                'timeframe': timeframe,
                'days_back': days_back,
                'analysis_period': analysis_period
            })
        
        print(f"📊 Valid pair/timeframe combinations: {len(valid_combinations)}")
        print(f"📊 Total optimization tests: {len(test_combinations) * 30:,}")
        
        # Run optimized parallel processing
        all_results = self.run_optimized_combination_parallel_tests(test_combinations)
        
        # Generate comprehensive Excel report
        if all_results:
            self.generate_combination_optimization_excel_report(all_results, analysis_period, period_config)
        
        return all_results
    
    def run_optimized_combination_parallel_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """
        Run combination optimization tests in parallel with memory management
        Each worker tests all 30 combinations for one pair/timeframe
        """
        print(f"\n🔄 OPTIMIZED PARALLEL COMBINATION TESTING")
        print(f"⚡ Workers: {self.max_workers}")
        print(f"📦 Chunk size: {self.chunk_size}")
        
        start_time = time.time()
        all_results = []
        
        # Process in chunks for memory management
        chunk_size = self.chunk_size
        total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(test_combinations))
            chunk_tests = test_combinations[chunk_start:chunk_end]
            
            print(f"\n📦 Chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_tests)} pair/timeframe combinations)")
            
            # Memory check
            memory_percent = psutil.virtual_memory().percent
            print(f"💾 Memory usage: {memory_percent:.1f}%")
            
            if memory_percent > self.memory_threshold * 100:
                print("⚠️  High memory usage, triggering cleanup...")
                gc.collect()
            
            # Process chunk with multiprocessing
            try:
                with Pool(processes=self.max_workers) as pool:
                    chunk_results = pool.map(run_combination_optimization_worker, chunk_tests)
                    
                    # Flatten results (each worker returns 30 combinations)
                    for worker_results in chunk_results:
                        all_results.extend(worker_results)
                
                # Progress tracking
                completed_combinations = chunk_end
                progress = (completed_combinations / len(test_combinations)) * 100
                total_tests_so_far = len(all_results)
                print(f"✅ Progress: {progress:.1f}% ({completed_combinations}/{len(test_combinations)} combinations)")
                print(f"   Total optimization tests completed: {total_tests_so_far:,}")
                
            except Exception as e:
                print(f"❌ Chunk {chunk_idx + 1} failed: {str(e)}")
                # Add empty results for failed chunk
                for test in chunk_tests:
                    for i in range(30):  # 30 combinations per test
                        all_results.append({
                            'pair': test['pair'],
                            'timeframe': test['timeframe'],
                            'combination': f'Failed_Combination_{i+1}',
                            'total_trades': 0,
                            'description': f"Parallel processing error: {str(e)}"
                        })
            
            # Memory cleanup after each chunk
            gc.collect()
        
        total_time = time.time() - start_time
        success_count = len([r for r in all_results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ PARALLEL COMBINATION OPTIMIZATION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🎯 Success rate: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")
        print(f"⚡ Speed: {len(all_results)/total_time:.1f} tests/second")
        
        return all_results
   
    def generate_combination_optimization_excel_report(self, all_results: List[Dict], 
                                                        analysis_period: str, period_config: Dict):
        """
        Generate comprehensive Excel report with combination optimization analysis
        
        Sheets:
        1. All_Results: Complete raw data
        2. Combination_Matrix: Performance by stop/target combinations
        3. Stop_Buffer_Analysis: Performance by stop buffer (averaged across targets)
        4. Target_Analysis: Performance by target level (averaged across stops)
        5. Heat_Map_Data: Profit factor matrix for visualization
        6. Top_Combinations: Best performers by different metrics
        7. Pair_Analysis: Performance by pair
        8. Timeframe_Analysis: Performance by timeframe
        """
        print(f"\n📊 GENERATING COMBINATION OPTIMIZATION EXCEL REPORT...")
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_results)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{analysis_period}_combination_optimization_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: All Results
                df_all.to_excel(writer, sheet_name='All_Results', index=False)
                print("   ✅ Sheet 1: All Results")
                
                # Filter successful results
                successful_df = df_all[df_all['total_trades'] > 0]
                
                if len(successful_df) > 0:
                    # SHEET 2: Combination Matrix
                    combination_matrix = self.create_combination_matrix_analysis(successful_df)
                    combination_matrix.to_excel(writer, sheet_name='Combination_Matrix', index=False)
                    print("   ✅ Sheet 2: Combination Matrix")
                    
                    # SHEET 3: Stop Buffer Analysis
                    stop_analysis = self.create_stop_buffer_analysis(successful_df)
                    stop_analysis.to_excel(writer, sheet_name='Stop_Buffer_Analysis', index=False)
                    print("   ✅ Sheet 3: Stop Buffer Analysis")
                    
                    # SHEET 4: Target Analysis
                    target_analysis = self.create_target_analysis(successful_df)
                    target_analysis.to_excel(writer, sheet_name='Target_Analysis', index=False)
                    print("   ✅ Sheet 4: Target Analysis")
                    
                    # SHEET 5: Heat Map Data
                    heat_map_data = self.create_heat_map_data(successful_df)
                    heat_map_data.to_excel(writer, sheet_name='Heat_Map_Data', index=False)
                    print("   ✅ Sheet 5: Heat Map Data")
                    
                    # SHEET 6: Top Combinations
                    top_combinations = self.create_top_combinations_analysis(successful_df)
                    top_combinations.to_excel(writer, sheet_name='Top_Combinations', index=False)
                    print("   ✅ Sheet 6: Top Combinations")
                    
                    # SHEET 7: Pair Analysis
                    pair_analysis = self.create_pair_analysis_optimized(successful_df)
                    pair_analysis.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ✅ Sheet 7: Pair Analysis")
                    
                    # SHEET 8: Timeframe Analysis
                    timeframe_analysis = self.create_timeframe_analysis_optimized(successful_df)
                    timeframe_analysis.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    print("   ✅ Sheet 8: Timeframe Analysis")
                    
                else:
                    # Create empty analysis sheets
                    empty_df = pd.DataFrame({'Note': ['No successful results to analyze']})
                    for sheet_name in ['Combination_Matrix', 'Stop_Buffer_Analysis', 'Target_Analysis', 
                                        'Heat_Map_Data', 'Top_Combinations', 'Pair_Analysis', 'Timeframe_Analysis']:
                        empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print("   ⚠️  Empty analysis sheets (no successful results)")
            
            print(f"\n📁 COMBINATION OPTIMIZATION EXCEL REPORT SAVED:")
            print(f"   File: {filename}")
            print(f"   📊 8 comprehensive analysis sheets created")
            print(f"   🎯 Ready for stop/target parameter optimization analysis")
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {str(e)}")
            # Fallback: Save as CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df_all.to_csv(csv_filename, index=False)
            print(f"📁 Fallback CSV saved: {csv_filename}")
    
    def create_combination_matrix_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combination performance matrix"""
        try:
            combo_analysis = df.groupby('combination').agg({
                'profit_factor': ['mean', 'count'],
                'win_rate': 'mean',
                'total_trades': 'sum',
                'total_return': 'mean',
                'avg_win_pips': 'mean',
                'avg_loss_pips': 'mean',
                'avg_trade_duration': 'mean',
                'stop_buffer': 'first',
                'target_multiplier': 'first'
            }).round(2)

            # Flatten column names
            combo_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                    'Total_Trades', 'Avg_Return', 'Avg_Win_Pips', 'Avg_Loss_Pips',
                                    'Avg_Duration_Days', 'Stop_Buffer_Pct', 'Target_Multiplier']
            combo_analysis = combo_analysis.sort_values('Avg_Profit_Factor', ascending=False)
            
            return combo_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Combination matrix analysis error: {str(e)}")
            return pd.DataFrame({'Combination': ['Error'], 'Note': [str(e)]})
    
    def create_stop_buffer_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create stop buffer performance analysis (averaged across all targets)"""
        try:
            stop_analysis = df.groupby('stop_key').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': ['mean', 'std'],
                'total_trades': ['sum', 'mean'],
                'total_return': ['mean', 'std'],
                'avg_trade_duration': ['mean', 'std'],
                'stop_buffer': 'first'
            }).round(2)

            # Flatten column names
            stop_analysis.columns = ['Avg_PF', 'StdDev_PF', 'Combination_Count', 'Avg_WR', 'StdDev_WR',
                                    'Total_Trades', 'Avg_Trades_Per_Combo', 'Avg_Return', 'StdDev_Return',
                                    'Avg_Duration', 'StdDev_Duration', 'Stop_Buffer_Pct']
            stop_analysis = stop_analysis.sort_values('Avg_PF', ascending=False)
            
            return stop_analysis.reset_index()
            
        except Exception as e:
            return pd.DataFrame({'Stop_Buffer': ['Error'], 'Note': [str(e)]})
   
    def create_target_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target level performance analysis (averaged across all stops)"""
        try:
            target_analysis = df.groupby('target_key').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': ['mean', 'std'],
                'total_trades': ['sum', 'mean'],
                'total_return': ['mean', 'std'],
                'avg_trade_duration': ['mean', 'std'],
                'target_multiplier': 'first'
            }).round(2)

            # Flatten column names
            target_analysis.columns = ['Avg_PF', 'StdDev_PF', 'Combination_Count', 'Avg_WR', 'StdDev_WR',
                                        'Total_Trades', 'Avg_Trades_Per_Combo', 'Avg_Return', 'StdDev_Return',
                                        'Avg_Duration', 'StdDev_Duration', 'Target_Multiplier']
            target_analysis = target_analysis.sort_values('Avg_PF', ascending=False)
            
            return target_analysis.reset_index()
            
        except Exception as e:
            return pd.DataFrame({'Target_Level': ['Error'], 'Note': [str(e)]})
   
    def create_heat_map_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create heat map data for stop/target combinations"""
        try:
            # Create pivot table for heat map visualization
            heat_map = df.groupby(['stop_key', 'target_key']).agg({
                'profit_factor': 'mean',
                'win_rate': 'mean',
                'total_trades': 'sum'
            }).round(2)
            
            # Reset index to make it a regular DataFrame
            heat_map = heat_map.reset_index()
            
            # Pivot for better visualization
            pf_pivot = heat_map.pivot(index='stop_key', columns='target_key', values='profit_factor')
            wr_pivot = heat_map.pivot(index='stop_key', columns='target_key', values='win_rate')
            trades_pivot = heat_map.pivot(index='stop_key', columns='target_key', values='total_trades')
            
            # Combine into single DataFrame with clear labels
            result_data = []
            
            # Add profit factor data
            result_data.append(['PROFIT_FACTOR_MATRIX', '', '', '', '', '', ''])
            result_data.append(['Stop\\Target'] + list(pf_pivot.columns))
            for stop_key in pf_pivot.index:
                row = [stop_key] + [pf_pivot.loc[stop_key, col] for col in pf_pivot.columns]
                result_data.append(row)
            
            result_data.append(['', '', '', '', '', '', ''])  # Separator
            
            # Add win rate data
            result_data.append(['WIN_RATE_MATRIX', '', '', '', '', '', ''])
            result_data.append(['Stop\\Target'] + list(wr_pivot.columns))
            for stop_key in wr_pivot.index:
                row = [stop_key] + [wr_pivot.loc[stop_key, col] for col in wr_pivot.columns]
                result_data.append(row)
            
            result_data.append(['', '', '', '', '', '', ''])  # Separator
            
            # Add trade count data
            result_data.append(['TRADE_COUNT_MATRIX', '', '', '', '', '', ''])
            result_data.append(['Stop\\Target'] + list(trades_pivot.columns))
            for stop_key in trades_pivot.index:
                row = [stop_key] + [trades_pivot.loc[stop_key, col] for col in trades_pivot.columns]
                result_data.append(row)
            
            # Convert to DataFrame
            max_cols = max(len(row) for row in result_data)
            for row in result_data:
                while len(row) < max_cols:
                    row.append('')
            
            columns = [f'Col_{i}' for i in range(max_cols)]
            heat_map_df = pd.DataFrame(result_data, columns=columns)
            
            return heat_map_df
            
        except Exception as e:
            return pd.DataFrame({'Heat_Map': ['Error'], 'Note': [str(e)]})
   
    def create_top_combinations_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create top combinations analysis by different metrics"""
        try:
            result_data = []
            
            # Top 5 by Profit Factor
            top_pf = df.nlargest(5, 'profit_factor')[['combination', 'profit_factor', 'win_rate', 'total_trades', 'total_return']]
            result_data.append(['TOP_5_BY_PROFIT_FACTOR', '', '', '', ''])
            result_data.append(['Combination', 'Profit_Factor', 'Win_Rate', 'Total_Trades', 'Return_Pct'])
            for _, row in top_pf.iterrows():
                result_data.append([row['combination'], row['profit_factor'], row['win_rate'], row['total_trades'], row['total_return']])
            
            result_data.append(['', '', '', '', ''])  # Separator
            
            # Top 5 by Win Rate
            top_wr = df.nlargest(5, 'win_rate')[['combination', 'win_rate', 'profit_factor', 'total_trades', 'total_return']]
            result_data.append(['TOP_5_BY_WIN_RATE', '', '', '', ''])
            result_data.append(['Combination', 'Win_Rate', 'Profit_Factor', 'Total_Trades', 'Return_Pct'])
            for _, row in top_wr.iterrows():
                result_data.append([row['combination'], row['win_rate'], row['profit_factor'], row['total_trades'], row['total_return']])
            
            result_data.append(['', '', '', '', ''])  # Separator
            
            # Top 5 by Total Return
            top_return = df.nlargest(5, 'total_return')[['combination', 'total_return', 'profit_factor', 'win_rate', 'total_trades']]
            result_data.append(['TOP_5_BY_RETURN', '', '', '', ''])
            result_data.append(['Combination', 'Return_Pct', 'Profit_Factor', 'Win_Rate', 'Total_Trades'])
            for _, row in top_return.iterrows():
                result_data.append([row['combination'], row['total_return'], row['profit_factor'], row['win_rate'], row['total_trades']])
            
            result_data.append(['', '', '', '', ''])  # Separator
            
            # Baseline comparison (33% stop, 2.5R target)
            baseline = df[(df['stop_key'] == '33pct') & (df['target_key'] == '2_5R')]
            if len(baseline) > 0:
                baseline_avg = baseline.agg({
                    'profit_factor': 'mean',
                    'win_rate': 'mean',
                    'total_trades': 'sum',
                    'total_return': 'mean'
                })
                result_data.append(['BASELINE_PERFORMANCE_33pct_2_5R', '', '', '', ''])
                result_data.append(['Metric', 'Value', '', '', ''])
                result_data.append(['Avg_Profit_Factor', baseline_avg['profit_factor'], '', '', ''])
                result_data.append(['Avg_Win_Rate', baseline_avg['win_rate'], '', '', ''])
                result_data.append(['Total_Trades', baseline_avg['total_trades'], '', '', ''])
                result_data.append(['Avg_Return', baseline_avg['total_return'], '', '', ''])
            
            # Convert to DataFrame
            columns = ['Col_1', 'Col_2', 'Col_3', 'Col_4', 'Col_5']
            top_combos_df = pd.DataFrame(result_data, columns=columns)
            
            return top_combos_df
            
        except Exception as e:
            return pd.DataFrame({'Top_Combinations': ['Error'], 'Note': [str(e)]})
   
    def create_pair_analysis_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pair performance analysis with combination optimization data"""
        try:
            pair_analysis = df.groupby('pair').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': ['mean', 'std'],
                'total_trades': ['sum', 'mean'],
                'total_return': ['mean', 'std'],
                'avg_trade_duration': ['mean', 'std']
            }).round(2)

            # Flatten column names
            pair_analysis.columns = ['Avg_PF', 'StdDev_PF', 'Combination_Count', 'Avg_WR', 'StdDev_WR',
                                    'Total_Trades', 'Avg_Trades_Per_Combo', 'Avg_Return', 'StdDev_Return',
                                    'Avg_Duration', 'StdDev_Duration']
            pair_analysis = pair_analysis.sort_values('Avg_PF', ascending=False)
            
            return pair_analysis.reset_index()
            
        except Exception as e:
            return pd.DataFrame({'Pair': ['Error'], 'Note': [str(e)]})
   
    def create_timeframe_analysis_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create timeframe performance analysis with combination optimization data"""
        try:
            tf_analysis = df.groupby('timeframe').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': ['mean', 'std'],
                'total_trades': ['sum', 'mean'],
                'total_return': ['mean', 'std'],
                'avg_trade_duration': ['mean', 'std']
            }).round(2)

            # Flatten column names
            tf_analysis.columns = ['Avg_PF', 'StdDev_PF', 'Combination_Count', 'Avg_WR', 'StdDev_WR',
                                    'Total_Trades', 'Avg_Trades_Per_Combo', 'Avg_Return', 'StdDev_Return',
                                    'Avg_Duration', 'StdDev_Duration']
            tf_analysis = tf_analysis.sort_values('Avg_PF', ascending=False)
            
            return tf_analysis.reset_index()
            
        except Exception as e:
            return pd.DataFrame({'Timeframe': ['Error'], 'Note': [str(e)]})

    def generate_single_pair_optimization_report(self, results: List[Dict], pair: str, timeframe: str):
        """
        Generate comprehensive Excel report for single pair optimization analysis
        Perfect for detailed parameter optimization analysis
        """
        print(f"\n📊 GENERATING SINGLE PAIR OPTIMIZATION REPORT...")
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/single_pair_optimization_{pair}_{timeframe}_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            # Convert to DataFrame
            df_all = pd.DataFrame(results)
            successful_df = df_all[df_all['total_trades'] > 0]
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: All Combinations Summary
                summary_data = []
                for result in results:
                    summary_data.append({
                        'Combination': result['combination'],
                        'Stop_Buffer': f"{result.get('stop_buffer', 0)*100:.0f}%",
                        'Target_Multiplier': f"{result.get('target_multiplier', 0):.1f}R",
                        'Total_Trades': result['total_trades'],
                        'Win_Rate': f"{result['win_rate']:.1f}%",
                        'Profit_Factor': result['profit_factor'],
                        'Total_Return': f"{result['total_return']:.2f}%",
                        'Avg_Duration': f"{result.get('avg_trade_duration', 0):.1f} days",
                        'Status': 'SUCCESS' if result['total_trades'] > 0 else 'NO_TRADES'
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='All_Combinations', index=False)
                print("   ✅ Sheet 1: All Combinations Summary")
                
                # SHEET 2: ALL TRADES FOR MANUAL CHART VALIDATION
                print("   🔄 Collecting all trades for manual validation...")
                all_trades_for_validation = []
                trade_counter = 1
                
                for result in results:
                    if result['total_trades'] > 0 and 'trades' in result:
                        for trade in result['trades']:
                            all_trades_for_validation.append({
                                'Trade_Number': trade_counter,
                                'Combination': result['combination'],
                                'Zone_Type': trade['zone_type'],
                                'Direction': trade['direction'],
                                'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                                'Exit_Date': trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date']),
                                'Entry_Price': f"{trade['entry_price']:.6f}",
                                'Exit_Price': f"{trade['exit_price']:.6f}",
                                'Zone_High': f"{trade.get('zone_high', 0):.6f}",
                                'Zone_Low': f"{trade.get('zone_low', 0):.6f}",
                                'Target_Price': f"{trade.get('target_price', 0):.6f}",
                                'Initial_Stop': f"{trade.get('initial_stop', 0):.6f}",
                                'Duration': trade['duration_days'],
                                'Result': trade['result'],
                                'Pips': f"{trade['pips']:+.1f}",
                                'PnL': f"${trade['pnl']:.2f}",
                                'Position_Size': f"{trade['position_size']:.4f}",
                                'Commission_Cost': f"${trade['commission_cost']:.2f}",
                                'Breakeven_Moved': 'Yes' if trade['breakeven_moved'] else 'No',
                                'Stop_Buffer': f"{result.get('stop_buffer', 0)*100:.0f}%",
                                'Target_Multiplier': f"{result.get('target_multiplier', 0):.1f}R"
                            })
                            trade_counter += 1
                
                if all_trades_for_validation:
                    trades_validation_df = pd.DataFrame(all_trades_for_validation)
                    trades_validation_df.to_excel(writer, sheet_name='All_Trades_Chart_Validation', index=False)
                    print(f"   ✅ Sheet 2: All Trades for Chart Validation ({len(all_trades_for_validation)} trades)")
                else:
                    empty_df = pd.DataFrame({'Note': ['No trades found across all combinations']})
                    empty_df.to_excel(writer, sheet_name='All_Trades_Chart_Validation', index=False)
                    print("   ⚠️  Sheet 2: No trades found for validation")
                
                if len(successful_df) > 0:
                    # SHEET 3: Performance Rankings
                    rankings_data = []
                    
                    # Top 5 by Profit Factor
                    top_pf = successful_df.nlargest(5, 'profit_factor')
                    rankings_data.append(['TOP_5_BY_PROFIT_FACTOR', '', '', '', ''])
                    rankings_data.append(['Rank', 'Combination', 'Profit_Factor', 'Win_Rate', 'Return'])
                    for i, (_, row) in enumerate(top_pf.iterrows(), 1):
                        rankings_data.append([i, row['combination'], row['profit_factor'], 
                                            f"{row['win_rate']:.1f}%", f"{row['total_return']:.2f}%"])
                    
                    rankings_data.append(['', '', '', '', ''])
                    
                    # Top 5 by Win Rate
                    top_wr = successful_df.nlargest(5, 'win_rate')
                    rankings_data.append(['TOP_5_BY_WIN_RATE', '', '', '', ''])
                    rankings_data.append(['Rank', 'Combination', 'Win_Rate', 'Profit_Factor', 'Return'])
                    for i, (_, row) in enumerate(top_wr.iterrows(), 1):
                        rankings_data.append([i, row['combination'], f"{row['win_rate']:.1f}%", 
                                            row['profit_factor'], f"{row['total_return']:.2f}%"])
                    
                    rankings_data.append(['', '', '', '', ''])
                    
                    # Baseline Performance
                    baseline = successful_df[successful_df['combination'] == 'Stop_33pct_Target_2_5R']
                    if len(baseline) > 0:
                        baseline_row = baseline.iloc[0]
                        rankings_data.append(['BASELINE_PERFORMANCE', '', '', '', ''])
                        rankings_data.append(['Metric', 'Value', '', '', ''])
                        rankings_data.append(['Combination', baseline_row['combination'], '', '', ''])
                        rankings_data.append(['Profit_Factor', baseline_row['profit_factor'], '', '', ''])
                        rankings_data.append(['Win_Rate', f"{baseline_row['win_rate']:.1f}%", '', '', ''])
                        rankings_data.append(['Total_Return', f"{baseline_row['total_return']:.2f}%", '', '', ''])
                        rankings_data.append(['Total_Trades', baseline_row['total_trades'], '', '', ''])
                    
                    rankings_df = pd.DataFrame(rankings_data, columns=['Col_1', 'Col_2', 'Col_3', 'Col_4', 'Col_5'])
                    rankings_df.to_excel(writer, sheet_name='Performance_Rankings', index=False)
                    print("   ✅ Sheet 3: Performance Rankings")
                    
                    # SHEET 4: Stop Buffer Analysis
                    if 'stop_key' in successful_df.columns:
                        stop_analysis = successful_df.groupby('stop_key').agg({
                            'profit_factor': ['mean', 'std', 'count'],
                            'win_rate': ['mean', 'std'],
                            'total_trades': 'sum',
                            'total_return': ['mean', 'std']
                        }).round(2)
                        
                        stop_analysis.columns = ['Avg_PF', 'StdDev_PF', 'Count', 'Avg_WR', 'StdDev_WR', 
                                            'Total_Trades', 'Avg_Return', 'StdDev_Return']
                        stop_analysis = stop_analysis.reset_index()
                        stop_analysis.to_excel(writer, sheet_name='Stop_Buffer_Analysis', index=False)
                        print("   ✅ Sheet 4: Stop Buffer Analysis")
                    
                    # SHEET 5: Target Analysis
                    if 'target_key' in successful_df.columns:
                        target_analysis = successful_df.groupby('target_key').agg({
                            'profit_factor': ['mean', 'std', 'count'],
                            'win_rate': ['mean', 'std'],
                            'total_trades': 'sum',
                            'total_return': ['mean', 'std']
                        }).round(2)
                        
                        target_analysis.columns = ['Avg_PF', 'StdDev_PF', 'Count', 'Avg_WR', 'StdDev_WR', 
                                                'Total_Trades', 'Avg_Return', 'StdDev_Return']
                        target_analysis = target_analysis.reset_index()
                        target_analysis.to_excel(writer, sheet_name='Target_Analysis', index=False)
                        print("   ✅ Sheet 5: Target Analysis")
                    
                    # SHEET 6: Best Combination Trade Details
                    best_combo = successful_df.loc[successful_df['profit_factor'].idxmax()]
                    if 'trades' in best_combo and best_combo['trades']:
                        trade_details = []
                        for i, trade in enumerate(best_combo['trades'], 1):
                            trade_details.append({
                                'Trade_Number': i,
                                'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                                'Exit_Date': trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date']),
                                'Zone_Type': trade['zone_type'],
                                'Direction': trade['direction'],
                                'Entry_Price': f"{trade['entry_price']:.6f}",
                                'Exit_Price': f"{trade['exit_price']:.6f}",
                                'Target_Price': f"{trade.get('target_price', 0):.6f}",
                                'Initial_Stop': f"{trade.get('initial_stop', 0):.6f}",
                                'Result': trade['result'],
                                'Pips': f"{trade['pips']:+.1f}",
                                'PnL': f"${trade['pnl']:.2f}",
                                'Duration_Days': trade['duration_days'],
                                'Breakeven_Moved': 'Yes' if trade['breakeven_moved'] else 'No'
                            })
                        
                        trades_df = pd.DataFrame(trade_details)
                        trades_df.to_excel(writer, sheet_name='Best_Combo_Trades', index=False)
                        print(f"   ✅ Sheet 6: Best Combination Trade Details ({best_combo['combination']})")
                
                else:
                    # No successful combinations
                    empty_df = pd.DataFrame({'Note': ['No successful combinations found for this pair/timeframe']})
                    for sheet_name in ['Performance_Rankings', 'Stop_Buffer_Analysis', 'Target_Analysis', 'Best_Combo_Trades']:
                        empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print("   ⚠️  Empty analysis sheets (no successful combinations)")
            
            print(f"\n📁 SINGLE PAIR OPTIMIZATION REPORT SAVED:")
            print(f"   File: {filename}")
            print(f"   📊 Comprehensive parameter optimization analysis")
            print(f"   🎯 ALL TRADES included for manual chart validation in Sheet 2")
            print(f"   📈 Use Sheet 2 to verify each trade against your TradingView charts")
            
        except Exception as e:
            print(f"❌ Error creating single pair optimization report: {str(e)}")
            # Fallback: Save basic CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df_all = pd.DataFrame(results)
            df_all.to_csv(csv_filename, index=False)
            print(f"📁 Fallback CSV saved: {csv_filename}")

# ============================================================================
# PARALLEL PROCESSING WORKER FUNCTION FOR COMBINATION OPTIMIZATION
# ============================================================================

def run_combination_optimization_worker(test_config: Dict) -> List[Dict]:
    """
    Worker function for parallel combination optimization
    Each worker tests ALL 30 combinations for one pair/timeframe
    """
    try:
        # Create fresh optimization engine instance for this worker
        engine = StopTargetOptimizationEngine()
        
        # Run all 30 combinations for this pair/timeframe
        results = engine.run_combination_optimization_test(
            test_config['pair'],
            test_config['timeframe'],
            test_config['days_back']
        )
        
        # Add analysis period info to each result
        for result in results:
            result['analysis_period'] = test_config['analysis_period']
        
        # Clean up
        del engine
        gc.collect()
        
        return results
        
    except Exception as e:
        gc.collect()
        # Return 30 empty results (one for each combination)
        empty_results = []
        for i in range(30):
            empty_results.append({
                'pair': test_config['pair'],
                'timeframe': test_config['timeframe'],
                'combination': f'Failed_Combination_{i+1}',
                'analysis_period': test_config['analysis_period'],
                'total_trades': 0,
                'description': f"Worker error: {str(e)}"
            })
        return empty_results

def main():
    """Main function for stop/target optimization analysis"""
    print("🎯 STOP/TARGET OPTIMIZATION ENGINE")
    print("=" * 60)
    
    # Check system resources
    engine = StopTargetOptimizationEngine()
    if not engine.check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    print("\n🎯 SELECT OPTIMIZATION MODE:")
    print("1. Quick Test (Single pair - EURUSD 3D, all 30 combinations)")
    print("2. Full Optimization - Priority 1 (2015-2025, All pairs/timeframes, all combinations)")
    print("3. Full Optimization - Priority 2 (2020-2025, All pairs/timeframes, all combinations)")
    print("4. Custom Single Pair Optimization")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Quick optimization test
        print("\n🧪 QUICK OPTIMIZATION TEST:")
        print("Testing EURUSD 3D with all 30 stop/target combinations...")
        
        start_time = time.time()
        results = engine.run_combination_optimization_test('EURUSD', '3D', 730)
        end_time = time.time()
        
        print(f"\n🕐 OPTIMIZATION TIME: {end_time - start_time:.1f} seconds")
        
        if results:
            successful_results = [r for r in results if r['total_trades'] > 0]
            print(f"\n📊 OPTIMIZATION RESULTS:")
            print(f"   Successful combinations: {len(successful_results)}/30")
            
            if successful_results:
                # Best performers
                best_pf = max(successful_results, key=lambda x: x['profit_factor'])
                best_wr = max(successful_results, key=lambda x: x['win_rate'])
                baseline = next((r for r in successful_results if r['combination'] == 'Stop_33pct_Target_2_5R'), None)
                
                print(f"   🏆 Best PF: {best_pf['combination']} - {best_pf['profit_factor']:.2f}")
                print(f"   🎯 Best WR: {best_wr['combination']} - {best_wr['win_rate']:.1f}%")
                if baseline:
                    print(f"   📊 Baseline: {baseline['combination']} - PF {baseline['profit_factor']:.2f}, WR {baseline['win_rate']:.1f}%")
                
                # Generate single-pair optimization report
                engine.generate_single_pair_optimization_report(results, 'EURUSD', '3D')
            else:
                print("   ❌ No successful combinations found")
        else:
            print("   ❌ No results generated")
    
    elif choice == '2':
        # Full optimization - Priority 1
        print("\n🚀 FULL OPTIMIZATION - PRIORITY 1")
        print("This will test ALL 30 combinations on ALL pairs and timeframes with 10 years of data")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_parallel_combination_optimization('priority_1')
        else:
            print("Optimization cancelled")
    
    elif choice == '3':
        # Full optimization - Priority 2
        print("\n🚀 FULL OPTIMIZATION - PRIORITY 2")
        print("This will test ALL 30 combinations on ALL pairs and timeframes with 4 years of data")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_parallel_combination_optimization('priority_2')
        else:
            print("Optimization cancelled")
    
    elif choice == '4':
        # Custom optimization
        print("\n🎯 CUSTOM SINGLE PAIR OPTIMIZATION:")
        pairs = engine.discover_all_pairs()
        print(f"Available pairs: {', '.join(pairs[:10])}..." if len(pairs) > 10 else f"Available pairs: {', '.join(pairs)}")
        
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        days_back = input("Enter days back (e.g., 730): ").strip()
        
        try:
            days_back = int(days_back)
            results = engine.run_combination_optimization_test(pair, timeframe, days_back)
            
            if results:
                successful_results = [r for r in results if r['total_trades'] > 0]
                print(f"\n📊 CUSTOM OPTIMIZATION RESULTS:")
                print(f"   Successful combinations: {len(successful_results)}/30")
                
                if successful_results:
                    # Best performers
                    best_pf = max(successful_results, key=lambda x: x['profit_factor'])
                    best_wr = max(successful_results, key=lambda x: x['win_rate'])
                    best_return = max(successful_results, key=lambda x: x['total_return'])
                    baseline = next((r for r in successful_results if r['combination'] == 'Stop_33pct_Target_2_5R'), None)
                    
                    print(f"   🏆 Best PF: {best_pf['combination']} - {best_pf['profit_factor']:.2f}")
                    print(f"   🎯 Best WR: {best_wr['combination']} - {best_wr['win_rate']:.1f}%")
                    print(f"   💰 Best Return: {best_return['combination']} - {best_return['total_return']:.2f}%")
                    if baseline:
                        print(f"   📊 Baseline: {baseline['combination']} - PF {baseline['profit_factor']:.2f}, WR {baseline['win_rate']:.1f}%")
                    
                    # Generate comprehensive single-pair optimization report
                    engine.generate_single_pair_optimization_report(results, pair, timeframe)
                else:
                    print("   ❌ No successful combinations found")
            else:
                print("   ❌ No results generated")
                
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()