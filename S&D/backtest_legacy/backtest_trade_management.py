"""
Advanced Trade Management Backtesting System
Tests multiple exit strategies across all timeframes and pairs
Includes volatility-adjusted exits and comprehensive Excel reporting
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager

class ATRCalculator:
    """Calculate Average True Range for volatility-adjusted exits"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR using standard formula"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = true_range.rolling(window=self.period).mean()
        
        return atr

class TradeManagementBacktester:
    """Comprehensive trade management backtesting system"""
    
    # TRADE MANAGEMENT CONFIGURATIONS
    MANAGEMENT_STRATEGIES = {
        # 1. SIMPLE STRATEGIES
        'Simple_1R': {
            'type': 'simple',
            'target': 1.0,
            'breakeven_at': None,
            'trailing': False,
            'description': 'Take profit at 1:1 R/R only'
        },
        
        'Simple_2R': {
            'type': 'simple', 
            'target': 2.0,
            'breakeven_at': None,
            'trailing': False,
            'description': 'Take profit at 1:2 R/R only'
        },
        
        'Simple_3R': {
            'type': 'simple',
            'target': 3.0,
            'breakeven_at': None,
            'trailing': False,
            'description': 'Take profit at 1:3 R/R only'
        },
        
        # 2. BREAK-EVEN STRATEGIES
        'BE_1R_TP_2R': {  # Your current baseline
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 2.0,
            'trailing': False,
            'description': 'Break-even at 1:1, TP at 1:2'
        },
        
        'BE_1R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 3.0,
            'trailing': False,
            'description': 'Break-even at 1:1, TP at 1:3'
        },
        
        'BE_2R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 2.0,
            'target': 3.0,
            'trailing': False,
            'description': 'Break-even at 1:2, TP at 1:3'
        },
        
        # 3. ZONE TRAILING STRATEGIES
        'Trail_Immediate_Both': {
            'type': 'zone_trailing',
            'breakeven_at': None,
            'target': None,
            'trailing': 'immediate',
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail stops with new zones from entry'
        },
        
        'Trail_1R_Both': {
            'type': 'zone_trailing',
            'breakeven_at': None,
            'target': None,
            'trailing': 'after_1R',
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail stops with new zones after 1:1'
        },
        
        'Trail_2R_Both': {
            'type': 'zone_trailing',
            'breakeven_at': None,
            'target': None,
            'trailing': 'after_2R',
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail stops with new zones after 1:2'
        },
        
        'Trail_1R_Momentum': {
            'type': 'zone_trailing',
            'breakeven_at': None,
            'target': None,
            'trailing': 'after_1R',
            'trail_zone_types': ['momentum'],
            'min_trail_distance': 2.5,
            'description': 'Trail stops with momentum zones after 1:1'
        },
        
        'Trail_1R_Reversal': {
            'type': 'zone_trailing',
            'breakeven_at': None,
            'target': None,
            'trailing': 'after_1R',
            'trail_zone_types': ['reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail stops with reversal zones after 1:1'
        },
        
        # 4. HYBRID STRATEGIES
        'BE_1R_Trail_Both': {
            'type': 'hybrid',
            'breakeven_at': 1.0,
            'target': None,
            'trailing': 'after_be',
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Break-even at 1:1, then trail with zones'
        },
        
        'BE_2R_Trail_Both': {
            'type': 'hybrid',
            'breakeven_at': 2.0,
            'target': None,
            'trailing': 'after_be',
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Break-even at 1:2, then trail with zones'
        },
        
        # 5. VOLATILITY-ADJUSTED STRATEGIES
        'ATR_Trail_2x': {
            'type': 'atr_trailing',
            'breakeven_at': 1.0,
            'target': None,
            'atr_multiplier': 2.0,
            'atr_period': 14,
            'description': 'Break-even at 1:1, then trail with 2x ATR'
        },
        
        'ATR_Trail_3x': {
            'type': 'atr_trailing',
            'breakeven_at': 1.0,
            'target': None,
            'atr_multiplier': 3.0,
            'atr_period': 14,
            'description': 'Break-even at 1:1, then trail with 3x ATR'
        },
        
        'ATR_Immediate_2x': {
            'type': 'atr_trailing',
            'breakeven_at': None,
            'target': None,
            'atr_multiplier': 2.0,
            'atr_period': 14,
            'description': 'Trail with 2x ATR from entry'
        },
        
        # 6. TIME-BASED STRATEGIES
        'BE_1R_TP_2R_Max_30D': {
            'type': 'time_limited',
            'breakeven_at': 1.0,
            'target': 2.0,
            'max_hold_days': 30,
            'trailing': False,
            'description': 'BE at 1:1, TP at 1:2, max 30 days'
        },
        
        'Trail_1R_Both_Max_45D': {
            'type': 'time_limited_trailing',
            'breakeven_at': None,
            'target': None,
            'trailing': 'after_1R',
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'max_hold_days': 45,
            'description': 'Trail after 1:1, max 45 days'
        }
    }
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.atr_calculator = ATRCalculator()
        self.results = []
        
    def get_all_available_data_files(self) -> List[Dict]:
        """Scan data directory for all available pairs and timeframes"""
        
        data_path = self.data_loader.raw_path
        if not os.path.exists(data_path):
            print(f"❌ Data path not found: {data_path}")
            return []
        
        # Scan for all CSV files
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        
        available_data = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Parse different file formats
            # Format 1: OANDA_EURUSD, 2D_*.csv
            # Format 2: EURUSD.raw_2D_*.csv
            # Format 3: EURUSD_2D.csv
            
            pair = None
            timeframe = None
            
            # Pattern matching for your file formats
            if 'OANDA_' in filename and ', ' in filename:
                # OANDA_EURUSD, 2D_*.csv format
                parts = filename.replace('OANDA_', '').replace('.csv', '').split(', ')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1].split('_')[0]  # Get 2D from 2D_something
                    
            elif '.raw_' in filename:
                # EURUSD.raw_2D_*.csv format
                parts = filename.split('.raw_')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1].split('_')[0]  # Get 2D from 2D_something
                    
            elif '_' in filename:
                # EURUSD_2D.csv format
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1]
            
            if pair and timeframe:
                # Normalize timeframe names
                timeframe_map = {
                    '1D': '1D', 'Daily': '1D',
                    '2D': '2D', '2Daily': '2D',
                    '3D': '3D', '3Daily': '3D', 
                    '4D': '4D', '4Daily': '4D',
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
        
        # Sort by pair, then timeframe
        unique_data.sort(key=lambda x: (x['pair'], x['timeframe']))
        
        return unique_data
    
    def run_single_backtest(self, pair: str, timeframe: str, strategy_name: str,
                          days_back: int = 730) -> Dict:
        """Run backtest for single pair/timeframe/strategy combination"""
        
        try:
            # Load data using your existing loader
            if timeframe == '1D':
                data = self.data_loader.load_pair_data(pair, 'Daily')
            elif timeframe == '2D':
                data = self.data_loader.load_pair_data(pair, '2Daily')
            elif timeframe == '3D':
                data = self.data_loader.load_pair_data(pair, '3Daily')
            elif timeframe == '4D':
                data = self.data_loader.load_pair_data(pair, '4Daily')
            else:
                data = self.data_loader.load_pair_data(pair, timeframe)
            
            if len(data) < 100:
                return self.empty_result(pair, timeframe, strategy_name, "Insufficient data")
                
            # Limit data to test period (unless using all data)
            if days_back < 99999 and days_back < len(data):
                data = data.iloc[-days_back:]
                print(f"   📊 Using last {days_back} days ({len(data)} candles)")
            else:
                print(f"   📊 Using ALL available data ({len(data)} candles)")
            
            # Initialize your existing trading components
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()               
            risk_manager = RiskManager(account_balance=10000)
            
            # Get strategy configuration
            strategy_config = self.MANAGEMENT_STRATEGIES[strategy_name]
            
            # Run backtest with enhanced trade management
            results = self.backtest_with_management(
                data, patterns, trend_data, risk_manager,
                strategy_config, pair, timeframe, strategy_name
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error testing {pair} {timeframe} {strategy_name}: {e}")
            return self.empty_result(pair, timeframe, strategy_name, str(e))
    
    def backtest_with_management(self, data: pd.DataFrame, patterns: Dict,
                             trend_data: pd.DataFrame, risk_manager: RiskManager,
                             strategy_config: Dict, pair: str, timeframe: str,
                             strategy_name: str) -> Dict:
        """Enhanced backtesting with proven zone selection and trade management"""

        # Step 1: Combine all patterns (momentum + reversal)
        momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
        reversal_patterns = patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', [])
        all_patterns = momentum_patterns + reversal_patterns

        # Step 2: Cumulative distance filtering (like distance_edge.py)
        valid_patterns = []
        for pattern in all_patterns:
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']:
                if pattern['leg_out']['ratio_to_base'] >= 2.0:  # Replace with 2.5 if preferred
                    valid_patterns.append(pattern)

        if not valid_patterns:
            return self.empty_result(pair, timeframe, strategy_name, "No valid patterns")

        # Step 3: Build zone activation schedule (chronological processing)
        zone_activation_schedule = []
        for pattern in valid_patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(data):
                activation_date = data.index[zone_end_idx]
                zone_activation_schedule.append({
                    'date': activation_date,
                    'pattern': pattern,
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
                })

        # Step 4: Sort by activation date (CRUCIAL)
        zone_activation_schedule.sort(key=lambda x: x['date'])

        # Step 5: Initialize tracking
        trades = []
        account_balance = 10000
        used_zones = set()

        # Step 6: ATR Calculation (if needed)
        atr_data = self.atr_calculator.calculate_atr(data) if strategy_config['type'] in ['atr_trailing', 'atr_immediate'] else None

        # Step 7: Loop through zones chronologically
        for zone_info in zone_activation_schedule:
            pattern = zone_info['pattern']
            zone_id = zone_info['zone_id']

            if zone_id in used_zones:
                continue

            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is None or zone_end_idx >= len(trend_data):
                continue

            # Step 8: Trend alignment logic
            current_trend = trend_data['trend'].iloc[zone_end_idx]
            is_aligned = (
                (pattern['type'] in ['R-B-R', 'D-B-R'] and current_trend == 'bullish') or
                (pattern['type'] in ['D-B-D', 'R-B-D'] and current_trend == 'bearish')
            )

            if not is_aligned:
                continue

            # Step 9: Execute trade
            trade_result = self.execute_trade_with_management(
                pattern, data, strategy_config, atr_data, zone_end_idx
            )

            if trade_result:
                trades.append(trade_result)
                account_balance += trade_result['pnl']
                used_zones.add(zone_id)

        # Step 10: Final performance calculation
        return self.calculate_enhanced_performance(
            trades, account_balance, pair, timeframe, strategy_name, strategy_config
        )

    
    def execute_trade_with_management(self, pattern: Dict, data: pd.DataFrame,
                                  strategy_config: Dict, atr_data: Optional[pd.Series],
                                  zone_end_idx: int) -> Optional[Dict]:
        """Execute trade using proven distance_edge.py logic"""
        
        # Extract zone boundaries
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        zone_range = zone_high - zone_low

        # Step 1: Entry and stop logic (5% front-run, 33% buffer)
        if pattern['type'] in ['R-B-R', 'D-B-R']:  # BUY
            entry_price = zone_low + (zone_range * 0.05)
            direction = 'BUY'
            initial_stop = zone_low - (zone_range * 0.33)
        else:  # SELL
            entry_price = zone_high - (zone_range * 0.05)
            direction = 'SELL'
            initial_stop = zone_high + (zone_range * 0.33)

        # Step 2: Fixed $500 risk per trade
        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            return None  # Invalid stop distance

        risk_amount = 500
        position_size = risk_amount / stop_distance

        # Step 3: Entry scanning logic (first candle to touch entry)
        entry_date = None
        entry_idx = None

        for i in range(zone_end_idx + 1, len(data)):
            candle = data.iloc[i]

            if direction == 'BUY' and candle['low'] <= entry_price:
                entry_date = data.index[i]
                entry_idx = i
                break
            elif direction == 'SELL' and candle['high'] >= entry_price:
                entry_date = data.index[i]
                entry_idx = i
                break

        if entry_idx is None:
            return None  # Entry never triggered

        # Step 4: Simulate trade with your management logic
        return self.simulate_enhanced_trade_outcome(
            pattern, entry_price, initial_stop, entry_idx, data,
            strategy_config, atr_data, direction, position_size, risk_amount
        )

    
    def simulate_enhanced_trade_outcome(self, zone: Dict, entry_price: float,
                                      initial_stop: float, entry_idx: int,
                                      data: pd.DataFrame, strategy_config: Dict,
                                      atr_data: Optional[pd.Series], direction: str,
                                      position_size: float, risk_amount: float) -> Dict:
        """Simulate trade with advanced management strategies"""
        
        # Initialize trade state
        current_stop = initial_stop
        current_target = None
        at_breakeven = False
        trailing_active = False
        trade_locked_profit = False
        
        # Calculate risk distance for R:R calculations
        risk_distance = abs(entry_price - initial_stop)
        
        # Set up strategy-specific parameters
        strategy_type = strategy_config['type']
        
        if strategy_type in ['simple', 'breakeven', 'time_limited']:
            if 'target' in strategy_config and strategy_config['target']:
                if direction == 'BUY':
                    current_target = entry_price + (risk_distance * strategy_config['target'])
                else:
                    current_target = entry_price - (risk_distance * strategy_config['target'])
        
        # Simulate trade progression
        for i in range(entry_idx + 1, len(data)):
            candle = data.iloc[i]
            current_date = data.index[i]
            days_held = i - entry_idx
            
            # Time-based exit check
            if 'max_hold_days' in strategy_config:
                if days_held >= strategy_config['max_hold_days']:
                    # Force exit at current price
                    exit_price = candle['close']
                    pnl = (exit_price - entry_price) * position_size if direction == 'BUY' else (entry_price - exit_price) * position_size
                    
                    return {
                        'entry_date': data.index[entry_idx],
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'direction': direction,
                        'pnl': pnl,
                        'exit_reason': 'time_limit',
                        'days_held': days_held,
                        'strategy': strategy_config['description']
                    }
            
            # Check stop loss hit
            if direction == 'BUY' and candle['low'] <= current_stop:
                pnl = (current_stop - entry_price) * position_size
                return {
                    'entry_date': data.index[entry_idx],
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_stop,
                    'direction': direction,
                    'pnl': pnl,
                    'exit_reason': 'stop_loss',
                    'days_held': days_held,
                    'strategy': strategy_config['description']
                }
            
            elif direction == 'SELL' and candle['high'] >= current_stop:
                pnl = (entry_price - current_stop) * position_size
                return {
                    'entry_date': data.index[entry_idx],
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_stop,
                    'direction': direction,
                    'pnl': pnl,
                    'exit_reason': 'stop_loss',
                    'days_held': days_held,
                    'strategy': strategy_config['description']
                }
            
            # Check target hit (simple strategies)
            if current_target is not None:
                if direction == 'BUY' and candle['high'] >= current_target:
                    pnl = (current_target - entry_price) * position_size
                    return {
                        'entry_date': data.index[entry_idx],
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': current_target,
                        'direction': direction,
                        'pnl': pnl,
                        'exit_reason': 'take_profit',
                        'days_held': days_held,
                        'strategy': strategy_config['description']
                    }
                
                elif direction == 'SELL' and candle['low'] <= current_target:
                    pnl = (entry_price - current_target) * position_size
                    return {
                        'entry_date': data.index[entry_idx],
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': current_target,
                        'direction': direction,
                        'pnl': pnl,
                        'exit_reason': 'take_profit',
                        'days_held': days_held,
                        'strategy': strategy_config['description']
                    }
            
            # Break-even management
            if 'breakeven_at' in strategy_config and strategy_config['breakeven_at'] and not at_breakeven:
                breakeven_level = strategy_config['breakeven_at']
                
                if direction == 'BUY':
                    breakeven_trigger = entry_price + (risk_distance * breakeven_level)
                    if candle['high'] >= breakeven_trigger:
                        current_stop = entry_price
                        at_breakeven = True
                else:
                    breakeven_trigger = entry_price - (risk_distance * breakeven_level)
                    if candle['low'] <= breakeven_trigger:
                        current_stop = entry_price
                        at_breakeven = True
            
            # ATR Trailing Logic
            if strategy_type in ['atr_trailing'] and atr_data is not None:
                should_trail = False
                
                if 'breakeven_at' not in strategy_config or strategy_config['breakeven_at'] is None:
                    should_trail = True  # Trail immediately
                elif at_breakeven:
                    should_trail = True  # Trail after break-even
                
                if should_trail and i < len(atr_data):
                    atr_value = atr_data.iloc[i]
                    atr_multiplier = strategy_config.get('atr_multiplier', 2.0)
                    
                    if direction == 'BUY':
                        atr_stop = candle['high'] - (atr_value * atr_multiplier)
                        if atr_stop > current_stop:
                            current_stop = atr_stop
                    else:
                        atr_stop = candle['low'] + (atr_value * atr_multiplier)
                        if atr_stop < current_stop:
                            current_stop = atr_stop
            
            # Zone Trailing Logic (your innovative feature)
            if strategy_type in ['zone_trailing', 'hybrid', 'time_limited_trailing']:
                should_trail = False
                
                if strategy_config.get('trailing') == 'immediate':
                    should_trail = True
                elif strategy_config.get('trailing') == 'after_1R':
                    rr_1_level = entry_price + risk_distance if direction == 'BUY' else entry_price - risk_distance
                    if (direction == 'BUY' and candle['high'] >= rr_1_level) or \
                       (direction == 'SELL' and candle['low'] <= rr_1_level):
                        trailing_active = True
                        should_trail = True
                elif strategy_config.get('trailing') == 'after_2R':
                    rr_2_level = entry_price + (risk_distance * 2) if direction == 'BUY' else entry_price - (risk_distance * 2)
                    if (direction == 'BUY' and candle['high'] >= rr_2_level) or \
                       (direction == 'SELL' and candle['low'] <= rr_2_level):
                        trailing_active = True
                        should_trail = True
                elif strategy_config.get('trailing') == 'after_be' and at_breakeven:
                    should_trail = True
                
                if should_trail or trailing_active:
                    # Look for new zones formed after entry
                    # This would require zone detection on-the-fly
                    # For now, simplified trailing based on recent highs/lows
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
        
        # If trade never hit stop or target, close at last price
        final_candle = data.iloc[-1]
        final_price = final_candle['close']
        pnl = (final_price - entry_price) * position_size if direction == 'BUY' else (entry_price - final_price) * position_size
        
        return {
            'entry_date': data.index[entry_idx],
            'entry_price': entry_price,
            'exit_date': data.index[-1],
            'exit_price': final_price,
            'direction': direction,
            'pnl': pnl,
            'exit_reason': 'end_of_data',
            'days_held': len(data) - entry_idx - 1,
            'strategy': strategy_config['description']
        }
    
    def calculate_enhanced_performance(self, trades: List[Dict], final_balance: float,
                                     pair: str, timeframe: str, strategy_name: str,
                                     strategy_config: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return self.empty_result(pair, timeframe, strategy_name, "No trades executed")
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        breakeven_trades = len([t for t in trades if t['pnl'] == 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100
        breakeven_rate = (breakeven_trades / total_trades) * 100
        loss_rate = (losing_trades / total_trades) * 100
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        total_return = ((final_balance / 10000) - 1) * 100

        # Risk metrics
        returns = [t['pnl'] for t in trades]
        max_drawdown = self.calculate_max_drawdown(trades)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        
        # Trade duration analysis
        durations = [t['days_held'] for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        # 🆕 NEW: Winner-specific duration analysis (FIXED)
        winning_trade_objects = [t for t in trades if t['pnl'] > 0]
        losing_trade_objects = [t for t in trades if t['pnl'] < 0]
        breakeven_trade_objects = [t for t in trades if t['pnl'] == 0]

        winner_durations = [t['days_held'] for t in winning_trade_objects]
        loser_durations = [t['days_held'] for t in losing_trade_objects]
        breakeven_durations = [t['days_held'] for t in breakeven_trade_objects]

        avg_winner_duration = np.mean(winner_durations) if winner_durations else 0
        avg_loser_duration = np.mean(loser_durations) if loser_durations else 0
        avg_breakeven_duration = np.mean(breakeven_durations) if breakeven_durations else 0
        
        # 🆕 Duration statistics
        median_duration = np.median(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': strategy_config['description'],
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'breakeven_trades': breakeven_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'breakeven_rate': round(breakeven_rate, 1),
            'loss_rate': round(loss_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_return': round(total_return, 1),
            'expectancy': round(expectancy, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'avg_duration_days': round(avg_duration, 1),
            'final_balance': round(final_balance, 2),
            'avg_winner_duration': round(avg_winner_duration, 1),
            'avg_loser_duration': round(avg_loser_duration, 1),
            'avg_breakeven_duration': round(avg_breakeven_duration, 1),
            'median_duration': round(median_duration, 1),
            'max_duration': round(max_duration, 1),
            'min_duration': round(min_duration, 1),
            'trades_data': trades
        }
    
    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown percentage"""
        if not trades:
            return 0
        
        cumulative_pnl = 0
        peak = 10000
        max_dd = 0
        
        for trade in trades:
            cumulative_pnl += trade['pnl']
            current_balance = 10000 + cumulative_pnl
            
            if current_balance > peak:
                peak = current_balance
            
            drawdown = ((peak - current_balance) / peak) * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (assuming 0% risk-free rate)"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualize (assuming average 50 trades per year)
        annual_return = mean_return * 50
        annual_std = std_return * np.sqrt(50)
        
        return annual_return / annual_std if annual_std > 0 else 0
    
    def empty_result(self, pair: str, timeframe: str, strategy_name: str, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': reason,
            'total_trades': 0,
            'winning_trades': 0,
            'breakeven_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'breakeven_rate': 0,
            'loss_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'expectancy': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_duration_days': 0,
            'final_balance': 10000,
            'avg_winner_duration': 0,
            'avg_loser_duration': 0,
            'avg_breakeven_duration': 0,
            'median_duration': 0,
            'max_duration': 0,
            'min_duration': 0,
            'trades_data': []
        }
    
    def run_comprehensive_analysis(self, test_all: bool = False, 
                                    pairs: List[str] = None, 
                                    timeframes: List[str] = None,
                                    strategies: List[str] = None,
                                    days_back: int = 730) -> pd.DataFrame:
        """Run comprehensive analysis across pairs/timeframes/strategies"""
        
        print("🚀 COMPREHENSIVE TRADE MANAGEMENT ANALYSIS")
        print("=" * 60)
        
        if test_all:
            # Auto-detect all available data
            available_data = self.get_all_available_data_files()
            
            if not available_data:
                print("❌ No data files found!")
                return pd.DataFrame()
            
            print(f"📊 Found {len(available_data)} data files:")
            for item in available_data:
                print(f"   {item['pair']} {item['timeframe']}")
            
            # Extract unique pairs and timeframes
            pairs = list(set([item['pair'] for item in available_data]))
            timeframes = list(set([item['timeframe'] for item in available_data]))
            
        else:
            # Use provided parameters
            if pairs is None:
                pairs = ['EURUSD']
            if timeframes is None:
                timeframes = ['1D']
        
        if strategies is None:
            strategies = list(self.MANAGEMENT_STRATEGIES.keys())
        
        print(f"\n📋 TEST CONFIGURATION:")
        print(f"   Pairs: {pairs}")
        print(f"   Timeframes: {timeframes}")
        print(f"   Strategies: {len(strategies)} total")
        print(f"   Period: {days_back} days")
        
        total_tests = len(pairs) * len(timeframes) * len(strategies)
        print(f"   Total tests: {total_tests}")
        
        confirm = input(f"\n🚀 Start comprehensive analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled.")
            return pd.DataFrame()
        
        # Run all combinations
        results = []
        test_count = 0
        
        for pair in pairs:
            for timeframe in timeframes:
                for strategy_name in strategies:
                    test_count += 1
                    print(f"\n🧪 Test {test_count}/{total_tests}: {pair} {timeframe} {strategy_name}")
                    
                    result = self.run_single_backtest(pair, timeframe, strategy_name, days_back)
                    results.append(result)
                    
                    if result['total_trades'] > 0:
                        print(f"   ✅ {result['total_trades']} trades, "
                                f"{result['win_rate']}% WR, "
                                f"PF: {result['profit_factor']}, "
                                f"Return: {result['total_return']}%")
                    else:
                        print(f"   ❌ {result['description']}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        self.save_results_to_excel(df)
        
        # Generate summary report
        self.generate_summary_report(df)
        
        return df
    
    def save_results_to_excel(self, df: pd.DataFrame):
        """Save comprehensive results to Excel file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = r"C:\Users\sim\Desktop\Quant\OTC\trading-strategy-automation\results\trademanagement"
        os.makedirs(export_path, exist_ok=True)
        filename = os.path.join(export_path, f"trade_management_analysis_{timestamp}.xlsx")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
            df_export = df[[
                'pair', 'timeframe', 'strategy', 'description',
                'total_trades', 'win_rate', 'breakeven_rate', 'loss_rate',
                'profit_factor', 'total_return', 'expectancy',
                'max_drawdown', 'sharpe_ratio', 
                'avg_duration_days', 'avg_winner_duration', 'avg_loser_duration',  # 🆕 NEW
                'median_duration', 'max_duration'  # 🆕 NEW
            ]].copy()
            
            df_export.to_excel(writer, sheet_name='All_Results', index=False)
            
            # 🆕 NEW: Duration Analysis Sheet
            duration_analysis = df[df['total_trades'] > 0][
                ['pair', 'timeframe', 'strategy', 'avg_winner_duration', 
                'avg_loser_duration', 'avg_duration_days', 'median_duration', 'max_duration']
            ].copy()
            duration_analysis.to_excel(writer, sheet_name='Duration_Analysis', index=False)
            
            # Top performers sheet (trades > 0, sorted by profit factor)
            top_performers = df[df['total_trades'] > 0].copy()
            top_performers = top_performers.sort_values('profit_factor', ascending=False)
            top_performers[['pair', 'timeframe', 'strategy', 'total_trades', 
                            'win_rate', 'profit_factor', 'total_return', 'max_drawdown']].to_excel(
                writer, sheet_name='Top_Performers', index=False)
            
            # Strategy comparison
            strategy_summary = df[df['total_trades'] > 0].groupby('strategy').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': 'mean',
                'total_return': 'mean',
                'max_drawdown': 'mean'
            }).round(2)
            strategy_summary.to_excel(writer, sheet_name='Strategy_Summary')
            
            # Pair comparison
            pair_summary = df[df['total_trades'] > 0].groupby('pair').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': 'mean',
                'total_return': 'mean',
                'total_trades': 'sum'
            }).round(2)
            pair_summary.to_excel(writer, sheet_name='Pair_Summary')
            
            # Timeframe comparison
            tf_summary = df[df['total_trades'] > 0].groupby('timeframe').agg({
                'profit_factor': ['mean', 'std', 'count'],
                'win_rate': 'mean',
                'total_return': 'mean',
                'total_trades': 'sum'
            }).round(2)
            tf_summary.to_excel(writer, sheet_name='Timeframe_Summary')
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate comprehensive summary report"""
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        total_tests = len(df)
        successful_tests = len(df[df['total_trades'] > 0])
        
        print(f"📋 OVERVIEW:")
        print(f"   Total tests: {total_tests}")
        print(f"   Successful tests: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")

        
        if successful_tests == 0:
            print("❌ No successful strategies found!")
            return
        
        successful_df = df[df['total_trades'] > 0].copy()
        
        # Top overall performers
        print(f"\n🏆 TOP 5 STRATEGIES (by Profit Factor):")
        top_5 = successful_df.nlargest(5, 'profit_factor')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. {row['pair']} {row['timeframe']} {row['strategy']}")
            print(f"      PF: {row['profit_factor']}, WR: {row['win_rate']}%, "
                    f"Return: {row['total_return']}%, Trades: {row['total_trades']}")
        
        # Best by category
        print(f"\n📈 CATEGORY LEADERS:")
        
        best_return = successful_df.loc[successful_df['total_return'].idxmax()]
        print(f"   💰 Highest Return: {best_return['pair']} {best_return['timeframe']} "
                f"{best_return['strategy']} → {best_return['total_return']}%")
        
        best_wr = successful_df.loc[successful_df['win_rate'].idxmax()]
        print(f"   🎯 Best Win Rate: {best_wr['pair']} {best_wr['timeframe']} "
                f"{best_wr['strategy']} → {best_wr['win_rate']}%")
        
        best_sharpe = successful_df.loc[successful_df['sharpe_ratio'].idxmax()]
        print(f"   📊 Best Sharpe: {best_sharpe['pair']} {best_sharpe['timeframe']} "
                f"{best_sharpe['strategy']} → {best_sharpe['sharpe_ratio']}")
        
        lowest_dd = successful_df.loc[successful_df['max_drawdown'].idxmin()]
        print(f"   🛡️  Lowest Drawdown: {lowest_dd['pair']} {lowest_dd['timeframe']} "
                f"{lowest_dd['strategy']} → {lowest_dd['max_drawdown']}%")
        
        # Strategy type analysis
        print(f"\n🎯 STRATEGY TYPE PERFORMANCE:")
        strategy_types = {}
        
        for _, row in successful_df.iterrows():
            strategy_name = row['strategy']
            strategy_config = self.MANAGEMENT_STRATEGIES[strategy_name]
            strategy_type = strategy_config['type']
            
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
            strategy_types[strategy_type].append(row['profit_factor'])
        
        for strategy_type, pf_list in strategy_types.items():
            avg_pf = np.mean(pf_list)
            count = len(pf_list)
            print(f"   {strategy_type.replace('_', ' ').title()}: "
                    f"{avg_pf:.2f} avg PF ({count} results)")
        
        # Pair recommendations
        print(f"\n💱 PAIR RECOMMENDATIONS:")
        pair_performance = successful_df.groupby('pair').agg({
            'profit_factor': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2)
        
        best_pairs = pair_performance.sort_values('profit_factor', ascending=False)
        for pair in best_pairs.index[:3]:
            row = best_pairs.loc[pair]
            print(f"   {pair}: {row['profit_factor']} avg PF, "
                    f"{row['win_rate']}% avg WR, {row['total_trades']} total trades")
        
        # Timeframe recommendations  
        print(f"\n⏰ TIMEFRAME RECOMMENDATIONS:")
        tf_performance = successful_df.groupby('timeframe').agg({
            'profit_factor': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2)
        
        best_tfs = tf_performance.sort_values('profit_factor', ascending=False)
        for tf in best_tfs.index:
            row = best_tfs.loc[tf]
            print(f"   {tf}: {row['profit_factor']} avg PF, "
                    f"{row['win_rate']}% avg WR, {row['total_trades']} total trades")
            
        # 🆕 NEW: Duration insights
        print(f"\n⏰ TRADE DURATION INSIGHTS:")
        
        if successful_tests > 0:
            avg_winner_time = successful_df['avg_winner_duration'].mean()
            avg_loser_time = successful_df['avg_loser_duration'].mean()
            
            print(f"   Average Winner Duration: {avg_winner_time:.1f} days")
            print(f"   Average Loser Duration: {avg_loser_time:.1f} days")
            print(f"   Winner vs Loser Ratio: {avg_winner_time/avg_loser_time:.1f}x" if avg_loser_time > 0 else "   Loser Duration: N/A")
            
            # Find fastest and slowest strategies
            fastest_winners = successful_df.loc[successful_df['avg_winner_duration'].idxmin()]
            slowest_winners = successful_df.loc[successful_df['avg_winner_duration'].idxmax()]
            
            print(f"   Fastest Winners: {fastest_winners['strategy']} → {fastest_winners['avg_winner_duration']:.1f} days")
            print(f"   Slowest Winners: {slowest_winners['strategy']} → {slowest_winners['avg_winner_duration']:.1f} days")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
class ParallelTradeManagementBacktester(TradeManagementBacktester):
    """Optimized parallel backtesting for i5-10400F (6C/12T)"""
    
    def __init__(self, max_workers: int = 10):
        super().__init__()
        # Optimal for i5-10400F: 10 workers (leave 2 threads for system)
        self.max_workers = min(max_workers, mp.cpu_count())
        
    def run_parallel_comprehensive_analysis(self, test_all: bool = False, 
                                          pairs: List[str] = None, 
                                          timeframes: List[str] = None,
                                          strategies: List[str] = None,
                                          days_back: int = 730) -> pd.DataFrame:
        """Run comprehensive analysis with parallel processing"""
        
        print("🚀 PARALLEL TRADE MANAGEMENT ANALYSIS")
        print("=" * 60)
        
        # Get test configurations
        test_combinations = self._prepare_test_combinations(
            test_all, pairs, timeframes, strategies, days_back
        )
        
        if not test_combinations:
            print("❌ No test combinations found!")
            return pd.DataFrame()
        
        total_tests = len(test_combinations)
        estimated_time_single = total_tests * 3.5  # seconds
        estimated_time_parallel = estimated_time_single / 9  # Expected 9x speedup
        
        print(f"\n📋 PERFORMANCE PROJECTION:")
        print(f"   Total tests: {total_tests}")
        print(f"   Workers: {self.max_workers}")
        print(f"   Single-threaded time: {estimated_time_single/60:.1f} minutes")
        print(f"   Parallel time estimate: {estimated_time_parallel/60:.1f} minutes")
        print(f"   Expected speedup: {estimated_time_single/estimated_time_parallel:.1f}x")
        
        confirm = input(f"\n🚀 Start parallel analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled.")
            return pd.DataFrame()
        
        # Run parallel processing
        results = self._run_parallel_tests(test_combinations)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        self.save_results_to_excel(df)
        self.generate_summary_report(df)
        
        return df
    
    def _prepare_test_combinations(self, test_all: bool, pairs: List[str], 
                                 timeframes: List[str], strategies: List[str],
                                 days_back: int) -> List[Dict]:
        """Prepare all test combinations for parallel execution"""
        
        if test_all:
            # Auto-detect all available data
            available_data = self.get_all_available_data_files()
            if not available_data:
                return []
            
            pairs = list(set([item['pair'] for item in available_data]))
            timeframes = list(set([item['timeframe'] for item in available_data]))
        else:
            if pairs is None:
                pairs = ['EURUSD']
            if timeframes is None:
                timeframes = ['3D']
        
        if strategies is None:
            strategies = list(self.MANAGEMENT_STRATEGIES.keys())
        
        # Create all combinations
        test_combinations = []
        for pair in pairs:
            for timeframe in timeframes:
                for strategy in strategies:
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'days_back': days_back
                    })
        
        return test_combinations
    
    def _run_parallel_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """Execute tests in parallel using ProcessPoolExecutor"""
        
        print(f"\n🔄 Starting parallel execution with {self.max_workers} workers...")
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_test = {
                executor.submit(self._run_single_test_static, test_config): test_config 
                for test_config in test_combinations
            }
            
            # Collect results as they complete
            completed = 0
            total_tests = len(test_combinations)
            
            for future in as_completed(future_to_test):
                test_config = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Progress tracking
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_tests - completed) / rate if rate > 0 else 0
                    
                    # Progress indicator
                    progress_percent = (completed / total_tests) * 100
                    print(f"🔄 Progress: {completed}/{total_tests} "
                          f"({progress_percent:.1f}%) - "
                          f"Rate: {rate:.1f}/sec - "
                          f"ETA: {eta/60:.1f}min", end='\r')
                    
                except Exception as e:
                    print(f"\n❌ Failed: {test_config['pair']} {test_config['timeframe']} "
                          f"{test_config['strategy']}: {e}")
                    # Add empty result for failed test
                    results.append(self.empty_result(
                        test_config['pair'], 
                        test_config['timeframe'], 
                        test_config['strategy'], 
                        f"Error: {str(e)}"
                    ))
                    completed += 1
        
        total_time = time.time() - start_time
        actual_speedup = (total_tests * 3.5) / total_time
        
        print(f"\n\n✅ Parallel execution complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Tests per second: {total_tests/total_time:.1f}")
        print(f"   Actual speedup: {actual_speedup:.1f}x")
        print(f"   CPU utilization: ~{(actual_speedup/12)*100:.0f}%")
        
        return results
    
    @staticmethod
    def _run_single_test_static(test_config: Dict) -> Dict:
        """Static method for parallel execution (required for multiprocessing)"""
        try:
            # Create fresh backtester instance for each process
            backtester = TradeManagementBacktester()
            
            return backtester.run_single_backtest(
                test_config['pair'],
                test_config['timeframe'], 
                test_config['strategy'],
                test_config['days_back']
            )
        except Exception as e:
            # Return error result
            return {
                'pair': test_config['pair'],
                'timeframe': test_config['timeframe'],
                'strategy': test_config['strategy'],
                'description': f"Error: {str(e)}",
                'total_trades': 0,
                'winning_trades': 0,
                'profit_factor': 0,
                'total_return': 0,
                'error': str(e)
            }


def get_days_back_input() -> int:
    """Enhanced days_back input with 'All Data' option"""
    
    print("\nSelect data period:")
    print("1. Last 730 days (2 years)")
    print("2. Last 1095 days (3 years)")
    print("3. Last 1825 days (5 years)")
    print("4. All available data (recommended for final analysis)")
    print("5. Custom days")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        return 730
    elif choice == '2':
        return 1095
    elif choice == '3':
        return 1825
    elif choice == '4':
        print("   📊 Will use ALL available data for each pair")
        print("   📈 XAUUSD 1D: ~14,552 candles")
        print("   📈 Other pairs: ~2,000-9,999 candles")
        return 99999  # Use large number to ensure all data
    elif choice == '5':
        days = int(input("Enter custom days: "))
        return days
    else:
        print("Invalid choice, using default 730 days")
        return 730

def main_parallel():
    """Main execution function with parallel processing"""
    
    print("🚀 PARALLEL TRADE MANAGEMENT BACKTESTING SYSTEM")
    print("⚡ Optimized for Intel i5-10400F (6C/12T)")
    print("=" * 60)
    
    backtester = ParallelTradeManagementBacktester(max_workers=10)
    
    print("\nSelect testing mode:")
    print("1. Single test (no parallelization)")
    print("2. Parallel: Specific pairs and timeframes")
    print("3. Parallel: ALL AVAILABLE DATA (Recommended)")
    print("4. Parallel: Custom configuration")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Single test (original method)
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (1D, 2D, 3D, 4D): ").strip()
        strategy_name = input("Enter strategy name: ").strip()
        days_back = get_days_back_input()
        
        start_time = time.time()
        result = backtester.run_single_backtest(pair, timeframe, strategy_name, days_back)
        elapsed = time.time() - start_time
        
        print(f"\nSingle test completed in {elapsed:.1f} seconds")
        print(f"Result: {result['total_trades']} trades, {result['win_rate']}% WR, "
              f"PF: {result['profit_factor']}")
    
    elif choice == '2':
        # Parallel: Specific pairs/timeframes
        pairs = input("Enter pairs (comma-separated): ").strip().upper().split(',')
        timeframes = input("Enter timeframes (comma-separated): ").strip().split(',')
        days_back = get_days_back_input()
        
        df = backtester.run_parallel_comprehensive_analysis(
            test_all=False, pairs=pairs, timeframes=timeframes, days_back=days_back
        )
    
    elif choice == '3':
        # Parallel: All available data
        days_back = get_days_back_input()
        
        df = backtester.run_parallel_comprehensive_analysis(
            test_all=True, days_back=days_back
        )
    
    elif choice == '4':
        # Parallel: Custom configuration
        pairs = input("Enter pairs (comma-separated): ").strip().upper().split(',')
        timeframes = input("Enter timeframes (comma-separated): ").strip().split(',')
        
        print("\nSelect strategies (enter numbers, comma-separated):")
        strategies_list = list(backtester.MANAGEMENT_STRATEGIES.keys())
        for i, strategy in enumerate(strategies_list, 1):
            print(f"   {i}. {strategy}")
        
        strategy_indices = input("\nStrategy numbers: ").strip().split(',')
        strategies = [strategies_list[int(i)-1] for i in strategy_indices]
        
        days_back = get_days_back_input()
        
        df = backtester.run_parallel_comprehensive_analysis(
            test_all=False, pairs=pairs, timeframes=timeframes, 
            strategies=strategies, days_back=days_back
        )
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main_parallel()