"""
FIXED: Enhanced Momentum vs Reversal Backtester
Compare D-B-D/R-B-R (momentum) vs D-B-R/R-B-D (reversal) performance
FIXED: Proper CandleClassifier integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager

class FixedMomentumVsReversalBacktester:
    """
    FIXED: Advanced backtester comparing momentum vs reversal strategies
    with proper CandleClassifier integration
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.distance_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.zones = None
        self.candle_classifier = None
        
    def load_data(self, days_back: int = 730, timeframe: str = 'Daily', pair: str = 'EURUSD'):
        """Load and prepare data for backtesting"""
        print(f"📊 Loading {pair} {timeframe} data ({days_back} days back)...")
        
        data_loader = DataLoader()
        self.data = data_loader.load_pair_data(pair, timeframe)
        
        # Debug: Check what type the index is
        print(f"🔍 Data index type: {type(self.data.index)}")
        print(f"🔍 Index sample: {self.data.index[:3]}")
        
        # Ensure the index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            print("⚠️  Converting index to datetime...")
            # If index is integers, assume it's a date column that needs to be converted
            if 'date' in self.data.columns:
                self.data.set_index('date', inplace=True)
                self.data.index = pd.to_datetime(self.data.index)
            elif '<DATE>' in self.data.columns:
                self.data.set_index('<DATE>', inplace=True)
                self.data.index = pd.to_datetime(self.data.index)
            else:
                # Try to convert the existing index
                try:
                    self.data.index = pd.to_datetime(self.data.index)
                except:
                    print("❌ Could not convert index to datetime. Using all data.")
                    self.test_data = self.data
                    return self.test_data
        
        # Now safely calculate date range
        try:
            if days_back >= 9999:  # "All available data"
                print("📊 Using all available data")
                self.test_data = self.data
            else:
                end_date = self.data.index[-1]  # Should now be datetime
                start_date = end_date - timedelta(days=days_back)
                
                print(f"🔍 End date: {end_date} (type: {type(end_date)})")
                print(f"🔍 Start date: {start_date} (type: {type(start_date)})")
                
                # Ensure we have enough lookback data (365 days minimum)
                if len(self.data) < days_back + 365:
                    print(f"⚠️  Limited data: Using {len(self.data)} candles available")
                    self.test_data = self.data
                else:
                    # Filter data by date range
                    self.test_data = self.data[self.data.index >= start_date]
        
        except Exception as e:
            print(f"⚠️  Date filtering failed ({e}), using all data")
            self.test_data = self.data
        
        print(f"✅ Loaded {len(self.test_data)} candles for testing")
        if hasattr(self.test_data.index[0], 'strftime'):
            print(f"📅 Test period: {self.test_data.index[0].strftime('%Y-%m-%d')} to {self.test_data.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"📅 Test period: {self.test_data.index[0]} to {self.test_data.index[-1]}")
        
        return self.test_data
    
    def detect_zones(self):
        """FIXED: Detect zones using the corrected zone detector"""
        print("🔍 Detecting zones with FIXED algorithms...")
        
        # FIXED: Initialize components properly
        self.candle_classifier = CandleClassifier(self.data)
        classified_data = self.candle_classifier.classify_all_candles()
        
        # FIXED: Use the corrected zone detector from our previous fix
        zone_detector = ZoneDetector(self.candle_classifier)
        self.zones = zone_detector.detect_all_patterns(classified_data)
        
        # Separate momentum and reversal patterns
        momentum_patterns = self.zones['dbd_patterns'] + self.zones['rbr_patterns']
        reversal_patterns = self.zones.get('dbr_patterns', []) + self.zones.get('rbd_patterns', [])
        
        print(f"✅ Zone detection complete:")
        print(f"   Momentum patterns (D-B-D + R-B-R): {len(momentum_patterns)}")
        print(f"   Reversal patterns (D-B-R + R-B-D): {len(reversal_patterns)}")
        
        return momentum_patterns, reversal_patterns
    
    def backtest_strategy(self, patterns: List[Dict], strategy_name: str, 
                     distance_threshold: float) -> Dict:
        """
        Backtest a specific strategy with distance threshold
        FIXED: Proper one-trade-per-zone logic
        """
        print(f"🧪 Testing {strategy_name} with {distance_threshold}x distance...")
        
        # Filter patterns by distance threshold
        valid_patterns = []
        for pattern in patterns:
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']:
                if pattern['leg_out']['ratio_to_base'] >= distance_threshold:
                    valid_patterns.append(pattern)
        
        if not valid_patterns:
            print(f"   ⚠️  No patterns meet {distance_threshold}x distance requirement")
            return self.empty_results()
        
        print(f"   📊 {len(valid_patterns)} patterns meet {distance_threshold}x requirement")
        
        # Initialize tracking
        trades = []
        used_zones = set()  # Track zones that have been used
        active_zones = []   # Zones waiting for entry
        account_balance = 10000
        
        # Pre-process zones to get activation dates
        zone_activation_schedule = []
        for pattern in valid_patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(self.data):
                activation_date = self.data.index[zone_end_idx]
                zone_activation_schedule.append({
                    'date': activation_date,
                    'pattern': pattern,
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
                })
        
        # Sort by activation date
        zone_activation_schedule.sort(key=lambda x: x['date'])
        
        # Process each candle for backtesting
        for i, (date, candle) in enumerate(self.test_data.iterrows()):
            current_price = candle['close']
            
            # Check for new zone activations
            for schedule_item in zone_activation_schedule:
                zone_id = schedule_item['zone_id']
                pattern = schedule_item['pattern']
                
                # Zone becomes active 1 day after formation
                if (date > schedule_item['date'] and 
                    zone_id not in used_zones and 
                    pattern not in active_zones):
                    active_zones.append(pattern)
            
            # Check for trade executions (process each zone only once)
            for zone in active_zones.copy():
                zone_id = f"{zone['type']}_{zone.get('end_idx', 0)}_{zone['zone_low']:.5f}"
                
                # Skip if this zone has already been used
                if zone_id in used_zones:
                    active_zones.remove(zone)
                    continue
                
                trade_result = self.check_trade_execution(zone, candle, date, current_price)
                
                if trade_result:
                    # Mark this zone as used regardless of outcome
                    used_zones.add(zone_id)
                    active_zones.remove(zone)
                    
                    if 'invalidated' not in trade_result:
                        trades.append(trade_result)
                        account_balance += trade_result['pnl']
                        print(f"      💰 Trade #{len(trades)}: {trade_result['result']} "
                            f"${trade_result['pnl']:.0f} ({trade_result['zone_type']})")
        
        print(f"   ✅ Completed: {len(trades)} trades from {len(valid_patterns)} zones")
        
        # Calculate performance metrics
        return self.calculate_performance(trades, account_balance, strategy_name, distance_threshold)
    
    def check_trade_execution(self, zone: Dict, candle: pd.Series, 
                        date: pd.Timestamp, current_price: float) -> Dict:
        """
        Check if a trade should be executed based on zone logic
        FIXED: More conservative entry and invalidation logic
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # More conservative invalidation (50% penetration instead of 33%)
        invalidation_threshold = zone_range * 0.50
        
        # Check for zone invalidation FIRST (before entries)
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones
            if candle['low'] <= zone_low - invalidation_threshold:
                return {'invalidated': True, 'zone_type': zone['type']}
        else:  # Supply zones  
            if candle['high'] >= zone_high + invalidation_threshold:
                return {'invalidated': True, 'zone_type': zone['type']}
        
        # 5% front-run entry logic (only if not invalidated)
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
            entry_price = zone_low + (zone_range * 0.05)  # 5% into zone
            
            # Check if price reached entry level
            if candle['low'] <= entry_price:
                return self.execute_buy_trade(zone, entry_price, date, candle)
                
        elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell)
            entry_price = zone_high - (zone_range * 0.05)  # 5% into zone
            
            # Check if price reached entry level
            if candle['high'] >= entry_price:
                return self.execute_sell_trade(zone, entry_price, date, candle)
        
        return None
    
    def execute_buy_trade(self, zone: Dict, entry_price: float, 
                     entry_date: pd.Timestamp, entry_candle: pd.Series) -> Dict:
        """Execute a buy trade with proper risk management"""
        zone_range = zone['zone_high'] - zone['zone_low']
        
        # Calculate stop loss (33% buffer beyond zone)
        stop_loss = zone['zone_low'] - (zone_range * 0.33)
        
        # Fixed $500 risk per trade
        risk_amount = 500
        stop_distance = entry_price - stop_loss
        
        if stop_distance <= 0:
            return None
        
        # Position size to risk exactly $500
        position_size = risk_amount / stop_distance
        
        # Calculate targets
        target_1 = entry_price + stop_distance  # 1:1 RR (break-even move)
        target_2 = entry_price + (stop_distance * 2)  # 1:2 RR (final target)
        
        # REALISTIC: Track actual trade outcome with proper management
        trade_outcome = self.simulate_realistic_trade_outcome(zone, entry_price, stop_loss, target_2, entry_date)
        
        return {
            'strategy': 'momentum' if zone['type'] in ['R-B-R', 'D-B-D'] else 'reversal',
            'zone_type': zone['type'],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_price': trade_outcome['exit_price'],
            'stop_loss': stop_loss,
            'breakeven_target': target_1,
            'final_target': target_2,
            'position_size': position_size,
            'pnl': trade_outcome['pnl'],
            'distance_ratio': zone['leg_out']['ratio_to_base'],
            'duration_days': trade_outcome.get('days_held', 0),
            'result': 'win' if trade_outcome['pnl'] > 0 else ('breakeven' if trade_outcome['pnl'] == 0 else 'loss'),
            'exit_reason': trade_outcome['exit_reason']
        }

    def execute_sell_trade(self, zone: Dict, entry_price: float, 
                        entry_date: pd.Timestamp, entry_candle: pd.Series) -> Dict:
        """Execute a sell trade with proper risk management"""
        zone_range = zone['zone_high'] - zone['zone_low']
        
        # Calculate stop loss (33% buffer beyond zone)
        stop_loss = zone['zone_high'] + (zone_range * 0.33)
        
        # Fixed $500 risk per trade
        risk_amount = 500
        stop_distance = stop_loss - entry_price
        
        if stop_distance <= 0:
            return None
        
        # Position size to risk exactly $500
        position_size = risk_amount / stop_distance
        
        # Calculate targets
        target_1 = entry_price - stop_distance  # 1:1 RR (break-even move)
        target_2 = entry_price - (stop_distance * 2)  # 1:2 RR (final target)
        
        # REALISTIC: Track actual trade outcome with proper management
        trade_outcome = self.simulate_realistic_trade_outcome(zone, entry_price, stop_loss, target_2, entry_date)
        
        return {
            'strategy': 'momentum' if zone['type'] in ['R-B-R', 'D-B-D'] else 'reversal',
            'zone_type': zone['type'],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_price': trade_outcome['exit_price'],
            'stop_loss': stop_loss,
            'breakeven_target': target_1,
            'final_target': target_2,
            'position_size': position_size,
            'pnl': trade_outcome['pnl'],
            'distance_ratio': zone['leg_out']['ratio_to_base'],
            'duration_days': trade_outcome.get('days_held', 0),
            'result': 'win' if trade_outcome['pnl'] > 0 else ('breakeven' if trade_outcome['pnl'] == 0 else 'loss'),
            'exit_reason': trade_outcome['exit_reason']
        }
    
    def calculate_performance(self, trades: List[Dict], final_balance: float, 
                            strategy_name: str, distance_threshold: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self.empty_results()
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            'strategy': strategy_name,
            'distance_threshold': distance_threshold,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'final_balance': round(final_balance, 2),
            'total_return': round(((final_balance / 10000) - 1) * 100, 1),
            'trades': trades
        }
    
    def simulate_realistic_trade_outcome(self, zone: Dict, entry_price: float, 
                               stop_loss: float, target_price: float, 
                               entry_date: pd.Timestamp) -> Dict:
        """
        REALISTIC: Simulate trade outcome with proper trade management
        - Hold trades indefinitely until stop or target hit
        - Move stop to break-even at 1:1 risk/reward
        - Final target at 1:2 risk/reward
        """
        # Calculate proper position size based on 5% risk ($500)
        risk_amount = 500  # Fixed $500 risk per trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return {'pnl': 0, 'exit_price': entry_price, 'exit_reason': 'invalid'}
        
        # Position size to risk exactly $500
        position_size = risk_amount / stop_distance
        
        # Calculate 1:1 target (break-even move level)
        if entry_price > stop_loss:  # Buy trade
            breakeven_target = entry_price + stop_distance  # 1:1 up
            final_target = entry_price + (stop_distance * 2)  # 1:2 up
        else:  # Sell trade
            breakeven_target = entry_price - stop_distance  # 1:1 down
            final_target = entry_price - (stop_distance * 2)  # 1:2 down
        
        # Find entry date index
        try:
            entry_idx = self.data.index.get_loc(entry_date)
        except KeyError:
            entry_idx = self.data.index.get_indexer([entry_date], method='nearest')[0]
        
        # Trade management state
        current_stop = stop_loss
        at_breakeven = False
        
        # Look forward indefinitely until stop or target hit
        for i in range(entry_idx + 1, len(self.data)):
            candle = self.data.iloc[i]
            
            # For buy trades
            if entry_price > stop_loss:
                
                # Phase 1: Check if original stop hit before 1:1
                if not at_breakeven and candle['low'] <= current_stop:
                    return {
                        'pnl': -risk_amount,  # Lose exactly $500
                        'exit_price': current_stop,
                        'exit_reason': 'original_stop',
                        'days_held': i - entry_idx
                    }
                
                # Phase 2: Check if 1:1 target hit (move to break-even)
                if not at_breakeven and candle['high'] >= breakeven_target:
                    current_stop = entry_price  # Move stop to break-even
                    at_breakeven = True
                    # Continue trading to 1:2 target
                    continue
                
                # Phase 3: After break-even move, check break-even stop
                if at_breakeven and candle['low'] <= current_stop:
                    return {
                        'pnl': 0,  # Break-even exit
                        'exit_price': current_stop,
                        'exit_reason': 'breakeven_stop',
                        'days_held': i - entry_idx
                    }
                
                # Phase 4: Check if final 1:2 target hit
                if candle['high'] >= final_target:
                    profit = (final_target - entry_price) * position_size
                    return {
                        'pnl': profit,  # Full profit at 1:2
                        'exit_price': final_target,
                        'exit_reason': 'final_target',
                        'days_held': i - entry_idx
                    }
            
            # For sell trades
            else:
                
                # Phase 1: Check if original stop hit before 1:1
                if not at_breakeven and candle['high'] >= current_stop:
                    return {
                        'pnl': -risk_amount,  # Lose exactly $500
                        'exit_price': current_stop,
                        'exit_reason': 'original_stop',
                        'days_held': i - entry_idx
                    }
                
                # Phase 2: Check if 1:1 target hit (move to break-even)
                if not at_breakeven and candle['low'] <= breakeven_target:
                    current_stop = entry_price  # Move stop to break-even
                    at_breakeven = True
                    # Continue trading to 1:2 target
                    continue
                
                # Phase 3: After break-even move, check break-even stop
                if at_breakeven and candle['high'] >= current_stop:
                    return {
                        'pnl': 0,  # Break-even exit
                        'exit_price': current_stop,
                        'exit_reason': 'breakeven_stop',
                        'days_held': i - entry_idx
                    }
                
                # Phase 4: Check if final 1:2 target hit
                if candle['low'] <= final_target:
                    profit = (entry_price - final_target) * position_size
                    return {
                        'pnl': profit,  # Full profit at 1:2
                        'exit_price': final_target,
                        'exit_reason': 'final_target',
                        'days_held': i - entry_idx
                    }
        
        # If we reach end of data and trade is still open
        final_candle = self.data.iloc[-1]
        final_price = final_candle['close']
        
        # Calculate PnL based on current price
        if entry_price > stop_loss:  # Buy trade
            current_pnl = (final_price - entry_price) * position_size
        else:  # Sell trade
            current_pnl = (entry_price - final_price) * position_size
        
        return {
            'pnl': current_pnl,
            'exit_price': final_price,
            'exit_reason': 'end_of_data',
            'days_held': len(self.data) - entry_idx - 1
        }

    def empty_results(self) -> Dict:
        """Return empty results for failed strategies"""
        return {
            'strategy': 'N/A',
            'distance_threshold': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'final_balance': 10000,
            'total_return': 0,
            'trades': []
        }
    
    def run_comprehensive_analysis(self, days_back: int = 730, timeframe: str = '1D', pair: str = 'AUDNZD') -> Dict:
        """Run comprehensive momentum vs reversal analysis"""
        print("🚀 ENHANCED: MOMENTUM VS REVERSAL COMPREHENSIVE ANALYSIS")
        print(f"💱 Pair: {pair}")
        print(f"⏰ Timeframe: {timeframe}")
        print("=" * 70)
        
        # Load data with specified parameters
        self.load_data(days_back, timeframe, pair)
        
        # Detect zones
        momentum_patterns, reversal_patterns = self.detect_zones()
        
        # Test all distance thresholds WITH CUMULATIVE LOGIC
        all_results = []
        
        for distance in self.distance_thresholds:
            print(f"\n📊 Testing {distance}x distance threshold...")
            
            # FIXED: Test momentum strategy with cumulative zones
            momentum_results = self.backtest_strategy_cumulative(momentum_patterns, 'Momentum', distance)
            all_results.append(momentum_results)
            
            # FIXED: Test reversal strategy with cumulative zones  
            reversal_results = self.backtest_strategy_cumulative(reversal_patterns, 'Reversal', distance)
            all_results.append(reversal_results)
            
            # Quick summary
            print(f"   Momentum: {momentum_results['total_trades']} trades, "
                f"{momentum_results['win_rate']}% WR, PF: {momentum_results['profit_factor']}")
            print(f"   Reversal: {reversal_results['total_trades']} trades, "
                f"{reversal_results['win_rate']}% WR, PF: {reversal_results['profit_factor']}")
        
        # Generate comprehensive report
        self.generate_analysis_report(all_results)
        self.create_performance_visualizations(all_results)
        
        return all_results

    def backtest_strategy_cumulative(self, patterns: List[Dict], strategy_name: str, 
                                distance_threshold: float) -> Dict:
        """
        Backtest strategy with CUMULATIVE distance logic
        All zones meeting higher thresholds are included in lower threshold tests
        """
        print(f"🧪 Testing {strategy_name} with {distance_threshold}x+ distance...")
        
        # FIXED: Filter patterns by distance threshold (>= not just ==)
        valid_patterns = []
        for pattern in patterns:
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']:
                # CRITICAL FIX: Use >= instead of == 
                if pattern['leg_out']['ratio_to_base'] >= distance_threshold:
                    valid_patterns.append(pattern)
        
        if not valid_patterns:
            print(f"   ⚠️  No patterns meet {distance_threshold}x+ distance requirement")
            return self.empty_results()
        
        print(f"   📊 {len(valid_patterns)} patterns meet {distance_threshold}x+ requirement")
        
        # Initialize tracking
        trades = []
        used_zones = set()  # Track zones that have been used
        active_zones = []   # Zones waiting for entry
        account_balance = 10000
        
        # Pre-process zones to get activation dates
        zone_activation_schedule = []
        for pattern in valid_patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(self.data):
                activation_date = self.data.index[zone_end_idx]
                zone_activation_schedule.append({
                    'date': activation_date,
                    'pattern': pattern,
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
                })
        
        # Sort by activation date
        zone_activation_schedule.sort(key=lambda x: x['date'])
        
        # Process each candle for backtesting
        for i, (date, candle) in enumerate(self.test_data.iterrows()):
            current_price = candle['close']
            
            # Check for new zone activations
            for schedule_item in zone_activation_schedule:
                zone_id = schedule_item['zone_id']
                pattern = schedule_item['pattern']
                
                # Zone becomes active 1 day after formation
                if (date > schedule_item['date'] and 
                    zone_id not in used_zones and 
                    pattern not in active_zones):
                    active_zones.append(pattern)
            
            # Check for trade executions (process each zone only once)
            for zone in active_zones.copy():
                zone_id = f"{zone['type']}_{zone.get('end_idx', 0)}_{zone['zone_low']:.5f}"
                
                # Skip if this zone has already been used
                if zone_id in used_zones:
                    active_zones.remove(zone)
                    continue
                
                trade_result = self.check_trade_execution(zone, candle, date, current_price)
                
                if trade_result:
                    # Mark this zone as used regardless of outcome
                    used_zones.add(zone_id)
                    active_zones.remove(zone)
                    
                    if 'invalidated' not in trade_result:
                        trades.append(trade_result)
                        account_balance += trade_result['pnl']
                        # Show the actual distance ratio for debugging
                        actual_distance = zone['leg_out']['ratio_to_base']
                        print(f"      💰 Trade #{len(trades)}: {trade_result['result']} "
                            f"${trade_result['pnl']:.0f} ({trade_result['zone_type']}) "
                            f"Distance: {actual_distance:.1f}x")
        
        print(f"   ✅ Completed: {len(trades)} trades from {len(valid_patterns)} zones")
        
        # Calculate performance metrics
        return self.calculate_performance(trades, account_balance, strategy_name, distance_threshold)
    
    def generate_analysis_report(self, results: List[Dict]):
        """Generate detailed analysis report"""
        print(f"\n📊 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        # Create summary table
        print(f"{'Strategy':<12} {'Distance':<10} {'Trades':<8} {'Win Rate':<10} "
              f"{'Profit Factor':<15} {'Total Return':<12} {'Expectancy':<12}")
        print("-" * 80)
        
        for result in results:
            if result['total_trades'] > 0:
                print(f"{result['strategy']:<12} {result['distance_threshold']:<10} "
                      f"{result['total_trades']:<8} {result['win_rate']:<10}% "
                      f"{result['profit_factor']:<15} {result['total_return']:<12}% "
                      f"${result['expectancy']:<12}")
        
        # Find best performers
        momentum_results = [r for r in results if r['strategy'] == 'Momentum' and r['total_trades'] > 0]
        reversal_results = [r for r in results if r['strategy'] == 'Reversal' and r['total_trades'] > 0]
        
        if momentum_results:
            best_momentum = max(momentum_results, key=lambda x: x['profit_factor'])
            print(f"\n🏆 Best Momentum: {best_momentum['distance_threshold']}x distance, "
                  f"PF: {best_momentum['profit_factor']}, Return: {best_momentum['total_return']}%")
        
        if reversal_results:
            best_reversal = max(reversal_results, key=lambda x: x['profit_factor'])
            print(f"🏆 Best Reversal: {best_reversal['distance_threshold']}x distance, "
                  f"PF: {best_reversal['profit_factor']}, Return: {best_reversal['total_return']}%")
        
        # Overall winner
        all_valid = [r for r in results if r['total_trades'] > 0]
        if all_valid:
            overall_best = max(all_valid, key=lambda x: x['profit_factor'])
            print(f"\n🎯 OVERALL WINNER: {overall_best['strategy']} with {overall_best['distance_threshold']}x distance")
            print(f"   Performance: {overall_best['total_trades']} trades, "
                  f"{overall_best['win_rate']}% WR, PF: {overall_best['profit_factor']}")
    
    def create_performance_visualizations(self, results: List[Dict]):
        """Create comprehensive performance visualizations"""
        print(f"\n📊 Creating performance visualizations...")
        
        # Filter valid results
        valid_results = [r for r in results if r['total_trades'] > 0]
        
        if not valid_results:
            print("❌ No valid results to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data for plotting
        momentum_data = [r for r in valid_results if r['strategy'] == 'Momentum']
        reversal_data = [r for r in valid_results if r['strategy'] == 'Reversal']
        
        # Plot 1: Profit Factor by Distance
        if momentum_data:
            ax1.plot([r['distance_threshold'] for r in momentum_data], 
                    [r['profit_factor'] for r in momentum_data], 
                    'o-', label='Momentum', linewidth=2, markersize=8)
        
        if reversal_data:
            ax1.plot([r['distance_threshold'] for r in reversal_data], 
                    [r['profit_factor'] for r in reversal_data], 
                    's-', label='Reversal', linewidth=2, markersize=8)
        
        ax1.set_title('Profit Factor by Distance Threshold', fontweight='bold')
        ax1.set_xlabel('Distance Threshold (x)')
        ax1.set_ylabel('Profit Factor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win Rate by Distance
        if momentum_data:
            ax2.plot([r['distance_threshold'] for r in momentum_data], 
                    [r['win_rate'] for r in momentum_data], 
                    'o-', label='Momentum', linewidth=2, markersize=8)
        
        if reversal_data:
            ax2.plot([r['distance_threshold'] for r in reversal_data], 
                    [r['win_rate'] for r in reversal_data], 
                    's-', label='Reversal', linewidth=2, markersize=8)
        
        ax2.set_title('Win Rate by Distance Threshold', fontweight='bold')
        ax2.set_xlabel('Distance Threshold (x)')
        ax2.set_ylabel('Win Rate (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total Return by Distance
        if momentum_data:
            ax3.plot([r['distance_threshold'] for r in momentum_data], 
                    [r['total_return'] for r in momentum_data], 
                    'o-', label='Momentum', linewidth=2, markersize=8)
        
        if reversal_data:
            ax3.plot([r['distance_threshold'] for r in reversal_data], 
                    [r['total_return'] for r in reversal_data], 
                    's-', label='Reversal', linewidth=2, markersize=8)
        
        ax3.set_title('Total Return by Distance Threshold', fontweight='bold')
        ax3.set_xlabel('Distance Threshold (x)')
        ax3.set_ylabel('Total Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Number of Trades by Distance
        if momentum_data:
            ax4.plot([r['distance_threshold'] for r in momentum_data], 
                    [r['total_trades'] for r in momentum_data], 
                    'o-', label='Momentum', linewidth=2, markersize=8)
        
        if reversal_data:
            ax4.plot([r['distance_threshold'] for r in reversal_data], 
                    [r['total_trades'] for r in reversal_data], 
                    's-', label='Reversal', linewidth=2, markersize=8)
        
        ax4.set_title('Trade Frequency by Distance Threshold', fontweight='bold')
        ax4.set_xlabel('Distance Threshold (x)')
        ax4.set_ylabel('Number of Trades')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/momentum_vs_reversal_FIXED_{timestamp}.png"
        os.makedirs('results', exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {filename}")
        
        plt.show()

def main():
    """Enhanced main function with multi-timeframe automation option"""
    print("🚀 ENHANCED: MOMENTUM VS REVERSAL BACKTESTING SYSTEM")
    print("=" * 60)
    
    # NEW: Add multi-timeframe option
    print("\n🎯 Select analysis mode:")
    print("   1. Single timeframe analysis (original)")
    print("   2. ALL timeframes automated (NEW!)")
    
    mode_choice = input("\nEnter mode choice (1-2): ").strip()
    
    if mode_choice == '2':
        # NEW: Multi-timeframe automation
        run_multi_timeframe_analysis()
        return
    
    # EXISTING: Original single timeframe analysis
    print("\n💱 Select currency pair:")
    print("   1. EURUSD")
    print("   2. AUDNZD")
    print("   3. Custom pair")
    
    pair_choice = input("\nEnter pair choice (1-3): ").strip()
    
    if pair_choice == '1':
        pair = 'EURUSD'
    elif pair_choice == '2':
        pair = 'AUDNZD'
    elif pair_choice == '3':
        pair = input("Enter custom pair (e.g., GBPUSD): ").strip().upper()
    else:
        print("⚠️  Invalid choice, using EURUSD")
        pair = 'EURUSD'
    
    # Get timeframe selection (YOUR ACTUAL TIMEFRAMES)
    print(f"\n⏰ Select timeframe for {pair}:")
    print("   1. 1D (Daily)")
    print("   2. 2D (2-Daily)")  
    print("   3. 3D (3-Daily)")
    print("   4. 4D (4-Daily)")
    
    tf_choice = input("\nEnter timeframe choice (1-4): ").strip()
    
    if tf_choice == '1':
        timeframe = '1D'
    elif tf_choice == '2':
        timeframe = '2D'
    elif tf_choice == '3':
        timeframe = '3D'
    elif tf_choice == '4':
        timeframe = '4D'
    else:
        print("⚠️  Invalid choice, using 1D")
        timeframe = '1D'
    
    # Get user input for backtest period
    print(f"\n📅 Select backtest period for {pair} {timeframe}:")
    print("   1. Last 6 months (180 days)")
    print("   2. Last 1 year (365 days)")  
    print("   3. Last 2 years (730 days)")
    print("   4. Last 3 years (1095 days)")
    print("   5. All available data")
    print("   6. Custom days")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        days_back = 180
    elif choice == '2':
        days_back = 365
    elif choice == '3':
        days_back = 730
    elif choice == '4':
        days_back = 1095
    elif choice == '5':
        days_back = 9999  # Use all available data
    elif choice == '6':
        try:
            days_back = int(input("Enter number of days back: "))
            if days_back < 100:
                print("⚠️  Minimum 100 days required, using 100")
                days_back = 100
        except ValueError:
            print("⚠️  Invalid input, using default 730 days")
            days_back = 730
    else:
        print("⚠️  Invalid choice, using default 730 days")
        days_back = 730
    
    # Run analysis with selected parameters
    print(f"\n🚀 Starting backtest: {pair} {timeframe} - {days_back} days")
    
    backtester = FixedMomentumVsReversalBacktester()
    results = backtester.run_comprehensive_analysis(days_back, timeframe, pair)
    
    print(f"\n✅ Analysis complete! Results saved to results/ directory")
    print(f"📊 Pair: {pair} | Timeframe: {timeframe} | Period: {days_back} days")


def run_multi_timeframe_analysis():
    """NEW: Multi-timeframe automation function"""
    print("\n🚀 MULTI-TIMEFRAME AUTOMATED ANALYSIS")
    print("=" * 50)
    
    # Get currency pair selection
    print("\n💱 Select currency pair:")
    print("   1. EURUSD")
    print("   2. AUDNZD") 
    print("   3. GBPUSD")
    print("   4. USDJPY")
    print("   5. Custom pair")
    
    pair_choice = input("\nEnter pair choice (1-5): ").strip()
    
    if pair_choice == '1':
        pair = 'EURUSD'
    elif pair_choice == '2':
        pair = 'AUDNZD'
    elif pair_choice == '3':
        pair = 'GBPUSD'
    elif pair_choice == '4':
        pair = 'USDJPY'
    elif pair_choice == '5':
        pair = input("Enter custom pair (e.g., CADJPY): ").strip().upper()
    else:
        print("⚠️  Invalid choice, using EURUSD")
        pair = 'EURUSD'
    
    # Get analysis period
    print(f"\n📅 Select analysis period for {pair}:")
    print("   1. Standard (2 years - 730 days)")
    print("   2. Extended (3 years - 1095 days)")
    print("   3. Maximum (All available data)")
    print("   4. Quick test (6 months - 180 days)")
    
    period_choice = input("\nEnter period choice (1-4): ").strip()
    
    if period_choice == '1':
        days_back = 730
    elif period_choice == '2':
        days_back = 1095
    elif period_choice == '3':
        days_back = 9999
    elif period_choice == '4':
        days_back = 180
    else:
        days_back = 730
    
    # Define YOUR ACTUAL timeframes to test
    timeframes = ['1D', '2D', '3D', '4D']
    
    print(f"\n✅ CONFIGURATION:")
    print(f"   Pair: {pair}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Period: {days_back} days")
    
    confirm = input(f"\n🚀 Start automated analysis? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Analysis cancelled")
        return
    
    # Run analysis for each timeframe
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING AUTOMATED MULTI-TIMEFRAME ANALYSIS")
    print(f"{'='*60}")
    
    for i, timeframe in enumerate(timeframes, 1):
        print(f"\n{'='*20} TIMEFRAME {i}/{len(timeframes)}: {timeframe} {'='*20}")
        
        try:
            backtester = FixedMomentumVsReversalBacktester()
            results = backtester.run_comprehensive_analysis(days_back, timeframe, pair)
            all_results[timeframe] = results
            
            # Quick summary
            if results:
                valid_results = [r for r in results if r['total_trades'] > 0]
                if valid_results:
                    best = max(valid_results, key=lambda x: x['profit_factor'])
                    print(f"✅ Best for {timeframe}: {best['strategy']} @ {best['distance_threshold']}x")
                    print(f"   PF: {best['profit_factor']}, WR: {best['win_rate']}%, {best['total_trades']} trades")
                else:
                    print(f"⚠️  No profitable strategies for {timeframe}")
            
        except Exception as e:
            print(f"❌ Error analyzing {timeframe}: {e}")
            all_results[timeframe] = None
            continue
    
    # Generate final comparison report
    generate_multi_timeframe_report(pair, all_results)
    
    # NEW: Create multi-timeframe visualization
    create_multi_timeframe_summary_chart(pair, all_results)


def generate_multi_timeframe_report(pair: str, all_results: Dict):
    """Generate comprehensive multi-timeframe comparison report"""
    print(f"\n{'='*80}")
    print(f"📊 FINAL MULTI-TIMEFRAME COMPARISON REPORT: {pair}")
    print(f"{'='*80}")
    
    # Create master comparison table
    print(f"\n{'Timeframe':<12} {'Best Strategy':<15} {'Distance':<10} {'Trades':<8} "
          f"{'Win Rate':<10} {'Profit Factor':<15} {'Return %':<12}")
    print("-" * 85)
    
    best_performers = []
    
    for timeframe, results in all_results.items():
        if not results:
            print(f"{timeframe:<12} {'NO DATA':<15}")
            continue
        
        # Find best performer for this timeframe
        valid_results = [r for r in results if r['total_trades'] > 0]
        
        if valid_results:
            best = max(valid_results, key=lambda x: x['profit_factor'])
            best_performers.append({
                'timeframe': timeframe,
                'strategy': best['strategy'],
                'distance': best['distance_threshold'],
                'trades': best['total_trades'],
                'win_rate': best['win_rate'],
                'profit_factor': best['profit_factor'],
                'return_pct': best['total_return']
            })
            
            print(f"{timeframe:<12} {best['strategy']:<15} {best['distance_threshold']:<10} "
                  f"{best['total_trades']:<8} {best['win_rate']:<10}% "
                  f"{best['profit_factor']:<15} {best['total_return']:<12}%")
        else:
            print(f"{timeframe:<12} {'NO TRADES':<15}")
    
    # Overall winners
    if best_performers:
        print(f"\n🏆 OVERALL WINNERS ACROSS ALL TIMEFRAMES:")
        
        # Highest profit factor
        best_pf = max(best_performers, key=lambda x: x['profit_factor'])
        print(f"   🥇 Highest Profit Factor: {best_pf['timeframe']} {best_pf['strategy']} "
              f"@ {best_pf['distance']}x → PF: {best_pf['profit_factor']}")
        
        # Highest return
        best_return = max(best_performers, key=lambda x: x['return_pct'])
        print(f"   💰 Highest Return: {best_return['timeframe']} {best_return['strategy']} "
              f"@ {best_return['distance']}x → {best_return['return_pct']}% return")
        
        # Best win rate
        best_wr = max(best_performers, key=lambda x: x['win_rate'])
        print(f"   🎯 Best Win Rate: {best_wr['timeframe']} {best_wr['strategy']} "
              f"@ {best_wr['distance']}x → {best_wr['win_rate']}% WR")
        
        # Most active
        most_trades = max(best_performers, key=lambda x: x['trades'])
        print(f"   📊 Most Active: {most_trades['timeframe']} {most_trades['strategy']} "
              f"@ {most_trades['distance']}x → {most_trades['trades']} trades")
        
        # Best timeframe overall
        tf_avg_pf = {}
        for perf in best_performers:
            tf = perf['timeframe']
            if tf not in tf_avg_pf:
                tf_avg_pf[tf] = []
            tf_avg_pf[tf].append(perf['profit_factor'])
        
        if tf_avg_pf:
            tf_averages = {tf: sum(pfs)/len(pfs) for tf, pfs in tf_avg_pf.items()}
            best_tf = max(tf_averages.keys(), key=lambda x: tf_averages[x])
            
            print(f"\n🎯 RECOMMENDED TIMEFRAME: {best_tf}")
            print(f"   Average performance: {tf_averages[best_tf]:.2f} PF")
            
            # Show all timeframe rankings
            print(f"\n📈 TIMEFRAME PERFORMANCE RANKING:")
            sorted_tf = sorted(tf_averages.items(), key=lambda x: x[1], reverse=True)
            for i, (tf, avg_pf) in enumerate(sorted_tf, 1):
                print(f"   {i}. {tf}: {avg_pf:.2f} PF")
    
    else:
        print(f"\n❌ No profitable strategies found across any timeframes")
        print(f"💡 Consider:")
        print(f"   • Longer analysis period")
        print(f"   • Different currency pair")
        print(f"   • Lower distance thresholds")
    
    print(f"\n✅ Multi-timeframe analysis complete for {pair}!")
    print(f"📁 Individual timeframe charts saved in results/ directory")


def create_multi_timeframe_summary_chart(pair: str, all_results: Dict):
    """Create summary visualization across all timeframes"""
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    print(f"\n📊 Creating multi-timeframe summary chart...")
    
    # Collect data for visualization
    timeframes = ['1D', '2D', '3D', '4D']
    momentum_pf = []
    reversal_pf = []
    momentum_trades = []
    reversal_trades = []
    
    for tf in timeframes:
        if tf in all_results and all_results[tf]:
            # Get best momentum and reversal results for this timeframe
            momentum_results = [r for r in all_results[tf] if r['strategy'] == 'Momentum' and r['total_trades'] > 0]
            reversal_results = [r for r in all_results[tf] if r['strategy'] == 'Reversal' and r['total_trades'] > 0]
            
            momentum_pf.append(max([r['profit_factor'] for r in momentum_results], default=0))
            reversal_pf.append(max([r['profit_factor'] for r in reversal_results], default=0))
            momentum_trades.append(sum([r['total_trades'] for r in momentum_results]))
            reversal_trades.append(sum([r['total_trades'] for r in reversal_results]))
        else:
            momentum_pf.append(0)
            reversal_pf.append(0)
            momentum_trades.append(0)
            reversal_trades.append(0)
    
    # Create the chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Profit Factor Comparison
    x = np.arange(len(timeframes))
    width = 0.35
    
    ax1.bar(x - width/2, momentum_pf, width, label='Momentum', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, reversal_pf, width, label='Reversal', color='darkorange', alpha=0.8)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    
    ax1.set_title(f'{pair}: Best Profit Factor by Timeframe', fontweight='bold')
    ax1.set_xlabel('Timeframe')
    ax1.set_ylabel('Profit Factor')
    ax1.set_xticks(x)
    ax1.set_xticklabels(timeframes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (m_pf, r_pf) in enumerate(zip(momentum_pf, reversal_pf)):
        if m_pf > 0:
            ax1.text(i - width/2, m_pf + 0.05, f'{m_pf:.1f}', ha='center', va='bottom', fontsize=9)
        if r_pf > 0:
            ax1.text(i + width/2, r_pf + 0.05, f'{r_pf:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Trade Count Comparison
    ax2.bar(x - width/2, momentum_trades, width, label='Momentum', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, reversal_trades, width, label='Reversal', color='darkorange', alpha=0.8)
    
    ax2.set_title(f'{pair}: Total Trades by Timeframe', fontweight='bold')
    ax2.set_xlabel('Timeframe')
    ax2.set_ylabel('Number of Trades')
    ax2.set_xticks(x)
    ax2.set_xticklabels(timeframes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Combined Performance Score
    combined_scores = []
    for i in range(len(timeframes)):
        # Simple scoring: PF * log(trades + 1) to favor both profitability and activity
        momentum_score = momentum_pf[i] * np.log(momentum_trades[i] + 1)
        reversal_score = reversal_pf[i] * np.log(reversal_trades[i] + 1)
        combined_scores.append(max(momentum_score, reversal_score))
    
    colors = ['gold' if score == max(combined_scores) else 'lightblue' for score in combined_scores]
    bars = ax3.bar(timeframes, combined_scores, color=colors, alpha=0.8, edgecolor='navy')
    
    ax3.set_title(f'{pair}: Overall Performance Score by Timeframe', fontweight='bold')
    ax3.set_xlabel('Timeframe')
    ax3.set_ylabel('Performance Score (PF × log(Trades))')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight best performer
    best_idx = combined_scores.index(max(combined_scores))
    if max(combined_scores) > 0:
        ax3.annotate(f'Best: {timeframes[best_idx]}', 
                    xy=(best_idx, combined_scores[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 4: Win Rate Heatmap (if we have that data)
    # For now, show strategy preference by timeframe
    strategy_preference = []
    for i in range(len(timeframes)):
        if momentum_pf[i] > reversal_pf[i]:
            strategy_preference.append(1)  # Momentum preferred
        elif reversal_pf[i] > momentum_pf[i]:
            strategy_preference.append(-1)  # Reversal preferred
        else:
            strategy_preference.append(0)  # Tie or no data
    
    colors_pref = ['steelblue' if pref == 1 else 'darkorange' if pref == -1 else 'gray' 
                   for pref in strategy_preference]
    
    ax4.bar(timeframes, [abs(p) for p in strategy_preference], color=colors_pref, alpha=0.8)
    ax4.set_title(f'{pair}: Preferred Strategy by Timeframe', fontweight='bold')
    ax4.set_xlabel('Timeframe')
    ax4.set_ylabel('Strategy Preference')
    ax4.set_ylim(0, 1.2)
    
    # Add strategy labels
    for i, (tf, pref) in enumerate(zip(timeframes, strategy_preference)):
        if pref == 1:
            ax4.text(i, 0.5, 'Momentum', ha='center', va='center', fontweight='bold', color='white')
        elif pref == -1:
            ax4.text(i, 0.5, 'Reversal', ha='center', va='center', fontweight='bold', color='white')
        else:
            ax4.text(i, 0.5, 'No Data', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/multi_timeframe_summary_{pair}_{timestamp}.png"
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Multi-timeframe summary chart saved: {filename}")
    
    plt.show()

if __name__ == "__main__":
    main()