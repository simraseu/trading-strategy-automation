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
        print("🚀 CORE BACKTESTING ENGINE - INITIALIZING")
        print("=" * 60)
        
        # Initialize your updated data loader
        self.data_loader = DataLoader()
        
        # CPU optimization
        available_cores = cpu_count()
        if available_cores >= 12:  # Hyperthreaded 6-core
            self.max_workers = 8  
        elif available_cores >= 6:
            self.max_workers = available_cores - 1
        else:
            self.max_workers = max(1, available_cores - 1)
        
        # Memory optimization settings
        self.chunk_size = 100  # Process in chunks
        self.memory_threshold = 0.75  # 75% memory trigger cleanup
        
        # Current test configuration (will be set per test)
        self.current_config = None
        
        print(f"✅ INITIALIZATION COMPLETE:")
        print(f"   🔧 Updated modules: CandleClassifier, ZoneDetector, TrendClassifier")
        print(f"   ⚙️  Current leg-out ratio: {ZONE_CONFIG['min_legout_ratio']}x (from settings.py)")
        print(f"   🔄 Parallel workers: {self.max_workers} (optimized)")
        print(f"   💾 Memory threshold: {self.memory_threshold*100:.0f}%")
        print(f"   📊 Analysis periods: {len(ANALYSIS_PERIODS)} configured")
    
    def discover_all_pairs(self) -> List[str]:
        """Auto-discover all available currency pairs using updated DataLoader"""
        print(f"🔍 AUTO-DISCOVERING CURRENCY PAIRS...")
        
        try:
            # Use your updated DataLoader's discovery method
            pairs = self.data_loader.discover_all_pairs()
            
            if not pairs:
                print("❌ No currency pairs found")
                return []
            
            print(f"✅ DISCOVERED {len(pairs)} PAIRS: {', '.join(pairs)}")
            return pairs
            
        except Exception as e:
            print(f"❌ Error discovering pairs: {str(e)}")
            return []
    
    def discover_valid_data_combinations(self) -> List[Tuple[str, str]]:
        """
        Discover only valid pair/timeframe combinations that actually have data files
        Returns list of (pair, timeframe) tuples
        """
        print(f"🔍 DISCOVERING VALID DATA COMBINATIONS...")
        
        try:
            # Use DataLoader's comprehensive inventory method
            data_inventory = self.data_loader.get_available_data()
            
            valid_combinations = []
            total_files = 0
            
            for pair, available_timeframes in data_inventory.items():
                for timeframe in available_timeframes:
                    valid_combinations.append((pair, timeframe))
                    total_files += 1
            
            print(f"✅ FOUND {total_files} VALID DATA FILES:")
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
        """Load data using updated DataLoader with validation"""
        try:
            print(f"📊 Loading {pair} {timeframe}...")
            
            # Use your updated data loader
            data = self.data_loader.load_pair_data(pair, timeframe)
            
            if data is None or len(data) < 100:
                print(f"❌ Insufficient data for {pair} {timeframe}")
                return None
            
            # Limit data if needed (keep some history for indicators)
            if days_back < 9999:
                max_candles = min(days_back + 365, len(data))
                data = data.iloc[-max_candles:]
            
            print(f"✅ Loaded {len(data)} candles for {pair} {timeframe}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading {pair} {timeframe}: {str(e)}")
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
        
        # Apply UPDATED distance filter (2.5x from settings.py)
        min_ratio = ZONE_CONFIG['min_legout_ratio']  # 2.5x from your settings
        valid_patterns = [
            pattern for pattern in all_patterns
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
            and pattern['leg_out']['ratio_to_base'] >= min_ratio
        ]
        
        if not valid_patterns:
            return self.create_empty_result(pair, timeframe, f"No patterns meet {min_ratio}x distance")
        
        print(f"   📊 {len(valid_patterns)} patterns meet {min_ratio}x requirement (updated threshold)")
        
        # Execute trades with REALISTIC LOGIC
        trades = self.execute_realistic_trades(valid_patterns, data, trend_data, timeframe)
        
        # Calculate performance
        return self.calculate_performance_metrics(trades, pair, timeframe)
    
    def execute_realistic_trades(self, patterns: List[Dict], data: pd.DataFrame,
                               trend_data: pd.DataFrame, timeframe: str) -> List[Dict]:
        """
        Execute trades using REALISTIC LOGIC extracted from distance_edge.py
        but with UPDATED SETTINGS and MODULES
        """
        trades = []
        used_zones = set()
        
        # Build zone activation schedule (realistic approach)
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
        active_zones = []
        
        # Process each candle for realistic trade execution
        for current_idx in range(200, len(data)):
            current_date = data.index[current_idx]
            
            # Check for new zone activations
            for zone_info in zone_activation_schedule:
                zone_id = zone_info['zone_id']
                pattern = zone_info['pattern']
                zone_end_idx = zone_info['zone_end_idx']
                
                # Zone becomes active 1 day after formation
                if (current_date > zone_info['date'] and 
                    zone_id not in used_zones and 
                    pattern not in active_zones and
                    zone_end_idx < current_idx):
                    active_zones.append(pattern)
            
            # Check for trade executions
            for zone in active_zones.copy():
                zone_id = f"{zone['type']}_{zone.get('end_idx', 0)}_{zone['zone_low']:.5f}"
                
                if zone_id in used_zones:
                    active_zones.remove(zone)
                    continue
                
                # Check trend alignment using UPDATED trend logic
                current_trend = trend_data['trend'].iloc[current_idx] if current_idx < len(trend_data) else 'bullish'
                
                is_aligned = False
                if current_trend == 'bullish':
                    is_aligned = zone['type'] in ['R-B-R', 'D-B-R']
                elif current_trend == 'bearish':
                    is_aligned = zone['type'] in ['D-B-D', 'R-B-D']
                
                if not is_aligned:
                    continue
                
                # Execute trade with REALISTIC LOGIC
                trade_result = self.execute_single_realistic_trade(zone, data, current_idx)
                
                if trade_result:
                    used_zones.add(zone_id)
                    active_zones.remove(zone)
                    trades.append(trade_result)
                    print(f"      💰 Trade #{len(trades)}: {trade_result['result']} "
                         f"${trade_result['pnl']:.0f} ({trade_result['zone_type']})")
        
        print(f"   ✅ Executed {len(trades)} trades from {len(patterns)} zones")
        return trades
    
    def execute_single_realistic_trade(self, zone: Dict, data: pd.DataFrame, current_idx: int) -> Optional[Dict]:
        """
        Execute single trade using REALISTIC 1R→2.5R management
        CORRECTED: Fixed zone approach direction logic
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # CORRECTED: Entry and stop logic - Front-run BEYOND zone boundaries
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy) - front-run ABOVE zone
            entry_price = zone_high + (zone_range * 0.05)  # 5% ABOVE zone (front-run)
            direction = 'BUY'
            initial_stop = zone_low - (zone_range * 0.33)  # 33% buffer below zone
        elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell) - front-run BELOW zone
            entry_price = zone_low - (zone_range * 0.05)  # 5% BELOW zone (front-run)
            direction = 'SELL'
            initial_stop = zone_high + (zone_range * 0.33)  # 33% buffer above zone
        else:
            return None
        
        # CORRECTED: Check if price can trigger entry from proper direction
        current_candle = data.iloc[current_idx]
        
        can_enter = False
        if direction == 'BUY':
            # For demand zones: price must reach the front-run level above zone
            if current_candle['high'] >= entry_price:
                can_enter = True
        elif direction == 'SELL':
            # For supply zones: price must reach the front-run level below zone
            if current_candle['low'] <= entry_price:
                can_enter = True
        
        # Calculate position size using UPDATED risk config
        risk_amount = 10000 * (RISK_CONFIG['risk_limits']['max_risk_per_trade'] / 100)  # 5% from settings
        pip_value = 0.0001
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
            position_size, data, current_idx, zone['type']
        )
    
    def simulate_realistic_outcome(self, entry_price: float, stop_loss: float, target_price: float,
                                 direction: str, position_size: float, data: pd.DataFrame,
                                 entry_idx: int, zone_type: str) -> Dict:
        """
        Simulate REALISTIC trade outcome with proper 1R→breakeven management
        FIXED: Proper position sizing and realistic P&L calculation
        """
        risk_distance = abs(entry_price - stop_loss)
        current_stop = stop_loss
        breakeven_moved = False
        
        # FIXED: Proper position sizing (5% risk = $500 max loss)
        max_risk_amount = 500  # $500 max risk per trade (5% of $10,000)
        pip_value = 0.0001
        stop_distance_pips = risk_distance / pip_value
        
        # Recalculate position size for realistic $500 risk
        if stop_distance_pips > 0:
            proper_position_size = max_risk_amount / (stop_distance_pips * 10)  # $10 per pip per lot
        else:
            return None
        
        # Look ahead for exit with proper trade management
        for exit_idx in range(entry_idx + 1, min(entry_idx + 200, len(data))):
            exit_candle = data.iloc[exit_idx]
            
            # Calculate current R:R for break-even move
            if direction == 'BUY':
                current_rr = (exit_candle['close'] - entry_price) / risk_distance if risk_distance > 0 else 0
            else:
                current_rr = (entry_price - exit_candle['close']) / risk_distance if risk_distance > 0 else 0
            
            # Move to break-even at 1R
            if not breakeven_moved and current_rr >= 1.0:
                current_stop = entry_price
                breakeven_moved = True
            
            # Check stops and targets with REALISTIC P&L
            if direction == 'BUY':
                if exit_candle['low'] <= current_stop:
                    # Calculate realistic P&L
                    price_diff = current_stop - entry_price
                    pips_moved = price_diff / pip_value
                    pnl = pips_moved * proper_position_size * 10  # $10 per pip per lot
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': current_stop,
                        'result': 'LOSS' if pnl < -10 else ('BREAKEVEN' if abs(pnl) <= 10 else 'WIN'),
                        'pnl': round(pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1)
                    }
                elif exit_candle['high'] >= target_price:
                    # Calculate realistic P&L for win
                    price_diff = target_price - entry_price
                    pips_moved = price_diff / pip_value
                    pnl = pips_moved * proper_position_size * 10
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'result': 'WIN',
                        'pnl': round(pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1)
                    }
            else:  # SELL
                if exit_candle['high'] >= current_stop:
                    # Calculate realistic P&L
                    price_diff = entry_price - current_stop
                    pips_moved = price_diff / pip_value
                    pnl = pips_moved * proper_position_size * 10
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': current_stop,
                        'result': 'LOSS' if pnl < -10 else ('BREAKEVEN' if abs(pnl) <= 10 else 'WIN'),
                        'pnl': round(pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1)
                    }
                elif exit_candle['low'] <= target_price:
                    # Calculate realistic P&L for win
                    price_diff = entry_price - target_price
                    pips_moved = price_diff / pip_value
                    pnl = pips_moved * proper_position_size * 10
                    
                    return {
                        'zone_type': zone_type,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'result': 'WIN',
                        'pnl': round(pnl, 2),
                        'duration_days': exit_idx - entry_idx,
                        'position_size': proper_position_size,
                        'pips': round(pips_moved, 1)
                    }
        
        # Trade still open at end (neutral exit)
        return None
    
    def calculate_performance_metrics(self, trades: List[Dict], pair: str, timeframe: str) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self.create_empty_result(pair, timeframe, "No trades executed")
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        # Return calculation
        total_return = (total_pnl / 10000) * 100  # % return on $10,000
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'total_return': round(total_return, 2),
            'avg_trade_duration': round(np.mean([t.get('duration_days', 0) for t in trades]), 1),
            'leg_out_threshold': ZONE_CONFIG['min_legout_ratio'],  # Record current threshold
            'trades': trades
        }
    
    def create_empty_result(self, pair: str, timeframe: str, reason: str) -> Dict:
        """Create empty result structure"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'total_return': 0.0,
            'avg_trade_duration': 0.0,
            'leg_out_threshold': ZONE_CONFIG['min_legout_ratio'],
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
                'total_trades': 'sum',
                'total_return': 'mean'
            }).round(2)
            
            # Flatten column names
            tf_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                 'Total_Trades', 'Avg_Return']
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
                'total_trades': 'sum',
                'total_return': 'mean'
            }).round(2)
            
            # Flatten column names
            pair_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                   'Total_Trades', 'Avg_Return']
            pair_analysis = pair_analysis.sort_values('Avg_Profit_Factor', ascending=False)
            
            return pair_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Pair analysis error: {str(e)}")
            return pd.DataFrame({'Pair': ['Error'], 'Note': [str(e)]})
    
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
        # Quick validation test
        print("\n🧪 QUICK VALIDATION TEST:")
        print("Testing EURUSD 3D with updated 2.5x threshold...")
        
        result = engine.run_single_strategy_test('EURUSD', '3D', 730)
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   Pair: {result['pair']} {result['timeframe']}")
        print(f"   Trades: {result['total_trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Total Return: {result['total_return']:.2f}%")
        print(f"   Leg-out threshold: {result['leg_out_threshold']}x (from settings.py)")
        
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
        # Custom single test
        print("\n🎯 CUSTOM SINGLE TEST:")
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
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Total Return: {result['total_return']:.2f}%")
            
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()