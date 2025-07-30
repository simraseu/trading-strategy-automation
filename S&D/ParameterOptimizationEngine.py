"""
Parameter Optimization Backtesting Engine
Tests different stop-loss buffers, break-even levels, and take-profit targets
Built on top of CoreBacktestEngine - inherits ALL validation logic
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import psutil
import warnings
warnings.filterwarnings('ignore')

# Import the PERFECT CoreBacktestEngine
from core_backtest_engine import CoreBacktestEngine, run_single_test_worker
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

# PARAMETER OPTIMIZATION CONFIGURATIONS
OPTIMIZATION_PARAMS = {
    'stop_buffers': [0.25, 0.33, 0.40, 0.50],  # 25%, 33%(current), 40%, 50%
    'breakeven_levels': [0.5, 1.0, 1.5],  # 0.5R, 1.0R(current), 1.5R
    'take_profits': [1.0, 1.5, 2.0, 2.5, 3.0]  # 1.0R to 3.0R
}

class ParameterOptimizationEngine(CoreBacktestEngine):
    """
    Parameter optimization engine that inherits EVERYTHING from CoreBacktestEngine
    Only overrides specific trade management parameters for testing
    """
    
    def __init__(self, stop_buffer=0.33, breakeven_level=1.0, take_profit=2.5):
        """
        Initialize with specific parameters to test
        
        Args:
            stop_buffer: Stop loss buffer percentage (0.33 = 33%)
            breakeven_level: R-multiple for break-even move (1.0 = 1R)
            take_profit: Final take profit R-multiple (2.5 = 2.5R)
        """
        # Initialize parent class with ALL its logic
        super().__init__()
        
        # Override ONLY the trade management parameters
        self.stop_buffer = stop_buffer
        self.breakeven_level = breakeven_level
        self.take_profit = take_profit
        
        # Track parameter combination being tested
        self.param_combo = f"SB{int(stop_buffer*100)}_BE{breakeven_level}_TP{take_profit}"
        
        print(f"🔧 Parameter Test: Stop Buffer={stop_buffer*100:.0f}%, BE={breakeven_level}R, TP={take_profit}R")
    
    def execute_single_realistic_trade(self, zone: Dict, data: pd.DataFrame, current_idx: int) -> Optional[Dict]:
        """
        OVERRIDE: Execute trade with CUSTOM parameters while keeping ALL other logic
        This is a surgical override - everything else stays exactly the same
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # Entry logic UNCHANGED - 5% front-run
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
            entry_price = zone_high + (zone_range * 0.05)  # 5% above zone
            direction = 'BUY'
            # CUSTOM STOP BUFFER (instead of hardcoded 0.33)
            initial_stop = zone_low - (zone_range * self.stop_buffer)
        elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell)
            entry_price = zone_low - (zone_range * 0.05)  # 5% below zone
            direction = 'SELL'
            # CUSTOM STOP BUFFER (instead of hardcoded 0.33)
            initial_stop = zone_high + (zone_range * self.stop_buffer)
        else:
            return None
        
        # Check if current candle can trigger entry (UNCHANGED)
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
        
        # Calculate position size (UNCHANGED)
        risk_amount = 10000 * (RISK_CONFIG['risk_limits']['max_risk_per_trade'] / 100)
        pip_value = self.get_pip_value_for_pair(zone.get('pair', 'EURUSD'))
        stop_distance_pips = abs(entry_price - initial_stop) / pip_value
        
        if stop_distance_pips <= 0:
            return None
        
        position_size = risk_amount / stop_distance_pips
        
        # Set CUSTOM target based on parameter
        risk_distance = abs(entry_price - initial_stop)
        target_rr = self.take_profit  # Use custom parameter instead of hardcoded 2.5
        
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * target_rr)
        else:
            target_price = entry_price - (risk_distance * target_rr)
        
        # Simulate outcome with CUSTOM parameters
        return self.simulate_realistic_outcome(
            entry_price, initial_stop, target_price, direction, 
            position_size, data, current_idx, zone['type'], stop_distance_pips, zone.get('pair', 'EURUSD'),
            zone_high, zone_low, zone
        )
    
    def simulate_realistic_outcome(self, entry_price: float, stop_loss: float, target_price: float,
                                   direction: str, position_size: float, data: pd.DataFrame,
                                   entry_idx: int, zone_type: str, stop_distance_pips: float, pair: str,
                                   zone_high: float = None, zone_low: float = None, zone: Dict = None) -> Dict:
        """
        OVERRIDE: Simulate outcome with CUSTOM break-even level
        All other logic (spreads, commissions, etc.) remains UNCHANGED
        """
        # Transaction costs UNCHANGED
        spread_pips = 2.0
        commission_per_lot = 7.0
        pip_value = self.get_pip_value_for_pair(pair)
        
        # Apply spread cost to entry UNCHANGED
        if direction == 'BUY':
            entry_price += (spread_pips * pip_value)
        else:
            entry_price -= (spread_pips * pip_value)
        
        risk_distance = abs(entry_price - stop_loss)
        current_stop = stop_loss
        breakeven_moved = False
        
        # Position sizing UNCHANGED
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
        
        # Look ahead for exit with CUSTOM break-even level
        for exit_idx in range(entry_idx + 1, min(entry_idx + 50, len(data))):
            exit_candle = data.iloc[exit_idx]
            
            # CUSTOM BREAK-EVEN LEVEL (instead of hardcoded 1R)
            breakeven_target = entry_price + (risk_distance * self.breakeven_level) if direction == 'BUY' else entry_price - (risk_distance * self.breakeven_level)
            
            # Check for break-even hit (wick-based) with CUSTOM level
            if not breakeven_moved:
                if direction == 'BUY' and exit_candle['high'] >= breakeven_target:
                    current_stop = entry_price  # Move to exact entry
                    breakeven_moved = True
                elif direction == 'SELL' and exit_candle['low'] <= breakeven_target:
                    current_stop = entry_price  # Move to exact entry
                    breakeven_moved = True
            
            # Exit logic UNCHANGED (wick-based)
            if direction == 'BUY':
                # Check stop loss hit
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
                    
                    trade_summary = f"{zone_type} zone - Params: {self.param_combo} - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
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
                        'zone_low': zone_low,
                        'zone_id': zone.get('zone_id', 'unknown') if zone else 'unknown',
                        'param_combo': self.param_combo
                    }
                # Check target hit
                elif exit_candle['high'] >= target_price:
                    price_diff = target_price - entry_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    trade_summary = f"{zone_type} zone - Params: {self.param_combo} - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
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
                        'zone_low': zone_low,
                        'param_combo': self.param_combo
                    }
            else:  # SELL
                # Check stop loss hit
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
                    
                    trade_summary = f"{zone_type} zone - Params: {self.param_combo} - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
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
                        'zone_low': zone_low,
                        'param_combo': self.param_combo
                    }
                # Check target hit
                elif exit_candle['low'] <= target_price:
                    price_diff = entry_price - target_price
                    pips_moved = price_diff / pip_value
                    gross_pnl = pips_moved * proper_position_size * pip_value_per_lot
                    total_commission = commission_per_lot * proper_position_size * 2
                    net_pnl = gross_pnl - total_commission
                    
                    trade_summary = f"{zone_type} zone - Params: {self.param_combo} - Entry: {entry_price:.6f}, Stop: {stop_loss:.6f}, Duration: {exit_idx - entry_idx} candles, Result: {pips_moved:+.0f} pips = ${net_pnl:.0f}"
                    
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
                        'zone_low': zone_low,
                        'param_combo': self.param_combo
                    }
        
        # Trade still open at end
        return None

def run_parameter_optimization(pair='EURUSD', timeframe='3D', days_back=730):  # Add parameters
    """
    Run comprehensive parameter optimization across all combinations
    Tests all stop buffer, break-even, and take-profit combinations
    """
    print("🚀 PARAMETER OPTIMIZATION BACKTESTING ENGINE")
    print("=" * 70)
    print("Testing combinations of:")
    print(f"- Stop Buffers: {[f'{x*100:.0f}%' for x in OPTIMIZATION_PARAMS['stop_buffers']]}")
    print(f"- Break-even Levels: {[f'{x}R' for x in OPTIMIZATION_PARAMS['breakeven_levels']]}")
    print(f"- Take Profits: {[f'{x}R' for x in OPTIMIZATION_PARAMS['take_profits']]}")
    print("=" * 70)
    
    # Check system resources
    if not check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    # Generate all parameter combinations
    test_combinations = []
    for stop_buffer in OPTIMIZATION_PARAMS['stop_buffers']:
        for breakeven_level in OPTIMIZATION_PARAMS['breakeven_levels']:
            for take_profit in OPTIMIZATION_PARAMS['take_profits']:
                test_combinations.append({
                    'stop_buffer': stop_buffer,
                    'breakeven_level': breakeven_level,
                    'take_profit': take_profit
                })
    
    total_combinations = len(test_combinations)
    print(f"\n📊 Total parameter combinations: {total_combinations}")
    
    # Use provided parameters (FIXED)
    test_pair = pair
    test_timeframe = timeframe
    test_days_back = days_back  # FIXED: Use parameter instead of undefined 'days'
    
    print(f"📊 Testing on: {test_pair} {test_timeframe} ({test_days_back} days)")
    
    # Store results
    all_results = []
    start_time = time.time()
    
    # Run each parameter combination
    for i, params in enumerate(test_combinations, 1):
        print(f"\n🔄 Testing combination {i}/{total_combinations}")
        
        try:
            # Create engine with specific parameters
            engine = ParameterOptimizationEngine(
                stop_buffer=params['stop_buffer'],
                breakeven_level=params['breakeven_level'],
                take_profit=params['take_profit']
            )
            
            # Run backtest (FIXED: Use test_days_back)
            result = engine.run_single_strategy_test(test_pair, test_timeframe, test_days_back)
            
            # Add parameter info to result
            result['stop_buffer'] = params['stop_buffer']
            result['breakeven_level'] = params['breakeven_level']
            result['take_profit'] = params['take_profit']
            result['param_combo'] = f"SB{int(params['stop_buffer']*100)}_BE{params['breakeven_level']}_TP{params['take_profit']}"
            
            # Calculate additional metrics
            if result['total_trades'] > 0:
                # Risk/Reward achieved
                avg_win = result['gross_profit'] / result['winning_trades'] if result['winning_trades'] > 0 else 0
                avg_loss = result['gross_loss'] / result['losing_trades'] if result['losing_trades'] > 0 else 0
                achieved_rr = avg_win / avg_loss if avg_loss > 0 else 0
                result['achieved_rr'] = round(achieved_rr, 2)
                
                # Expectancy per trade
                result['expectancy'] = round(result['total_pnl'] / result['total_trades'], 2)
            else:
                result['achieved_rr'] = 0
                result['expectancy'] = 0
            
            all_results.append(result)
            
            # Quick summary
            print(f"   ✅ Trades: {result['total_trades']}, PF: {result['profit_factor']:.2f}, WR: {result['win_rate']:.1f}%, Return: {result['total_return']:.2f}%")
            
            # Memory cleanup
            del engine
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            error_result = {
                'pair': test_pair,
                'timeframe': test_timeframe,
                'stop_buffer': params['stop_buffer'],
                'breakeven_level': params['breakeven_level'],
                'take_profit': params['take_profit'],
                'param_combo': f"SB{int(params['stop_buffer']*100)}_BE{params['breakeven_level']}_TP{params['take_profit']}",
                'total_trades': 0,
                'error': str(e)
            }
            all_results.append(error_result)
    
    # Analysis complete
    total_time = time.time() - start_time
    print(f"\n✅ OPTIMIZATION COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"⚡ Speed: {total_combinations/total_time:.1f} tests/second")
    
    # Generate comprehensive report
    generate_optimization_report(all_results, test_pair, test_timeframe)
    
    # Find best parameter combination
    valid_results = [r for r in all_results if r.get('total_trades', 0) > 0]
    if valid_results:
        # Best by profit factor
        best_pf = max(valid_results, key=lambda x: x['profit_factor'])
        print(f"\n🏆 BEST PROFIT FACTOR: {best_pf['param_combo']}")
        print(f"   PF: {best_pf['profit_factor']:.2f}, WR: {best_pf['win_rate']:.1f}%, Return: {best_pf['total_return']:.2f}%")
        
        # Best by total return
        best_return = max(valid_results, key=lambda x: x['total_return'])
        print(f"\n💰 BEST TOTAL RETURN: {best_return['param_combo']}")
        print(f"   Return: {best_return['total_return']:.2f}%, PF: {best_return['profit_factor']:.2f}, WR: {best_return['win_rate']:.1f}%")
        
        # Most balanced (high PF with reasonable win rate)
        balanced_results = [r for r in valid_results if r['win_rate'] >= 30 and r['profit_factor'] >= 1.5]
        if balanced_results:
            best_balanced = max(balanced_results, key=lambda x: x['profit_factor'] * (x['win_rate']/100))
            print(f"\n⚖️  MOST BALANCED: {best_balanced['param_combo']}")
            print(f"   PF: {best_balanced['profit_factor']:.2f}, WR: {best_balanced['win_rate']:.1f}%, Return: {best_balanced['total_return']:.2f}%")

def generate_optimization_report(results: List[Dict], pair: str, timeframe: str):
    """
    Generate comprehensive Excel report for parameter optimization
    """
    print(f"\n📊 GENERATING OPTIMIZATION REPORT...")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/parameter_optimization_{pair}_{timeframe}_{timestamp}.xlsx"
    os.makedirs('results', exist_ok=True)
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # SHEET 1: All Results
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # SHEET 2: Summary by Stop Buffer
            if 'stop_buffer' in df.columns:
                stop_summary = df.groupby('stop_buffer').agg({
                    'total_trades': 'mean',
                    'profit_factor': 'mean',
                    'win_rate': 'mean',
                    'total_return': 'mean',
                    'avg_trade_duration': 'mean',
                    'expectancy': 'mean'
                }).round(2)
                stop_summary.index = [f"{int(x*100)}%" for x in stop_summary.index]
                stop_summary.to_excel(writer, sheet_name='Stop_Buffer_Analysis')
            
            # SHEET 3: Summary by Break-even Level
            if 'breakeven_level' in df.columns:
                be_summary = df.groupby('breakeven_level').agg({
                    'total_trades': 'mean',
                    'profit_factor': 'mean',
                    'win_rate': 'mean',
                    'total_return': 'mean',
                    'breakeven_trades': 'mean',
                    'expectancy': 'mean'
                }).round(2)
                be_summary.index = [f"{x}R" for x in be_summary.index]
                be_summary.to_excel(writer, sheet_name='Breakeven_Analysis')
            
            # SHEET 4: Summary by Take Profit
            if 'take_profit' in df.columns:
                tp_summary = df.groupby('take_profit').agg({
                    'total_trades': 'mean',
                    'profit_factor': 'mean',
                    'win_rate': 'mean',
                    'total_return': 'mean',
                    'achieved_rr': 'mean',
                    'expectancy': 'mean'
                }).round(2)
                tp_summary.index = [f"{x}R" for x in tp_summary.index]
                tp_summary.to_excel(writer, sheet_name='TakeProfit_Analysis')
            
            # SHEET 5: Top 10 Combinations
            valid_df = df[df['total_trades'] > 0].copy()
            if len(valid_df) > 0:
                top_10 = valid_df.nlargest(10, 'profit_factor')[
                    ['param_combo', 'profit_factor', 'win_rate', 'total_return', 
                     'total_trades', 'expectancy', 'avg_trade_duration']
                ]
                top_10.to_excel(writer, sheet_name='Top_10_Combinations', index=False)
            
            # SHEET 6: Heatmap Data (for visualization)
            if all(col in df.columns for col in ['stop_buffer', 'take_profit', 'profit_factor']):
                # Create pivot table for each break-even level
                for be_level in df['breakeven_level'].unique():
                    be_data = df[df['breakeven_level'] == be_level]
                    heatmap = be_data.pivot_table(
                        values='profit_factor',
                        index='stop_buffer',
                        columns='take_profit',
                        aggfunc='mean'
                    ).round(2)
                    
                    # Convert index and columns to percentages/R-values
                    heatmap.index = [f"{int(x*100)}%" for x in heatmap.index]
                    heatmap.columns = [f"{x}R" for x in heatmap.columns]
                    
                    sheet_name = f'Heatmap_BE_{be_level}R'[:31]  # Excel sheet name limit
                    heatmap.to_excel(writer, sheet_name=sheet_name)
        
        print(f"📁 OPTIMIZATION REPORT SAVED: {filename}")
        print(f"📊 Multiple analysis sheets created for parameter comparison")
        
    except Exception as e:
        print(f"❌ Error creating Excel report: {str(e)}")
        # Fallback: Save as CSV
        csv_filename = filename.replace('.xlsx', '.csv')
        df.to_csv(csv_filename, index=False)
        print(f"📁 Fallback CSV saved: {csv_filename}")

def check_system_resources() -> bool:
    """Check system resources before optimization"""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = cpu_count()
    memory_percent = psutil.virtual_memory().percent
    
    print(f"\n💻 SYSTEM RESOURCES CHECK:")
    print(f"   RAM: {memory_gb:.1f} GB available")
    print(f"   CPU: {cpu_cores} cores")
    print(f"   Current memory usage: {memory_percent:.1f}%")
    
    if memory_gb < 8:
        print("⚠️  WARNING: Less than 8GB RAM. Consider testing fewer combinations.")
    
    if memory_percent > 60:
        print("⚠️  WARNING: High memory usage. Close other applications.")
    
    return memory_gb >= 4  # Minimum 4GB required

def run_parallel_optimization(pair='EURUSD', timeframe='3D', days_back=3847):
    """
    Run parameter optimization in parallel for faster execution
    """
    print("🚀 PARALLEL PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Check system resources
    if not check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    # Generate all parameter combinations
    test_combinations = []
    for stop_buffer in OPTIMIZATION_PARAMS['stop_buffers']:
        for breakeven_level in OPTIMIZATION_PARAMS['breakeven_levels']:
            for take_profit in OPTIMIZATION_PARAMS['take_profits']:
                test_combinations.append({
                    'pair': pair,
                    'timeframe': timeframe,
                    'days_back': days_back,
                    'stop_buffer': stop_buffer,
                    'breakeven_level': breakeven_level,
                    'take_profit': take_profit
                })
    
    print(f"📊 Total combinations: {len(test_combinations)}")
    print(f"📊 Testing on: {pair} {timeframe} ({days_back} days)")
    
    # Determine worker count
    available_cores = cpu_count()
    if available_cores >= 12:
        max_workers = 6
    elif available_cores >= 5:
        max_workers = available_cores - 2
    else:
        max_workers = max(1, available_cores - 1)
    
    print(f"⚡ Using {max_workers} parallel workers")
    
    # Run parallel optimization
    start_time = time.time()
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(run_optimization_test_worker, test_combinations)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ PARALLEL OPTIMIZATION COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"⚡ Speed: {len(test_combinations)/total_time:.1f} tests/second")
    
    # Generate report
    generate_optimization_report(results, pair, timeframe)
    
    # Find best combinations
    valid_results = [r for r in results if r.get('total_trades', 0) > 0]
    if valid_results:
        # Best by profit factor
        best_pf = max(valid_results, key=lambda x: x['profit_factor'])
        print(f"\n🏆 BEST PROFIT FACTOR: {best_pf['param_combo']}")
        print(f"   PF: {best_pf['profit_factor']:.2f}, WR: {best_pf['win_rate']:.1f}%, Return: {best_pf['total_return']:.2f}%")
        
        # Best by total return
        best_return = max(valid_results, key=lambda x: x['total_return'])
        print(f"\n💰 BEST TOTAL RETURN: {best_return['param_combo']}")
        print(f"   Return: {best_return['total_return']:.2f}%, PF: {best_return['profit_factor']:.2f}, WR: {best_return['win_rate']:.1f}%")
    
    # At the end, ALWAYS return results (even if empty)
    if 'results' not in locals():
        return []
    
    return results  # Make sure this line exists

def run_optimization_test_worker(test_config: Dict) -> Dict:
   """
   Worker function for parallel parameter optimization
   """
   try:
       # Create engine with specific parameters
       engine = ParameterOptimizationEngine(
           stop_buffer=test_config['stop_buffer'],
           breakeven_level=test_config['breakeven_level'],
           take_profit=test_config['take_profit']
       )
       
       # Run backtest
       result = engine.run_single_strategy_test(
           test_config['pair'],
           test_config['timeframe'],
           test_config['days_back']
       )
       
       # Add parameter info
       result['stop_buffer'] = test_config['stop_buffer']
       result['breakeven_level'] = test_config['breakeven_level']
       result['take_profit'] = test_config['take_profit']
       result['param_combo'] = f"SB{int(test_config['stop_buffer']*100)}_BE{test_config['breakeven_level']}_TP{test_config['take_profit']}"
       
       # Calculate additional metrics
       if result['total_trades'] > 0:
           avg_win = result['gross_profit'] / result['winning_trades'] if result['winning_trades'] > 0 else 0
           avg_loss = result['gross_loss'] / result['losing_trades'] if result['losing_trades'] > 0 else 0
           achieved_rr = avg_win / avg_loss if avg_loss > 0 else 0
           result['achieved_rr'] = round(achieved_rr, 2)
           result['expectancy'] = round(result['total_pnl'] / result['total_trades'], 2)
       else:
           result['achieved_rr'] = 0
           result['expectancy'] = 0
       
       # Clean up
       del engine
       gc.collect()
       
       return result
       
   except Exception as e:
       gc.collect()
       return {
           'pair': test_config['pair'],
           'timeframe': test_config['timeframe'],
           'stop_buffer': test_config['stop_buffer'],
           'breakeven_level': test_config['breakeven_level'],
           'take_profit': test_config['take_profit'],
           'param_combo': f"SB{int(test_config['stop_buffer']*100)}_BE{test_config['breakeven_level']}_TP{test_config['take_profit']}",
           'total_trades': 0,
           'error': str(e)
       }

def run_focused_optimization(pair='EURUSD', timeframe='3D'):
   """
   Run focused optimization on most promising parameter ranges
   Based on initial results, test refined parameter combinations
   """
   print("🎯 FOCUSED PARAMETER OPTIMIZATION")
   print("=" * 70)
   
   # Refined parameters based on typical optimization results
   FOCUSED_PARAMS = {
       'stop_buffers': [0.25, 0.30, 0.33, 0.35, 0.40],  # Refined around 33%
       'breakeven_levels': [0.75, 1.0, 1.25],  # Refined around 1R
       'take_profits': [2.0, 2.25, 2.5, 2.75, 3.0]  # Refined around 2.5R
   }
   
   print("Testing refined combinations:")
   print(f"- Stop Buffers: {[f'{x*100:.0f}%' for x in FOCUSED_PARAMS['stop_buffers']]}")
   print(f"- Break-even Levels: {[f'{x}R' for x in FOCUSED_PARAMS['breakeven_levels']]}")
   print(f"- Take Profits: {[f'{x}R' for x in FOCUSED_PARAMS['take_profits']]}")
   
   # Run with parallel processing
   test_combinations = []
   for stop_buffer in FOCUSED_PARAMS['stop_buffers']:
       for breakeven_level in FOCUSED_PARAMS['breakeven_levels']:
           for take_profit in FOCUSED_PARAMS['take_profits']:
               test_combinations.append({
                   'pair': pair,
                   'timeframe': timeframe,
                   'days_back': 3847,  # 10 years
                   'stop_buffer': stop_buffer,
                   'breakeven_level': breakeven_level,
                   'take_profit': take_profit
               })
   
   print(f"\n📊 Total refined combinations: {len(test_combinations)}")
   
   # Use parallel processing
   run_parallel_optimization(pair, timeframe, 3847)

def run_batch_optimization(pairs: List[str], timeframes: List[str], days_back: int):
    """
    Run parameter optimization across multiple pairs and timeframes
    Creates ONE consolidated Excel file with all results - ERROR RESISTANT
    """
    print("🚀 BATCH PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    all_batch_results = []  # Store ALL parameter test results
    batch_summary = []      # Store batch-level summary
    total_tests = len(pairs) * len(timeframes)
    current_test = 0
    
    start_time = time.time()
    
    for pair in pairs:
        for timeframe in timeframes:
            current_test += 1
            print(f"\n📊 TEST {current_test}/{total_tests}: {pair} {timeframe}")
            print("-" * 50)
            
            try:
                # Run parallel optimization for this pair/timeframe
                results = run_parallel_optimization(pair, timeframe, days_back)
                
                # CRITICAL FIX: Handle None return
                if results is None:
                    print(f"⚠️  {pair} {timeframe} returned None - treating as no results")
                    results = []
                
                # CRITICAL FIX: Ensure results is a list
                if not isinstance(results, list):
                    print(f"⚠️  {pair} {timeframe} returned {type(results)} - converting to list")
                    results = []
                
                # Add batch identifiers to each result
                for result in results:
                    if isinstance(result, dict):  # Safety check
                        result['batch_pair'] = pair
                        result['batch_timeframe'] = timeframe
                        result['batch_test_number'] = current_test
                        all_batch_results.append(result)
                
                # Create batch summary entry
                valid_results = [r for r in results if isinstance(r, dict) and r.get('total_trades', 0) > 0]
                
                if valid_results:
                    best_result = max(valid_results, key=lambda x: x.get('profit_factor', 0))
                    avg_pf = sum(r.get('profit_factor', 0) for r in valid_results) / len(valid_results)
                    avg_wr = sum(r.get('win_rate', 0) for r in valid_results) / len(valid_results)
                    
                    batch_info = {
                        'test_number': current_test,
                        'pair': pair,
                        'timeframe': timeframe,
                        'days_back': days_back,
                        'total_parameter_combinations': len(results),
                        'successful_combinations': len(valid_results),
                        'success_rate': len(valid_results) / len(results) * 100 if results else 0,
                        'best_profit_factor': best_result.get('profit_factor', 0),
                        'best_param_combo': best_result.get('param_combo', 'Unknown'),
                        'avg_profit_factor': round(avg_pf, 2),
                        'avg_win_rate': round(avg_wr, 1),
                        'status': 'SUCCESS'
                    }
                else:
                    batch_info = {
                        'test_number': current_test,
                        'pair': pair,
                        'timeframe': timeframe,
                        'days_back': days_back,
                        'total_parameter_combinations': len(results),
                        'successful_combinations': 0,
                        'success_rate': 0,
                        'best_profit_factor': 0,
                        'best_param_combo': 'None',
                        'avg_profit_factor': 0,
                        'avg_win_rate': 0,
                        'status': 'NO_TRADES'
                    }
                
                batch_summary.append(batch_info)
                print(f"✅ {pair} {timeframe} completed - {len(valid_results)}/{len(results)} successful combinations")
                
            except Exception as e:
                print(f"❌ {pair} {timeframe} failed: {str(e)}")
                
                # Add error results to maintain data structure
                error_result = {
                    'batch_pair': pair,
                    'batch_timeframe': timeframe,
                    'batch_test_number': current_test,
                    'pair': pair,
                    'timeframe': timeframe,
                    'total_trades': 0,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'param_combo': 'ERROR',
                    'error': str(e)
                }
                all_batch_results.append(error_result)
                
                error_info = {
                    'test_number': current_test,
                    'pair': pair,
                    'timeframe': timeframe,
                    'days_back': days_back,
                    'total_parameter_combinations': 0,
                    'successful_combinations': 0,
                    'success_rate': 0,
                    'best_profit_factor': 0,
                    'best_param_combo': 'ERROR',
                    'avg_profit_factor': 0,
                    'avg_win_rate': 0,
                    'status': 'FAILED',
                    'error': str(e)
                }
                batch_summary.append(error_info)
    
    # Create consolidated Excel report
    total_time = time.time() - start_time
    successful_tests = sum(1 for r in batch_summary if r['status'] == 'SUCCESS')
    
    print(f"\n✅ BATCH OPTIMIZATION COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📊 Successful tests: {successful_tests}/{total_tests}")
    
    # ALWAYS generate report even if some tests failed
    generate_consolidated_batch_report(all_batch_results, batch_summary, pairs, timeframes, days_back)

def generate_consolidated_batch_report(all_results: List[Dict], batch_summary: List[Dict], 
                                     pairs: List[str], timeframes: List[str], days_back: int):
    """
    Generate ONE comprehensive Excel file - DEFENSIVE VERSION
    """
    print(f"\n📊 GENERATING CONSOLIDATED BATCH REPORT...")
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/batch_parameter_optimization_{timestamp}.xlsx"
    os.makedirs('results', exist_ok=True)
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # SHEET 1: Executive Summary
            if batch_summary:
                exec_summary = pd.DataFrame(batch_summary)
                exec_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
                print("   ✅ Sheet 1: Executive Summary")
            else:
                pd.DataFrame({'Note': ['No batch summary data']}).to_excel(writer, sheet_name='Executive_Summary', index=False)
                print("   ⚠️  Sheet 1: No summary data")
            
            # SHEET 2: All Parameter Test Results
            if all_results:
                all_results_df = pd.DataFrame(all_results)
                all_results_df.to_excel(writer, sheet_name='All_Parameter_Results', index=False)
                print("   ✅ Sheet 2: All Parameter Results")
            else:
                pd.DataFrame({'Note': ['No parameter results']}).to_excel(writer, sheet_name='All_Parameter_Results', index=False)
                print("   ⚠️  Sheet 2: No parameter results")
            
            # SHEET 3: Best Combinations Only
            successful_results = [r for r in all_results if isinstance(r, dict) and r.get('total_trades', 0) > 0]
            
            if successful_results:
                try:
                    best_results_df = pd.DataFrame(successful_results)
                    # Sort by profit factor descending
                    best_results_df = best_results_df.sort_values('profit_factor', ascending=False)
                    best_results_df.to_excel(writer, sheet_name='Best_Combinations', index=False)
                    print("   ✅ Sheet 3: Best Combinations")
                    
                    # SHEET 4: Top 20 Overall (simplified)
                    try:
                        top_20 = best_results_df.head(20)[
                            ['batch_pair', 'batch_timeframe', 'param_combo', 'profit_factor', 
                             'win_rate', 'total_return', 'total_trades']
                        ].copy()
                        top_20.to_excel(writer, sheet_name='Top_20_Overall', index=False)
                        print("   ✅ Sheet 4: Top 20 Overall")
                    except Exception as e:
                        pd.DataFrame({'Error': [f'Top 20 analysis failed: {str(e)}']}).to_excel(writer, sheet_name='Top_20_Overall', index=False)
                        print(f"   ⚠️  Sheet 4: Top 20 analysis failed")
                    
                except Exception as e:
                    pd.DataFrame({'Error': [f'Best combinations analysis failed: {str(e)}']}).to_excel(writer, sheet_name='Best_Combinations', index=False)
                    pd.DataFrame({'Error': [f'Analysis failed: {str(e)}']}).to_excel(writer, sheet_name='Top_20_Overall', index=False)
                    print(f"   ⚠️  Sheets 3-4: Analysis failed: {str(e)}")
                    
            else:
                # No successful results
                empty_df = pd.DataFrame({'Note': ['No successful parameter combinations found']})
                empty_df.to_excel(writer, sheet_name='Best_Combinations', index=False)
                empty_df.to_excel(writer, sheet_name='Top_20_Overall', index=False)
                print("   ⚠️  Sheets 3-4: No successful results")
        
        print(f"\n📁 CONSOLIDATED BATCH REPORT SAVED:")
        print(f"   File: {filename}")
        print(f"   📊 {len(pairs)} pairs × {len(timeframes)} timeframes = {len(pairs) * len(timeframes)} tests")
        print(f"   📊 {len(all_results):,} total parameter combinations tested")
        
        # Print key insights if available
        if successful_results:
            best_overall = max(successful_results, key=lambda x: x.get('profit_factor', 0))
            print(f"\n🏆 BEST OVERALL COMBINATION:")
            print(f"   {best_overall.get('batch_pair', 'Unknown')} {best_overall.get('batch_timeframe', 'Unknown')} - {best_overall.get('param_combo', 'Unknown')}")
            print(f"   PF: {best_overall.get('profit_factor', 0):.2f}, WR: {best_overall.get('win_rate', 0):.1f}%, Return: {best_overall.get('total_return', 0):.2f}%")
        else:
            print(f"\n⚠️  NO SUCCESSFUL COMBINATIONS FOUND")
        
    except Exception as e:
        print(f"❌ Error creating consolidated report: {str(e)}")
        # Emergency fallback: Save whatever we have as CSV
        try:
            if all_results:
                csv_filename = filename.replace('.xlsx', '_emergency.csv')
                pd.DataFrame(all_results).to_csv(csv_filename, index=False)
                print(f"📁 Emergency CSV saved: {csv_filename}")
        except:
            print("❌ Could not save emergency backup")

def create_batch_pair_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create pair-level analysis across all timeframes and parameters"""
    try:
        analysis = df.groupby('batch_pair').agg({
            'profit_factor': ['mean', 'max', 'count'],
            'win_rate': 'mean',
            'total_return': 'mean',
            'total_trades': 'sum',
            'expectancy': 'mean'
        }).round(2)
        
        # Flatten column names
        analysis.columns = ['Avg_PF', 'Max_PF', 'Combinations_Count', 'Avg_WR', 'Avg_Return', 'Total_Trades', 'Avg_Expectancy']
        return analysis.sort_values('Avg_PF', ascending=False).reset_index()
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

def create_batch_timeframe_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create timeframe-level analysis across all pairs and parameters"""
    try:
        analysis = df.groupby('batch_timeframe').agg({
            'profit_factor': ['mean', 'max', 'count'],
            'win_rate': 'mean',
            'total_return': 'mean',
            'total_trades': 'sum',
            'expectancy': 'mean'
        }).round(2)
        
        # Flatten column names
        analysis.columns = ['Avg_PF', 'Max_PF', 'Combinations_Count', 'Avg_WR', 'Avg_Return', 'Total_Trades', 'Avg_Expectancy']
        return analysis.sort_values('Avg_PF', ascending=False).reset_index()
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

def create_parameter_effectiveness_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze which parameter combinations work best overall"""
    try:
        analysis = df.groupby('param_combo').agg({
            'profit_factor': ['mean', 'count'],
            'win_rate': 'mean',
            'total_return': 'mean',
            'total_trades': 'sum'
        }).round(2)
        
        # Flatten column names
        analysis.columns = ['Avg_PF', 'Occurrences', 'Avg_WR', 'Avg_Return', 'Total_Trades']
        return analysis.sort_values('Avg_PF', ascending=False).reset_index()
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

def main():
    """Main function with menu for different optimization approaches"""
    print("🎯 PARAMETER OPTIMIZATION BACKTESTING ENGINE")
    print("=" * 60)
    print("Built on CoreBacktestEngine - Testing trade management parameters")
    print("=" * 60)
    
    print("\n🎯 SELECT OPTIMIZATION MODE:")
    print("1. Sequential Optimization (All combinations)")
    print("2. Parallel Optimization (Faster, all combinations)")
    print("3. Focused Optimization (Refined parameter ranges)")
    print("4. Single Parameter Test (Custom values)")
    print("5. Quick Validation (Current vs one alternative)")
    print("6. Batch Optimization (Multiple pairs/timeframes)") 
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        # Sequential optimization
        print("\n📊 SEQUENTIAL OPTIMIZATION")
        confirm = input("This will test 60 combinations sequentially. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            run_parameter_optimization()
    
    elif choice == '2':
        # Parallel optimization with user inputs
        print("\n⚡ PARALLEL OPTIMIZATION")
        
        # Get user inputs with validation
        try:
            # Pair input
            pair = input("Enter pair (e.g., EURUSD, GBPUSD): ").strip().upper()
            if not pair:
                pair = 'EURUSD'
                print(f"✓ Using default: {pair}")
            
            # Timeframe input
            timeframe = input("Enter timeframe (1H, 4H, 1D, 3D, 1W): ").strip().upper()
            if not timeframe:
                timeframe = '3D'
                print(f"✓ Using default: {timeframe}")
            
            # Days back input
            days_input = input("Enter days back (730=2yr, 1825=5yr, 3847=10yr): ").strip()
            if not days_input:
                days_back = 3847
                print(f"✓ Using default: {days_back} days (10 years)")
            else:
                days_back = int(days_input)
                years = round(days_back / 365.25, 1)
                print(f"✓ Testing: {days_back} days ({years} years)")
            
            # Configuration summary
            print(f"\n📊 CONFIGURATION:")
            print(f"   Pair: {pair}")
            print(f"   Timeframe: {timeframe}")
            print(f"   Period: {days_back} days")
            print(f"   Combinations: 60 (4×3×5 parameters)")
            
            confirm = input("\nProceed with parallel optimization? (y/n): ").strip().lower()
            
            if confirm == 'y':
                run_parallel_optimization(pair, timeframe, days_back)
            else:
                print("❌ Optimization cancelled")
                
        except ValueError:
            print("❌ Invalid input for days back. Must be a number.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    elif choice == '3':
        # Focused optimization
        print("\n🎯 FOCUSED OPTIMIZATION")
        confirm = input("This will test refined parameter ranges. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            run_focused_optimization()
    
    elif choice == '4':
        # Single custom test
        print("\n🔧 SINGLE PARAMETER TEST")
        try:
            stop_buffer = float(input("Enter stop buffer % (e.g., 33): ").strip()) / 100
            breakeven_level = float(input("Enter break-even R level (e.g., 1.0): ").strip())
            take_profit = float(input("Enter take profit R level (e.g., 2.5): ").strip())
            
            engine = ParameterOptimizationEngine(stop_buffer, breakeven_level, take_profit)
            result = engine.run_single_strategy_test('EURUSD', '3D', 3847)
            
            print(f"\n📊 CUSTOM TEST RESULTS:")
            print(f"   Parameters: SB{int(stop_buffer*100)}%_BE{breakeven_level}R_TP{take_profit}R")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Total Return: {result['total_return']:.2f}%")
            print(f"   Avg Duration: {result['avg_trade_duration']:.1f} days")
            
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == '5':
        # Quick validation
        print("\n✅ QUICK VALIDATION TEST")
        print("Comparing current settings vs alternative...")
        
        # Current settings
        print("\n📊 Current Settings (SB33%_BE1.0R_TP2.5R):")
        engine1 = ParameterOptimizationEngine(0.33, 1.0, 2.5)
        result1 = engine1.run_single_strategy_test('EURUSD', '3D', 730)  # 2 years for quick test
        
        print(f"   Trades: {result1['total_trades']}, PF: {result1['profit_factor']:.2f}, WR: {result1['win_rate']:.1f}%")
        
        # Alternative settings
        print("\n📊 Alternative Settings (SB25%_BE0.75R_TP2.0R):")
        engine2 = ParameterOptimizationEngine(0.25, 0.75, 2.0)
        result2 = engine2.run_single_strategy_test('EURUSD', '3D', 730)
        
        print(f"   Trades: {result2['total_trades']}, PF: {result2['profit_factor']:.2f}, WR: {result2['win_rate']:.1f}%")
        
        # Comparison
        print("\n📊 COMPARISON:")
        print(f"   Trade Count: {result1['total_trades']} vs {result2['total_trades']} ({result2['total_trades']-result1['total_trades']:+d})")
        print(f"   Profit Factor: {result1['profit_factor']:.2f} vs {result2['profit_factor']:.2f} ({result2['profit_factor']-result1['profit_factor']:+.2f})")
        print(f"   Win Rate: {result1['win_rate']:.1f}% vs {result2['win_rate']:.1f}% ({result2['win_rate']-result1['win_rate']:+.1f}%)")
        print(f"   Total Return: {result1['total_return']:.2f}% vs {result2['total_return']:.2f}% ({result2['total_return']-result1['total_return']:+.2f}%)")
    elif choice == '6':
        # Batch optimization across multiple pairs/timeframes
        print("\n🔄 BATCH OPTIMIZATION")
        
        # Define test combinations
        pairs = ['AUDCAD', 'USDJPY', 'EURGBP', 'GBPCHF', 'GBPAUD', 'EURNZD', 'AUDCHF']
        timeframes = ['1D', '2D', '4D', '5D']
        days_back = 1825  # 5 years
        
        print(f"📊 BATCH CONFIGURATION:")
        print(f"   Pairs: {pairs}")
        print(f"   Timeframes: {timeframes}")  
        print(f"   Period: {days_back} days")
        print(f"   Total tests: {len(pairs)} × {len(timeframes)} = {len(pairs) * len(timeframes)}")
        print(f"   Parameter combinations per test: 60")
        
        confirm = input(f"\nRun {len(pairs) * len(timeframes)} separate optimizations? (y/n): ").strip().lower()
        
        if confirm == 'y':
            run_batch_optimization(pairs, timeframes, days_back)


    else:
        print("❌ Invalid choice")
    
    
    

if __name__ == "__main__":
   main()