"""
Zone Penetration Backtesting - Research Extension
Tests different wick penetration thresholds for zone invalidation
Extends CoreBacktestEngine with penetration strategy variations
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core_backtest_engine import CoreBacktestEngine
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import time

# Define penetration strategies
PENETRATION_STRATEGIES = {
    'strict_25': {
        'threshold': 0.25,
        'description': 'Very strict: Only 25% wick penetration allowed',
        'method': 'wick'
    },
    'moderate_33': {
        'threshold': 0.33,
        'description': 'Moderate: 33% wick penetration (common standard)',
        'method': 'wick'
    },
    'default_50': {
        'threshold': 0.50,
        'description': 'Current baseline: 50% wick penetration',
        'method': 'wick'
    },
    'lenient_66': {
        'threshold': 0.66,
        'description': 'Lenient: 66% wick penetration allowed',
        'method': 'wick'
    },
    'ultra_lenient_75': {
        'threshold': 0.75,
        'description': 'Very forgiving: 75% wick penetration',
        'method': 'wick'
    },
    'full_body': {
        'threshold': 1.0,
        'description': 'Body-based: Full candle body must penetrate zone',
        'method': 'body'
    },
    'close_beyond': {
        'threshold': 1.0,
        'description': 'Close-based: Close must be beyond zone boundary',
        'method': 'close'
    }
}

class ZonePenetrationBacktester(CoreBacktestEngine):
    """
    Extension of CoreBacktestEngine to test different zone penetration strategies
    Only modifies invalidation logic, inherits everything else
    """
    
    def __init__(self, penetration_strategy: str = 'default_50'):
        """
        Initialize with specific penetration strategy
        
        Args:
            penetration_strategy: One of the PENETRATION_STRATEGIES keys
        """
        super().__init__()
        
        if penetration_strategy not in PENETRATION_STRATEGIES:
            raise ValueError(f"Invalid strategy: {penetration_strategy}. Must be one of {list(PENETRATION_STRATEGIES.keys())}")
        
        self.penetration_strategy = penetration_strategy
        self.strategy_config = PENETRATION_STRATEGIES[penetration_strategy]
        
        print(f"🎯 Zone Penetration Backtester initialized:")
        print(f"   Strategy: {penetration_strategy}")
        print(f"   Threshold: {self.strategy_config['threshold']*100:.0f}%")
        print(f"   Method: {self.strategy_config['method']}")
        print(f"   Description: {self.strategy_config['description']}")
    
    def precalculate_invalidation_level(self, zone: Dict) -> float:
        """
        Override: Calculate invalidation level based on selected penetration strategy
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_size = zone_high - zone_low
        zone_type = zone['type']
        threshold = self.strategy_config['threshold']
        
        if zone_type in ['R-B-R', 'D-B-R']:  # DEMAND zones
            # Invalidated if penetrates X% below zone_low
            return zone_low - (zone_size * threshold)
        else:  # SUPPLY zones  
            # Invalidated if penetrates X% above zone_high
            return zone_high + (zone_size * threshold)
    
    def fast_wick_invalidation_check(self, zone: Dict, current_candle: pd.Series, 
                                    zone_id: str, invalidated_zones: set) -> bool:
        """
        Override: Check invalidation based on selected penetration strategy
        """
        invalidation_level = zone['invalidation_level']
        zone_type = zone['type']
        method = self.strategy_config['method']
        
        if method == 'wick':
            # Standard wick-based invalidation
            if zone_type in ['R-B-R', 'D-B-R']:  # DEMAND zones
                if current_candle['low'] < invalidation_level:
                    invalidated_zones.add(zone_id)
                    return True
            else:  # SUPPLY zones
                if current_candle['high'] > invalidation_level:
                    invalidated_zones.add(zone_id)
                    return True
                    
        elif method == 'body':
            # Body-based invalidation (requires full body penetration)
            body_high = max(current_candle['open'], current_candle['close'])
            body_low = min(current_candle['open'], current_candle['close'])
            
            if zone_type in ['R-B-R', 'D-B-R']:  # DEMAND zones
                if body_low < zone['zone_low']:  # Full body below zone
                    invalidated_zones.add(zone_id)
                    return True
            else:  # SUPPLY zones
                if body_high > zone['zone_high']:  # Full body above zone
                    invalidated_zones.add(zone_id)
                    return True
                    
        elif method == 'close':
            # Close-based invalidation (only close price matters)
            if zone_type in ['R-B-R', 'D-B-R']:  # DEMAND zones
                if current_candle['close'] < zone['zone_low']:
                    invalidated_zones.add(zone_id)
                    return True
            else:  # SUPPLY zones
                if current_candle['close'] > zone['zone_high']:
                    invalidated_zones.add(zone_id)
                    return True
        
        return False
    
    def calculate_performance_metrics(self, trades: List[Dict], pair: str, timeframe: str) -> Dict:
        """
        Override: Add penetration strategy to results
        """
        # Get base metrics from parent
        metrics = super().calculate_performance_metrics(trades, pair, timeframe)
        
        # Add penetration strategy info
        metrics['penetration_strategy'] = self.penetration_strategy
        metrics['penetration_threshold'] = self.strategy_config['threshold']
        metrics['penetration_method'] = self.strategy_config['method']
        metrics['strategy_description'] = self.strategy_config['description']
        
        return metrics
    
    def create_empty_result(self, pair: str, timeframe: str, reason: str) -> Dict:
        """
        Override: Add penetration strategy to empty results
        """
        result = super().create_empty_result(pair, timeframe, reason)
        
        # Add penetration strategy info
        result['penetration_strategy'] = self.penetration_strategy
        result['penetration_threshold'] = self.strategy_config['threshold']
        result['penetration_method'] = self.strategy_config['method']
        result['strategy_description'] = self.strategy_config['description']
        
        return result


def run_penetration_strategy_comparison(pair: str = 'AUDCAD', timeframe: str = '2D', 
                                      days_back: int = 8000) -> pd.DataFrame:
    """
    Run all penetration strategies and compare results
    """
    print(f"\n🔬 ZONE PENETRATION STRATEGY COMPARISON")
    print(f"📊 Testing: {pair} {timeframe} ({days_back} days)")
    print("=" * 70)
    
    results = []
    
    for strategy_name in PENETRATION_STRATEGIES.keys():
        print(f"\n🧪 Testing {strategy_name}...")
        start_time = time.time()
        
        try:
            # Create backtester with specific strategy
            backtester = ZonePenetrationBacktester(penetration_strategy=strategy_name)
            
            # Run single test
            result = backtester.run_single_strategy_test(pair, timeframe, days_back)
            results.append(result)
            
            # Quick summary
            elapsed = time.time() - start_time
            print(f"   ✅ Complete in {elapsed:.1f}s")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            # Add failed result
            results.append({
                'pair': pair,
                'timeframe': timeframe,
                'penetration_strategy': strategy_name,
                'total_trades': 0,
                'description': f"Error: {str(e)}"
            })
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Generate Excel report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/penetration_strategy_comparison_{pair}_{timeframe}_{timestamp}.xlsx"
    os.makedirs('results', exist_ok=True)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Strategy comparison
        df.to_excel(writer, sheet_name='Strategy_Comparison', index=False)
        
        # Summary statistics
        if len(df[df['total_trades'] > 0]) > 0:
            summary = df[df['total_trades'] > 0].groupby('penetration_strategy').agg({
                'total_trades': 'mean',
                'win_rate': 'mean',
                'profit_factor': 'mean',
                'total_return': 'mean'
            }).round(2)
            summary.to_excel(writer, sheet_name='Summary_Stats')
    
    print(f"\n📊 COMPARISON COMPLETE!")
    print(f"📁 Report saved: {filename}")
    
    # Display summary
    print("\n📈 STRATEGY PERFORMANCE SUMMARY:")
    print("-" * 70)
    for _, row in df.iterrows():
        if row['total_trades'] > 0:
            print(f"{row['penetration_strategy']:15} | Trades: {row['total_trades']:3} | "
                  f"WR: {row['win_rate']:5.1f}% | PF: {row['profit_factor']:4.2f} | "
                  f"Return: {row['total_return']:6.2f}%")
    
    return df


def main():
    """Main function for penetration strategy testing"""
    print("🎯 ZONE PENETRATION STRATEGY RESEARCH")
    print("=" * 60)
    
    print("\n📊 SELECT TESTING MODE:")
    print("1. Quick Test - Single Strategy (AUDCAD 2D)")
    print("2. Strategy Comparison - All 7 Strategies (AUDCAD 2D)")
    print("3. Custom Test - Choose pair/timeframe/strategy")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Quick single strategy test
        print("\n🧪 QUICK STRATEGY TEST")
        print("Available strategies:", ', '.join(PENETRATION_STRATEGIES.keys()))
        strategy = input("Enter strategy name (or press Enter for default_50): ").strip() or 'default_50'
        
        backtester = ZonePenetrationBacktester(penetration_strategy=strategy)
        result = backtester.run_single_strategy_test('EURUSD', '3D', 730)
        
        print(f"\n📊 RESULTS FOR {strategy}:")
        print(f"   Trades: {result['total_trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Total Return: {result['total_return']:.2f}%")
        
    elif choice == '2':
        # Run full comparison
        print("\n🔬 RUNNING FULL STRATEGY COMPARISON...")
        run_penetration_strategy_comparison()
        
    elif choice == '3':
        # Custom test
        print("\n🎯 CUSTOM PENETRATION TEST")
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        days_back = int(input("Enter days back (e.g., 730): ").strip())
        
        print("\nAvailable strategies:", ', '.join(PENETRATION_STRATEGIES.keys()))
        strategy = input("Enter strategy name: ").strip()
        
        backtester = ZonePenetrationBacktester(penetration_strategy=strategy)
        result = backtester.run_single_strategy_test(pair, timeframe, days_back)
        
        print(f"\n📊 RESULTS:")
        print(f"   Strategy: {strategy}")
        print(f"   Trades: {result['total_trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Total Return: {result['total_return']:.2f}%")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()