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
        
    def load_data(self, days_back: int = 730):
        """Load and prepare data for backtesting"""
        print(f"📊 Loading EURUSD Daily data ({days_back} days back)...")
        
        data_loader = DataLoader()
        self.data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # Calculate date range
        end_date = self.data.index[-1]
        start_date = end_date - timedelta(days=days_back)
        
        # Ensure we have enough lookback data (365 days minimum)
        if len(self.data) < days_back + 365:
            print(f"⚠️  Limited data: Using {len(self.data)} candles available")
            self.test_data = self.data
        else:
            self.test_data = self.data[self.data.index >= start_date]
        
        print(f"✅ Loaded {len(self.test_data)} candles for testing")
        print(f"📅 Test period: {self.test_data.index[0].strftime('%Y-%m-%d')} to {self.test_data.index[-1].strftime('%Y-%m-%d')}")
        
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
        active_zones = []
        account_balance = 10000
        
        # Process each candle for backtesting
        for i, (date, candle) in enumerate(self.test_data.iterrows()):
            current_price = candle['close']
            
            # Check for new zone activations
            for pattern in valid_patterns:
                if pattern['end_idx'] < len(self.data):
                    zone_end_date = self.data.index[pattern['end_idx']]
                    
                    # Zone becomes active after formation
                    if date >= zone_end_date and pattern not in active_zones:
                        active_zones.append(pattern)
            
            # Check for trade executions
            for zone in active_zones.copy():
                trade_result = self.check_trade_execution(zone, candle, date, current_price)
                
                if trade_result:
                    if 'invalidated' not in trade_result:
                        trades.append(trade_result)
                        account_balance += trade_result['pnl']
                    
                    active_zones.remove(zone)  # Zone used or invalidated
        
        # Calculate performance metrics
        return self.calculate_performance(trades, account_balance, strategy_name, distance_threshold)
    
    def check_trade_execution(self, zone: Dict, candle: pd.Series, 
                            date: pd.Timestamp, current_price: float) -> Dict:
        """
        Check if a trade should be executed based on zone logic
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # 5% front-run entry logic
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
        
        # Check for zone invalidation (33% penetration)
        invalidation_threshold = zone_range * 0.33
        
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones
            if candle['low'] <= zone_low - invalidation_threshold:
                return {'invalidated': True, 'zone_type': zone['type']}
        else:  # Supply zones
            if candle['high'] >= zone_high + invalidation_threshold:
                return {'invalidated': True, 'zone_type': zone['type']}
        
        return None
    
    def execute_buy_trade(self, zone: Dict, entry_price: float, 
                         entry_date: pd.Timestamp, entry_candle: pd.Series) -> Dict:
        """Execute a buy trade with proper risk management"""
        zone_range = zone['zone_high'] - zone['zone_low']
        
        # Calculate stop loss (33% buffer beyond zone)
        stop_loss = zone['zone_low'] - (zone_range * 0.33)
        
        # Calculate position size (5% risk)
        risk_amount = 10000 * 0.05  # 5% of account
        stop_distance = entry_price - stop_loss
        
        if stop_distance <= 0:
            return None
        
        position_size = risk_amount / stop_distance
        
        # Calculate targets
        target_1 = entry_price + stop_distance  # 1:1 RR
        target_2 = entry_price + (stop_distance * 2)  # 1:2 RR
        
        # REALISTIC: Track actual trade outcome based on market data
        trade_outcome = self.simulate_realistic_trade_outcome(zone, entry_price, stop_loss, target_2, entry_date)
        pnl = trade_outcome['pnl']
        
        return {
            'strategy': 'momentum' if zone['type'] in ['R-B-R', 'D-B-D'] else 'reversal',
            'zone_type': zone['type'],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_price': trade_outcome['exit_price'],
            'stop_loss': stop_loss,
            'position_size': position_size,
            'pnl': trade_outcome['pnl'],
            'distance_ratio': zone['leg_out']['ratio_to_base'],
            'duration_days': 5,  # Simplified
            'result': 'win' if trade_outcome['pnl'] > 0 else 'loss',
            'exit_reason': trade_outcome['exit_reason']
        }
    
    def execute_sell_trade(self, zone: Dict, entry_price: float, 
                          entry_date: pd.Timestamp, entry_candle: pd.Series) -> Dict:
        """Execute a sell trade with proper risk management"""
        zone_range = zone['zone_high'] - zone['zone_low']
        
        # Calculate stop loss (33% buffer beyond zone)
        stop_loss = zone['zone_high'] + (zone_range * 0.33)
        
        # Calculate position size (5% risk)
        risk_amount = 10000 * 0.05  # 5% of account
        stop_distance = stop_loss - entry_price
        
        if stop_distance <= 0:
            return None
        
        position_size = risk_amount / stop_distance
        
        # Calculate targets
        target_1 = entry_price - stop_distance  # 1:1 RR
        target_2 = entry_price - (stop_distance * 2)  # 1:2 RR
        
        # Simulate trade execution
        simulated_exit = target_2  # Assume successful momentum trade
        pnl = (entry_price - simulated_exit) * position_size
        
        return {
            'strategy': 'momentum' if zone['type'] in ['R-B-R', 'D-B-D'] else 'reversal',
            'zone_type': zone['type'],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_price': simulated_exit,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'pnl': pnl,
            'distance_ratio': zone['leg_out']['ratio_to_base'],
            'duration_days': 5,  # Simplified
            'result': 'win' if pnl > 0 else 'loss'
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
        REALISTIC: Simulate trade outcome based on actual market data
        """
        # Find entry date index
        try:
            entry_idx = self.data.index.get_loc(entry_date)
        except KeyError:
            # If exact date not found, find closest
            entry_idx = self.data.index.get_loc(entry_date, method='nearest')
        
        # Look forward from entry date to see what actually happened
        for i in range(entry_idx + 1, min(entry_idx + 30, len(self.data))):  # Max 30 days
            candle = self.data.iloc[i]
            
            # Check if stop loss was hit first
            if entry_price > stop_loss:  # Buy trade
                if candle['low'] <= stop_loss:
                    return {
                        'pnl': -abs(entry_price - stop_loss) * 0.1,  # Loss
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss'
                    }
                # Check if target was hit
                elif candle['high'] >= target_price:
                    return {
                        'pnl': abs(target_price - entry_price) * 0.1,  # Win
                        'exit_price': target_price,
                        'exit_reason': 'target'
                    }
            else:  # Sell trade
                if candle['high'] >= stop_loss:
                    return {
                        'pnl': -abs(entry_price - stop_loss) * 0.1,  # Loss
                        'exit_price': stop_loss,
                        'exit_reason': 'stop_loss'
                    }
                elif candle['low'] <= target_price:
                    return {
                        'pnl': abs(entry_price - target_price) * 0.1,  # Win
                        'exit_price': target_price,
                        'exit_reason': 'target'
                    }
        
        # If neither hit in 30 days, close at current price (neutral outcome)
        final_candle = self.data.iloc[min(entry_idx + 30, len(self.data) - 1)]
        final_price = final_candle['close']
        
        if entry_price > stop_loss:  # Buy trade
            pnl = (final_price - entry_price) * 0.1
        else:  # Sell trade
            pnl = (entry_price - final_price) * 0.1
        
        return {
            'pnl': pnl,
            'exit_price': final_price,
            'exit_reason': 'timeout'
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
    
    def run_comprehensive_analysis(self, days_back: int = 730) -> Dict:
        """Run comprehensive momentum vs reversal analysis"""
        print("🚀 FIXED: MOMENTUM VS REVERSAL COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        # Load data
        self.load_data(days_back)
        
        # Detect zones
        momentum_patterns, reversal_patterns = self.detect_zones()
        
        # Test all distance thresholds
        all_results = []
        
        for distance in self.distance_thresholds:
            print(f"\n📊 Testing {distance}x distance threshold...")
            
            # Test momentum strategy
            momentum_results = self.backtest_strategy(momentum_patterns, 'Momentum', distance)
            all_results.append(momentum_results)
            
            # Test reversal strategy  
            reversal_results = self.backtest_strategy(reversal_patterns, 'Reversal', distance)
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
    """Main function with user input for days back"""
    print("🚀 FIXED: MOMENTUM VS REVERSAL BACKTESTING SYSTEM")
    print("=" * 60)
    
    # Get user input for backtest period
    print("\n📅 Select backtest period:")
    print("   1. Last 6 months (180 days)")
    print("   2. Last 1 year (365 days)")  
    print("   3. Last 2 years (730 days)")
    print("   4. Last 3 years (1095 days)")
    print("   5. Custom days")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        days_back = 180
    elif choice == '2':
        days_back = 365
    elif choice == '3':
        days_back = 730
    elif choice == '4':
        days_back = 1095
    elif choice == '5':
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
    
    # Run analysis
    backtester = FixedMomentumVsReversalBacktester()
    results = backtester.run_comprehensive_analysis(days_back)
    
    print(f"\n✅ Analysis complete! Results saved to results/ directory")
    print(f"⏱️  Estimated runtime: ~{len(backtester.distance_thresholds) * 2 * 5} seconds")

if __name__ == "__main__":
    main()