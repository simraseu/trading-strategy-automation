"""
Distance Edge Backtester - Test pure pattern distance performance
No trend filters, no complex risk management - just distance validation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

class DistanceEdgeBacktester:
    """
    Pure distance edge testing - isolate the distance factor only
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.results = {}
        
    def run_distance_analysis(self):
        """Run comprehensive distance edge analysis"""
        print("🎯 DISTANCE EDGE BACKTESTER")
        print("=" * 50)
        
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # All period for testing
        test_data = data.copy()
        test_data = test_data.reset_index()
        
        print(f"📊 Test period: {test_data['date'].iloc[0]} to {test_data['date'].iloc[-1]}")
        print(f"📈 Candles: {len(test_data)}")
        
        # Detect all patterns
        candle_classifier = CandleClassifier(test_data)
        classified_data = candle_classifier.classify_all_candles()
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        # Test configurations
        distance_configs = {
            '1.0x': {'multiplier': 1.0, 'name': '1.0x Distance'},
            '1.5x': {'multiplier': 1.5, 'name': '1.5x Distance'},  # Current system
            '2.0x': {'multiplier': 2.0, 'name': '2.0x Distance'},
            '2.5x': {'multiplier': 2.5, 'name': '2.5x Distance'},
            '3.0x': {'multiplier': 3.0, 'name': '3.0x Distance'},
            '4.0x': {'multiplier': 4.0, 'name': '4.0x Distance'},
            '5.0x': {'multiplier': 5.0, 'name': '5.0x Distance'}
        }
        
        pattern_configs = {
            'momentum': {
                'name': 'Momentum Patterns',
                'patterns': patterns['dbd_patterns'] + patterns['rbr_patterns'],
                'description': 'D-B-D + R-B-R (trend continuation)'
            },
            'reversal': {
                'name': 'Reversal Patterns', 
                'patterns': patterns['dbr_patterns'] + patterns['rbd_patterns'],
                'description': 'D-B-R + R-B-D (trend reversal)'
            }
        }
        
        # Run tests for each combination
        all_results = {}
        
        for pattern_type, pattern_config in pattern_configs.items():
            print(f"\n🔄 Testing {pattern_config['name']}:")
            print(f"   Patterns found: {len(pattern_config['patterns'])}")
            
            pattern_results = {}
            
            for distance_key, distance_config in distance_configs.items():
                print(f"   Testing {distance_config['name']}...")
                
                # Filter patterns by distance
                qualified_patterns = self.filter_patterns_by_distance(
                    pattern_config['patterns'], distance_config['multiplier']
                )
                
                if qualified_patterns:
                    # Run simple backtest
                    backtest_results = self.run_simple_backtest(
                        qualified_patterns, test_data, distance_config['multiplier']
                    )
                    pattern_results[distance_key] = backtest_results
                    
                    print(f"      Qualified: {len(qualified_patterns)} | "
                          f"Win Rate: {backtest_results['win_rate']:.1f}% | "
                          f"Avg Return: {backtest_results['avg_return']:.1f}%")
                else:
                    pattern_results[distance_key] = self.empty_result()
                    print(f"      No patterns qualified")
            
            all_results[pattern_type] = pattern_results
        
        # Generate analysis
        self.generate_distance_analysis(all_results, distance_configs, pattern_configs)
        self.create_distance_charts(all_results, distance_configs, pattern_configs)
        
        return all_results
    
    def filter_patterns_by_distance(self, patterns: list, distance_multiplier: float) -> list:
        """Filter patterns that meet minimum distance requirement"""
        qualified = []
        
        for pattern in patterns:
            # Calculate actual distance ratio
            base_range = pattern['base']['range']
            leg_out_range = pattern['leg_out']['range']
            
            if base_range > 0:
                actual_ratio = leg_out_range / base_range
                
                # Check if meets minimum distance requirement
                if actual_ratio >= distance_multiplier:
                    pattern['distance_ratio'] = actual_ratio
                    qualified.append(pattern)
        
        return qualified
    
    def run_simple_backtest(self, patterns: list, data: pd.DataFrame, 
                           distance_multiplier: float) -> dict:
        """
        Run simple backtest - pure pattern performance testing
        NO trend filters, NO complex risk management
        """
        trades = []
        
        for pattern in patterns:
            # Simple trade simulation
            trade_result = self.simulate_pattern_trade(pattern, data)
            if trade_result:
                trade_result['distance_multiplier'] = distance_multiplier
                trade_result['distance_ratio'] = pattern['distance_ratio']
                trades.append(trade_result)
        
        # Calculate performance metrics
        return self.calculate_simple_metrics(trades)
    
    def simulate_pattern_trade(self, pattern: dict, data: pd.DataFrame) -> dict:
        """
        Simulate single pattern trade with simple logic
        Entry: At pattern completion
        Exit: Fixed 1:2 risk/reward OR 20-candle timeout
        """
        try:
            pattern_end_idx = pattern['end_idx']
            
            # Skip if not enough data after pattern
            if pattern_end_idx + 25 >= len(data):
                return None
            
            # Entry price: Close of pattern completion candle
            entry_candle = data.iloc[pattern_end_idx + 1]  # Next candle after pattern
            entry_price = entry_candle['open']  # Enter at open
            
            # Direction and stop/target calculation
            zone_high = pattern['zone_high']
            zone_low = pattern['zone_low']
            pattern_type = pattern['type']
            
            if pattern_type in ['R-B-R', 'D-B-R']:  # Buy patterns
                direction = 'BUY'
                stop_loss = zone_low * 0.995  # 5 pips below zone low
                risk_distance = abs(entry_price - stop_loss)
                take_profit = entry_price + (risk_distance * 2)  # 1:2 RR
            else:  # Sell patterns
                direction = 'SELL'
                stop_loss = zone_high * 1.005  # 5 pips above zone high
                risk_distance = abs(entry_price - stop_loss)
                take_profit = entry_price - (risk_distance * 2)  # 1:2 RR
            
            # Simulate trade outcome
            for i in range(pattern_end_idx + 2, min(pattern_end_idx + 22, len(data))):
                candle = data.iloc[i]
                
                # Check for exit conditions
                if direction == 'BUY':
                    if candle['low'] <= stop_loss:
                        # Stop loss hit
                        return {
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'direction': direction,
                            'outcome': 'LOSS',
                            'return_pct': -100,  # Fixed -1R loss
                            'candles_held': i - (pattern_end_idx + 1),
                            'pattern_type': pattern_type
                        }
                    elif candle['high'] >= take_profit:
                        # Take profit hit
                        return {
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'direction': direction,
                            'outcome': 'WIN',
                            'return_pct': 200,  # Fixed +2R win
                            'candles_held': i - (pattern_end_idx + 1),
                            'pattern_type': pattern_type
                        }
                else:  # SELL
                    if candle['high'] >= stop_loss:
                        # Stop loss hit
                        return {
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'direction': direction,
                            'outcome': 'LOSS',
                            'return_pct': -100,  # Fixed -1R loss
                            'candles_held': i - (pattern_end_idx + 1),
                            'pattern_type': pattern_type
                        }
                    elif candle['low'] <= take_profit:
                        # Take profit hit
                        return {
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'direction': direction,
                            'outcome': 'WIN',
                            'return_pct': 200,  # Fixed +2R win
                            'candles_held': i - (pattern_end_idx + 1),
                            'pattern_type': pattern_type
                        }
            
            # Timeout - close at break-even
            timeout_candle = data.iloc[min(pattern_end_idx + 21, len(data) - 1)]
            return {
                'entry_price': entry_price,
                'exit_price': entry_price,  # Break-even
                'direction': direction,
                'outcome': 'TIMEOUT',
                'return_pct': 0,
                'candles_held': 20,
                'pattern_type': pattern_type
            }
            
        except Exception as e:
            return None
    
    def calculate_simple_metrics(self, trades: list) -> dict:
        """Calculate simple performance metrics"""
        if not trades:
            return self.empty_result()
        
        total_trades = len(trades)
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']
        timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']
        
        win_rate = (len(wins) / total_trades) * 100
        loss_rate = (len(losses) / total_trades) * 100
        timeout_rate = (len(timeouts) / total_trades) * 100
        
        # Calculate returns
        total_return = sum(t['return_pct'] for t in trades)
        avg_return = total_return / total_trades
        
        # Calculate expectancy (per trade)
        expectancy = (len(wins) * 200 + len(losses) * (-100) + len(timeouts) * 0) / total_trades
        
        return {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'timeouts': len(timeouts),
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'timeout_rate': timeout_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'expectancy': expectancy,
            'avg_hold_time': np.mean([t['candles_held'] for t in trades])
        }
    
    def empty_result(self) -> dict:
        """Return empty result for no trades"""
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'timeouts': 0,
            'win_rate': 0, 'loss_rate': 0, 'timeout_rate': 0,
            'total_return': 0, 'avg_return': 0, 'expectancy': 0,
            'avg_hold_time': 0
        }
    
    def generate_distance_analysis(self, results: dict, distance_configs: dict, pattern_configs: dict):
        """Generate comprehensive distance analysis report"""
        print(f"\n📊 DISTANCE EDGE ANALYSIS REPORT")
        print("=" * 70)
        
        # Create comparison table
        print(f"\n📈 PERFORMANCE BY DISTANCE THRESHOLD:")
        print(f"{'Distance':<10} {'Pattern':<12} {'Trades':<8} {'Win%':<8} {'Exp':<8} {'AvgRet':<10}")
        print("-" * 60)
        
        for distance_key in distance_configs.keys():
            for pattern_type in results.keys():
                result = results[pattern_type].get(distance_key, self.empty_result())
                
                pattern_name = pattern_type.title()
                trades = result['total_trades']
                win_rate = result['win_rate']
                expectancy = result['expectancy']
                avg_return = result['avg_return']
                
                print(f"{distance_key:<10} {pattern_name:<12} {trades:<8} {win_rate:<7.1f}% {expectancy:<7.1f}% {avg_return:<9.1f}%")
        
        # Find best performers
        print(f"\n🏆 BEST PERFORMERS:")
        
        best_expectancy = -999
        best_combo = None
        
        for pattern_type, pattern_results in results.items():
            for distance_key, result in pattern_results.items():
                if result['expectancy'] > best_expectancy and result['total_trades'] >= 5:
                    best_expectancy = result['expectancy']
                    best_combo = (pattern_type, distance_key, result)
        
        if best_combo:
            pattern_type, distance_key, result = best_combo
            print(f"   🥇 {pattern_type.title()} at {distance_key}: {result['expectancy']:.1f}% expectancy")
            print(f"      {result['total_trades']} trades, {result['win_rate']:.1f}% win rate")
        
        # Save results
        self.save_distance_results(results, distance_configs, pattern_configs)
    
    def create_distance_charts(self, results: dict, distance_configs: dict, pattern_configs: dict):
        """Create distance performance visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distance Edge Analysis', fontsize=16, fontweight='bold')
        
        distances = list(distance_configs.keys())
        momentum_data = [results['momentum'].get(d, self.empty_result()) for d in distances]
        reversal_data = [results['reversal'].get(d, self.empty_result()) for d in distances]
        
        # Chart 1: Win Rate by Distance
        momentum_wr = [d['win_rate'] for d in momentum_data]
        reversal_wr = [d['win_rate'] for d in reversal_data]
        
        x = np.arange(len(distances))
        width = 0.35
        
        ax1.bar(x - width/2, momentum_wr, width, label='Momentum', alpha=0.8, color='#2E86AB')
        ax1.bar(x + width/2, reversal_wr, width, label='Reversal', alpha=0.8, color='#A23B72')
        ax1.set_title('Win Rate by Distance')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(distances)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Expectancy by Distance
        momentum_exp = [d['expectancy'] for d in momentum_data]
        reversal_exp = [d['expectancy'] for d in reversal_data]
        
        ax2.bar(x - width/2, momentum_exp, width, label='Momentum', alpha=0.8, color='#2E86AB')
        ax2.bar(x + width/2, reversal_exp, width, label='Reversal', alpha=0.8, color='#A23B72')
        ax2.set_title('Expectancy by Distance')
        ax2.set_ylabel('Expectancy (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(distances)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Chart 3: Trade Count by Distance
        momentum_trades = [d['total_trades'] for d in momentum_data]
        reversal_trades = [d['total_trades'] for d in reversal_data]
        
        ax3.bar(x - width/2, momentum_trades, width, label='Momentum', alpha=0.8, color='#2E86AB')
        ax3.bar(x + width/2, reversal_trades, width, label='Reversal', alpha=0.8, color='#A23B72')
        ax3.set_title('Trade Count by Distance')
        ax3.set_ylabel('Number of Trades')
        ax3.set_xticks(x)
        ax3.set_xticklabels(distances)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Average Return by Distance
        momentum_ret = [d['avg_return'] for d in momentum_data]
        reversal_ret = [d['avg_return'] for d in reversal_data]
        
        ax4.bar(x - width/2, momentum_ret, width, label='Momentum', alpha=0.8, color='#2E86AB')
        ax4.bar(x + width/2, reversal_ret, width, label='Reversal', alpha=0.8, color='#A23B72')
        ax4.set_title('Average Return by Distance')
        ax4.set_ylabel('Average Return (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(distances)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save chart
        os.makedirs('results/distance_analysis', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/distance_analysis/distance_edge_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Charts saved: {filename}")
        
        plt.show()
    
    def save_distance_results(self, results: dict, distance_configs: dict, pattern_configs: dict):
        """Save detailed results to CSV"""
        all_data = []
        
        for pattern_type, pattern_results in results.items():
            for distance_key, result in pattern_results.items():
                all_data.append({
                    'pattern_type': pattern_type,
                    'distance_threshold': distance_key,
                    'total_trades': result['total_trades'],
                    'wins': result['wins'],
                    'losses': result['losses'],
                    'timeouts': result['timeouts'],
                    'win_rate': result['win_rate'],
                    'loss_rate': result['loss_rate'],
                    'timeout_rate': result['timeout_rate'],
                    'expectancy': result['expectancy'],
                    'avg_return': result['avg_return'],
                    'avg_hold_time': result['avg_hold_time']
                })
        
        df = pd.DataFrame(all_data)
        
        os.makedirs('results/distance_analysis', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/distance_analysis/distance_results_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"💾 Results saved: {filename}")

def main():
    """Run distance edge analysis"""
    print("🚀 DISTANCE EDGE BACKTESTER")
    print("=" * 40)
    
    backtester = DistanceEdgeBacktester()
    results = backtester.run_distance_analysis()
    
    print(f"\n🎉 Distance analysis complete!")
    print(f"📁 Results saved in: results/distance_analysis/")

if __name__ == "__main__":
    main()