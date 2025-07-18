"""
Pattern Performance Comparison Backtester
Compare momentum vs reversal pattern performance with auto-date detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator
from modules.backtester import TradingBacktester

class PatternPerformanceAnalyzer:
    """
    Compare performance across different pattern types with smart date detection
    """
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.results = {}
        self.data = None
        
    def load_and_analyze_data(self):
        """Load data and determine optimal test period"""
        print("📊 Loading and analyzing data...")
        
        data_loader = DataLoader()
        self.data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        print(f"✅ Loaded {len(self.data)} candles")
        print(f"📅 Data range: {self.data.index[0]} to {self.data.index[-1]}")
        
        # Determine optimal backtest period (last 2 years or available data)
        end_date = self.data.index[-30]  # 30 days before end for safety
        
        # Try to get 2 years of data, but ensure we have enough lookback
        days_back = min(730, len(self.data) - 400)  # 2 years or max available minus 400 for lookback
        start_date = self.data.index[-days_back]
        
        # Ensure we have enough data before start_date for 365-day lookback
        min_start_idx = 365
        if self.data.index.get_loc(start_date) < min_start_idx:
            start_date = self.data.index[min_start_idx]
        
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')
        
        test_days = (end_date - start_date).days
        print(f"🎯 Test period: {self.start_date} to {self.end_date} ({test_days} days)")
        
        return self.data
    
    def run_pattern_comparison(self):
        """Run comprehensive pattern performance comparison"""
        print("🎯 PATTERN PERFORMANCE COMPARISON")
        print("=" * 60)
        
        # Load data and determine test period
        data = self.load_and_analyze_data()
        
        # Initialize components
        print("🔧 Initializing trading components...")
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(data)
        risk_manager = RiskManager(account_balance=self.initial_balance)
        
        # Define test configurations
        test_configs = {
            'momentum_only': {
                'name': 'Momentum Patterns',
                'patterns': ['D-B-D', 'R-B-R'],
                'description': 'Trend continuation (D-B-D + R-B-R)',
                'color': '#2E86AB'
            },
            'reversal_only': {
                'name': 'Reversal Patterns', 
                'patterns': ['D-B-R', 'R-B-D'],
                'description': 'Trend reversal (D-B-R + R-B-D)',
                'color': '#A23B72'
            },
            'bullish_only': {
                'name': 'Bullish Patterns',
                'patterns': ['R-B-R', 'D-B-R'],
                'description': 'Buy signals only (R-B-R + D-B-R)',
                'color': '#27AE60'
            },
            'bearish_only': {
                'name': 'Bearish Patterns',
                'patterns': ['D-B-D', 'R-B-D'], 
                'description': 'Sell signals only (D-B-D + R-B-D)',
                'color': '#C0392B'
            },
            'all_patterns': {
                'name': 'Complete System',
                'patterns': ['D-B-D', 'R-B-R', 'D-B-R', 'R-B-D'],
                'description': 'All pattern types combined',
                'color': '#4ECDC4'
            }
        }
        
        # Run backtest for each configuration
        successful_tests = 0
        
        for config_name, config in test_configs.items():
            print(f"\n🔄 Testing: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Patterns: {', '.join(config['patterns'])}")
            
            try:
                # Create filtered signal generator
                filtered_signal_generator = FilteredSignalGenerator(
                    zone_detector, trend_classifier, risk_manager, 
                    allowed_patterns=config['patterns']
                )
                
                # Run backtest
                backtester = TradingBacktester(
                    filtered_signal_generator, 
                    initial_balance=self.initial_balance
                )
                
                # Execute backtest with our determined dates
                results = backtester.run_walk_forward_backtest(
                    classified_data, self.start_date, self.end_date, 365, 'EURUSD'
                )
                
                # Store results
                self.results[config_name] = {
                    'config': config,
                    'results': results,
                    'performance_metrics': self.calculate_performance_metrics(results)
                }
                
                successful_tests += 1
                
                print(f"   ✅ Success: {results['total_trades']} trades, "
                      f"{results['win_rate']}% WR, PF: {results['profit_factor']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                self.results[config_name] = None
        
        # Generate analysis if we have results
        if successful_tests > 0:
            print(f"\n📊 Analysis based on {successful_tests} successful tests...")
            self.generate_comparison_report()
            
            if successful_tests >= 2:
                self.create_performance_charts()
            
            # Pattern-specific insights
            self.generate_pattern_insights()
        else:
            print("\n❌ No successful backtests to analyze")
        
        return self.results
    
    def calculate_performance_metrics(self, results):
        """Calculate enhanced performance metrics"""
        if results['total_trades'] == 0:
            return self.empty_metrics()
        
        trades = results.get('closed_trades', [])
        if not trades:
            return self.empty_metrics()
        
        # Extract P&L values
        pnls = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
        
        if not pnls:
            return self.empty_metrics()
        
        # Core metrics
        total_return = (results['final_balance'] / self.initial_balance - 1) * 100
        avg_trade_return = np.mean(pnls)
        std_trade_return = np.std(pnls) if len(pnls) > 1 else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = avg_trade_return / std_trade_return if std_trade_return > 0 else 0
        
        # Win/Loss analysis
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Trading frequency
        test_period = pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)
        test_days = test_period.days
        trades_per_month = (results['total_trades'] / test_days) * 30 if test_days > 0 else 0
        
        # Consistency score
        consistency_score = len(wins) / len(pnls) if pnls else 0
        
        # Risk metrics
        max_dd_pct = results.get('max_drawdown_pct', 0)
        
        return {
            'total_return_pct': total_return,
            'avg_trade_return': avg_trade_return,
            'sharpe_ratio': sharpe_ratio,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades_per_month': trades_per_month,
            'consistency_score': consistency_score,
            'max_drawdown_pct': max_dd_pct,
            'risk_adjusted_return': total_return / max(max_dd_pct, 1)  # Return/DD ratio
        }
    
    def empty_metrics(self):
        """Return empty metrics for failed backtests"""
        return {
            'total_return_pct': 0,
            'avg_trade_return': 0, 
            'sharpe_ratio': 0,
            'win_loss_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'trades_per_month': 0,
            'consistency_score': 0,
            'max_drawdown_pct': 0,
            'risk_adjusted_return': 0
        }
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        print(f"\n📊 PATTERN PERFORMANCE COMPARISON REPORT")
        print("=" * 80)
        
        # Filter valid results
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            print("❌ No valid results to analyze")
            return
        
        # Create comparison table
        comparison_data = []
        
        for config_name, data in valid_results.items():
            results = data['results']
            metrics = data['performance_metrics']
            config = data['config']
            
            comparison_data.append({
                'Strategy': config['name'],
                'Trades': results['total_trades'],
                'Win Rate': f"{results['win_rate']:.1f}%",
                'Profit Factor': f"{results['profit_factor']:.2f}",
                'Total Return': f"{metrics['total_return_pct']:.1f}%",
                'Avg Trade': f"${metrics['avg_trade_return']:.0f}",
                'Trades/Month': f"{metrics['trades_per_month']:.1f}",
                'Max DD': f"{metrics['max_drawdown_pct']:.1f}%",
                'Risk/Return': f"{metrics['risk_adjusted_return']:.1f}"
            })
        
        # Display and save results
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("\n" + df.to_string(index=False))
            
            # Save results
            os.makedirs('results/pattern_comparison', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results/pattern_comparison/pattern_comparison_{timestamp}.csv'
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved: {filename}")
        
        # Identify best performers
        self.identify_best_performers(valid_results)
    
    def identify_best_performers(self, valid_results):
        """Identify top performing strategies"""
        print(f"\n🏆 PERFORMANCE RANKINGS:")
        print("=" * 50)
        
        # Rankings by different metrics
        rankings = {
            'Highest Profit Factor': ('profit_factor', 'results', True),
            'Highest Win Rate': ('win_rate', 'results', True),
            'Highest Total Return': ('total_return_pct', 'performance_metrics', True),
            'Best Risk-Adjusted Return': ('risk_adjusted_return', 'performance_metrics', True),
            'Most Active (Trades/Month)': ('trades_per_month', 'performance_metrics', True),
            'Lowest Drawdown': ('max_drawdown_pct', 'performance_metrics', False)
        }
        
        for metric_name, (metric_key, data_source, higher_better) in rankings.items():
            ranked_configs = []
            
            for config_name, data in valid_results.items():
                value = data[data_source][metric_key]
                ranked_configs.append((config_name, value, data['config']['name']))
            
            # Sort by metric
            ranked_configs.sort(key=lambda x: x[1], reverse=higher_better)
            
            if ranked_configs:
                winner = ranked_configs[0]
                print(f"🥇 {metric_name}: {winner[2]} ({winner[1]:.2f})")
    
    def generate_pattern_insights(self):
        """Generate specific insights about pattern performance"""
        print(f"\n💡 PATTERN INSIGHTS:")
        print("=" * 50)
        
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        # Compare momentum vs reversal
        momentum_data = valid_results.get('momentum_only')
        reversal_data = valid_results.get('reversal_only')
        
        if momentum_data and reversal_data:
            mom_pf = momentum_data['results']['profit_factor']
            rev_pf = reversal_data['results']['profit_factor']
            mom_wr = momentum_data['results']['win_rate']
            rev_wr = reversal_data['results']['win_rate']
            mom_trades = momentum_data['results']['total_trades']
            rev_trades = reversal_data['results']['total_trades']
            
            print(f"📈 Momentum vs Reversal Comparison:")
            print(f"   Momentum: {mom_trades} trades, {mom_wr:.1f}% WR, {mom_pf:.2f} PF")
            print(f"   Reversal: {rev_trades} trades, {rev_wr:.1f}% WR, {rev_pf:.2f} PF")
            
            if mom_pf > rev_pf:
                print(f"   💪 Momentum patterns outperformed by {((mom_pf/rev_pf-1)*100):.1f}%")
            else:
                print(f"   🔄 Reversal patterns outperformed by {((rev_pf/mom_pf-1)*100):.1f}%")
        
        # Compare bullish vs bearish
        bullish_data = valid_results.get('bullish_only')
        bearish_data = valid_results.get('bearish_only')
        
        if bullish_data and bearish_data:
            bull_pf = bullish_data['results']['profit_factor']
            bear_pf = bearish_data['results']['profit_factor']
            
            print(f"\n📊 Bullish vs Bearish Comparison:")
            print(f"   Bullish: {bullish_data['results']['total_trades']} trades, PF: {bull_pf:.2f}")
            print(f"   Bearish: {bearish_data['results']['total_trades']} trades, PF: {bear_pf:.2f}")
            
            if bull_pf > bear_pf:
                print(f"   📈 Bullish patterns performed better")
            else:
                print(f"   📉 Bearish patterns performed better")
        
        # Overall recommendation
        all_data = valid_results.get('all_patterns')
        if all_data:
            print(f"\n🎯 RECOMMENDATION:")
            if len(valid_results) > 1:
                # Find best individual strategy
                best_strategy = max(valid_results.items(), 
                                  key=lambda x: x[1]['results']['profit_factor'])
                best_name = best_strategy[1]['config']['name']
                best_pf = best_strategy[1]['results']['profit_factor']
                all_pf = all_data['results']['profit_factor']
                
                if all_pf >= best_pf * 0.9:  # Within 10% of best
                    print(f"   ✅ Use Complete System - provides diversification with minimal performance loss")
                else:
                    print(f"   🎯 Focus on {best_name} - significantly outperforms (PF: {best_pf:.2f} vs {all_pf:.2f})")
            else:
                print(f"   📊 Only one strategy tested successfully")
    
    def create_performance_charts(self):
        """Create comprehensive performance visualization"""
        print(f"\n📈 Creating performance charts...")
        
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if len(valid_results) < 2:
            print("⚠️  Need at least 2 strategies for meaningful charts")
            return
        
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pattern Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        names = [data['config']['name'] for data in valid_results.values()]
        colors = [data['config']['color'] for data in valid_results.values()]
        
        profit_factors = [data['results']['profit_factor'] for data in valid_results.values()]
        win_rates = [data['results']['win_rate'] for data in valid_results.values()]
        total_returns = [data['performance_metrics']['total_return_pct'] for data in valid_results.values()]
        max_drawdowns = [data['performance_metrics']['max_drawdown_pct'] for data in valid_results.values()]
        
        # Chart 1: Profit Factor
        bars1 = ax1.bar(names, profit_factors, color=colors, alpha=0.8)
        ax1.set_title('Profit Factor by Strategy', fontweight='bold')
        ax1.set_ylabel('Profit Factor')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, profit_factors):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Win Rate
        bars2 = ax2.bar(names, win_rates, color=colors, alpha=0.8)
        ax2.set_title('Win Rate by Strategy', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, win_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: Total Return
        bars3 = ax3.bar(names, total_returns, color=colors, alpha=0.8)
        ax3.set_title('Total Return by Strategy', fontweight='bold')
        ax3.set_ylabel('Total Return (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, total_returns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(total_returns)*0.02),
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Chart 4: Maximum Drawdown
        bars4 = ax4.bar(names, max_drawdowns, color=colors, alpha=0.8)
        ax4.set_title('Maximum Drawdown by Strategy', fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, max_drawdowns):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(max_drawdowns)*0.02),
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        os.makedirs('results/pattern_comparison', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_filename = f'results/pattern_comparison/pattern_performance_{timestamp}.png'
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Charts saved: {chart_filename}")
        
        plt.show()

class FilteredSignalGenerator:
    """Signal generator that filters to specific pattern types"""
    
    def __init__(self, zone_detector, trend_classifier, risk_manager, allowed_patterns):
        self.zone_detector = zone_detector
        self.trend_classifier = trend_classifier  
        self.risk_manager = risk_manager
        self.allowed_patterns = set(allowed_patterns)  # Use set for faster lookup
        
    def generate_signals(self, data, timeframe, pair):
        """Generate signals using only specified pattern types"""
        
        # Detect all patterns first
        all_patterns = self.zone_detector.detect_all_patterns(data)
        
        # Filter to only allowed patterns
        filtered_zones = []
        
        # Add momentum patterns if allowed
        if 'D-B-D' in self.allowed_patterns:
            filtered_zones.extend(all_patterns.get('dbd_patterns', []))
        if 'R-B-R' in self.allowed_patterns:
            filtered_zones.extend(all_patterns.get('rbr_patterns', []))
            
        # Add reversal patterns if allowed
        if 'D-B-R' in self.allowed_patterns:
            filtered_zones.extend(all_patterns.get('dbr_patterns', []))
        if 'R-B-D' in self.allowed_patterns:
            filtered_zones.extend(all_patterns.get('rbd_patterns', []))
        
        print(f"   Filtered to {len(filtered_zones)} zones from patterns: {', '.join(self.allowed_patterns)}")
        
        if not filtered_zones:
            return []
        
        # Create temporary signal generator with filtered patterns
        from modules.signal_generator import SignalGenerator
        temp_generator = SignalGenerator(self.zone_detector, self.trend_classifier, self.risk_manager)
        
        # Mock the pattern detection to return only our filtered zones
        original_detect_method = self.zone_detector.detect_all_patterns
        
        def mock_detect_patterns(data_input):
            # Sort zones by type for the mock response
            mock_response = {
                'dbd_patterns': [z for z in filtered_zones if z['type'] == 'D-B-D'],
                'rbr_patterns': [z for z in filtered_zones if z['type'] == 'R-B-R'],
                'dbr_patterns': [z for z in filtered_zones if z['type'] == 'D-B-R'],
                'rbd_patterns': [z for z in filtered_zones if z['type'] == 'R-B-D'],
                'total_patterns': len(filtered_zones)
            }
            return mock_response
        
        # Temporarily replace the detection method
        self.zone_detector.detect_all_patterns = mock_detect_patterns
        
        try:
            # Generate signals with filtered patterns
            signals = temp_generator.generate_signals(data, timeframe, pair)
        finally:
            # Always restore the original method
            self.zone_detector.detect_all_patterns = original_detect_method
        
        return signals

def main():
    """Main execution function"""
    print("🚀 PATTERN PERFORMANCE ANALYZER")
    print("=" * 40)
    
    try:
        analyzer = PatternPerformanceAnalyzer(initial_balance=10000)
        results = analyzer.run_pattern_comparison()
        
        print(f"\n🎉 Pattern comparison complete!")
        print(f"📁 Results saved in: results/pattern_comparison/")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()