import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

class StrategyOptimizer:
    """
    Comprehensive optimizer for Frankfurt Opening Range Strategy
    Tests all parameter combinations systematically
    """
    
    def __init__(self, strategy_class, data_file):
        self.strategy_class = strategy_class
        self.data_file = data_file
        self.optimization_results = []
        
    def define_parameter_ranges(self):
        """Define all parameters to optimize with their ranges"""
        return {
            # Core strategy parameters
            'target_multiplier': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5],
            'stop_type': ['range_opposite', 'fixed_ratio'],  # Two stop loss methods
            'stop_multiplier': [0.8, 1.0, 1.2, 1.5, 2.0],  # For fixed ratio stops
            
            # Entry parameters
            'breakout_buffer_pips': [0.5, 1.0, 2.0, 3.0, 5.0],
            
            # Range filtering
            'min_range_pips': [3, 5, 8, 10, 15],
            'max_range_pips': [30, 40, 50, 60, 80],
            
            # Time parameters
            'entry_delay_minutes': [0, 15, 30, 60],
            'trade_cutoff_hour': [15, 16, 17, 18, 20],  # GMT+3
        }
    
    def optimize_strategy(self, optimization_metric='total_pips'):
        """
        Run optimization across all parameter combinations
        """
        param_ranges = self.define_parameter_ranges()
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        
        print(f"🔍 Starting optimization with {total_combinations:,} combinations...")
        print(f"📊 Optimization metric: {optimization_metric}")
        
        combination_count = 0
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in product(*param_values):
            combination_count += 1
            
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Progress tracking
            if combination_count % 100 == 0:
                progress = (combination_count / total_combinations) * 100
                print(f"Progress: {combination_count:,}/{total_combinations:,} ({progress:.1f}%)")
            
            # Run backtest with these parameters
            try:
                result = self.run_single_backtest(params)
                if result and result['total_trades'] >= 10:  # Minimum trade threshold
                    
                    # Add parameters to result
                    result.update(params)
                    result['combination_id'] = combination_count
                    
                    self.optimization_results.append(result)
                    
            except Exception as e:
                print(f"Error in combination {combination_count}: {e}")
                continue
        
        print(f"✅ Optimization complete! {len(self.optimization_results)} valid combinations tested")
        return self.analyze_optimization_results(optimization_metric)
    
    def run_single_backtest(self, params):
        """Run backtest with specific parameter set"""
        
        # Create modified strategy class
        strategy = ModifiedFrankfurtStrategy(self.data_file, params)
        results = strategy.backtest(
            min_range_pips=params['min_range_pips'],
            max_range_pips=params['max_range_pips']
        )
        
        return results
    
    def analyze_optimization_results(self, metric):
        """Analyze and rank optimization results"""
        
        if not self.optimization_results:
            print("❌ No valid optimization results found")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.optimization_results)
        
        # Sort by optimization metric (descending for profit metrics)
        if metric in ['total_pips', 'win_rate', 'risk_reward_ratio']:
            df = df.sort_values(metric, ascending=False)
        else:
            df = df.sort_values(metric, ascending=True)
        
        # Display top 10 results
        print("\n" + "="*80)
        print(f"🏆 TOP 10 PARAMETER COMBINATIONS (by {metric})")
        print("="*80)
        
        display_cols = [
            'total_trades', 'win_rate', 'total_pips', 'risk_reward_ratio',
            'target_multiplier', 'stop_type', 'stop_multiplier', 
            'min_range_pips', 'max_range_pips', 'breakout_buffer_pips'
        ]
        
        print(df[display_cols].head(10).to_string(index=False))
        
        # Save full results
        df.to_csv('optimization_results.csv', index=False)
        print(f"\n📁 Full results saved to: optimization_results.csv")
        
        # Create visualization
        self.create_optimization_plots(df)
        
        return df
    
    def create_optimization_plots(self, df):
        """Create visualization of optimization results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Total Pips vs Win Rate scatter
        axes[0,0].scatter(df['win_rate'], df['total_pips'], alpha=0.6)
        axes[0,0].set_xlabel('Win Rate (%)')
        axes[0,0].set_ylabel('Total Pips')
        axes[0,0].set_title('Win Rate vs Total Pips')
        
        # 2. Target Multiplier vs Performance
        target_perf = df.groupby('target_multiplier')['total_pips'].mean()
        axes[0,1].bar(target_perf.index, target_perf.values)
        axes[0,1].set_xlabel('Target Multiplier')
        axes[0,1].set_ylabel('Average Total Pips')
        axes[0,1].set_title('Target Multiplier Performance')
        
        # 3. Range Size vs Performance
        range_perf = df.groupby(['min_range_pips', 'max_range_pips'])['total_pips'].mean().reset_index()
        pivot_data = range_perf.pivot(index='min_range_pips', columns='max_range_pips', values='total_pips')
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', ax=axes[1,0])
        axes[1,0].set_title('Range Size Optimization Heatmap')
        
        # 4. Risk-Reward Distribution
        axes[1,1].hist(df['risk_reward_ratio'], bins=20, alpha=0.7)
        axes[1,1].set_xlabel('Risk-Reward Ratio')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Risk-Reward Distribution')
        
        plt.tight_layout()
        plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Optimization charts saved as: optimization_analysis.png")

class ModifiedFrankfurtStrategy(FrankfurtOpeningRangeStrategy):
    """
    Modified strategy class that accepts optimization parameters
    """
    
    def __init__(self, data_file, params):
        super().__init__(data_file)
        self.params = params
    
    def execute_trade(self, direction, entry_price, support_resistance_level, remaining_data, entry_time, date, range_pips):
        """Modified execute_trade with optimizable parameters"""
        
        # Use optimized parameters
        target_multiplier = self.params['target_multiplier']
        range_size = range_pips / 10000
        
        # Calculate target
        if direction == 'LONG':
            target = entry_price + (target_multiplier * range_size)
        else:
            target = entry_price - (target_multiplier * range_size)
        
        # Calculate stop loss based on stop type
        if self.params['stop_type'] == 'range_opposite':
            stop_loss = support_resistance_level
        else:  # fixed_ratio
            stop_multiplier = self.params['stop_multiplier']
            if direction == 'LONG':
                stop_loss = entry_price - (stop_multiplier * range_size)
            else:
                stop_loss = entry_price + (stop_multiplier * range_size)
        
        # Apply trade cutoff hour
        cutoff_hour = self.params['trade_cutoff_hour']
        remaining_data = remaining_data[remaining_data['hour'] <= cutoff_hour]
        
        # Apply entry delay
        delay_minutes = self.params['entry_delay_minutes']
        if delay_minutes > 0:
            entry_datetime = pd.to_datetime(f"{date} {entry_time}")
            cutoff_time = entry_datetime + pd.Timedelta(minutes=delay_minutes)
            remaining_data = remaining_data[remaining_data['datetime'] >= cutoff_time]
        
        # Rest of trade execution logic remains the same...
        return super().execute_trade(direction, entry_price, stop_loss, remaining_data, entry_time, date, range_pips)
    
    def check_for_breakout(self, breakout_data, opening_range, date, range_pips):
        """Modified breakout detection with optimized buffer"""
        
        buffer_pips = self.params['breakout_buffer_pips']
        buffer_price = buffer_pips / 10000
        
        high_break_price = opening_range['high'] + buffer_price
        low_break_price = opening_range['low'] - buffer_price
        
        for _, candle in breakout_data.iterrows():
            if candle['HIGH'] >= high_break_price:
                return self.execute_trade(
                    'LONG', high_break_price, opening_range['low'],
                    breakout_data, candle['TIME'], date, range_pips
                )
            elif candle['LOW'] <= low_break_price:
                return self.execute_trade(
                    'SHORT', low_break_price, opening_range['high'],
                    breakout_data, candle['TIME'], date, range_pips
                )
        
        return None

# USAGE EXAMPLE
if __name__ == "__main__":
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(
        ModifiedFrankfurtStrategy,
        'C:/path/to/your/EURUSD_data.csv'
    )
    
    # Run optimization (this will take time!)
    best_results = optimizer.optimize_strategy(optimization_metric='total_pips')
    
    # Alternative metrics you can optimize for:
    # - 'win_rate': Maximize win percentage
    # - 'risk_reward_ratio': Maximize risk-reward
    # - 'total_pips': Maximize total profit