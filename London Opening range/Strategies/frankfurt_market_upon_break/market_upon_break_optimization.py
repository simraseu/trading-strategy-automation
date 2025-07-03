import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import time

class SmartOptimizer:
    """
    SMART OPTIMIZER - Focused on best-performing parameters
    Based on 24-test results: 2,880 combinations in ~4.4 hours
    """
    
    def __init__(self, data_file):
        print("📊 Loading and optimizing data structure for SMART optimization...")
        start_time = time.time()
        self.base_data = self.load_and_optimize_data(data_file)
        load_time = time.time() - start_time
        print(f"✅ Data optimized in {load_time:.1f} seconds")
        self.optimization_results = []
    
    def load_and_optimize_data(self, file_path):
        """Load data and create optimized structure"""
        # Load raw data
        df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None,
                         names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD'])
        
        # Pre-process everything
        df[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)
        df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        print(f"   📊 Loaded {len(df):,} rows spanning {df['DATE'].nunique():,} unique dates")
        
        # Pre-group by date to eliminate repeated filtering
        print("   🔧 Pre-grouping data by date...")
        group_start = time.time()
        grouped_data = {}
        
        for date, day_data in df.groupby('DATE'):
            grouped_data[date] = day_data.reset_index(drop=True)
        
        group_time = time.time() - group_start
        print(f"   ✅ Pre-grouped {len(grouped_data):,} dates in {group_time:.2f}s")
        
        return grouped_data
    
    def define_smart_parameter_ranges(self):
        """SMART parameter ranges based on 24-test insights"""
        return {
            # Based on results: 0.6 target was best, test around it
            'target_multiplier': [0.4, 0.5, 0.6, 0.7, 0.8],         # 5 values (focus on winners)
            
            # Test both stop methods (range_opposite won in 24-test, but test fixed_ratio)
            'stop_type': ['range_opposite', 'fixed_ratio'],           # 2 values
            'stop_multiplier': [0.8, 1.0, 1.2, 1.5],                # 4 values (for fixed_ratio)
            
            # 1.0 buffer was slightly better, but test around it
            'breakout_buffer_pips': [1.0, 2.0, 3.0],                # 3 values
            
            # Keep successful range filters
            'min_range_pips': [5, 8],                                # 2 values (both worked)
            'max_range_pips': [40, 50, 60],                          # 3 values (50+ was best)
            
            # Test minimal entry delays (0 was good)
            'entry_delay_minutes': [0, 15],                          # 2 values
            
            # Test reasonable cutoff times
            'trade_cutoff_hour': [17, 18],                           # 2 values
        }
        # Total: 5×2×4×3×2×3×2×2 = 2,880 combinations
        # Estimated time: 2,880 × 5.55s = 4.4 hours
    
    def run_smart_optimization(self, optimization_metric='total_pips'):
        """Run smart optimization with performance tracking"""
        param_ranges = self.define_smart_parameter_ranges()
        
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        
        print(f"\n🧠 SMART OPTIMIZATION STARTING")
        print(f"📊 Total combinations: {total_combinations:,}")
        print(f"⏱️  Estimated time: {total_combinations * 5.55 / 3600:.1f} hours")
        print(f"🎯 Focus: Best-performing parameters from 24-test results")
        print("="*70)
        
        start_time = time.time()
        combination_count = 0
        valid_results = 0
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Progress tracking
        next_progress_update = 100
        
        for combination in product(*param_values):
            combination_count += 1
            params = dict(zip(param_names, combination))
            
            # Run test
            try:
                result = self.run_single_test(params)
                
                if result and result.get('total_trades', 0) >= 10:
                    result.update(params)
                    result['combination_id'] = combination_count
                    self.optimization_results.append(result)
                    valid_results += 1
                    
            except Exception as e:
                # Skip errors silently for speed
                pass
            
            # Progress updates
            if combination_count >= next_progress_update:
                elapsed = time.time() - start_time
                progress_pct = (combination_count / total_combinations) * 100
                
                if combination_count > 0:
                    avg_time = elapsed / combination_count
                    eta_seconds = avg_time * (total_combinations - combination_count)
                    eta_hours = eta_seconds / 3600
                    
                    print(f"📊 Progress: {combination_count:,}/{total_combinations:,} ({progress_pct:.1f}%) | "
                          f"Valid: {valid_results:,} | "
                          f"Speed: {combination_count/elapsed:.1f}/sec | "
                          f"ETA: {eta_hours:.1f}h")
                
                # Update intervals: 100, 200, 500, 1000, then every 500
                if next_progress_update < 500:
                    next_progress_update = min(next_progress_update * 2, 500)
                else:
                    next_progress_update += 500
        
        total_time = time.time() - start_time
        avg_time = total_time / total_combinations
        
        print("\n" + "="*70)
        print(f"🎉 SMART OPTIMIZATION COMPLETE!")
        print(f"⏱️  Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        print(f"⚡ Average per test: {avg_time:.2f} seconds")
        print(f"📊 Valid results: {valid_results:,}/{total_combinations:,}")
        print(f"✅ Success rate: {valid_results/total_combinations*100:.1f}%")
        print("="*70)
        
        return self.analyze_smart_results(optimization_metric)
    
    def run_single_test(self, params):
        """Run single test efficiently"""
        try:
            strategy = OptimizedFrankfurtStrategy(self.base_data, params)
            return strategy.optimized_backtest(
                min_range_pips=params['min_range_pips'],
                max_range_pips=params['max_range_pips']
            )
        except:
            return None
    
    def analyze_smart_results(self, metric):
        """Comprehensive analysis of smart optimization results"""
        if not self.optimization_results:
            print("❌ No valid results found")
            return None
        
        df = pd.DataFrame(self.optimization_results)
        df = df.sort_values(metric, ascending=False)
        
        print(f"\n🏆 TOP 20 SMART OPTIMIZATION RESULTS (by {metric}):")
        print("="*120)
        
        display_cols = ['total_trades', 'win_rate', 'total_pips', 'risk_reward_ratio',
                       'target_multiplier', 'stop_type', 'stop_multiplier',
                       'breakout_buffer_pips', 'min_range_pips', 'max_range_pips',
                       'entry_delay_minutes', 'trade_cutoff_hour']
        
        top_results = df[display_cols].head(20).copy()
        top_results['win_rate'] = top_results['win_rate'].round(1)
        top_results['total_pips'] = top_results['total_pips'].round(1)
        top_results['risk_reward_ratio'] = top_results['risk_reward_ratio'].round(2)
        
        print(top_results.to_string(index=False))
        
        # Save results
        df.to_csv('smart_optimization_results.csv', index=False)
        print(f"\n📁 Results saved to: smart_optimization_results.csv")
        
        # Comprehensive analysis
        print(f"\n📈 COMPREHENSIVE ANALYSIS:")
        print(f"🥇 Best Total Pips: {df['total_pips'].max():.1f}")
        print(f"🏆 Best Win Rate: {df['win_rate'].max():.1f}%")
        print(f"💎 Best Risk-Reward: {df['risk_reward_ratio'].max():.2f}")
        print(f"📊 Average Total Pips: {df['total_pips'].mean():.1f}")
        print(f"📏 Standard Deviation: {df['total_pips'].std():.1f}")
        print(f"🎯 Median Total Pips: {df['total_pips'].median():.1f}")
        
        # Parameter insights
        print(f"\n🧠 PARAMETER INSIGHTS:")
        
        # Best target multipliers
        target_analysis = df.groupby('target_multiplier')['total_pips'].agg(['mean', 'max', 'count']).round(1)
        print(f"\n🎯 Target Multiplier Analysis:")
        print(target_analysis.to_string())
        
        # Best stop types
        stop_analysis = df.groupby('stop_type')['total_pips'].agg(['mean', 'max', 'count']).round(1)
        print(f"\n🛑 Stop Type Analysis:")
        print(stop_analysis.to_string())
        
        # Best range sizes
        range_analysis = df.groupby(['min_range_pips', 'max_range_pips'])['total_pips'].mean().round(1)
        print(f"\n📏 Range Size Analysis (top 10):")
        print(range_analysis.sort_values(ascending=False).head(10).to_string())
        
        # Create visualizations
        self.create_smart_visualizations(df)
        
        return df
    
    def create_smart_visualizations(self, df):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Performance distribution
        axes[0,0].hist(df['total_pips'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].axvline(df['total_pips'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["total_pips"].mean():.0f}')
        axes[0,0].set_xlabel('Total Pips')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Total Pips Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Target multiplier performance
        target_perf = df.groupby('target_multiplier')['total_pips'].mean()
        axes[0,1].bar(target_perf.index, target_perf.values, color='green', alpha=0.7)
        axes[0,1].set_xlabel('Target Multiplier')
        axes[0,1].set_ylabel('Average Total Pips')
        axes[0,1].set_title('Target Multiplier Performance')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Stop type comparison
        stop_perf = df.groupby('stop_type')['total_pips'].mean()
        axes[0,2].bar(stop_perf.index, stop_perf.values, color='orange', alpha=0.7)
        axes[0,2].set_xlabel('Stop Type')
        axes[0,2].set_ylabel('Average Total Pips')
        axes[0,2].set_title('Stop Type Performance')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Win rate vs Total pips
        axes[1,0].scatter(df['win_rate'], df['total_pips'], alpha=0.6, c='purple')
        axes[1,0].set_xlabel('Win Rate (%)')
        axes[1,0].set_ylabel('Total Pips')
        axes[1,0].set_title('Win Rate vs Total Pips')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Range size impact
        range_perf = df.groupby('max_range_pips')['total_pips'].mean()
        axes[1,1].bar(range_perf.index, range_perf.values, color='brown', alpha=0.7)
        axes[1,1].set_xlabel('Max Range Pips')
        axes[1,1].set_ylabel('Average Total Pips')
        axes[1,1].set_title('Max Range Size Impact')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Risk-reward distribution
        axes[1,2].hist(df['risk_reward_ratio'], bins=20, alpha=0.7, color='red')
        axes[1,2].axvline(df['risk_reward_ratio'].mean(), color='blue', linestyle='--',
                         label=f'Mean: {df["risk_reward_ratio"].mean():.2f}')
        axes[1,2].set_xlabel('Risk-Reward Ratio')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Risk-Reward Distribution')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('smart_optimization_analysis.png', dpi=300, bbox_inches='tight')
        
        print("📊 Comprehensive analysis charts saved as: smart_optimization_analysis.png")
        return fig

class OptimizedFrankfurtStrategy:
    """
    Optimized Frankfurt strategy using pre-grouped data
    """
    
    def __init__(self, pre_grouped_data, params):
        self.grouped_data = pre_grouped_data
        self.params = params
        self.trades = []
    
    def optimized_backtest(self, min_range_pips=5, max_range_pips=50):
        """Ultra-fast backtest using pre-grouped data"""
        
        # Process all pre-grouped dates efficiently
        for date, day_data in self.grouped_data.items():
            
            session_hours = self.get_frankfurt_session_hours(date)
            
            opening_range = self.calculate_opening_range(day_data, session_hours)
            if not opening_range:
                continue
                
            range_pips = opening_range['range_size'] * 10000
            if range_pips < min_range_pips or range_pips > max_range_pips:
                continue
            
            breakout_data = day_data[day_data['hour'] >= session_hours['trade_start']]
            
            trade_result = self.check_for_breakout(breakout_data, opening_range, date, range_pips)
            
            if trade_result:
                self.trades.append(trade_result)
        
        return self.calculate_results()
    
    def get_frankfurt_session_hours(self, date):
        """Get Frankfurt session hours adjusted for DST"""
        dt = pd.to_datetime(date)
        month = dt.month
        
        if 3 <= month <= 10:  # Summer
            return {
                'range_start': 9,
                'range_end': 12,
                'trade_start': 12
            }
        else:  # Winter
            return {
                'range_start': 10,
                'range_end': 13,
                'trade_start': 13
            }
    
    def calculate_opening_range(self, day_data, session_hours):
        """Calculate the Frankfurt opening range"""
        range_data = day_data[
            (day_data['hour'] >= session_hours['range_start']) & 
            (day_data['hour'] < session_hours['range_end'])
        ]
        
        if len(range_data) == 0:
            return None
            
        return {
            'high': range_data['HIGH'].max(),
            'low': range_data['LOW'].min(),
            'range_size': range_data['HIGH'].max() - range_data['LOW'].min()
        }
    
    def check_for_breakout(self, breakout_data, opening_range, date, range_pips):
        """Check for range breakout"""
        
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
    
    def execute_trade(self, direction, entry_price, support_resistance_level, remaining_data, entry_time, date, range_pips):
        """Execute trade with optimizable parameters"""
        
        target_multiplier = self.params['target_multiplier']
        range_size = range_pips / 10000
        
        if direction == 'LONG':
            target = entry_price + (target_multiplier * range_size)
        else:
            target = entry_price - (target_multiplier * range_size)
        
        # Calculate stop loss
        if self.params['stop_type'] == 'range_opposite':
            stop_loss = support_resistance_level
        else:  # fixed_ratio
            stop_multiplier = self.params['stop_multiplier']
            if direction == 'LONG':
                stop_loss = entry_price - (stop_multiplier * range_size)
            else:
                stop_loss = entry_price + (stop_multiplier * range_size)
        
        # Apply filters
        cutoff_hour = self.params['trade_cutoff_hour']
        remaining_data = remaining_data[remaining_data['hour'] <= cutoff_hour]
        
        delay_minutes = self.params['entry_delay_minutes']
        if delay_minutes > 0:
            entry_datetime = pd.to_datetime(f"{date} {entry_time}")
            cutoff_time = entry_datetime + pd.Timedelta(minutes=delay_minutes)
            remaining_data = remaining_data[remaining_data['datetime'] >= cutoff_time]
        
        # Track trade
        future_data = remaining_data[remaining_data['TIME'] > entry_time]
        
        for _, candle in future_data.iterrows():
            # Check target
            if direction == 'LONG' and candle['HIGH'] >= target:
                return {
                    'date': date, 'direction': direction, 'entry_time': entry_time,
                    'entry_price': entry_price, 'exit_time': candle['TIME'],
                    'exit_price': target, 'exit_reason': 'TARGET', 'result': 'WIN',
                    'pips': (target - entry_price) * 10000, 'range_pips': range_pips
                }
            elif direction == 'SHORT' and candle['LOW'] <= target:
                return {
                    'date': date, 'direction': direction, 'entry_time': entry_time,
                    'entry_price': entry_price, 'exit_time': candle['TIME'],
                    'exit_price': target, 'exit_reason': 'TARGET', 'result': 'WIN',
                    'pips': (entry_price - target) * 10000, 'range_pips': range_pips
                }
            
            # Check stop
            if direction == 'LONG' and candle['LOW'] <= stop_loss:
                return {
                    'date': date, 'direction': direction, 'entry_time': entry_time,
                    'entry_price': entry_price, 'exit_time': candle['TIME'],
                    'exit_price': stop_loss, 'exit_reason': 'STOP_LOSS', 'result': 'LOSS',
                    'pips': (stop_loss - entry_price) * 10000, 'range_pips': range_pips
                }
            elif direction == 'SHORT' and candle['HIGH'] >= stop_loss:
                return {
                    'date': date, 'direction': direction, 'entry_time': entry_time,
                    'entry_price': entry_price, 'exit_time': candle['TIME'],
                    'exit_price': stop_loss, 'exit_reason': 'STOP_LOSS', 'result': 'LOSS',
                    'pips': (entry_price - stop_loss) * 10000, 'range_pips': range_pips
                }
        
        return None
    
    def calculate_results(self):
        """Calculate results efficiently"""
        if not self.trades:
            return None
            
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['result'] == 'WIN'])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pips = df_trades['pips'].sum()
        avg_win = df_trades[df_trades['result'] == 'WIN']['pips'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['result'] == 'LOSS']['pips'].mean() if total_trades > winning_trades else 0
        
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'risk_reward_ratio': risk_reward
        }

if __name__ == "__main__":
    import os
    
    data_path = r'C:\Users\sim\Desktop\Quant\European Opening Range\trading-strategy-automation\London Opening range\Data\EURUSD.raw_M15_201701100000_202507021745.csv'
    
    print("🧠 SMART OPTIMIZATION")
    print("💡 Focused on best-performing parameters from 24-test results")
    print("🎯 2,880 combinations in ~4.4 hours")
    print("="*70)
    
    if not os.path.exists(data_path):
        print("❌ File not found! Please check the path.")
        exit()
    
    print("✅ File found! Starting smart optimization...")
    
    # Run smart optimization
    optimizer = SmartOptimizer(data_path)
    results = optimizer.run_smart_optimization()
    
    print("\n🎉 SMART OPTIMIZATION COMPLETE!")
    print("📊 Check the results above and the saved CSV file for detailed analysis")