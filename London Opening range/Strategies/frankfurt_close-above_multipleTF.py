import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from pathlib import Path
import seaborn as sns

class FrankfurtOpeningRangeStrategy:
   """
   Frankfurt Opening Range Breakout Strategy - CLOSE ABOVE VERSION
   Range: 08:00-10:59 GMT+3 (adjusted for DST)
   Entry: Close above/below range + buy/sell on next candle open
   Target: 0.5x range size continuation
   Stop: Range opposite (support/resistance)
   """
   
   def __init__(self, data_file=None):
       if data_file:
           self.df = self.load_data(data_file)
       self.trades = []
       self.results = {}
       
   def load_data(self, file_path):
       """Load and prepare forex data"""
       df = pd.read_csv(file_path, 
                        sep='\t',
                        skiprows=1,
                        header=None,
                        names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD'])
       
       # Convert price columns to numeric
       price_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
       df[price_cols] = df[price_cols].astype(float)
       
       # Create datetime
       df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
       df['hour'] = df['datetime'].dt.hour
       df['minute'] = df['datetime'].dt.minute
       
       return df
   
   def get_frankfurt_session_hours(self, date):
       """Get Frankfurt session hours adjusted for DST"""
       dt = pd.to_datetime(date)
       month = dt.month
       
       if 3 <= month <= 10:  # Summer
           return {
               'range_start': 9,   # 08:00 Frankfurt = 09:00 GMT+3 in summer
               'range_end': 12,    # 10:59 Frankfurt = 11:59 GMT+3 in summer  
               'trade_start': 12   # Start looking for breakouts at 12:00
           }
       else:  # Winter
           return {
               'range_start': 10,  # 08:00 Frankfurt = 10:00 GMT+3 in winter
               'range_end': 13,    # 10:59 Frankfurt = 12:59 GMT+3 in winter
               'trade_start': 13   # Start looking for breakouts at 13:00
           }
   
   def calculate_opening_range(self, day_data, session_hours):
       """Calculate the Frankfurt opening range (08:00-10:59)"""
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
   
   def backtest(self, min_range_pips=5, max_range_pips=50):
       """
       Run the Frankfurt Opening Range strategy backtest
       NEW: Close above/below range + entry on next candle open
       """
       print("🚀 Starting Frankfurt Opening Range Strategy - CLOSE ABOVE VERSION...")
       print("📊 Strategy: Close above/below range + entry on next candle open")
       
       for date in self.df['DATE'].unique():
           day_data = self.df[self.df['DATE'] == date].copy()
           session_hours = self.get_frankfurt_session_hours(date)
           
           # Calculate opening range
           opening_range = self.calculate_opening_range(day_data, session_hours)
           if not opening_range:
               continue
               
           # Filter by range size (avoid choppy/too wide ranges)
           range_pips = opening_range['range_size'] * 10000  # Convert to pips
           if range_pips < min_range_pips or range_pips > max_range_pips:
               continue
           
           # Look for close above/below range after range period
           breakout_data = day_data[day_data['hour'] >= session_hours['trade_start']]
           
           trade_result = self.check_for_close_breakout(
               breakout_data, opening_range, date, range_pips
           )
           
           if trade_result:
               self.trades.append(trade_result)
       
       self.analyze_results()
       return self.results
   
   def check_for_close_breakout(self, breakout_data, opening_range, date, range_pips):
       """Check for close above/below range and execute on next candle open"""
       
       range_high = opening_range['high']
       range_low = opening_range['low']
       
       breakout_data_list = breakout_data.reset_index(drop=True)
       
       for i, (_, candle) in enumerate(breakout_data_list.iterrows()):
           # Check if we have a next candle for entry
           if i >= len(breakout_data_list) - 1:
               continue
               
           next_candle = breakout_data_list.iloc[i + 1]
           
           # Long signal: Close above range high
           if candle['CLOSE'] > range_high:
               return self.execute_trade_on_next_open(
                   'LONG', next_candle, opening_range['low'],
                   breakout_data_list[i+1:], date, range_pips, candle['TIME']
               )
           
           # Short signal: Close below range low
           elif candle['CLOSE'] < range_low:
               return self.execute_trade_on_next_open(
                   'SHORT', next_candle, opening_range['high'],
                   breakout_data_list[i+1:], date, range_pips, candle['TIME']
               )
       
       return None
   
   def execute_trade_on_next_open(self, direction, entry_candle, support_resistance_level, 
                                 remaining_data, date, range_pips, signal_time):
       """Execute trade on next candle open after close breakout signal"""
       
       entry_price = entry_candle['OPEN']
       entry_time = entry_candle['TIME']
       
       # STRATEGY PARAMETERS (OPTIMIZED)
       target_multiplier = 0.5    # Target = 0.5x range size
       range_size = range_pips / 10000
       
       # Calculate target and stop loss
       if direction == 'LONG':
           target = entry_price + (target_multiplier * range_size)
           stop_loss = support_resistance_level  # Range low (opposite end)
       else:
           target = entry_price - (target_multiplier * range_size)
           stop_loss = support_resistance_level  # Range high (opposite end)
       
       # Track trade through rest of day (starting from entry candle)
       for _, candle in remaining_data.iterrows():
           # Skip the entry candle itself
           if candle['TIME'] == entry_time:
               continue
               
           # Check for target hit first
           if direction == 'LONG' and candle['HIGH'] >= target:
               pips_profit = (target - entry_price) * 10000
               return {
                   'date': date,
                   'direction': direction,
                   'signal_time': signal_time,
                   'entry_time': entry_time,
                   'entry_price': entry_price,
                   'exit_time': candle['TIME'],
                   'exit_price': target,
                   'exit_reason': 'TARGET',
                   'result': 'WIN',
                   'pips': pips_profit,
                   'range_pips': range_pips,
                   'target_price': target,
                   'stop_loss_price': stop_loss
               }
           
           elif direction == 'SHORT' and candle['LOW'] <= target:
               pips_profit = (entry_price - target) * 10000
               return {
                   'date': date,
                   'direction': direction,
                   'signal_time': signal_time,
                   'entry_time': entry_time,
                   'entry_price': entry_price,
                   'exit_time': candle['TIME'],
                   'exit_price': target,
                   'exit_reason': 'TARGET',
                   'result': 'WIN',
                   'pips': pips_profit,
                   'range_pips': range_pips,
                   'target_price': target,
                   'stop_loss_price': stop_loss
               }
           
           # Check for stop loss hit
           if direction == 'LONG' and candle['LOW'] <= stop_loss:
               pips_loss = (stop_loss - entry_price) * 10000
               return {
                   'date': date,
                   'direction': direction,
                   'signal_time': signal_time,
                   'entry_time': entry_time,
                   'entry_price': entry_price,
                   'exit_time': candle['TIME'],
                   'exit_price': stop_loss,
                   'exit_reason': 'STOP_LOSS',
                   'result': 'LOSS',
                   'pips': pips_loss,
                   'range_pips': range_pips,
                   'target_price': target,
                   'stop_loss_price': stop_loss
               }
               
           elif direction == 'SHORT' and candle['HIGH'] >= stop_loss:
               pips_loss = (entry_price - stop_loss) * 10000
               return {
                   'date': date,
                   'direction': direction,
                   'signal_time': signal_time,
                   'entry_time': entry_time,
                   'entry_price': entry_price,
                   'exit_time': candle['TIME'],
                   'exit_price': stop_loss,
                   'exit_reason': 'STOP_LOSS',
                   'result': 'LOSS',
                   'pips': pips_loss,
                   'range_pips': range_pips,
                   'target_price': target,
                   'stop_loss_price': stop_loss
               }
       
       return None
   
   def calculate_advanced_kpis(self, trades_df):
       """Calculate advanced KPIs including the requested metrics"""
       if len(trades_df) == 0:
           return {}
       
       # Basic calculations
       total_pips = trades_df['pips'].sum()
       winning_trades = trades_df[trades_df['result'] == 'WIN']
       losing_trades = trades_df[trades_df['result'] == 'LOSS']
       
       total_wins_pips = winning_trades['pips'].sum() if len(winning_trades) > 0 else 0
       total_losses_pips = abs(losing_trades['pips'].sum()) if len(losing_trades) > 0 else 0
       
       # 1. PROFIT FACTOR
       profit_factor = total_wins_pips / total_losses_pips if total_losses_pips > 0 else float('inf')
       
       # 2. AVERAGE RETURN PER TRADE
       avg_return_per_trade = total_pips / len(trades_df)
       
       # 3. MAX DRAWDOWN
       trades_df = trades_df.copy()
       trades_df['cumulative_pips'] = trades_df['pips'].cumsum()
       trades_df['running_max'] = trades_df['cumulative_pips'].expanding().max()
       trades_df['drawdown'] = trades_df['cumulative_pips'] - trades_df['running_max']
       max_drawdown = abs(trades_df['drawdown'].min())
       
       # 4. CONSECUTIVE LOSSES IN A ROW
       consecutive_losses = 0
       max_consecutive_losses = 0
       
       for _, trade in trades_df.iterrows():
           if trade['result'] == 'LOSS':
               consecutive_losses += 1
               max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
           else:
               consecutive_losses = 0
       
       return {
           'profit_factor': profit_factor,
           'avg_return_per_trade': avg_return_per_trade,
           'max_drawdown': max_drawdown,
           'max_consecutive_losses': max_consecutive_losses,
           'total_wins_pips': total_wins_pips,
           'total_losses_pips': total_losses_pips,
           'drawdown_series': trades_df['drawdown'].values,
           'cumulative_pips': trades_df['cumulative_pips'].values
       }
   
   def analyze_results(self):
       """Analyze backtest results with detailed breakdown including advanced KPIs"""
       if not self.trades:
           print("❌ No trades executed")
           return
           
       df_trades = pd.DataFrame(self.trades)
       
       total_trades = len(df_trades)
       winning_trades = len(df_trades[df_trades['result'] == 'WIN'])
       losing_trades = total_trades - winning_trades
       win_rate = (winning_trades / total_trades) * 100
       
       total_pips = df_trades['pips'].sum()
       avg_win = df_trades[df_trades['result'] == 'WIN']['pips'].mean() if winning_trades > 0 else 0
       avg_loss = df_trades[df_trades['result'] == 'LOSS']['pips'].mean() if losing_trades > 0 else 0
       
       # Calculate risk-reward ratio
       risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
       
       # Calculate advanced KPIs
       advanced_kpis = self.calculate_advanced_kpis(df_trades)
       
       self.results = {
           'total_trades': total_trades,
           'winning_trades': winning_trades,
           'losing_trades': losing_trades,
           'win_rate': win_rate,
           'total_pips': total_pips,
           'avg_win_pips': avg_win,
           'avg_loss_pips': avg_loss,
           'risk_reward_ratio': risk_reward,
           'trades_df': df_trades,
           **advanced_kpis  # Add advanced KPIs
       }
       
       # Print comprehensive results
       print("\n" + "="*70)
       print("📊 FRANKFURT OPENING RANGE STRATEGY RESULTS - CLOSE ABOVE VERSION")
       print("="*70)
       print(f"Strategy: Close above/below range + entry on next candle open")
       print(f"Data Period: {df_trades['date'].min()} to {df_trades['date'].max()}")
       print("-" * 70)
       print(f"Total Trades: {total_trades}")
       print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
       print(f"Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
       print(f"Total Pips: {total_pips:.1f}")
       print(f"Average Win: {avg_win:.1f} pips")
       print(f"Average Loss: {avg_loss:.1f} pips")
       print(f"Risk-Reward Ratio: {risk_reward:.2f}")
       print("-" * 70)
       print("🎯 ADVANCED KPIs:")
       print(f"Profit Factor: {advanced_kpis['profit_factor']:.2f}")
       print(f"Avg Return Per Trade: {advanced_kpis['avg_return_per_trade']:.1f} pips")
       print(f"Max Drawdown: {advanced_kpis['max_drawdown']:.1f} pips")
       print(f"Max Consecutive Losses: {advanced_kpis['max_consecutive_losses']}")
       print("="*70)


class MultiFrankfurtStrategy(FrankfurtOpeningRangeStrategy):
   """
   Multi-file backtesting with visual analysis and advanced KPIs
   Updated with new timeframes: H1, M5, M6, M10, M12, M15, M20, M30
   """
   
   def __init__(self):
       self.batch_results = {}
       # Set matplotlib style for better looking plots
       plt.style.use('default')
       sns.set_palette("husl")
   
   def batch_backtest(self, data_directory, file_pattern="*.csv", 
                     min_range_pips=5, max_range_pips=50):
       """Test multiple data files and generate visual analysis"""
       
       data_files = list(Path(data_directory).glob(file_pattern))
       
       if not data_files:
           print(f"❌ No files found matching pattern: {file_pattern}")
           return {}
       
       print(f"🔄 Testing {len(data_files)} files with pattern: {file_pattern}")
       print("-" * 80)
       
       for file_path in data_files:
           print(f"\n📊 Testing: {file_path.name}")
           
           try:
               self.df = self.load_data(str(file_path))
               self.trades = []
               self.results = {}
               
               results = self.backtest(min_range_pips, max_range_pips)
               
               self.batch_results[file_path.name] = {
                   'file_path': str(file_path),
                   'results': results,
                   'trades': self.trades.copy()
               }
               
           except Exception as e:
               print(f"❌ Error processing {file_path.name}: {str(e)}")
               self.batch_results[file_path.name] = {
                   'file_path': str(file_path),
                   'results': None,
                   'trades': [],
                   'error': str(e)
               }
               continue
       
       # Generate visual analysis
       self.generate_visual_analysis()
       return self.batch_results
   
   def generate_visual_analysis(self):
       """Generate comprehensive visual analysis with charts and summary"""
       
       if not self.batch_results:
           print("❌ No results to analyze")
           return
       
       # Prepare data for analysis
       analysis_data = []
       
       for filename, data in self.batch_results.items():
           if data['results'] and data['results'].get('total_trades', 0) > 0:
               results = data['results']
               
               # Extract timeframe and currency from filename
               timeframe = self.extract_timeframe(filename)
               currency = self.extract_currency(filename)
               
               analysis_data.append({
                   'File': filename[:30] + "..." if len(filename) > 30 else filename,
                   'Timeframe': timeframe,
                   'Currency': currency,
                   'Total_Trades': results['total_trades'],
                   'Win_Rate': results['win_rate'],
                   'Total_Pips': results['total_pips'],
                   'Profit_Factor': results['profit_factor'],
                   'Avg_Return_Per_Trade': results['avg_return_per_trade'],
                   'Max_Drawdown': results['max_drawdown'],
                   'Max_Consecutive_Losses': results['max_consecutive_losses'],
                   'Risk_Reward': results['risk_reward_ratio'],
                   'drawdown_series': results.get('drawdown_series', []),
                   'cumulative_pips': results.get('cumulative_pips', [])
               })
       
       if not analysis_data:
           print("❌ No successful trades to visualize")
           return
       
       df_analysis = pd.DataFrame(analysis_data)
       
       # Create comprehensive visual dashboard
       self.create_visual_dashboard(df_analysis)
       
       # Print summary table
       self.print_kpi_summary(df_analysis)
   
   def extract_timeframe(self, filename):
       """Extract timeframe from filename - Updated with new timeframes"""
       # Updated timeframe list to match your files
       timeframes = ['H1', 'M30', 'M20', 'M15', 'M12', 'M10', 'M6', 'M5', 'M1', 'H4', 'D1']
       for tf in timeframes:
           if tf in filename:
               return tf
       return 'Unknown'
   
   def extract_currency(self, filename):
       """Extract currency pair from filename"""
       currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY']
       for curr in currencies:
           if curr in filename:
               return curr
       return 'Unknown'
   
   def create_visual_dashboard(self, df_analysis):
       """Create comprehensive visual dashboard"""
       
       # Set up the figure with subplots
       fig = plt.figure(figsize=(20, 16))
       fig.suptitle('📊 FRANKFURT OPENING RANGE STRATEGY - CLOSE ABOVE VERSION - PERFORMANCE DASHBOARD', 
                    fontsize=18, fontweight='bold', y=0.98)
       
       # 1. Performance Overview Bar Chart
       ax1 = plt.subplot(3, 3, 1)
       top_performers = df_analysis.nlargest(10, 'Total_Pips')
       bars = ax1.bar(range(len(top_performers)), top_performers['Total_Pips'], 
                      color=['green' if x > 0 else 'red' for x in top_performers['Total_Pips']])
       ax1.set_title('🏆 Top 10 Performers (Total Pips)', fontweight='bold')
       ax1.set_xlabel('Dataset')
       ax1.set_ylabel('Total Pips')
       ax1.set_xticks(range(len(top_performers)))
       ax1.set_xticklabels([f"{row['Currency']}\n{row['Timeframe']}" for _, row in top_performers.iterrows()], 
                          rotation=45, ha='right')
       
       # Add value labels on bars
       for i, bar in enumerate(bars):
           height = bar.get_height()
           ax1.text(bar.get_x() + bar.get_width()/2., height + (height*0.01),
                   f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
       
       # 2. Win Rate Distribution
       ax2 = plt.subplot(3, 3, 2)
       ax2.hist(df_analysis['Win_Rate'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
       ax2.axvline(df_analysis['Win_Rate'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df_analysis["Win_Rate"].mean():.1f}%')
       ax2.set_title('📈 Win Rate Distribution', fontweight='bold')
       ax2.set_xlabel('Win Rate (%)')
       ax2.set_ylabel('Frequency')
       ax2.legend()
       
       # 3. Profit Factor Analysis
       ax3 = plt.subplot(3, 3, 3)
       profit_factors = df_analysis['Profit_Factor'].replace([np.inf, -np.inf], np.nan).dropna()
       if len(profit_factors) > 0:
           ax3.hist(profit_factors, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
           ax3.axvline(profit_factors.mean(), color='red', linestyle='--', 
                      label=f'Mean: {profit_factors.mean():.2f}')
           ax3.axvline(1.0, color='orange', linestyle='-', alpha=0.8, 
                      label='Break-even (1.0)')
       ax3.set_title('💰 Profit Factor Distribution', fontweight='bold')
       ax3.set_xlabel('Profit Factor')
       ax3.set_ylabel('Frequency')
       ax3.legend()
       
       # 4. Timeframe Performance Comparison
       ax4 = plt.subplot(3, 3, 4)
       if len(df_analysis['Timeframe'].unique()) > 1:
           timeframe_perf = df_analysis.groupby('Timeframe').agg({
               'Total_Pips': 'mean',
               'Win_Rate': 'mean',
               'Total_Trades': 'sum'
           }).round(1)
           
           # Sort timeframes logically
           timeframe_order = ['M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H4', 'D1']
           timeframe_perf = timeframe_perf.reindex([tf for tf in timeframe_order if tf in timeframe_perf.index])
           
           bars = ax4.bar(timeframe_perf.index, timeframe_perf['Total_Pips'], 
                         color=['green' if x > 0 else 'red' for x in timeframe_perf['Total_Pips']])
           ax4.set_title('⏰ Average Performance by Timeframe', fontweight='bold')
           ax4.set_xlabel('Timeframe')
           ax4.set_ylabel('Average Total Pips')
           
           # Add value labels
           for i, bar in enumerate(bars):
               height = bar.get_height()
               ax4.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.01),
                       f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
       else:
           ax4.text(0.5, 0.5, 'Only one timeframe\navailable', ha='center', va='center', 
                   transform=ax4.transAxes, fontsize=12)
           ax4.set_title('⏰ Timeframe Analysis', fontweight='bold')
       
       # 5. Max Drawdown vs Total Pips Scatter
       ax5 = plt.subplot(3, 3, 5)
       colors = ['green' if x > 0 else 'red' for x in df_analysis['Total_Pips']]
       scatter = ax5.scatter(df_analysis['Max_Drawdown'], df_analysis['Total_Pips'], 
                            c=colors, alpha=0.7, s=60)
       ax5.set_title('📉 Risk vs Reward Analysis', fontweight='bold')
       ax5.set_xlabel('Max Drawdown (Pips)')
       ax5.set_ylabel('Total Pips')
       ax5.grid(True, alpha=0.3)
       ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
       
       # 6. Consecutive Losses Analysis
       ax6 = plt.subplot(3, 3, 6)
       ax6.hist(df_analysis['Max_Consecutive_Losses'], bins=range(0, int(df_analysis['Max_Consecutive_Losses'].max())+2),
               alpha=0.7, color='salmon', edgecolor='black', align='left')
       ax6.set_title('🔴 Max Consecutive Losses Distribution', fontweight='bold')
       ax6.set_xlabel('Max Consecutive Losses')
       ax6.set_ylabel('Frequency')
       ax6.axvline(df_analysis['Max_Consecutive_Losses'].mean(), color='red', linestyle='--',
                  label=f'Mean: {df_analysis["Max_Consecutive_Losses"].mean():.1f}')
       ax6.legend()
       
       # 7. Currency Performance (if multiple currencies)
       ax7 = plt.subplot(3, 3, 7)
       if len(df_analysis['Currency'].unique()) > 1:
           currency_perf = df_analysis.groupby('Currency')['Total_Pips'].mean().sort_values(ascending=False)
           bars = ax7.bar(currency_perf.index, currency_perf.values,
                         color=['green' if x > 0 else 'red' for x in currency_perf.values])
           ax7.set_title('💱 Average Performance by Currency', fontweight='bold')
           ax7.set_xlabel('Currency Pair')
           ax7.set_ylabel('Average Total Pips')
           plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
           
           # Add value labels
           for i, bar in enumerate(bars):
               height = bar.get_height()
               ax7.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.01),
                       f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
       else:
           ax7.text(0.5, 0.5, 'Only one currency\navailable', ha='center', va='center',
                   transform=ax7.transAxes, fontsize=12)
           ax7.set_title('💱 Currency Analysis', fontweight='bold')
       
       # 8. Risk-Reward Ratio Distribution
       ax8 = plt.subplot(3, 3, 8)
       risk_rewards = df_analysis['Risk_Reward'].replace([np.inf, -np.inf], np.nan).dropna()
       if len(risk_rewards) > 0:
           ax8.hist(risk_rewards, bins=15, alpha=0.7, color='gold', edgecolor='black')
           ax8.axvline(risk_rewards.mean(), color='red', linestyle='--',
                      label=f'Mean: {risk_rewards.mean():.2f}')
           ax8.axvline(1.0, color='orange', linestyle='-', alpha=0.8,
                      label='Break-even (1.0)')
       ax8.set_title('⚖️ Risk-Reward Ratio Distribution', fontweight='bold')
       ax8.set_xlabel('Risk-Reward Ratio')
       ax8.set_ylabel('Frequency')
       ax8.legend()
       
       # 9. Average Return Per Trade
       ax9 = plt.subplot(3, 3, 9)
       ax9.hist(df_analysis['Avg_Return_Per_Trade'], bins=15, alpha=0.7, color='purple', edgecolor='black')
       ax9.axvline(df_analysis['Avg_Return_Per_Trade'].mean(), color='red', linestyle='--',
                  label=f'Mean: {df_analysis["Avg_Return_Per_Trade"].mean():.1f}')
       ax9.axvline(0, color='orange', linestyle='-', alpha=0.8, label='Break-even')
       ax9.set_title('📊 Avg Return Per Trade Distribution', fontweight='bold')
       ax9.set_xlabel('Average Return Per Trade (Pips)')
       ax9.set_ylabel('Frequency')
       ax9.legend()
       
       plt.tight_layout()
       plt.subplots_adjust(top=0.95)
       plt.show()
   
   def print_kpi_summary(self, df_analysis):
       """Print detailed KPI summary table"""
       
       print("\n" + "="*140)
       print("📊 COMPREHENSIVE KPI SUMMARY - FRANKFURT OPENING RANGE STRATEGY (CLOSE ABOVE VERSION)")
       print("="*140)
       
       # Sort by Total Pips (best performers first)
       df_summary = df_analysis.sort_values('Total_Pips', ascending=False)
       
       # Create summary table
       summary_cols = ['File', 'Timeframe', 'Currency', 'Total_Trades', 'Win_Rate', 
                      'Total_Pips', 'Profit_Factor', 'Avg_Return_Per_Trade', 
                      'Max_Drawdown', 'Max_Consecutive_Losses']
       
       # Format the data for better display
       display_df = df_summary[summary_cols].copy()
       display_df['Win_Rate'] = display_df['Win_Rate'].round(1).astype(str) + '%'
       display_df['Total_Pips'] = display_df['Total_Pips'].round(1)
       display_df['Profit_Factor'] = display_df['Profit_Factor'].replace([np.inf], 999.0).round(2)
       display_df['Avg_Return_Per_Trade'] = display_df['Avg_Return_Per_Trade'].round(1)
       display_df['Max_Drawdown'] = display_df['Max_Drawdown'].round(1)
       
       print(display_df.to_string(index=False))
       
       # Overall statistics
       print("\n" + "="*80)
       print("📈 OVERALL STATISTICS")
       print("="*80)
       successful_tests = len(df_analysis)
       profitable_tests = len(df_analysis[df_analysis['Total_Pips'] > 0])
       avg_total_pips = df_analysis['Total_Pips'].mean()
       avg_win_rate = df_analysis['Win_Rate'].mean()
       avg_profit_factor = df_analysis['Profit_Factor'].replace([np.inf, -np.inf], np.nan).mean()
       avg_max_drawdown = df_analysis['Max_Drawdown'].mean()
       avg_consecutive_losses = df_analysis['Max_Consecutive_Losses'].mean()
       
       print(f"Files Tested Successfully: {successful_tests}")
       print(f"Profitable Strategies: {profitable_tests} ({(profitable_tests/successful_tests)*100:.1f}%)")
       print(f"Average Win Rate: {avg_win_rate:.1f}%")
       print(f"Average Total Pips: {avg_total_pips:.1f}")
       print(f"Average Profit Factor: {avg_profit_factor:.2f}")
       print(f"Average Max Drawdown: {avg_max_drawdown:.1f} pips")
       print(f"Average Max Consecutive Losses: {avg_consecutive_losses:.1f}")
       
       # Best and worst performers
       best_performer = df_summary.iloc[0]
       worst_performer = df_summary.iloc[-1]
       
       print(f"\n🏆 BEST PERFORMER:")
       print(f"  File: {best_performer['File']}")
       print(f"  Currency: {best_performer['Currency']} | Timeframe: {best_performer['Timeframe']}")
       print(f"  Total Pips: {best_performer['Total_Pips']:.1f} | Win Rate: {best_performer['Win_Rate']:.1f}%")
       print(f"  Profit Factor: {best_performer['Profit_Factor']:.2f} | Max Drawdown: {best_performer['Max_Drawdown']:.1f}")
       
       print(f"\n📉 WORST PERFORMER:")
       print(f"  File: {worst_performer['File']}")
       print(f"  Currency: {worst_performer['Currency']} | Timeframe: {worst_performer['Timeframe']}")
       print(f"  Total Pips: {worst_performer['Total_Pips']:.1f} | Win Rate: {worst_performer['Win_Rate']:.1f}%")
       print(f"  Profit Factor: {worst_performer['Profit_Factor']:.2f} | Max Drawdown: {worst_performer['Max_Drawdown']:.1f}")
       
       # Timeframe analysis
       if len(df_analysis['Timeframe'].unique()) > 1:
           print(f"\n⏰ TIMEFRAME ANALYSIS:")
           timeframe_stats = df_analysis.groupby('Timeframe').agg({
               'Total_Pips': ['mean', 'count'],
               'Win_Rate': 'mean',
               'Profit_Factor': lambda x: x.replace([np.inf, -np.inf], np.nan).mean(),
               'Max_Drawdown': 'mean'
           }).round(2)
           
           # Sort timeframes logically
           timeframe_order = ['M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H4', 'D1']
           available_timeframes = [tf for tf in timeframe_order if tf in timeframe_stats.index]
           
           for timeframe in available_timeframes:
               count = int(timeframe_stats.loc[timeframe, ('Total_Pips', 'count')])
               avg_pips = timeframe_stats.loc[timeframe, ('Total_Pips', 'mean')]
               avg_win_rate = timeframe_stats.loc[timeframe, ('Win_Rate', 'mean')]
               avg_pf = timeframe_stats.loc[timeframe, ('Profit_Factor', '<lambda>')]
               avg_dd = timeframe_stats.loc[timeframe, ('Max_Drawdown', 'mean')]
               
               print(f"  {timeframe}: {count} files | Avg Pips: {avg_pips:.1f} | "
                     f"Avg Win Rate: {avg_win_rate:.1f}% | Avg PF: {avg_pf:.2f} | Avg DD: {avg_dd:.1f}")
       
       # Currency analysis
       if len(df_analysis['Currency'].unique()) > 1:
           print(f"\n💱 CURRENCY ANALYSIS:")
           currency_stats = df_analysis.groupby('Currency').agg({
               'Total_Pips': ['mean', 'count'],
               'Win_Rate': 'mean',
               'Profit_Factor': lambda x: x.replace([np.inf, -np.inf], np.nan).mean(),
               'Max_Drawdown': 'mean'
           }).round(2)
           
           for currency in currency_stats.index:
               count = int(currency_stats.loc[currency, ('Total_Pips', 'count')])
               avg_pips = currency_stats.loc[currency, ('Total_Pips', 'mean')]
               avg_win_rate = currency_stats.loc[currency, ('Win_Rate', 'mean')]
               avg_pf = currency_stats.loc[currency, ('Profit_Factor', '<lambda>')]
               avg_dd = currency_stats.loc[currency, ('Max_Drawdown', 'mean')]
               
               print(f"  {currency}: {count} files | Avg Pips: {avg_pips:.1f} | "
                     f"Avg Win Rate: {avg_win_rate:.1f}% | Avg PF: {avg_pf:.2f} | Avg DD: {avg_dd:.1f}")
       
       # Strategy recommendations
       print(f"\n💡 STRATEGY RECOMMENDATIONS:")
       print("-" * 50)
       
       # Find strategies with good risk-adjusted returns
       good_strategies = df_analysis[
           (df_analysis['Total_Pips'] > 0) & 
           (df_analysis['Win_Rate'] > 40) & 
           (df_analysis['Profit_Factor'] > 1.2) &
           (df_analysis['Max_Consecutive_Losses'] <= 5)
       ]
       
       if len(good_strategies) > 0:
           print(f"✅ {len(good_strategies)} strategies meet quality criteria:")
           print("   - Positive total pips")
           print("   - Win rate > 40%")
           print("   - Profit factor > 1.2")
           print("   - Max consecutive losses ≤ 5")
           print("\nTop recommendations:")
           for i, (_, strategy) in enumerate(good_strategies.head(3).iterrows()):
               print(f"  {i+1}. {strategy['Currency']} {strategy['Timeframe']} - "
                     f"{strategy['Total_Pips']:.1f} pips, {strategy['Win_Rate']:.1f}% win rate")
       else:
           print("❌ No strategies meet all quality criteria")
           print("Consider:")
           print("  - Adjusting range size filters")
           print("  - Testing different timeframes")
           print("  - Modifying target/stop parameters")
       
       # Close above vs breakout comparison note
       print(f"\n📝 STRATEGY NOTES:")
       print("-" * 50)
       print("🔄 CLOSE ABOVE VERSION CHANGES:")
       print("  - Signal: Close above/below range (instead of breakout)")
       print("  - Entry: Next candle open (instead of immediate)")
       print("  - Better entry timing but potentially fewer signals")
       print("  - May reduce false breakouts and whipsaws")
       
       print("="*140)


# Main execution
if __name__ == "__main__":
   
   # Initialize multi-file strategy
   multi_strategy = MultiFrankfurtStrategy()
   
   # =============================================================================
   # CONFIGURATION - UPDATE THIS PATH TO YOUR DATA FOLDER
   # =============================================================================
   
   data_folder = "C:/Users/sim/Desktop/Quant/European Opening Range/trading-strategy-automation/London Opening range/Data/"
   
   # Strategy parameters
   min_range_pips = 5      # Minimum range size to trade
   max_range_pips = 50     # Maximum range size to trade
   
   # =============================================================================
   # BATCH TESTING - TEST ALL CSV FILES IN FOLDER
   # =============================================================================
   
   print("🚀 FRANKFURT OPENING RANGE STRATEGY - CLOSE ABOVE VERSION - BATCH TESTING")
   print("=" * 90)
   print(f"📁 Data Directory: {data_folder}")
   print(f"📊 Range Filter: {min_range_pips}-{max_range_pips} pips")
   print(f"🎯 Strategy: Close above/below range + entry on next candle open")
   print(f"🕒 Available Timeframes: H1, M30, M20, M15, M12, M10, M6, M5")
   print("=" * 90)
   
   # Test all CSV files in the folder
   results = multi_strategy.batch_backtest(
       data_directory=data_folder,
       file_pattern="*.csv",
       min_range_pips=min_range_pips,
       max_range_pips=max_range_pips
   )
   
   print("\n✅ Batch testing complete!")
   print("📊 Visual dashboard and KPI summary generated above.")
  