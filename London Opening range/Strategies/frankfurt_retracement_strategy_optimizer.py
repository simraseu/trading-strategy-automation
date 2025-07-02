import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

class FrankfurtRetracementStrategy:
    """
    Frankfurt Opening Range Retracement Strategy - ENHANCED VERSION
    
    Phase 1: Range Formation (08:00-10:59 GMT+3)
    Phase 2: Breakout Detection (after 11:00)
    Phase 3: Retracement Entry (wait for pullback)
    Phase 4: Risk-Reward Based Exit
    
    Entry: Retracement to specified level after breakout
    Stop: Range opposite (support/resistance)
    Target: Risk-Reward ratio based
    """
    
    def __init__(self, data_file):
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
    
    def backtest(self, retracement_pct=50, risk_reward_ratio=1.0, max_wait_hours=6, 
                 min_range_pips=5, max_range_pips=50):
        """
        Run the enhanced retracement strategy backtest
        
        Parameters:
        - retracement_pct: Percentage retracement to wait for (30-70)
        - risk_reward_ratio: Target = risk * this ratio (0.5-3.0)
        - max_wait_hours: Maximum hours to wait for retracement (2-8)
        - min_range_pips: Minimum range size in pips (5-20)
        - max_range_pips: Maximum range size in pips (30-100)
        """
        print(f"🚀 Starting Frankfurt Retracement Strategy...")
        print(f"📊 Parameters: {retracement_pct}% retracement, {risk_reward_ratio}:1 R/R, {max_wait_hours}h max wait")
        
        for date in self.df['DATE'].unique():
            day_data = self.df[self.df['DATE'] == date].copy()
            session_hours = self.get_frankfurt_session_hours(date)
            
            # Calculate opening range
            opening_range = self.calculate_opening_range(day_data, session_hours)
            if not opening_range:
                continue
                
            # Filter by range size
            range_pips = opening_range['range_size'] * 10000
            if range_pips < min_range_pips or range_pips > max_range_pips:
                continue
            
            # Look for breakouts and retracement entries
            breakout_data = day_data[day_data['hour'] >= session_hours['trade_start']].copy()
            
            trade_result = self.check_for_retracement_setup(
                breakout_data, opening_range, date, range_pips,
                retracement_pct, risk_reward_ratio, max_wait_hours
            )
            
            if trade_result:
                self.trades.append(trade_result)
        
        self.analyze_results()
        return self.results
    
    def check_for_retracement_setup(self, breakout_data, opening_range, date, range_pips,
                                   retracement_pct, risk_reward_ratio, max_wait_hours):
        """Check for breakout and subsequent retracement entry"""
        
        high_break_price = opening_range['high'] + 0.0001  # 1 pip above range high
        low_break_price = opening_range['low'] - 0.0001    # 1 pip below range low
        
        breakout_detected = None
        breakout_extreme = None
        breakout_time = None
        
        # Phase 1: Detect initial breakout
        for _, candle in breakout_data.iterrows():
            # Long breakout detection
            if candle['HIGH'] >= high_break_price and not breakout_detected:
                breakout_detected = 'LONG'
                breakout_extreme = candle['HIGH']  # Track highest point after breakout
                breakout_time = candle['TIME']
                break
            
            # Short breakout detection
            elif candle['LOW'] <= low_break_price and not breakout_detected:
                breakout_detected = 'SHORT'
                breakout_extreme = candle['LOW']  # Track lowest point after breakout
                breakout_time = candle['TIME']
                break
        
        if not breakout_detected:
            return None
        
        # Phase 2: Wait for retracement entry
        return self.wait_for_retracement_entry(
            breakout_data, opening_range, breakout_detected, breakout_extreme,
            breakout_time, date, range_pips, retracement_pct, risk_reward_ratio, max_wait_hours
        )
    
    def wait_for_retracement_entry(self, breakout_data, opening_range, direction, 
                                  breakout_extreme, breakout_time, date, range_pips,
                                  retracement_pct, risk_reward_ratio, max_wait_hours):
        """Wait for retracement to specified level and enter trade"""
        
        retracement_decimal = retracement_pct / 100.0
        
        # Calculate retracement entry level
        if direction == 'LONG':
            # Retracement from range_low to breakout_extreme
            move_size = breakout_extreme - opening_range['low']
            retracement_level = breakout_extreme - (move_size * retracement_decimal)
            stop_loss = opening_range['low']  # Range opposite
            
        else:  # SHORT
            # Retracement from range_high to breakout_extreme  
            move_size = opening_range['high'] - breakout_extreme
            retracement_level = breakout_extreme + (move_size * retracement_decimal)
            stop_loss = opening_range['high']  # Range opposite
        
        # Filter data to only look after breakout time and within time limit
        breakout_time_dt = pd.to_datetime(f"{date} {breakout_time}")
        max_wait_time_dt = breakout_time_dt + pd.Timedelta(hours=max_wait_hours)
        
        remaining_data = breakout_data[
            (breakout_data['TIME'] > breakout_time) &
            (pd.to_datetime(breakout_data['DATE'] + ' ' + breakout_data['TIME']) <= max_wait_time_dt)
        ].copy()
        
        # Look for retracement entry
        for _, candle in remaining_data.iterrows():
            entry_triggered = False
            entry_price = None
            
            if direction == 'LONG' and candle['LOW'] <= retracement_level:
                entry_triggered = True
                entry_price = retracement_level
                
            elif direction == 'SHORT' and candle['HIGH'] >= retracement_level:
                entry_triggered = True
                entry_price = retracement_level
            
            if entry_triggered:
                # Calculate target based on risk-reward ratio
                risk_amount = abs(entry_price - stop_loss)
                
                if direction == 'LONG':
                    target = entry_price + (risk_amount * risk_reward_ratio)
                else:
                    target = entry_price - (risk_amount * risk_reward_ratio)
                
                # Execute the retracement trade
                return self.execute_retracement_trade(
                    direction, entry_price, target, stop_loss,
                    remaining_data, candle['TIME'], date, range_pips,
                    retracement_level, breakout_extreme, retracement_pct, risk_reward_ratio
                )
        
        # No retracement entry found within time limit
        return None
    
    def execute_retracement_trade(self, direction, entry_price, target, stop_loss,
                                 remaining_data, entry_time, date, range_pips,
                                 retracement_level, breakout_extreme, retracement_pct, risk_reward_ratio):
        """Execute retracement-based trade and track to completion"""
        
        # Track trade through remaining data
        future_data = remaining_data[remaining_data['TIME'] > entry_time]
        
        for _, candle in future_data.iterrows():
            # Check for target hit
            if direction == 'LONG' and candle['HIGH'] >= target:
                pips_profit = (target - entry_price) * 10000
                return {
                    'date': date,
                    'direction': direction,
                    'entry_type': 'RETRACEMENT',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': target,
                    'exit_reason': 'TARGET',
                    'result': 'WIN',
                    'pips': pips_profit,
                    'range_pips': range_pips,
                    'target_price': target,
                    'stop_loss_price': stop_loss,
                    'retracement_level': retracement_level,
                    'breakout_extreme': breakout_extreme,
                    'retracement_pct': retracement_pct,
                    'risk_reward_ratio': risk_reward_ratio
                }
            
            elif direction == 'SHORT' and candle['LOW'] <= target:
                pips_profit = (entry_price - target) * 10000
                return {
                    'date': date,
                    'direction': direction,
                    'entry_type': 'RETRACEMENT',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': target,
                    'exit_reason': 'TARGET',
                    'result': 'WIN',
                    'pips': pips_profit,
                    'range_pips': range_pips,
                    'target_price': target,
                    'stop_loss_price': stop_loss,
                    'retracement_level': retracement_level,
                    'breakout_extreme': breakout_extreme,
                    'retracement_pct': retracement_pct,
                    'risk_reward_ratio': risk_reward_ratio
                }
            
            # Check for stop loss hit
            if direction == 'LONG' and candle['LOW'] <= stop_loss:
                pips_loss = (stop_loss - entry_price) * 10000
                return {
                    'date': date,
                    'direction': direction,
                    'entry_type': 'RETRACEMENT',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': stop_loss,
                    'exit_reason': 'STOP_LOSS',
                    'result': 'LOSS',
                    'pips': pips_loss,
                    'range_pips': range_pips,
                    'target_price': target,
                    'stop_loss_price': stop_loss,
                    'retracement_level': retracement_level,
                    'breakout_extreme': breakout_extreme,
                    'retracement_pct': retracement_pct,
                    'risk_reward_ratio': risk_reward_ratio
                }
                
            elif direction == 'SHORT' and candle['HIGH'] >= stop_loss:
                pips_loss = (entry_price - stop_loss) * 10000
                return {
                    'date': date,
                    'direction': direction,
                    'entry_type': 'RETRACEMENT',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': stop_loss,
                    'exit_reason': 'STOP_LOSS',
                    'result': 'LOSS',
                    'pips': pips_loss,
                    'range_pips': range_pips,
                    'target_price': target,
                    'stop_loss_price': stop_loss,
                    'retracement_level': retracement_level,
                    'breakout_extreme': breakout_extreme,
                    'retracement_pct': retracement_pct,
                    'risk_reward_ratio': risk_reward_ratio
                }
        
        # No exit found - trade remains open (don't record)
        return None
    
    def optimize_parameters(self):
        """Test multiple parameter combinations to find optimal settings"""
        
        # Define parameter ranges to test
        retracement_percentages = [30, 40, 50, 60, 70]
        risk_reward_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        max_wait_hours_options = [2, 4, 6, 8]
        
        best_result = None
        best_params = None
        all_results = []
        
        print("🔍 OPTIMIZING RETRACEMENT STRATEGY PARAMETERS...")
        print("Testing combinations...")
        
        total_combinations = len(retracement_percentages) * len(risk_reward_ratios) * len(max_wait_hours_options)
        current_test = 0
        
        for retr_pct in retracement_percentages:
            for rr_ratio in risk_reward_ratios:
                for max_wait in max_wait_hours_options:
                    
                    current_test += 1
                    print(f"Testing {current_test}/{total_combinations}: {retr_pct}% retr, {rr_ratio}:1 R/R, {max_wait}h wait")
                    
                    # Run backtest with these parameters
                    result = self.backtest_with_params(retr_pct, rr_ratio, max_wait)
                    
                    if result and result['total_trades'] > 5:  # Only consider if enough trades
                        
                        # Score calculation (prioritize total pips and win rate)
                        score = (result['total_pips'] * result['win_rate'] / 100) + (result['risk_reward_ratio'] * 20)
                        
                        result_data = {
                            'retracement_pct': retr_pct,
                            'risk_reward_ratio': rr_ratio,
                            'max_wait_hours': max_wait,
                            'total_trades': result['total_trades'],
                            'win_rate': result['win_rate'],
                            'total_pips': result['total_pips'],
                            'avg_win': result['avg_win_pips'],
                            'avg_loss': result['avg_loss_pips'],
                            'calculated_rr': result['risk_reward_ratio'],
                            'score': score
                        }
                        
                        all_results.append(result_data)
                        
                        # Track best result
                        if best_result is None or score > best_result['score']:
                            best_result = result_data
                            best_params = (retr_pct, rr_ratio, max_wait)
        
        # Display results
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n" + "="*100)
        print("🏆 RETRACEMENT STRATEGY OPTIMIZATION RESULTS (TOP 10)")
        print("="*100)
        print(results_df.head(10).to_string(index=False))
        
        if best_params:
            print(f"\n🥇 BEST PARAMETERS:")
            print(f"Retracement Percentage: {best_params[0]}%")
            print(f"Risk-Reward Ratio: {best_params[1]}:1")
            print(f"Max Wait Hours: {best_params[2]}h")
            print(f"Score: {best_result['score']:.2f}")
        
        # Save results
        results_df.to_csv('retracement_strategy_optimization.csv', index=False)
        
        return best_params, results_df
    
    def backtest_with_params(self, retracement_pct, risk_reward_ratio, max_wait_hours):
        """Run backtest with specific parameters for optimization"""
        
        trades_temp = []
        
        for date in self.df['DATE'].unique():
            day_data = self.df[self.df['DATE'] == date].copy()
            session_hours = self.get_frankfurt_session_hours(date)
            
            opening_range = self.calculate_opening_range(day_data, session_hours)
            if not opening_range:
                continue
                
            range_pips = opening_range['range_size'] * 10000
            if range_pips < 5 or range_pips > 50:
                continue
            
            breakout_data = day_data[day_data['hour'] >= session_hours['trade_start']].copy()
            
            trade_result = self.check_for_retracement_setup(
                breakout_data, opening_range, date, range_pips,
                retracement_pct, risk_reward_ratio, max_wait_hours
            )
            
            if trade_result:
                trades_temp.append(trade_result)
        
        if not trades_temp:
            return None
            
        df_trades = pd.DataFrame(trades_temp)
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['result'] == 'WIN'])
        win_rate = (winning_trades / total_trades) * 100
        total_pips = df_trades['pips'].sum()
        avg_win = df_trades[df_trades['result'] == 'WIN']['pips'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['result'] == 'LOSS']['pips'].mean() if (total_trades - winning_trades) > 0 else 0
        calculated_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'risk_reward_ratio': calculated_rr
        }
    
    def analyze_results(self):
        """Analyze backtest results with enhanced metrics"""
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
        
        calculated_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Enhanced analysis
        exit_reasons = df_trades['exit_reason'].value_counts()
        long_trades = df_trades[df_trades['direction'] == 'LONG']
        short_trades = df_trades[df_trades['direction'] == 'SHORT']
        
        # Retracement analysis
        avg_retracement_pct = df_trades['retracement_pct'].mean() if 'retracement_pct' in df_trades.columns else 0
        avg_rr_ratio = df_trades['risk_reward_ratio'].mean() if 'risk_reward_ratio' in df_trades.columns else 0
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'risk_reward_ratio': calculated_rr,
            'trades_df': df_trades
        }
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("📊 FRANKFURT RETRACEMENT STRATEGY RESULTS")
        print("="*70)
        print(f"Strategy: Retracement entries with R/R based exits")
        print(f"Data Period: {df_trades['date'].min()} to {df_trades['date'].max()}")
        print("-" * 70)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"Total Pips: {total_pips:.1f}")
        print(f"Average Win: {avg_win:.1f} pips")
        print(f"Average Loss: {avg_loss:.1f} pips")
        print(f"Calculated R/R Ratio: {calculated_rr:.2f}")
        print(f"Average Retracement %: {avg_retracement_pct:.1f}%")
        print(f"Average Target R/R: {avg_rr_ratio:.1f}:1")
        print("-" * 70)
        
        # Exit reason breakdown
        print("EXIT REASONS:")
        for reason, count in exit_reasons.items():
            percentage = (count / total_trades) * 100
            print(f"  {reason}: {count} trades ({percentage:.1f}%)")
        
        # Direction breakdown
        print(f"\nDIRECTION BREAKDOWN:")
        if len(long_trades) > 0:
            long_win_rate = (len(long_trades[long_trades['result'] == 'WIN']) / len(long_trades)) * 100
            print(f"  LONG: {len(long_trades)} trades, {long_win_rate:.1f}% win rate")
        
        if len(short_trades) > 0:
            short_win_rate = (len(short_trades[short_trades['result'] == 'WIN']) / len(short_trades)) * 100
            print(f"  SHORT: {len(short_trades)} trades, {short_win_rate:.1f}% win rate")
        
        print("="*70)
        
        # Save detailed results
        df_trades.to_csv('frankfurt_retracement_trades.csv', index=False)
        print(f"📁 Detailed trades saved to: frankfurt_retracement_trades.csv")

# Run the strategy
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR FILE LOCATION
    strategy = FrankfurtRetracementStrategy('C:/Users/sim/Desktop/Quant/London Opening range/Strategies/EURUSD.raw_M15_202106170000_202507021515.csv')
    
    # Choose what to run:
    
    # Option 1: Single backtest with specific parameters
    # results = strategy.backtest(retracement_pct=50, risk_reward_ratio=1.0, max_wait_hours=6)
    
    # Option 2: Full parameter optimization (recommended)
    best_params, all_results = strategy.optimize_parameters()