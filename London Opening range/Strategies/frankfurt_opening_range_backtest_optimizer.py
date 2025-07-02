import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

class FrankfurtOpeningRangeStrategy:
    """
    Frankfurt Opening Range Breakout Strategy
    Range: 08:00-10:59 GMT+3 (adjusted for DST)
    Entry: Breakout confirmation after 11:00
    Target: Continuation in breakout direction
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
    
    def backtest(self, min_range_pips=5, max_range_pips=50):
        """
        Run the Frankfurt Opening Range strategy backtest
        """
        print("🚀 Starting Frankfurt Opening Range Backtest...")
        
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
            
            # Look for breakouts after range period
            breakout_data = day_data[day_data['hour'] >= session_hours['trade_start']]
            
            trade_result = self.check_for_breakout(
                breakout_data, opening_range, date, range_pips
            )
            
            if trade_result:
                self.trades.append(trade_result)
        
        self.analyze_results()
        return self.results
    
    def check_for_breakout(self, breakout_data, opening_range, date, range_pips):
        """Check for range breakout and simulate trade"""
        
        high_break_price = opening_range['high'] + 0.0001  # 1 pip above range high
        low_break_price = opening_range['low'] - 0.0001    # 1 pip below range low
        
        for _, candle in breakout_data.iterrows():
            # Long breakout (above range high)
            if candle['HIGH'] >= high_break_price:
                return self.execute_trade(
                    'LONG', high_break_price, opening_range['low'],  # Range low = support level
                    breakout_data, candle['TIME'], date, range_pips
                )
            
            # Short breakout (below range low)  
            elif candle['LOW'] <= low_break_price:
                return self.execute_trade(
                    'SHORT', low_break_price, opening_range['high'],  # Range high = resistance level
                    breakout_data, candle['TIME'], date, range_pips
                )
        
        return None
    
    def execute_trade(self, direction, entry_price, support_resistance_level, remaining_data, entry_time, date, range_pips):
        """Execute and track trade to end of day"""
        
        # CONFIGURABLE VARIABLES - EDIT THESE TO TEST DIFFERENT SETUPS
        range_size = range_pips / 10000  # Convert back to price
        target_multiplier = 0.5    # Target = 1x range size (EDIT THIS)
        stop_loss_level = 'range_opposite'  # Options: 'range_opposite', 'range_mid', 'fixed_pips' (EDIT THIS)
        fixed_stop_pips = 15       # If using 'fixed_pips' option (EDIT THIS)
        
        # Calculate target and stop loss based on breakout direction
        if direction == 'LONG':
            # LONG: Expect continuation higher
            target = entry_price + (target_multiplier * range_size)
            
            # Stop loss options
            if stop_loss_level == 'range_opposite':
                stop_loss = support_resistance_level  # Range low
            elif stop_loss_level == 'range_mid':
                stop_loss = support_resistance_level + (range_size / 2)  # Range middle
            elif stop_loss_level == 'fixed_pips':
                stop_loss = entry_price - (fixed_stop_pips / 10000)
                
        else:  # SHORT
            # SHORT: Expect continuation lower
            target = entry_price - (target_multiplier * range_size)
            
            # Stop loss options
            if stop_loss_level == 'range_opposite':
                stop_loss = support_resistance_level  # Range high
            elif stop_loss_level == 'range_mid':
                stop_loss = support_resistance_level - (range_size / 2)  # Range middle
            elif stop_loss_level == 'fixed_pips':
                stop_loss = entry_price + (fixed_stop_pips / 10000)
        
        # Track trade through rest of day
        future_data = remaining_data[remaining_data['TIME'] > entry_time]
        
        for _, candle in future_data.iterrows():
            # Check for target hit
            if direction == 'LONG' and candle['HIGH'] >= target:
                pips_profit = (target - entry_price) * 10000
                return {
                    'date': date,
                    'direction': direction,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': target,
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
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': target,
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
                    'entry_time': entry_time, 
                    'entry_price': entry_price,
                    'exit_time': candle['TIME'],
                    'exit_price': stop_loss,
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
                    'entry_time': entry_time,
                    'entry_price': entry_price, 
                    'exit_time': candle['TIME'],
                    'exit_price': stop_loss,
                    'result': 'LOSS',
                    'pips': pips_loss,
                    'range_pips': range_pips,
                    'target_price': target,
                    'stop_loss_price': stop_loss
                }
        
        # End of day - close at market
        last_candle = future_data.iloc[-1] if len(future_data) > 0 else remaining_data.iloc[-1]
        if direction == 'LONG':
            pips_result = (last_candle['CLOSE'] - entry_price) * 10000
        else:
            pips_result = (entry_price - last_candle['CLOSE']) * 10000
            
        return {
            'date': date,
            'direction': direction,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': last_candle['TIME'],
            'exit_price': last_candle['CLOSE'],
            'result': 'WIN' if pips_result > 0 else 'LOSS',
            'pips': pips_result,
            'range_pips': range_pips,
            'target_price': target,
            'stop_loss_price': stop_loss
        }

    def optimize_parameters(self):
        """Test multiple parameter combinations to find optimal settings"""
        
        # Define parameter ranges to test
        target_multipliers = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        stop_loss_levels = ['range_opposite', 'range_mid', 'fixed_pips']
        fixed_stop_pips_options = [10, 15, 20, 25, 30]
        
        best_result = None
        best_params = None
        all_results = []
        
        print("🔍 OPTIMIZING PARAMETERS...")
        print("Testing combinations...")
        
        total_combinations = len(target_multipliers) * len(stop_loss_levels) * len(fixed_stop_pips_options)
        current_test = 0
        
        for target_mult in target_multipliers:
            for stop_level in stop_loss_levels:
                for fixed_pips in fixed_stop_pips_options:
                    
                    current_test += 1
                    print(f"Testing {current_test}/{total_combinations}: Target={target_mult}, Stop={stop_level}, Pips={fixed_pips}")
                    
                    # Run backtest with these parameters
                    result = self.backtest_with_params(target_mult, stop_level, fixed_pips)
                    
                    if result and result['total_trades'] > 10:  # Only consider if enough trades
                        
                        # Score calculation (customize this formula)
                        score = (result['total_pips'] * result['win_rate'] / 100) + (result['risk_reward_ratio'] * 10)
                        
                        result_data = {
                            'target_multiplier': target_mult,
                            'stop_loss_level': stop_level,
                            'fixed_stop_pips': fixed_pips,
                            'total_trades': result['total_trades'],
                            'win_rate': result['win_rate'],
                            'total_pips': result['total_pips'],
                            'avg_win': result['avg_win_pips'],
                            'avg_loss': result['avg_loss_pips'],
                            'risk_reward': result['risk_reward_ratio'],
                            'score': score
                        }
                        
                        all_results.append(result_data)
                        
                        # Track best result
                        if best_result is None or score > best_result['score']:
                            best_result = result_data
                            best_params = (target_mult, stop_level, fixed_pips)
        
        # Display results
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n" + "="*80)
        print("🏆 OPTIMIZATION RESULTS (TOP 10)")
        print("="*80)
        print(results_df.head(10).to_string(index=False))
        
        print(f"\n🥇 BEST PARAMETERS:")
        print(f"Target Multiplier: {best_params[0]}")
        print(f"Stop Loss Level: {best_params[1]}")
        print(f"Fixed Stop Pips: {best_params[2]}")
        print(f"Score: {best_result['score']:.2f}")
        
        # Save results
        results_df.to_csv('parameter_optimization_results.csv', index=False)
        
        return best_params, results_df

    def backtest_with_params(self, target_multiplier, stop_loss_level, fixed_stop_pips, min_range_pips=5, max_range_pips=50):
        """Run backtest with specific parameters"""
        
        trades_temp = []  # Use temporary list
        
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
            
            # Look for breakouts
            breakout_data = day_data[day_data['hour'] >= session_hours['trade_start']]
            
            trade_result = self.check_for_breakout_optimized(
                breakout_data, opening_range, date, range_pips, 
                target_multiplier, stop_loss_level, fixed_stop_pips
            )
            
            if trade_result:
                trades_temp.append(trade_result)
        
        # Calculate results
        if not trades_temp:
            return None
            
        df_trades = pd.DataFrame(trades_temp)
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['result'] == 'WIN'])
        win_rate = (winning_trades / total_trades) * 100
        total_pips = df_trades['pips'].sum()
        avg_win = df_trades[df_trades['result'] == 'WIN']['pips'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['result'] == 'LOSS']['pips'].mean() if (total_trades - winning_trades) > 0 else 0
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'risk_reward_ratio': risk_reward
        }

    def check_for_breakout_optimized(self, breakout_data, opening_range, date, range_pips, 
                                   target_multiplier, stop_loss_level, fixed_stop_pips):
        """Optimized version that accepts parameters"""
        
        high_break_price = opening_range['high'] + 0.0001
        low_break_price = opening_range['low'] - 0.0001
        
        for _, candle in breakout_data.iterrows():
            if candle['HIGH'] >= high_break_price:
                return self.execute_trade_optimized(
                    'LONG', high_break_price, opening_range['low'],
                    breakout_data, candle['TIME'], date, range_pips,
                    target_multiplier, stop_loss_level, fixed_stop_pips
                )
            elif candle['LOW'] <= low_break_price:
                return self.execute_trade_optimized(
                    'SHORT', low_break_price, opening_range['high'],
                    breakout_data, candle['TIME'], date, range_pips,
                    target_multiplier, stop_loss_level, fixed_stop_pips
                )
        return None

    def execute_trade_optimized(self, direction, entry_price, support_resistance_level, 
                              remaining_data, entry_time, date, range_pips,
                              target_multiplier, stop_loss_level, fixed_stop_pips):
        """Execute trade with specified parameters"""
        
        range_size = range_pips / 10000
        
        # Calculate target and stop loss
        if direction == 'LONG':
            target = entry_price + (target_multiplier * range_size)
            if stop_loss_level == 'range_opposite':
                stop_loss = support_resistance_level
            elif stop_loss_level == 'range_mid':
                stop_loss = support_resistance_level + (range_size / 2)
            elif stop_loss_level == 'fixed_pips':
                stop_loss = entry_price - (fixed_stop_pips / 10000)
        else:
            target = entry_price - (target_multiplier * range_size)
            if stop_loss_level == 'range_opposite':
                stop_loss = support_resistance_level
            elif stop_loss_level == 'range_mid':
                stop_loss = support_resistance_level - (range_size / 2)
            elif stop_loss_level == 'fixed_pips':
                stop_loss = entry_price + (fixed_stop_pips / 10000)
        
        # Track trade execution
        future_data = remaining_data[remaining_data['TIME'] > entry_time]
        
        for _, candle in future_data.iterrows():
            # Check target/stop loss hits
            if direction == 'LONG':
                if candle['HIGH'] >= target:
                    return {'date': date, 'direction': direction, 'result': 'WIN', 
                           'pips': (target - entry_price) * 10000, 'range_pips': range_pips}
                elif candle['LOW'] <= stop_loss:
                    return {'date': date, 'direction': direction, 'result': 'LOSS',
                           'pips': (stop_loss - entry_price) * 10000, 'range_pips': range_pips}
            else:
                if candle['LOW'] <= target:
                    return {'date': date, 'direction': direction, 'result': 'WIN',
                           'pips': (entry_price - target) * 10000, 'range_pips': range_pips}
                elif candle['HIGH'] >= stop_loss:
                    return {'date': date, 'direction': direction, 'result': 'LOSS',
                           'pips': (entry_price - stop_loss) * 10000, 'range_pips': range_pips}
        
        # End of day close
        last_candle = future_data.iloc[-1] if len(future_data) > 0 else remaining_data.iloc[-1]
        pips_result = ((last_candle['CLOSE'] - entry_price) if direction == 'LONG' 
                       else (entry_price - last_candle['CLOSE'])) * 10000
        
        return {'date': date, 'direction': direction, 
               'result': 'WIN' if pips_result > 0 else 'LOSS',
               'pips': pips_result, 'range_pips': range_pips}
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.trades:
            print("❌ No trades executed")
            return
            
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['result'] == 'WIN'])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pips = df_trades['pips'].sum()
        avg_win = df_trades[df_trades['result'] == 'WIN']['pips'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['result'] == 'LOSS']['pips'].mean() if (total_trades - winning_trades) > 0 else 0
        
        # Calculate risk-reward ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'risk_reward_ratio': risk_reward,
            'trades_df': df_trades
        }
        
        # Print results
        print("\n" + "="*50)
        print("📊 FRANKFURT OPENING RANGE BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {total_trades - winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Pips: {total_pips:.1f}")
        print(f"Average Win: {avg_win:.1f} pips")
        print(f"Average Loss: {avg_loss:.1f} pips")
        print(f"Risk-Reward Ratio: {risk_reward:.2f}")
        print("="*50)
        
        # Show sample trades
        print("\nSAMPLE TRADES:")
        print(df_trades[['date', 'direction', 'entry_time', 'result', 'pips']].head(10))

# Run the backtest
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR FILE LOCATION
    strategy = FrankfurtOpeningRangeStrategy('C:/Users/sim/Desktop/Quant/London Opening range/Strategies/EURUSD.raw_M15_202106170000_202507021515.csv')
    
    # Choose what to run:
    
    # Option 1: Single backtest with current parameters
    # results = strategy.backtest()
    
    # Option 2: Full parameter optimization (recommended)
    best_params, all_results = strategy.optimize_parameters()