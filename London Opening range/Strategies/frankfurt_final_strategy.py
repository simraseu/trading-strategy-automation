import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

class FrankfurtOpeningRangeStrategy:
    """
    Frankfurt Opening Range Breakout Strategy - FINAL VERSION
    Range: 08:00-10:59 GMT+3 (adjusted for DST)
    Entry: Breakout confirmation after 11:00
    Target: 0.5x range size continuation
    Stop: Range opposite (support/resistance)
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
        Fixed Parameters: Target=0.5x range, Stop=range_opposite
        """
        print("🚀 Starting Frankfurt Opening Range Strategy...")
        print("📊 Strategy: Target=0.5x range, Stop=range opposite")
        
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
                    'LONG', high_break_price, opening_range['low'],
                    breakout_data, candle['TIME'], date, range_pips
                )
            
            # Short breakout (below range low)  
            elif candle['LOW'] <= low_break_price:
                return self.execute_trade(
                    'SHORT', low_break_price, opening_range['high'],
                    breakout_data, candle['TIME'], date, range_pips
                )
        
        return None
    
    def execute_trade(self, direction, entry_price, support_resistance_level, remaining_data, entry_time, date, range_pips):
        """Execute trade with fixed strategy parameters"""
        
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
        
        # Track trade through rest of day
        future_data = remaining_data[remaining_data['TIME'] > entry_time]
        
        for _, candle in future_data.iterrows():
            # Check for target hit first
            if direction == 'LONG' and candle['HIGH'] >= target:
                pips_profit = (target - entry_price) * 10000
                return {
                    'date': date,
                    'direction': direction,
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
        
        
    
    def analyze_results(self):
        """Analyze backtest results with detailed breakdown"""
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
        
        # Exit reason breakdown
        exit_reasons = df_trades['exit_reason'].value_counts()
        
        # Direction breakdown
        long_trades = df_trades[df_trades['direction'] == 'LONG']
        short_trades = df_trades[df_trades['direction'] == 'SHORT']
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'risk_reward_ratio': risk_reward,
            'trades_df': df_trades
        }
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("📊 FRANKFURT OPENING RANGE STRATEGY RESULTS")
        print("="*60)
        print(f"Strategy: Target=0.5x range, Stop=range opposite")
        print(f"Data Period: {df_trades['date'].min()} to {df_trades['date'].max()}")
        print("-" * 60)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"Total Pips: {total_pips:.1f}")
        print(f"Average Win: {avg_win:.1f} pips")
        print(f"Average Loss: {avg_loss:.1f} pips")
        print(f"Risk-Reward Ratio: {risk_reward:.2f}")
        print("-" * 60)
        
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
        
        print("="*60)
        
        # Save detailed results
        df_trades.to_csv('frankfurt_strategy_trades.csv', index=False)
        print(f"📁 Detailed trades saved to: frankfurt_strategy_trades.csv")
        
        # Show sample trades
        print("\nSAMPLE TRADES:")
        print(df_trades[['date', 'direction', 'entry_time', 'exit_reason', 'result', 'pips']].head(10))

# Run the strategy
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR FILE LOCATION
    strategy = FrankfurtOpeningRangeStrategy('C:/Users/sim/Desktop/Quant/European Opening Range/trading-strategy-automation/London Opening range/Data/EURUSD.raw_M5_201701100005_202507021745.csv')
    results = strategy.backtest()

    