"""
Professional Backtesting Engine - Module 6 (FIXED VERSION)
Walk-forward historical validation with PROPER LIMIT ORDER LOGIC
Now waits for price to return to zones before executing trades
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os

# Import required modules
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator

class TradingBacktester:
   """
   Professional walk-forward backtesting engine with FIXED entry logic
   Now properly waits for price to return to zones before entering trades
   """
   
   def __init__(self, signal_generator, initial_balance: float = 10000, config: Dict = None):
       """
       Initialize backtesting engine with proper limit order tracking
       
       Args:
           signal_generator: SignalGenerator instance
           initial_balance: Starting account balance
           config: Backtesting configuration
       """
       self.signal_generator = signal_generator
       self.initial_balance = initial_balance
       self.current_balance = initial_balance
       self.config = config or self.default_config()
       
       # Trade tracking
       self.open_trades = []
       self.closed_trades = []
       self.pending_orders = []  # NEW: Track pending limit orders
       self.equity_curve = []
       self.daily_pnl = []
       
       # Performance metrics
       self.total_trades = 0
       self.total_pending_orders = 0
       self.winning_trades = 0
       self.losing_trades = 0
       self.max_drawdown = 0
       self.peak_balance = initial_balance
       self.current_drawdown = 0
       
       # Risk tracking
       self.max_concurrent_trades = 0
       self.current_exposure = 0
       
       self.logger = logging.getLogger(__name__)
       
       print(f"🛡️  Backtesting Engine Initialized (FIXED VERSION):")
       print(f"   Initial Balance: ${self.initial_balance:,.2f}")
       print(f"   Max Concurrent Trades: {self.config['max_concurrent_trades']}")
       print(f"   Slippage: {self.config['slippage_pips']} pips")
       print(f"   Entry Method: Limit orders (wait for price to return to zones)")
   
   def default_config(self) -> Dict:
       """Default backtesting configuration"""
       return {
           'max_concurrent_trades': 3,
           'max_pending_orders': 10,
           'slippage_pips': 2,
           'commission_per_lot': 7.0,
           'signal_generation_frequency': 'weekly',
           'order_expiry_days': 180,
           'break_even_trigger': 1.0,
           'partial_close_at_1r': False,
           'weekend_gap_protection': True,
           'max_spread_pips': 3
       }
   
   def run_walk_forward_backtest(self, data: pd.DataFrame, 
                                start_date: str, end_date: str,
                                lookback_days: int = 365,
                                pair: str = 'EURUSD') -> Dict:
       """
       FIXED: Execute walk-forward backtesting with proper limit order logic
       Now waits for price to return to zones before entering trades
       
       Args:
           data: Complete OHLC dataset
           start_date: Backtest start date
           end_date: Backtest end date  
           lookback_days: Days of history for signal generation
           pair: Currency pair being tested
           
       Returns:
           Complete backtest results
       """
       print(f"\n🔄 STARTING WALK-FORWARD BACKTEST (FIXED ENTRY LOGIC)")
       print(f"=" * 60)
       print(f"📊 Pair: {pair}")
       print(f"📅 Period: {start_date} to {end_date}")
       print(f"🔙 Lookback: {lookback_days} days")
       print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
       print(f"🎯 Entry Method: Wait for price to return to zones")
       
       # Convert dates and validate
       try:
           start_idx = data.index.get_loc(start_date)
           end_idx = data.index.get_loc(end_date)
       except KeyError as e:
           raise ValueError(f"Date not found in data: {e}")
       
       # Ensure sufficient lookback data
       if start_idx < lookback_days:
           raise ValueError(f"Insufficient lookback data. Need {lookback_days} days before {start_date}")
       
       total_days = end_idx - start_idx + 1
       print(f"📈 Trading Days: {total_days}")
       
       # Initialize tracking
       last_signal_date = None
       days_processed = 0
       
       # Walk forward through each trading day
       for current_idx in range(start_idx, end_idx + 1):
           current_date = data.index[current_idx]
           current_data = data.iloc[current_idx]
           
           # Progress tracking
           days_processed += 1
           if days_processed % 100 == 0:
               progress = (days_processed / total_days) * 100
               print(f"   Progress: {progress:.1f}% ({days_processed}/{total_days} days)")
           
           # STEP 1: CRITICAL - Check pending limit orders FIRST
           self.manage_pending_orders(current_data, current_date)
           
           # STEP 2: Generate new signals (weekly frequency to avoid over-trading)
           if self.should_generate_signals(current_date, last_signal_date):
               # Get historical window for signal generation (only past data)
               history_start = max(0, current_idx - lookback_days)
               historical_data = data.iloc[history_start:current_idx]  # Exclude current day
               
               new_signals = self.generate_historical_signals(
                   historical_data, current_date, pair
               )
               
               if new_signals:
                   print(f"📡 {current_date.strftime('%Y-%m-%d')}: {len(new_signals)} signals generated")
               
               # STEP 3: Create pending limit orders (NO IMMEDIATE EXECUTION)
               for signal in new_signals:
                   if self.can_create_pending_order(signal):
                       self.create_pending_order_from_signal(signal, current_data, current_date)
               
               last_signal_date = current_date
           
           # STEP 4: Manage existing open trades
           self.manage_open_trades(current_data, current_date)
           
           # STEP 5: Clean up expired pending orders
           self.cleanup_expired_orders(current_date)
           
           # STEP 6: Record daily performance
           self.record_daily_performance(current_date, current_data)
       
       # Close any remaining open trades at backtest end
       self.close_remaining_trades(data.iloc[end_idx], data.index[end_idx])
       
       # Calculate final metrics
       results = self.calculate_final_metrics(start_date, end_date, pair)
       
       print(f"\n✅ BACKTEST COMPLETE")
       print(f"   Total Trades: {results['total_trades']}")
       print(f"   Pending Orders Created: {self.total_pending_orders}")
       print(f"   Orders Executed: {results['total_trades']}")
       print(f"   Win Rate: {results['win_rate']}%")
       print(f"   Profit Factor: {results['profit_factor']}")
       print(f"   Final Balance: ${results['final_balance']:,.2f}")
       
       return results
   
   def manage_pending_orders(self, current_data: pd.Series, current_date: pd.Timestamp):
       """
       CRITICAL: Check if any pending limit orders can be executed
       This is where trades actually get opened when price returns to zones
       """
       orders_to_execute = []
       
       for order in self.pending_orders:
           if self.can_execute_limit_order(order, current_data):
               orders_to_execute.append(order)
       
       # Execute orders that can be filled
       for order in orders_to_execute:
           if self.execute_limit_order(order, current_data, current_date):
               self.pending_orders.remove(order)
               print(f"   ✅ Limit order executed: {order['direction']} at {order['limit_price']:.5f}")
               print(f"      Zone: {order['zone_low']:.5f} - {order['zone_high']:.5f}")
   
   def can_execute_limit_order(self, order: Dict, current_data: pd.Series) -> bool:
       """
       Check if current candle can execute the limit order
       BUY limits execute when price drops TO the limit price
       SELL limits execute when price rises TO the limit price
       """
       limit_price = order['limit_price']
       current_high = current_data['high']
       current_low = current_data['low']
       
       # Check if we already have max concurrent trades
       if len(self.open_trades) >= self.config['max_concurrent_trades']:
           return False
       
       if order['direction'] == 'BUY':
           # Buy limit: executes when price drops down to limit price
           return current_low <= limit_price <= current_high
       else:
           # Sell limit: executes when price rises up to limit price  
           return current_low <= limit_price <= current_high
   
   def create_pending_order_from_signal(self, signal: Dict, current_data: pd.Series, 
                                      current_date: pd.Timestamp):
       """
       Create pending limit order from signal (NO IMMEDIATE EXECUTION)
       """
       # Calculate limit order price (5% beyond zone boundary)
       limit_price = self.calculate_limit_order_price(signal)
       
       # Create pending order
       order = {
           'order_id': f"PO{len(self.pending_orders) + 1}",
           'signal_id': signal['signal_id'],
           'direction': signal['direction'],
           'limit_price': limit_price,
           'zone_high': signal['zone_high'],
           'zone_low': signal['zone_low'],
           'stop_loss': signal['stop_loss'],
           'take_profit_1': signal['take_profit_1'],
           'take_profit_2': signal['take_profit_2'],
           'position_size': signal['position_size'],
           'risk_amount': signal['risk_amount'],
           'order_date': current_date,
           'expires_after_days': self.config['order_expiry_days'],
           'status': 'pending'
       }
       
       self.pending_orders.append(order)
       self.total_pending_orders += 1
       
       print(f"   📋 Pending order: {signal['direction']} limit at {limit_price:.5f}")
       print(f"      Waiting for price to return to zone {signal['zone_low']:.5f}-{signal['zone_high']:.5f}")
   
   def calculate_limit_order_price(self, signal: Dict) -> float:
       """
       Calculate limit order price: 5% beyond zone boundary
       R-B-R (BUY): 5% above zone high
       D-B-D (SELL): 5% below zone low
       """
       zone_high = signal['zone_high']
       zone_low = signal['zone_low']
       zone_size = zone_high - zone_low
       buffer = zone_size * 0.05  # 5% of zone size
       
       if signal['direction'] == 'BUY':
           # R-B-R: Limit order 5% above zone high
           return zone_high + buffer
       else:
           # D-B-D: Limit order 5% below zone low  
           return zone_low - buffer
   
   def execute_limit_order(self, order: Dict, current_data: pd.Series, 
                          current_date: pd.Timestamp) -> bool:
       """
       Execute limit order when price reaches the limit price
       """
       try:
           # Apply slippage to limit price
           entry_price = self.apply_slippage(order['limit_price'], order['direction'])
           
           # Calculate commission
           commission = self.calculate_commission(order['position_size'])
           
           # Create trade record
           trade = {
               'trade_id': f"T{len(self.closed_trades) + len(self.open_trades) + 1}",
               'signal_id': order['signal_id'],
               'pair': 'EURUSD',
               'direction': order['direction'],
               'entry_date': current_date,
               'entry_price': entry_price,
               'position_size': order['position_size'],
               'stop_loss': order['stop_loss'],
               'take_profit_1': order['take_profit_1'],
               'take_profit_2': order['take_profit_2'],
               'initial_risk': order['risk_amount'],
               'commission': commission,
               'break_even_moved': False,
               'partial_closed': False,
               'status': 'open',
               'zone_high': order['zone_high'],
               'zone_low': order['zone_low'],
               'entry_method': 'limit_order'
           }
           
           # Update account balance for commission
           self.current_balance -= commission
           
           # Add to open trades
           self.open_trades.append(trade)
           self.total_trades += 1
           
           # Update tracking
           self.max_concurrent_trades = max(self.max_concurrent_trades, len(self.open_trades))
           
           self.logger.info(f"Limit order executed: {trade['trade_id']} - {trade['direction']} at {entry_price:.5f}")
           return True
           
       except Exception as e:
           self.logger.error(f"Limit order execution failed: {str(e)}")
           return False
   
   def can_create_pending_order(self, signal: Dict) -> bool:
       """
       Check if we can create a pending order (not immediate execution check)
       """
       # Check maximum pending orders
       if len(self.pending_orders) >= self.config['max_pending_orders']:
           return False
       
       # Check if we already have a pending order for this zone
       for existing_order in self.pending_orders:
           if (abs(existing_order['zone_high'] - signal['zone_high']) < 0.00001 and
               abs(existing_order['zone_low'] - signal['zone_low']) < 0.00001):
               return False  # Already have order for this zone
       
       return True
   
   def cleanup_expired_orders(self, current_date: pd.Timestamp):
       """
       Remove expired pending orders
       """
       orders_to_remove = []
       
       for order in self.pending_orders:
           days_pending = (current_date - order['order_date']).days
           if days_pending > order['expires_after_days']:
               orders_to_remove.append(order)
       
       for order in orders_to_remove:
           self.pending_orders.remove(order)
           print(f"   ❌ Expired order: {order['direction']} limit at {order['limit_price']:.5f}")
   
   def should_generate_signals(self, current_date: pd.Timestamp, 
                              last_signal_date: Optional[pd.Timestamp]) -> bool:
       """
       Determine if signals should be generated (weekly frequency)
       """
       if last_signal_date is None:
           return True  # First signal generation
       
       # Weekly signal generation (every 7 days)
       days_since_last = (current_date - last_signal_date).days
       return days_since_last >= 7
   
   def generate_historical_signals(self, historical_data: pd.DataFrame, 
                                  current_date: pd.Timestamp,
                                  pair: str) -> List[Dict]:
       """
       Generate signals using only historical data (no future bias)
       """
       try:
           # Ensure we have enough data for signal generation
           if len(historical_data) < 200:  # Need EMA200
               return []
           
           # Use the existing signal generator but with historical data window
           signals = self.signal_generator.generate_signals(
               historical_data, 'Daily', pair
           )
           
           # Add generation timestamp and filter by distance
           current_market_price = historical_data['close'].iloc[-1]
           valid_signals = []
           
           for signal in signals:
               signal['generated_date'] = current_date
               signal['data_end_date'] = historical_data.index[-1]
               signal['historical_price'] = current_market_price
               
               # Filter signals that are too far from current market price
               price_distance = abs(signal['entry_price'] - current_market_price)
               max_distance = 0.10  # 1000 pips maximum distance
               
               if price_distance <= max_distance:
                   valid_signals.append(signal)
               else:
                   print(f"   Signal filtered: Entry {signal['entry_price']:.5f} too far from market {current_market_price:.5f}")
           
           return valid_signals
           
       except Exception as e:
           self.logger.warning(f"Signal generation failed on {current_date}: {str(e)}")
           return []
   
   def manage_open_trades(self, current_data: pd.Series, current_date: pd.Timestamp):
       """
       Manage all open positions with proper exit logic
       """
       trades_to_close = []
       
       for trade in self.open_trades:
           # Debug trade management
           print(f"   Managing trade {trade['trade_id']}: {trade['direction']} at {trade['entry_price']:.5f}")
           print(f"      Current: H={current_data['high']:.5f} L={current_data['low']:.5f} C={current_data['close']:.5f}")
           print(f"      Stop: {trade['stop_loss']:.5f}, TP1: {trade['take_profit_1']:.5f}, TP2: {trade['take_profit_2']:.5f}")
           
           # Check stop loss hit
           if self.check_stop_loss_hit(trade, current_data):
               exit_price = self.get_stop_loss_exit_price(trade, current_data)
               trades_to_close.append((trade, 'stop_loss', exit_price))
               print(f"      ❌ STOP LOSS HIT at {exit_price:.5f}")
               continue
           
           # Check take profit hit
           tp_result = self.check_take_profit_hit(trade, current_data)
           if tp_result:
               exit_price = tp_result['exit_price']
               trades_to_close.append((trade, 'take_profit', exit_price))
               print(f"      ✅ TAKE PROFIT HIT at {exit_price:.5f} ({tp_result['level']})")
               continue
           
           # Move to break-even at 1:1 risk/reward
           if not trade['break_even_moved']:
               if self.should_move_to_breakeven(trade, current_data):
                   self.move_to_breakeven(trade)
                   print(f"      🔄 MOVED TO BREAK-EVEN")
       
       # Close trades that hit exit conditions
       for trade, exit_reason, exit_price in trades_to_close:
           self.close_trade(trade, exit_reason, exit_price, current_date)
   
   def check_stop_loss_hit(self, trade: Dict, current_data: pd.Series) -> bool:
       """Check if stop loss was hit"""
       direction = trade['direction']
       stop_loss = trade['stop_loss']
       
       if direction == 'BUY':
           # Long trade: stop hit if low goes below stop
           return current_data['low'] <= stop_loss
       else:
           # Short trade: stop hit if high goes above stop
           return current_data['high'] >= stop_loss
   
   def get_stop_loss_exit_price(self, trade: Dict, current_data: pd.Series) -> float:
       """Get realistic exit price for stop loss"""
       return trade['stop_loss']
   
   def check_take_profit_hit(self, trade: Dict, current_data: pd.Series) -> Optional[Dict]:
       """Check if take profit was hit"""
       direction = trade['direction']
       tp1 = trade['take_profit_1']
       tp2 = trade['take_profit_2']
       
       if direction == 'BUY':
           # Long trade: TP hit if high reaches target
           if current_data['high'] >= tp2:
               return {'exit_price': tp2, 'level': 'tp2'}
           elif current_data['high'] >= tp1:
               return {'exit_price': tp1, 'level': 'tp1'}
       else:
           # Short trade: TP hit if low reaches target
           if current_data['low'] <= tp2:
               return {'exit_price': tp2, 'level': 'tp2'}
           elif current_data['low'] <= tp1:
               return {'exit_price': tp1, 'level': 'tp1'}
       
       return None
   
   def should_move_to_breakeven(self, trade: Dict, current_data: pd.Series) -> bool:
       """
       Check if trade should move to break-even (1:1 risk/reward hit)
       """
       direction = trade['direction']
       entry_price = trade['entry_price']
       initial_stop = trade['stop_loss']
       
       # Calculate 1:1 risk/reward level
       risk_distance = abs(entry_price - initial_stop)
       
       if direction == 'BUY':
           breakeven_trigger = entry_price + risk_distance
           return current_data['high'] >= breakeven_trigger
       else:
           breakeven_trigger = entry_price - risk_distance
           return current_data['low'] <= breakeven_trigger
   
   def move_to_breakeven(self, trade: Dict):
       """
       Move stop loss to break-even (entry price)
       """
       trade['stop_loss'] = trade['entry_price']
       trade['break_even_moved'] = True
       
       self.logger.info(f"Break-even moved: {trade['trade_id']}")
   
   def close_trade(self, trade: Dict, exit_reason: str, 
                  exit_price: float, exit_date: pd.Timestamp):
       """
       Close trade and calculate P&L
       """
       # Apply slippage to exit
       if exit_reason == 'stop_loss':
           # Stop losses get worse slippage
           final_exit_price = self.apply_exit_slippage(exit_price, trade['direction'], True)
       else:
           # Take profits get normal slippage
           final_exit_price = self.apply_exit_slippage(exit_price, trade['direction'], False)
       
       # Calculate P&L
       pnl = self.calculate_trade_pnl(trade, final_exit_price)
       return_percent = (pnl / trade['initial_risk']) * 100
       
       # Update trade record
       trade.update({
           'exit_date': exit_date,
           'exit_price': final_exit_price,
           'exit_reason': exit_reason,
           'pnl': pnl,
           'return_percent': return_percent,
           'days_held': (exit_date - trade['entry_date']).days,
           'status': 'closed'
       })
       
       # Update account balance
       self.current_balance += pnl
       
       # Track performance
       if pnl > 0:
           self.winning_trades += 1
       else:
           self.losing_trades += 1
       
       # Update drawdown tracking
       if self.current_balance > self.peak_balance:
           self.peak_balance = self.current_balance
           self.current_drawdown = 0
       else:
           self.current_drawdown = self.peak_balance - self.current_balance
           self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
       
       # Move to closed trades
       self.closed_trades.append(trade)
       self.open_trades.remove(trade)
       
       # Log trade closure
       self.logger.info(f"Trade closed: {trade['trade_id']} - {exit_reason} - P&L: ${pnl:.2f}")
   
   def apply_slippage(self, entry_price: float, direction: str) -> float:
       """Apply realistic slippage to entry price"""
       pip_value = 0.0001  # EURUSD
       slippage_pips = self.config['slippage_pips']
       slippage_amount = slippage_pips * pip_value
       
       if direction == 'BUY':
           # Buy orders get filled at higher price (worse for trader)
           return entry_price + slippage_amount
       else:
           # Sell orders get filled at lower price (worse for trader)
           return entry_price - slippage_amount
   
   def apply_exit_slippage(self, exit_price: float, direction: str, is_stop_loss: bool) -> float:
       """Apply slippage to exit price"""
       pip_value = 0.0001
       
       # Stop losses get worse slippage
       slippage_pips = self.config['slippage_pips'] * (2 if is_stop_loss else 1)
       slippage_amount = slippage_pips * pip_value
       
       if direction == 'BUY':
           # Selling: get worse price (lower)
           return exit_price - slippage_amount
       else:
           # Covering short: get worse price (higher)
           return exit_price + slippage_amount
   
   def calculate_commission(self, position_size: float) -> float:
       """Calculate commission for the trade"""
       return position_size * self.config['commission_per_lot']
   
   def calculate_trade_pnl(self, trade: Dict, exit_price: float) -> float:
       """
       Calculate trade P&L in account currency
       """
       entry_price = trade['entry_price']
       position_size = trade['position_size']
       direction = trade['direction']
       
       # Calculate price difference
       if direction == 'BUY':
           price_diff = exit_price - entry_price
       else:
           price_diff = entry_price - exit_price
       
       # Convert to account currency (USD)
       # For EURUSD: 1 pip = $10 per standard lot
       pip_value_usd = 10.0
       pip_difference = price_diff / 0.0001  # Convert to pips
       
       gross_pnl = pip_difference * pip_value_usd * position_size
       
       # Subtract exit commission
       exit_commission = self.calculate_commission(position_size)
       net_pnl = gross_pnl - exit_commission
       
       return net_pnl
   
   def record_daily_performance(self, current_date: pd.Timestamp, current_data: pd.Series):
       """
       Record daily account performance for equity curve
       """
       # Calculate unrealized P&L for open trades
       unrealized_pnl = 0
       for trade in self.open_trades:
           current_price = current_data['close']
           unrealized_pnl += self.calculate_unrealized_pnl(trade, current_price)
       
       # Total equity (realized + unrealized)
       total_equity = self.current_balance + unrealized_pnl
       
       # Record equity point
       equity_point = {
           'date': current_date,
           'balance': self.current_balance,
           'unrealized_pnl': unrealized_pnl,
           'total_equity': total_equity,
           'open_trades': len(self.open_trades),
           'pending_orders': len(self.pending_orders),
           'drawdown': self.current_drawdown
       }
       
       self.equity_curve.append(equity_point)
   
   def calculate_unrealized_pnl(self, trade: Dict, current_price: float) -> float:
       """Calculate unrealized P&L for open trade"""
       if trade['direction'] == 'BUY':
           price_diff = current_price - trade['entry_price']
       else:
           price_diff = trade['entry_price'] - current_price
       
       pip_difference = price_diff / 0.0001
       return pip_difference * 10.0 * trade['position_size']
   
   def close_remaining_trades(self, final_data: pd.Series, final_date: pd.Timestamp):
       """Close any remaining open trades at backtest end"""
       final_price = final_data['close']
       
       for trade in self.open_trades.copy():
           self.close_trade(trade, 'backtest_end', final_price, final_date)
   
   def calculate_final_metrics(self, start_date: str, end_date: str, pair: str) -> Dict:
       """
       Calculate comprehensive performance metrics
       """
       if not self.closed_trades:
           return self.empty_results()
       
       # Basic metrics
       total_trades = len(self.closed_trades)
       winning_trades = self.winning_trades
       losing_trades = self.losing_trades
       
       win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
       
       # P&L calculations
       all_pnls = [trade['pnl'] for trade in self.closed_trades]
       winning_pnls = [pnl for pnl in all_pnls if pnl > 0]
       losing_pnls = [pnl for pnl in all_pnls if pnl < 0]
       
       gross_profit = sum(winning_pnls) if winning_pnls else 0
       gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
       net_profit = gross_profit - gross_loss
       
       profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
       
       # Return metrics
       total_return_pct = ((self.current_balance / self.initial_balance) - 1) * 100
       avg_return_per_trade = net_profit / total_trades if total_trades > 0 else 0
       
       # Risk metrics
       max_drawdown_pct = (self.max_drawdown / self.peak_balance) * 100 if self.peak_balance > 0 else 0
       
       # Trade duration analysis
       trade_durations = [trade['days_held'] for trade in self.closed_trades if 'days_held' in trade]
       avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
       
       # Manual strategy comparison
       manual_pf = 5.3
       manual_wr = 53.0
       
       pf_accuracy = abs(profit_factor - manual_pf) / manual_pf * 100 if profit_factor != float('inf') else 100
       wr_accuracy = abs(win_rate - manual_wr) / manual_wr * 100
       
       # Expectancy calculation
       avg_win = np.mean(winning_pnls) if winning_pnls else 0
       avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0
       expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
       expectancy_pct = (expectancy / self.initial_balance) * 100
       
       return {
           # Test Parameters
           'pair': pair,
           'start_date': start_date,
           'end_date': end_date,
           'initial_balance': self.initial_balance,
           'final_balance': round(self.current_balance, 2),
           
           # Core Performance
           'total_trades': total_trades,
           'winning_trades': winning_trades,
           'losing_trades': losing_trades,
           'win_rate': round(win_rate, 1),
           
           # Order Statistics
           'total_pending_orders': self.total_pending_orders,
           'execution_rate': round((total_trades / self.total_pending_orders) * 100, 1) if self.total_pending_orders > 0 else 0,
           
           # Profitability
           'net_profit': round(net_profit, 2),
           'gross_profit': round(gross_profit, 2),
           'gross_loss': round(gross_loss, 2),
           'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
           'total_return_pct': round(total_return_pct, 1),
           'avg_return_per_trade': round(avg_return_per_trade, 2),
           'expectancy': round(expectancy, 2),
           'expectancy_pct': round(expectancy_pct, 3),
           
           # Risk Metrics
           'max_drawdown': round(self.max_drawdown, 2),
           'max_drawdown_pct': round(max_drawdown_pct, 1),
           'max_concurrent_trades': self.max_concurrent_trades,
           
           # Trade Analysis
           'avg_trade_duration_days': round(avg_trade_duration, 1),
           'avg_winning_trade': round(avg_win, 2) if avg_win > 0 else 0,
           'avg_losing_trade': round(avg_loss, 2) if avg_loss > 0 else 0,
           
           # Manual Strategy Comparison
           'manual_pf_baseline': manual_pf,
           'manual_wr_baseline': manual_wr,
           'pf_vs_manual': f"{profit_factor:.2f} vs {manual_pf} ({pf_accuracy:.1f}% diff)",
           'wr_vs_manual': f"{win_rate:.1f}% vs {manual_wr}% ({wr_accuracy:.1f}% diff)",
           'within_15pct_tolerance': pf_accuracy <= 15 and wr_accuracy <= 15,
           
           # Raw Data
           'equity_curve': self.equity_curve,
           'closed_trades': self.closed_trades,
           'trade_pnls': all_pnls
       }
   
   def empty_results(self) -> Dict:
       """Return empty results structure when no trades executed"""
       return {
           'total_trades': 0,
           'total_pending_orders': self.total_pending_orders,
           'execution_rate': 0,
           'winning_trades': 0,
           'losing_trades': 0,
           'win_rate': 0,
           'net_profit': 0,
           'profit_factor': 0,
           'final_balance': self.initial_balance,
           'within_15pct_tolerance': False,
           'equity_curve': [],
           'closed_trades': []
       }