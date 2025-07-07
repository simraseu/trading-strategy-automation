"""
Professional Backtesting Engine - Module 6 (100% UPDATED VERSION)
Fully synchronized with all modules using exact same logic
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
   100% SYNCHRONIZED backtesting engine using exact same logic as all modules
   """
   
   def __init__(self, signal_generator, initial_balance: float = 10000, config: Dict = None):
       """
       Initialize backtesting engine with complete module synchronization
       
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
       self.pending_orders = []
       self.equity_curve = []
       
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
       
       # Trend tracking
       self.last_trend = None
       
       self.logger = logging.getLogger(__name__)
       
       print(f"🛡️  Backtesting Engine Initialized (100% SYNCHRONIZED):")
       print(f"   Initial Balance: ${self.initial_balance:,.2f}")
       print(f"   Max Concurrent Trades: {self.config['max_concurrent_trades']}")
       print(f"   Slippage: {self.config['slippage_pips']} pips")
       print(f"   Integration: All modules synchronized")
   
   def default_config(self) -> Dict:
       """Default backtesting configuration"""
       return {
           'max_concurrent_trades': 3,
           'max_pending_orders': 10,
           'slippage_pips': 2,
           'commission_per_lot': 7.0,
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
       100% SYNCHRONIZED walk-forward backtest using exact module logic
       
       Args:
           data: Complete OHLC dataset
           start_date: Backtest start date
           end_date: Backtest end date  
           lookback_days: Days of history for signal generation
           pair: Currency pair being tested
           
       Returns:
           Complete backtest results
       """
       print(f"\n🔄 STARTING 100% SYNCHRONIZED BACKTEST")
       print(f"=" * 60)
       print(f"📊 Pair: {pair}")
       print(f"📅 Period: {start_date} to {end_date}")
       print(f"🔙 Lookback: {lookback_days} days")
       print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
       print(f"🎯 Strategy: Exact module logic replication")
       
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
       days_processed = 0
       
       # Walk forward through EVERY data point
       for current_idx in range(start_idx, end_idx + 1):
           current_date = data.index[current_idx]
           current_data = data.iloc[current_idx]
           
           # Progress tracking
           days_processed += 1
           if days_processed % 100 == 0:
               progress = (days_processed / total_days) * 100
               print(f"   Progress: {progress:.1f}% ({days_processed}/{total_days} days)")
           
           # STEP 1: Manage pending orders (check for fills)
           self.manage_pending_orders(current_data, current_date)
           
           # STEP 2: Generate new signals using EXACT module logic
           history_start = max(0, current_idx - lookback_days)
           historical_data = data.iloc[history_start:current_idx]
           
           new_signals = self.generate_signals_exact_logic(
               historical_data, current_data, current_date, pair
           )
           
           # STEP 3: Create pending orders from new signals
           for signal in new_signals:
               if self.can_create_pending_order(signal):
                   self.create_pending_order_exact_logic(signal, current_data, current_date)
           
           # STEP 4: Manage open trades
           self.manage_open_trades(current_data, current_date)
           
           # STEP 5: Clean up expired orders
           self.cleanup_expired_orders(current_date)
           
           # STEP 6: Record daily performance
           self.record_daily_performance(current_date, current_data)
       
       # Close remaining trades
       self.close_remaining_trades(data.iloc[end_idx], data.index[end_idx])
       
       # Calculate final metrics
       results = self.calculate_final_metrics(start_date, end_date, pair)
       
       print(f"\n✅ 100% SYNCHRONIZED BACKTEST COMPLETE")
       print(f"   Total Trades: {results['total_trades']}")
       print(f"   Pending Orders Created: {self.total_pending_orders}")
       print(f"   Orders Executed: {results['total_trades']}")
       print(f"   Win Rate: {results['win_rate']}%")
       print(f"   Profit Factor: {results['profit_factor']}")
       print(f"   Final Balance: ${results['final_balance']:,.2f}")
       
       return results
   
   def generate_signals_exact_logic(self, historical_data: pd.DataFrame, 
                                  current_data: pd.Series, current_date: pd.Timestamp,
                                  pair: str) -> List[Dict]:
       """
       Generate signals using EXACT same logic as modules
       """
       try:
           if len(historical_data) < 200:  # Need EMA200
               return []
           
           # 1. Initialize components with EXACT same logic
           candle_classifier = CandleClassifier(historical_data)
           classified_data = candle_classifier.classify_all_candles()
           
           zone_detector = ZoneDetector(candle_classifier)
           trend_classifier = TrendClassifier(historical_data)
           risk_manager = RiskManager(account_balance=self.current_balance)
           
           # 2. Get current trend using EXACT trend logic
           trend_data = trend_classifier.classify_trend_with_filter()
           current_trend = trend_data['trend_filtered'].iloc[-1]
           current_price = historical_data['close'].iloc[-1]
           
           # 3. Skip ranging markets (exact logic)
           if current_trend == 'ranging':
               return []
           
           # 4. Track trend changes (exact logic)
           if self.last_trend is not None and self.last_trend != current_trend:
               print(f"   📊 Trend changed from {self.last_trend} to {current_trend} - no new signals")
               self.last_trend = current_trend
               return []
           
           self.last_trend = current_trend
           
           # 5. Detect zones using EXACT zone logic
           patterns = zone_detector.detect_all_patterns(classified_data)
           
           # 6. Filter by trend alignment (exact logic)
           valid_zones = []
           bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
           
           if current_trend in bullish_trends:
               valid_zones = patterns['rbr_patterns']
           else:
               valid_zones = patterns['dbd_patterns']
           
           # 7. Find closest valid zone (exact logic)
           if not valid_zones:
               return []
           
           closest_zone = self.find_closest_valid_zone_exact_logic(
               valid_zones, current_price, historical_data
           )
           
           if closest_zone:
               # 8. Validate using EXACT risk logic
               risk_validation = risk_manager.validate_zone_for_trading(
                   closest_zone, current_price, pair, historical_data
               )
               
               if risk_validation['is_tradeable']:
                   signal = self.create_signal_exact_logic(
                       closest_zone, risk_validation, current_trend, pair, current_date
                   )
                   return [signal]
           
           return []
           
       except Exception as e:
           self.logger.warning(f"Signal generation failed on {current_date}: {str(e)}")
           return []
   
   def find_closest_valid_zone_exact_logic(self, zones: List[Dict], current_price: float, 
                                         data: pd.DataFrame) -> Optional[Dict]:
       """
       Find closest valid zone using EXACT risk manager logic
       """
       valid_zones = []
       
       for zone in zones:
           # Use EXACT same zone testing logic as risk_manager
           risk_manager = RiskManager()
           is_untested, _ = risk_manager.check_zone_testing(zone, data)
           
           if is_untested:
               zone_center = (zone['zone_high'] + zone['zone_low']) / 2
               distance = abs(zone_center - current_price)
               
               # Use EXACT same distance tolerance as risk_manager
               if distance <= 0.10:  # Same as risk_manager max_distance
                   valid_zones.append((zone, distance))
       
       if not valid_zones:
           return None
       
       # Return closest zone
       valid_zones.sort(key=lambda x: x[1])
       return valid_zones[0][0]
   
   def create_signal_exact_logic(self, zone: Dict, risk_validation: Dict, 
                               trend: str, pair: str, current_date: pd.Timestamp) -> Dict:
       """
       Create signal using EXACT same logic as signal_generator
       """
       signal_id = f"{pair}_{current_date.strftime('%Y%m%d_%H%M%S')}"
       
       return {
           'signal_id': signal_id,
           'pair': pair,
           'direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL',
           'entry_price': risk_validation['entry_price'],  # From risk_manager
           'stop_loss': risk_validation['stop_loss_price'],  # From risk_manager
           'take_profit_1': risk_validation['take_profit_1'],  # From risk_manager
           'take_profit_2': risk_validation['take_profit_2'],  # From risk_manager
           'position_size': risk_validation['position_size'],  # From risk_manager
           'risk_amount': risk_validation['risk_amount'],  # From risk_manager
           'zone_high': zone['zone_high'],  # From zone_detector
           'zone_low': zone['zone_low'],  # From zone_detector
           'generated_date': current_date,
           'trend': trend,
           'zone_score': zone.get('strength', 0.5) * 100
       }
   
   def create_pending_order_exact_logic(self, signal: Dict, current_data: pd.Series, 
                                      current_date: pd.Timestamp):
       """
       Create pending order using EXACT same entry logic as risk_manager
       """
       # Use EXACT same entry price as calculated by risk_manager
       limit_price = signal['entry_price']  # Already calculated by risk_manager
       
       # Create pending order with EXACT parameters
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
       print(f"      Zone: {signal['zone_low']:.5f}-{signal['zone_high']:.5f}")
   
   def manage_pending_orders(self, current_data: pd.Series, current_date: pd.Timestamp):
       """
       Check for order fills using realistic execution logic
       """
       orders_to_execute = []
       
       for order in self.pending_orders:
           if self.can_execute_limit_order(order, current_data):
               orders_to_execute.append(order)
       
       # Execute orders that can be filled
       for order in orders_to_execute:
           if self.execute_limit_order(order, current_data, current_date):
               self.pending_orders.remove(order)
               print(f"   ✅ Order executed: {order['direction']} at {order['limit_price']:.5f}")
   
   def can_execute_limit_order(self, order: Dict, current_data: pd.Series) -> bool:
       """
       Check if limit order can be executed (realistic fill logic)
       """
       limit_price = order['limit_price']
       current_high = current_data['high']
       current_low = current_data['low']
       
       # Check concurrent trade limit
       if len(self.open_trades) >= self.config['max_concurrent_trades']:
           return False
       
       # Check if price touched limit level
       if order['direction'] == 'BUY':
           # Buy limit executes when price drops TO limit price
           return current_low <= limit_price <= current_high
       else:
           # Sell limit executes when price rises TO limit price  
           return current_low <= limit_price <= current_high
   
   def execute_limit_order(self, order: Dict, current_data: pd.Series, 
                          current_date: pd.Timestamp) -> bool:
       """
       Execute limit order with realistic slippage and commission
       """
       try:
           # Apply realistic slippage
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
           
           return True
           
       except Exception as e:
           self.logger.error(f"Order execution failed: {str(e)}")
           return False
   
   def can_create_pending_order(self, signal: Dict) -> bool:
       """
       Check if we can create a pending order
       """
       # Check maximum pending orders
       if len(self.pending_orders) >= self.config['max_pending_orders']:
           return False
       
       # Check for duplicate zones
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
   
   def manage_open_trades(self, current_data: pd.Series, current_date: pd.Timestamp):
       """
       Manage open trades with EXACT exit logic
       """
       trades_to_close = []
       
       for trade in self.open_trades:
           # Check stop loss hit
           if self.check_stop_loss_hit(trade, current_data):
               exit_price = trade['stop_loss']
               trades_to_close.append((trade, 'stop_loss', exit_price))
               continue
           
           # Check take profit hit (2:1 target only)
           if self.check_take_profit_hit(trade, current_data):
               exit_price = trade['take_profit_2']
               trades_to_close.append((trade, 'take_profit', exit_price))
               continue
           
           # Move to break-even at 1:1 (exact logic)
           if not trade['break_even_moved']:
               if self.should_move_to_breakeven(trade, current_data):
                   self.move_to_breakeven(trade)
       
       # Close trades
       for trade, exit_reason, exit_price in trades_to_close:
           self.close_trade(trade, exit_reason, exit_price, current_date)
   
   def check_stop_loss_hit(self, trade: Dict, current_data: pd.Series) -> bool:
       """Check if stop loss was hit"""
       direction = trade['direction']
       stop_loss = trade['stop_loss']
       
       if direction == 'BUY':
           return current_data['low'] <= stop_loss
       else:
           return current_data['high'] >= stop_loss
   
   def check_take_profit_hit(self, trade: Dict, current_data: pd.Series) -> bool:
       """Check if take profit was hit"""
       direction = trade['direction']
       tp2 = trade['take_profit_2']  # 2:1 target only
       
       if direction == 'BUY':
           return current_data['high'] >= tp2
       else:
           return current_data['low'] <= tp2
   
   def should_move_to_breakeven(self, trade: Dict, current_data: pd.Series) -> bool:
       """
       Check if trade should move to break-even (1:1 risk/reward)
       """
       direction = trade['direction']
       entry_price = trade['entry_price']
       initial_stop = trade['stop_loss']
       
       # Calculate 1:1 level
       risk_distance = abs(entry_price - initial_stop)
       
       if direction == 'BUY':
           breakeven_trigger = entry_price + risk_distance
           return current_data['high'] >= breakeven_trigger
       else:
           breakeven_trigger = entry_price - risk_distance
           return current_data['low'] <= breakeven_trigger
   
   def move_to_breakeven(self, trade: Dict):
       """Move stop to break-even"""
       trade['stop_loss'] = trade['entry_price']
       trade['break_even_moved'] = True
   
   def close_trade(self, trade: Dict, exit_reason: str, 
                  exit_price: float, exit_date: pd.Timestamp):
       """
       Close trade and calculate P&L
       """
       # Apply exit slippage
       if exit_reason == 'stop_loss':
           final_exit_price = self.apply_exit_slippage(exit_price, trade['direction'], True)
       else:
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
   
   def apply_slippage(self, entry_price: float, direction: str) -> float:
       """Apply realistic entry slippage"""
       pip_value = 0.0001
       slippage_pips = self.config['slippage_pips']
       slippage_amount = slippage_pips * pip_value
       
       if direction == 'BUY':
           return entry_price + slippage_amount
       else:
           return entry_price - slippage_amount
   
   def apply_exit_slippage(self, exit_price: float, direction: str, is_stop_loss: bool) -> float:
       """Apply realistic exit slippage"""
       pip_value = 0.0001
       slippage_pips = self.config['slippage_pips'] * (2 if is_stop_loss else 1)
       slippage_amount = slippage_pips * pip_value
       
       if direction == 'BUY':
           return exit_price - slippage_amount
       else:
           return exit_price + slippage_amount
   
   def calculate_commission(self, position_size: float) -> float:
       """Calculate commission"""
       return position_size * self.config['commission_per_lot']
   
   def calculate_trade_pnl(self, trade: Dict, exit_price: float) -> float:
       """Calculate trade P&L"""
       entry_price = trade['entry_price']
       position_size = trade['position_size']
       direction = trade['direction']
       
       # Price difference
       if direction == 'BUY':
           price_diff = exit_price - entry_price
       else:
           price_diff = entry_price - exit_price
       
       # Convert to USD (EURUSD: $10 per pip per lot)
       pip_difference = price_diff / 0.0001
       gross_pnl = pip_difference * 10.0 * position_size
       
       # Subtract exit commission
       exit_commission = self.calculate_commission(position_size)
       net_pnl = gross_pnl - exit_commission
       
       return net_pnl
   
   def record_daily_performance(self, current_date: pd.Timestamp, current_data: pd.Series):
       """Record daily performance for equity curve"""
       # Calculate unrealized P&L
       unrealized_pnl = 0
       for trade in self.open_trades:
           current_price = current_data['close']
           if trade['direction'] == 'BUY':
               price_diff = current_price - trade['entry_price']
           else:
               price_diff = trade['entry_price'] - current_price
           
           pip_difference = price_diff / 0.0001
           unrealized_pnl += pip_difference * 10.0 * trade['position_size']
       
       # Total equity
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
   
   def close_remaining_trades(self, final_data: pd.Series, final_date: pd.Timestamp):
       """Close remaining trades at backtest end"""
       final_price = final_data['close']
       
       for trade in self.open_trades.copy():
           self.close_trade(trade, 'backtest_end', final_price, final_date)
   
   def calculate_final_metrics(self, start_date: str, end_date: str, pair: str) -> Dict:
       """Calculate comprehensive performance metrics"""
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
       
       # Trade analysis
       trade_durations = [trade['days_held'] for trade in self.closed_trades if 'days_held' in trade]
       avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
       
       # Manual strategy comparison
       manual_pf = 5.3
       manual_wr = 53.0
       
       pf_accuracy = abs(profit_factor - manual_pf) / manual_pf * 100 if profit_factor != float('inf') else 100
       wr_accuracy = abs(win_rate - manual_wr) / manual_wr * 100
       
       # Expectancy
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
       """Return empty results when no trades executed"""
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