"""
Professional Backtesting Engine - Module 6 (CORRECTED VERSION)
Walk-forward historical validation with EVERY DATA POINT signal generation
Trades closest zones in trend direction only
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
    Professional walk-forward backtesting engine with CORRECTED logic
    Checks every data point for new zones and trades closest zones only
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
        self.pending_orders = []
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
        
        # Trend tracking for signal generation
        self.last_trend = None
        
        self.logger = logging.getLogger(__name__)
        
        print(f"🛡️  Backtesting Engine Initialized (CORRECTED VERSION):")
        print(f"   Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   Max Concurrent Trades: {self.config['max_concurrent_trades']}")
        print(f"   Slippage: {self.config['slippage_pips']} pips")
        print(f"   Signal Generation: Every data point (closest zone only)")
    
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
        CORRECTED: Check every data point for new zones and signals
        Trade closest zones in trend direction only
        
        Args:
            data: Complete OHLC dataset
            start_date: Backtest start date
            end_date: Backtest end date  
            lookback_days: Days of history for signal generation
            pair: Currency pair being tested
            
        Returns:
            Complete backtest results
        """
        print(f"\n🔄 STARTING WALK-FORWARD BACKTEST (CORRECTED LOGIC)")
        print(f"=" * 60)
        print(f"📊 Pair: {pair}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🔙 Lookback: {lookback_days} days")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"🎯 Strategy: Check every data point, trade closest zones only")
        
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
            
            # STEP 1: Check pending limit orders FIRST
            self.manage_pending_orders(current_data, current_date)
            
            # STEP 2: Generate signals EVERY data point (not weekly)
            # Get historical window for signal generation (only past data)
            history_start = max(0, current_idx - lookback_days)
            historical_data = data.iloc[history_start:current_idx]
            
            new_signals = self.generate_closest_zone_signals(
                historical_data, current_data, current_date, pair
            )
            
            # STEP 3: Create pending limit orders for new signals
            for signal in new_signals:
                if self.can_create_pending_order(signal):
                    self.create_pending_order_from_signal(signal, current_data, current_date)
            
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
    
    def generate_closest_zone_signals(self, historical_data: pd.DataFrame, 
                                    current_data: pd.Series, current_date: pd.Timestamp,
                                    pair: str) -> List[Dict]:
        """
        Generate signals for CLOSEST valid zone in trend direction EVERY data point
        """
        try:
            if len(historical_data) < 200:  # Need EMA200
                return []
            
            # 1. Get current trend bias
            trend_classifier = TrendClassifier(historical_data)
            trend_data = trend_classifier.classify_trend_with_filter()
            current_trend = trend_data['trend_filtered'].iloc[-1]
            current_price = historical_data['close'].iloc[-1]
            
            # 2. Only proceed if we have a clear trend (not ranging)
            if current_trend == 'ranging':
                return []
            
            # 3. Check if trend changed - if so, stop creating new orders but keep existing trades
            if self.last_trend is not None and self.last_trend != current_trend:
                print(f"   📊 Trend changed from {self.last_trend} to {current_trend} - no new signals")
                self.last_trend = current_trend
                return []
            
            self.last_trend = current_trend
            
            # 4. Generate all zones
            candle_classifier = CandleClassifier(historical_data)
            classified_data = candle_classifier.classify_all_candles()
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            # 5. Filter by trend direction only
            valid_zones = []
            bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
            
            if current_trend in bullish_trends:
                # Bullish trend = only R-B-R zones
                valid_zones = patterns['rbr_patterns']
            else:
                # Bearish trend = only D-B-D zones  
                valid_zones = patterns['dbd_patterns']
            
            # 6. Find CLOSEST valid zone
            if not valid_zones:
                return []
            
            closest_zone = self.find_closest_valid_zone(valid_zones, current_price, historical_data)
            
            if closest_zone:
                # 7. Create signal for closest zone only
                risk_manager = RiskManager(account_balance=self.current_balance)
                risk_validation = risk_manager.validate_zone_for_trading(
                    closest_zone, current_price, pair, historical_data
                )
                
                if risk_validation['is_tradeable']:
                    signal = self.create_signal_from_zone(closest_zone, risk_validation, current_trend, pair, current_date)
                    return [signal]
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Signal generation failed on {current_date}: {str(e)}")
            return []
    
    def find_closest_valid_zone(self, zones: List[Dict], current_price: float, 
                               data: pd.DataFrame) -> Optional[Dict]:
        """
        Find the closest untested zone to current price
        """
        valid_zones = []
        
        for zone in zones:
            # Check if zone is untested
            risk_manager = RiskManager()
            is_untested, _ = risk_manager.check_zone_testing(zone, data)
            
            if is_untested:
                zone_center = (zone['zone_high'] + zone['zone_low']) / 2
                distance = abs(zone_center - current_price)
                
                # Only consider zones within reasonable distance (500 pips)
                if distance <= 0.05:  # 500 pips
                    valid_zones.append((zone, distance))
        
        if not valid_zones:
            return None
        
        # Return closest zone
        valid_zones.sort(key=lambda x: x[1])  # Sort by distance
        return valid_zones[0][0]  # Return zone (not distance)
    
    def create_signal_from_zone(self, zone: Dict, risk_validation: Dict, 
                               trend: str, pair: str, current_date: pd.Timestamp) -> Dict:
        """
        Create signal from validated zone
        """
        signal_id = f"{pair}_{current_date.strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'signal_id': signal_id,
            'pair': pair,
            'direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL',
            'entry_price': risk_validation['entry_price'],
            'stop_loss': risk_validation['stop_loss_price'],
            'take_profit_1': risk_validation['take_profit_1'],
            'take_profit_2': risk_validation['take_profit_2'],
            'position_size': risk_validation['position_size'],
            'risk_amount': risk_validation['risk_amount'],
            'zone_high': zone['zone_high'],
            'zone_low': zone['zone_low'],
            'generated_date': current_date,
            'trend': trend,
            'zone_score': zone.get('strength', 0.5) * 100
        }
    
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
        Check if we can create a pending order - allow multiple orders in trend direction
        """
        # Check maximum pending orders
        if len(self.pending_orders) >= self.config['max_pending_orders']:
            return False
        
        # Check if we already have a pending order for this exact zone
        for existing_order in self.pending_orders:
            if (abs(existing_order['zone_high'] - signal['zone_high']) < 0.00001 and
                abs(existing_order['zone_low'] - signal['zone_low']) < 0.00001):
                return False  # Already have order for this zone
        
        # Allow multiple orders for different zones in same direction
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
    
    def manage_open_trades(self, current_data: pd.Series, current_date: pd.Timestamp):
        """
        Manage all open positions with proper exit logic
        """
        trades_to_close = []
        
        for trade in self.open_trades:
            # Check stop loss hit
            if self.check_stop_loss_hit(trade, current_data):
                exit_price = self.get_stop_loss_exit_price(trade, current_data)
                trades_to_close.append((trade, 'stop_loss', exit_price))
                continue
            
            # Check take profit hit (2:1 only - simplified)
            tp_result = self.check_take_profit_hit(trade, current_data)
            if tp_result:
                exit_price = tp_result['exit_price']
                trades_to_close.append((trade, 'take_profit', exit_price))
                continue
            
            # Move to break-even at 1:1 risk/reward
            if not trade['break_even_moved']:
                if self.should_move_to_breakeven(trade, current_data):
                    self.move_to_breakeven(trade)
        
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
        """Check if take profit was hit - use 2:1 target only"""
        direction = trade['direction']
        tp2 = trade['take_profit_2']  # 2:1 target
        
        if direction == 'BUY':
            # Long trade: TP hit if high reaches target
            if current_data['high'] >= tp2:
                return {'exit_price': tp2, 'level': 'tp2'}
        else:
            # Short trade: TP hit if low reaches target
            if current_data['low'] <= tp2:
                return {'exit_price': tp2, 'level': 'tp2'}
        
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