"""
Risk Management System - Module 4
Professional position sizing, stop loss, and take profit calculation
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from config.settings import RISK_CONFIG

class RiskManager:
    """
    Professional risk management system for forex trading automation
    Handles position sizing, stop losses, take profits, and risk validation
    """
    
    def __init__(self, account_balance: float = 10000, config: Dict = None):
        """
        Initialize risk manager with account settings
        
        Args:
            account_balance: Starting account balance in USD
            config: Risk configuration dictionary
        """
        self.account_balance = account_balance
        self.starting_balance = account_balance
        self.config = config or RISK_CONFIG
        
        # Risk tracking
        self.current_exposure = 0.0
        self.open_positions = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        self.logger = logging.getLogger(__name__)
        
        print(f"🛡️  Risk Manager initialized:")
        print(f"   Account Balance: ${self.account_balance:,.2f}")
        print(f"   Max Risk Per Trade: {self.config['risk_limits']['max_risk_per_trade']}%")
        print(f"   Daily Risk: No limit (single trade focus)")
    
    def check_zone_testing(self, zone: Dict, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if zone has been tested (33% penetration rule)
        
        Args:
            zone: Zone dictionary  
            data: Full OHLC data
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            zone_end_idx = zone['end_idx']
            zone_high = zone['zone_high']
            zone_low = zone['zone_low']
            zone_size = zone_high - zone_low
            zone_type = zone['type']
            
            # Check candles after zone formation
            for i in range(zone_end_idx + 1, len(data)):
                candle = data.iloc[i]
                
                if zone_type == 'R-B-R':  # Demand zone
                    # 33% down from zone top = tested
                    test_level = zone_high - (zone_size * 0.33)
                    if candle['close'] < test_level:
                        return False, f"Demand zone tested - price closed at {candle['close']:.5f} below test level {test_level:.5f}"
                        
                else:  # D-B-D Supply zone  
                    # 33% up from zone bottom = tested
                    test_level = zone_low + (zone_size * 0.33)
                    if candle['close'] > test_level:
                        return False, f"Supply zone tested - price closed at {candle['close']:.5f} above test level {test_level:.5f}"
            
            return True, "Zone untested"
            
        except Exception as e:
            return False, f"Error checking zone testing: {str(e)}"
        
    def validate_zone_for_trading(self, zone: Dict, current_price: float, 
                         pair: str = 'EURUSD', data: pd.DataFrame = None) -> Dict:
        """
        FIXED: Increase distance tolerance for historical backtesting
        """
        try:
            # Calculate entry and stop using manual methods
            entry_price = self.calculate_entry_price_manual(zone)
            stop_loss_price = self.calculate_stop_loss_manual(zone)
            
            # Calculate stop distance from your entry (not current price)
            pip_value = self.get_pip_value(pair)
            stop_distance_pips = abs(entry_price - stop_loss_price) / pip_value
            
            # Calculate position size based on YOUR 5% risk
            position_size = self.calculate_position_size(stop_distance_pips, pair)
            
            # Validate position size
            if position_size < self.config['position_sizing']['min_lot_size']:
                return {
                    'is_tradeable': False,
                    'reason': f"Position size {position_size} below minimum",
                    'position_size': position_size
                }
            
            # max_distance filter removed - zones valid regardless of current price distance
            
            # Calculate YOUR take profits
            take_profits = self.calculate_take_profits_manual(entry_price, stop_loss_price, zone['type'])
            
            # Calculate risk amount
            risk_amount = self.calculate_risk_amount(stop_distance_pips, position_size, pair)
            
            # Return complete trading parameters using YOUR strategy
            return {
                'is_tradeable': True,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'stop_distance_pips': stop_distance_pips,
                'take_profit_1': take_profits['tp1'],  # 1:1 break-even
                'take_profit_2': take_profits['tp2'],  # 1:2 final target
                'risk_reward_ratio': 2.5,
                'entry_method': 'manual_strategy',
                'trade_direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL'
            }
            
        except Exception as e:
            self.logger.error(f"Error in zone validation: {str(e)}")
            return {
                'is_tradeable': False,
                'reason': f"Validation error: {str(e)}"
            }

    def calculate_entry_price_manual(self, zone: Dict) -> float:
        """
        Your 5% front-running entry method using CORRECT zone logic
        
        Args:
            zone: Zone dictionary
            
        Returns:
            Entry price with 5% front-running from proper zone boundary
        """
        zone_size = zone['zone_high'] - zone['zone_low']
        front_run_distance = zone_size * 0.05  # 5% of zone size
        
        if zone['type'] == 'R-B-R':  # Bullish demand zone
            # Entry 5% above the HIGHEST CLOSE/OPEN boundary
            # Since zone detection already calculated the proper boundary, use zone_high
            entry_price = zone['zone_high'] + front_run_distance
            
        else:  # D-B-D bearish supply zone
            # Entry 5% below the LOWEST CLOSE/OPEN boundary  
            # Since zone detection already calculated the proper boundary, use zone_low
            entry_price = zone['zone_low'] - front_run_distance
        
        return entry_price

    def calculate_take_profits_manual(self, entry_price: float, stop_loss_price: float, zone_type: str) -> Dict:
        """Your 1:1 and 1:2.5 take profit method"""
        risk_distance = abs(entry_price - stop_loss_price)
        
        if zone_type == 'R-B-R':  # BUY trade
            tp1 = entry_price + risk_distance        # 1:1 (break-even move)
            tp2 = entry_price + (risk_distance * 2.5) # 1:2.5 (final target)
        else:  # SELL trade
            tp1 = entry_price - risk_distance        # 1:1 (break-even move)
            tp2 = entry_price - (risk_distance * 2.5) # 1:2.5 (final target)
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'risk_distance': risk_distance
        }
    
    def calculate_stop_loss_manual(self, zone: Dict) -> float:
        """
        Your 33% zone buffer stop loss method using CORRECT zone logic
        
        Args:
            zone: Zone dictionary
            
        Returns:
            Stop loss price with 33% buffer from proper zone boundary
        """
        if zone['type'] == 'R-B-R':  # Bullish demand zone
            # For bullish zones, stop goes 33% below the LOWEST WICK (zone_low)
            zone_boundary = zone['zone_low']  # This is the lowest wick already
            zone_size = zone['zone_high'] - zone['zone_low']
            buffer_distance = zone_size * 0.33
            stop_loss_price = zone_boundary - buffer_distance
            
        else:  # D-B-D bearish supply zone
            # For bearish zones, stop goes 33% above the HIGHEST WICK (zone_high)  
            zone_boundary = zone['zone_high']  # This is the highest wick already
            zone_size = zone['zone_high'] - zone['zone_low']
            buffer_distance = zone_size * 0.33
            stop_loss_price = zone_boundary + buffer_distance
        
        return stop_loss_price
    
    def calculate_position_size(self, stop_distance_pips: float, pair: str = 'EURUSD') -> float:
        """
        Calculate position size using your 5% risk method
        """
        # Risk amount in account currency (5% of balance)
        risk_amount = self.account_balance * (self.config['risk_limits']['max_risk_per_trade'] / 100)
        
        # Get pip value in account currency
        pip_value_usd = self.get_pip_value_usd(pair)
        
        # Position size formula: Risk Amount / (Stop Distance × Pip Value)
        position_size = risk_amount / (stop_distance_pips * pip_value_usd)
        
        # Round to broker increment
        increment = self.config['position_sizing']['lot_size_increment']
        position_size = round(position_size / increment) * increment
        
        # Apply limits
        min_size = self.config['position_sizing']['min_lot_size']
        max_size = self.config['position_sizing']['max_lot_size']
        
        position_size = max(min_size, min(max_size, position_size))
        
        return position_size
    
    
    def get_pip_value(self, pair: str = 'EURUSD') -> float:
        """
        Get pip value for currency pair
        
        Args:
            pair: Currency pair
            
        Returns:
            Pip value
        """
        if 'JPY' in pair:
            return 0.01  # JPY pairs
        else:
            return 0.0001  # Major pairs
    
    def get_pip_value_usd(self, pair: str = 'EURUSD') -> float:
        """
        Get pip value in USD for position sizing
        
        Args:
            pair: Currency pair
            
        Returns:
            Pip value in USD per standard lot
        """
        # Standard pip values for 1 standard lot
        pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'AUDUSD': 10.0,
            'NZDUSD': 10.0,
            'USDCAD': 10.0,  # Approximate
            'USDCHF': 10.0,  # Approximate
            'USDJPY': 10.0   # Approximate
        }
        
        return pip_values.get(pair, 10.0)  # Default to $10
    
    def calculate_risk_amount(self, stop_distance_pips: float, position_size: float, 
                            pair: str = 'EURUSD') -> float:
        """
        Calculate actual risk amount for the trade
        
        Args:
            stop_distance_pips: Stop distance in pips
            position_size: Position size in lots
            pair: Currency pair
            
        Returns:
            Risk amount in account currency
        """
        pip_value_usd = self.get_pip_value_usd(pair)
        return stop_distance_pips * pip_value_usd * position_size
    
    
    def update_account_balance(self, pnl: float):
        """
        Update account balance after trade close
        
        Args:
            pnl: Profit/Loss from closed trade
        """
        self.account_balance += pnl
        self.total_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Log performance
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        
        print(f"💰 Account Update:")
        print(f"   Trade P&L: ${pnl:,.2f}")
        print(f"   Account Balance: ${self.account_balance:,.2f}")
        print(f"   Total P&L: ${self.total_pnl:,.2f}")
        print(f"   Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})")
    
    def get_risk_summary(self) -> Dict:
        """
        Get comprehensive risk management summary
        
        Returns:
            Dictionary with risk statistics
        """
        return {
            'account_balance': self.account_balance,
            'starting_balance': self.starting_balance,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'current_exposure': self.current_exposure,
            'risk_per_trade_percent': self.config['risk_limits']['max_risk_per_trade'],
            'max_daily_risk_percent': None
        }