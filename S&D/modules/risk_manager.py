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
        self.daily_risk_used = 0.0
        self.open_positions = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        self.logger = logging.getLogger(__name__)
        
        print(f"🛡️  Risk Manager initialized:")
        print(f"   Account Balance: ${self.account_balance:,.2f}")
        print(f"   Max Risk Per Trade: {self.config['risk_limits']['max_risk_per_trade']}%")
        print(f"   Max Daily Risk: {self.config['risk_limits']['max_daily_risk']}%")
        
    def validate_zone_for_trading(self, zone: Dict, current_price: float, 
                                 pair: str = 'EURUSD') -> Dict:
        """
        CRITICAL: Validate if zone meets all risk management criteria
        
        Args:
            zone: Zone dictionary from zone detector
            current_price: Current market price
            pair: Currency pair
            
        Returns:
            Dictionary with validation results and trading parameters
        """
        try:
            # Get pip value for the pair
            pip_value = self.get_pip_value(pair)
            
            # Calculate stop loss placement
            stop_loss_data = self.calculate_stop_loss(zone, current_price, pip_value)
            
            # Validate stop loss distance
            stop_distance_valid = self.validate_stop_distance(stop_loss_data['distance_pips'])
            
            if not stop_distance_valid:
                return {
                    'is_tradeable': False,
                    'reason': f"Stop distance {stop_loss_data['distance_pips']:.1f} pips outside limits",
                    'stop_distance_pips': stop_loss_data['distance_pips']
                }
            
            # Calculate position size
            position_size = self.calculate_position_size(stop_loss_data['distance_pips'], pair)
            
            # Validate position size
            if position_size < self.config['position_sizing']['min_lot_size']:
                return {
                    'is_tradeable': False,
                    'reason': f"Position size {position_size} below minimum",
                    'position_size': position_size
                }
            
            # Check risk budget availability
            risk_amount = self.calculate_risk_amount(stop_loss_data['distance_pips'], position_size, pair)
            
            if not self.check_risk_budget(risk_amount):
                return {
                    'is_tradeable': False,
                    'reason': "Daily risk budget exceeded",
                    'risk_amount': risk_amount
                }
            
            # Calculate take profits
            take_profits = self.calculate_take_profits(
                zone, current_price, stop_loss_data['price'], pair
            )
            
            # Everything passed - create complete trading parameters
            return {
                'is_tradeable': True,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_loss_price': stop_loss_data['price'],
                'stop_distance_pips': stop_loss_data['distance_pips'],
                'take_profit_1': take_profits['tp1'],
                'take_profit_2': take_profits['tp2'],
                'take_profit_3': take_profits['tp3'],
                'risk_reward_ratio': take_profits['risk_reward_ratio'],
                'entry_method': self.determine_entry_method(zone),
                'trade_direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL'
            }
            
        except Exception as e:
            self.logger.error(f"Error in zone validation: {str(e)}")
            return {
                'is_tradeable': False,
                'reason': f"Validation error: {str(e)}"
            }
    
    def calculate_stop_loss(self, zone: Dict, current_price: float, pip_value: float) -> Dict:
        """
        Calculate stop loss placement based on zone boundaries
        
        Args:
            zone: Zone dictionary
            current_price: Current market price
            pip_value: Pip value for the pair
            
        Returns:
            Dictionary with stop loss price and distance
        """
        buffer_pips = self.config['stop_loss_rules']['buffer_pips']
        buffer_price = buffer_pips * pip_value
        
        if zone['type'] == 'R-B-R':  # Demand zone - stop below
            stop_loss_price = zone['zone_low'] - buffer_price
            direction = 'BUY'
        else:  # D-B-D Supply zone - stop above
            stop_loss_price = zone['zone_high'] + buffer_price
            direction = 'SELL'
        
        # Calculate distance in pips
        if direction == 'BUY':
            distance_pips = (current_price - stop_loss_price) / pip_value
        else:
            distance_pips = (stop_loss_price - current_price) / pip_value
        
        return {
            'price': stop_loss_price,
            'distance_pips': abs(distance_pips),
            'direction': direction
        }
    
    def calculate_position_size(self, stop_distance_pips: float, pair: str = 'EURUSD') -> float:
        """
        Calculate position size using fixed risk percentage method
        
        Args:
            stop_distance_pips: Stop loss distance in pips
            pair: Currency pair
            
        Returns:
            Position size in lots
        """
        # Risk amount in account currency
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
    
    def calculate_take_profits(self, zone: Dict, current_price: float, 
                             stop_loss_price: float, pair: str = 'EURUSD') -> Dict:
        """
        Calculate take profit levels based on risk-reward ratios
        
        Args:
            zone: Zone dictionary
            current_price: Current market price
            stop_loss_price: Stop loss price
            pair: Currency pair
            
        Returns:
            Dictionary with take profit levels
        """
        # Calculate risk distance
        risk_distance = abs(current_price - stop_loss_price)
        
        # Get risk-reward ratios from config
        rr_ratios = self.config['take_profit_rules']['scale_levels']
        
        if zone['type'] == 'R-B-R':  # BUY trade
            tp1 = current_price + (risk_distance * rr_ratios[0])
            tp2 = current_price + (risk_distance * rr_ratios[1])
            tp3 = current_price + (risk_distance * rr_ratios[2])
        else:  # SELL trade
            tp1 = current_price - (risk_distance * rr_ratios[0])
            tp2 = current_price - (risk_distance * rr_ratios[1])
            tp3 = current_price - (risk_distance * rr_ratios[2])
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'risk_reward_ratio': rr_ratios[1],  # Main target is 2R
            'risk_distance': risk_distance
        }
    
    def validate_stop_distance(self, stop_distance_pips: float) -> bool:
        """
        Validate stop loss distance against limits
        
        Args:
            stop_distance_pips: Stop distance in pips
            
        Returns:
            Boolean indicating if distance is valid
        """
        min_stop = self.config['stop_loss_rules']['min_stop_distance']
        max_stop = self.config['stop_loss_rules']['max_stop_distance']
        
        return min_stop <= stop_distance_pips <= max_stop
    
    def check_risk_budget(self, risk_amount: float) -> bool:
        """
        Check if trade fits within daily risk budget
        
        Args:
            risk_amount: Risk amount for the trade
            
        Returns:
            Boolean indicating if trade fits budget
        """
        max_daily_risk = self.account_balance * (self.config['risk_limits']['max_daily_risk'] / 100)
        
        return (self.daily_risk_used + risk_amount) <= max_daily_risk
    
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
    
    def determine_entry_method(self, zone: Dict) -> str:
        """
        Determine best entry method for the zone
        
        Args:
            zone: Zone dictionary
            
        Returns:
            Entry method string
        """
        zone_score = self.calculate_zone_score(zone)
        
        if zone_score >= 80:
            return 'market_entry'  # High quality - enter at market
        elif zone_score >= 65:
            return 'limit_entry'   # Medium quality - use limit order
        else:
            return 'wait_retest'   # Lower quality - wait for retest
    
    def calculate_zone_score(self, zone: Dict) -> float:
        """
        Calculate zone quality score (same as visualizer for consistency)
        
        Args:
            zone: Zone dictionary
            
        Returns:
            Zone score (0-100)
        """
        # Base scoring components
        leg_in_score = zone['leg_in']['strength'] * 20      # 0-20 points
        base_score = zone['base']['quality_score'] * 30     # 0-30 points  
        leg_out_score = zone['leg_out']['strength'] * 25    # 0-25 points
        
        # Distance bonus (most important for momentum)
        distance_ratio = zone['leg_out']['ratio_to_base']
        distance_score = min(distance_ratio / 2.0, 1.0) * 20  # 0-20 points
        
        # Base candle bonus (1-2 candles optimal)
        base_candles = zone['base']['candle_count']
        if base_candles <= 2:
            base_bonus = 5
        elif base_candles == 3:
            base_bonus = 3
        else:
            base_bonus = 0
            
        total_score = leg_in_score + base_score + leg_out_score + distance_score + base_bonus
        return min(total_score, 100)  # Cap at 100
    
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
            'daily_risk_used': self.daily_risk_used,
            'current_exposure': self.current_exposure,
            'risk_per_trade_percent': self.config['risk_limits']['max_risk_per_trade'],
            'max_daily_risk_percent': self.config['risk_limits']['max_daily_risk']
        }