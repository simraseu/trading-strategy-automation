"""
Signal Generation System - Module 5
Risk-aware signal generation combining zones + trends
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from modules.risk_manager import RiskManager
from config.settings import SIGNAL_CONFIG

class SignalGenerator:
    """
    Professional signal generation system that combines:
    - Zone detection (configurable timeframes)
    - Trend classification (always Daily)
    - Risk management (pre-validated trades only)
    """
    
    def __init__(self, zone_detector, trend_classifier, risk_manager, config=None):
        """
        Initialize signal generator with all required components
        
        Args:
            zone_detector: ZoneDetector instance
            trend_classifier: TrendClassifier instance
            risk_manager: RiskManager instance
            config: Signal configuration dictionary
        """
        self.zone_detector = zone_detector
        self.trend_classifier = trend_classifier
        self.risk_manager = risk_manager
        self.config = config or SIGNAL_CONFIG
        
        self.signals = []
        self.signal_history = []
        
        self.logger = logging.getLogger(__name__)
        
        print(f"🎯 Signal Generator initialized:")
        print(f"   Zone Timeframes: {self.config['zone_timeframes']}")
        print(f"   Trend Timeframe: {self.config['trend_timeframe']}")
        print(f"   Min Zone Score: {self.config['quality_thresholds']['min_zone_score']}")
        print(f"   Risk Integration: ✅ Enabled")
    
    def generate_signals(self, data: pd.DataFrame, zone_timeframe: str = 'Daily', 
                        pair: str = 'EURUSD') -> List[Dict]:
        """
        Generate risk-validated trading signals
        
        Args:
            data: OHLC data for the specified timeframe
            zone_timeframe: Timeframe for zone detection
            pair: Currency pair
            
        Returns:
            List of validated trading signals
        """
        try:
            # STORE DATA for date access
            self.data = data.copy()
            
            print(f"\n🎯 Generating {pair} signals ({zone_timeframe} zones + Daily trend)")
            print("=" * 60)
            
            # Step 1: Detect zones on specified timeframe
            print("🔍 Step 1: Detecting zones...")
            zones = self.zone_detector.detect_all_patterns(data)
            print(f"   Found {zones['total_patterns']} total zones")
            
            # Step 2: Get current trend (always Daily timeframe)
            print("📊 Step 2: Analyzing trend...")
            trend_data = self.trend_classifier.classify_trend_with_filter()
            current_trend = trend_data['trend_filtered'].iloc[-1]
            trend_strength = trend_data['ema_separation'].iloc[-1]
            print(f"   Current trend: {current_trend.upper()}")
            print(f"   Trend strength: {trend_strength:.3f}")
            
            # Step 3: No trend filtering - use all zones
            print("🎯 Step 3: No trend filtering applied...")
            aligned_zones = self.filter_zones_by_trend(zones, current_trend)
            print(f"   Available zones: {len(aligned_zones)}")
            
            # Step 4: Risk validation (CRITICAL FILTER)
            print("🛡️  Step 4: Risk validation...")
            current_price = data['close'].iloc[-1]
            tradeable_signals = []

            for zone in aligned_zones:
                # PASS DATA to risk validation
                risk_validation = self.risk_manager.validate_zone_for_trading(
                    zone, current_price, pair, data  # ← ADD data parameter
                )
                
                if risk_validation['is_tradeable']:
                    signal = self.create_signal(zone, risk_validation, trend_data, pair, zone_timeframe)
                    tradeable_signals.append(signal)
            
            print(f"   Risk-validated signals: {len(tradeable_signals)}")
            
            print(f"✅ Final signals generated: {len(tradeable_signals)}")
            
            # Store signals
            self.signals = tradeable_signals
            
            return tradeable_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def filter_zones_by_trend(self, zones: Dict, current_trend: str) -> List[Dict]:
        """
        NO TREND FILTER: Return all zones regardless of trend direction
        
        Args:
            zones: Dictionary with zone patterns
            current_trend: Current trend classification (unused)
            
        Returns:
            List of all zones (no filtering)
        """
        # Get all zone types including reversals
        all_zones = (zones['dbd_patterns'] + zones['rbr_patterns'] + 
                    zones.get('dbr_patterns', []) + zones.get('rbd_patterns', []))
        
        print(f"   NO TREND FILTER: Using all {len(all_zones)} zones")
        
        # Return all zones without filtering
        for zone in all_zones:
            zone_type = zone['type']
            
            # Calculate zone age for debugging
            if hasattr(self, 'data') and self.data is not None:
                try:
                    # More robust date handling
                    if zone['end_idx'] < len(self.data):
                        zone_end_idx = zone['end_idx']
                        current_idx = len(self.data) - 1
                        candles_ago = current_idx - zone_end_idx
                        print(f"   KEPT: {zone_type} zone ({candles_ago} candles ago)")
                    else:
                        print(f"   KEPT: {zone_type} zone (recent formation)")
                except Exception:
                    print(f"   KEPT: {zone_type} zone")
        
        print(f"   RESULT: {len(all_zones)} zones available (no trend filtering)")
        
        return all_zones
    
    def create_signal(self, zone: Dict, risk_data: Dict, trend_data: pd.DataFrame, 
                 pair: str, zone_timeframe: str) -> Dict:
        """
        Create complete trading signal with all parameters INCLUDING DATES
        """
        signal_id = f"{pair}_{zone_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine signal direction based on pattern type
        if zone['type'] in ['R-B-R', 'D-B-R']:
            signal_direction = 'BUY'
        elif zone['type'] in ['D-B-D', 'R-B-D']:
            signal_direction = 'SELL'
        else:
            signal_direction = 'BUY' if zone['type'] == 'R-B-R' else 'SELL'  # Fallback
        
        # GET ZONE FORMATION DATES from the stored data
        # FIXED: Handle both datetime index and integer index
        try:
            zone_start_date = self.data.index[zone['start_idx']]
            zone_end_date = self.data.index[zone['end_idx']]
            base_start_date = self.data.index[zone['base']['start_idx']]
            base_end_date = self.data.index[zone['base']['end_idx']]
            
            # Ensure we have datetime objects
            if isinstance(zone_start_date, int):
                zone_start_date = f"Index_{zone_start_date}"
                zone_end_date = f"Index_{zone_end_date}"
                base_start_date = f"Index_{base_start_date}"
                base_end_date = f"Index_{base_end_date}"
                zone_formation_period = f"Index {zone['start_idx']} to {zone['end_idx']}"
            else:
                zone_formation_period = f"{zone_start_date.strftime('%Y-%m-%d')} to {zone_end_date.strftime('%Y-%m-%d')}"
                
        except (KeyError, AttributeError, IndexError) as e:
            # Fallback for problematic indices
            zone_start_date = f"Index_{zone.get('start_idx', 'unknown')}"
            zone_end_date = f"Index_{zone.get('end_idx', 'unknown')}"
            base_start_date = f"Index_{zone['base'].get('start_idx', 'unknown')}"
            base_end_date = f"Index_{zone['base'].get('end_idx', 'unknown')}"
            zone_formation_period = f"Formation indices: {zone.get('start_idx', 'unknown')}-{zone.get('end_idx', 'unknown')}"
        
        signal = {
            # Signal Identification WITH DATES
            'signal_id': signal_id,
            'pair': pair,
            'timeframe': zone_timeframe,
            'timestamp': datetime.now(),
            
            # ZONE DATES - Critical for manual validation
            'zone_start_date': zone_start_date,
            'zone_end_date': zone_end_date,
            'base_start_date': base_start_date,
            'base_end_date': base_end_date,
            'zone_formation_period': zone_formation_period,
            
            # Trade Direction & Type
            'direction': signal_direction,
            'signal_type': zone['type'],
            'entry_method': risk_data.get('entry_method', 'manual_strategy'),
            
            # Price Levels
            'entry_price': risk_data['entry_price'],
            'stop_loss': risk_data['stop_loss_price'],
            'take_profit_1': risk_data['take_profit_1'],
            'take_profit_2': risk_data['take_profit_2'],
            
            # Risk Management
            'position_size': risk_data['position_size'],
            'risk_amount': risk_data['risk_amount'],
            'risk_reward_ratio': risk_data['risk_reward_ratio'],
            'stop_distance_pips': risk_data['stop_distance_pips'],
            
            # Zone Information
            'zone_high': zone['zone_high'],
            'zone_low': zone['zone_low'],
            'zone_range': zone['zone_range'],
            
            # Trend Context
            'trend': trend_data['trend_filtered'].iloc[-1],
            'trend_strength': trend_data['ema_separation'].iloc[-1],
            'ema_50': trend_data['ema_50'].iloc[-1],
            'ema_200': trend_data['ema_200'].iloc[-1],
            
            # Signal Quality - Basic information only
            'signal_type': 'zone_entry'
        }
        
        return signal
    
    def calculate_entry_price(self, zone: Dict, risk_data: Dict) -> float:
        """
        Calculate entry price using your 5% front-running method
        
        Args:
            zone: Zone dictionary
            risk_data: Risk validation data
            
        Returns:
            Entry price
        """
        zone_size = zone['zone_high'] - zone['zone_low']
        front_run_distance = zone_size * 0.05  # 5% of zone size
        
        if zone['type'] == 'R-B-R':  # Demand zone - enter 5% above zone top
            entry_price = zone['zone_high'] + front_run_distance
        else:  # D-B-D Supply zone - enter 5% below zone bottom  
            entry_price = zone['zone_low'] - front_run_distance
        
        return entry_price
    
    def get_signal_summary(self) -> Dict:
        """
        Get summary of generated signals
        
        Returns:
            Signal summary dictionary
        """
        if not self.signals:
            return {
                'total_signals': 0,
                'by_direction': {'BUY': 0, 'SELL': 0},
                'by_pattern': {'D-B-D': 0, 'R-B-R': 0, 'D-B-R': 0, 'R-B-D': 0},
                'avg_risk_reward': 0
            }
        
        total = len(self.signals)
        
        # Count by direction
        direction_counts = {'BUY': 0, 'SELL': 0}
        for signal in self.signals:
            direction_counts[signal['direction']] += 1
        
        # Count by pattern type
        pattern_counts = {'D-B-D': 0, 'R-B-R': 0, 'D-B-R': 0, 'R-B-D': 0}
        for signal in self.signals:
            pattern_type = signal['signal_type']
            if pattern_type in pattern_counts:
                pattern_counts[pattern_type] += 1
        
        # Calculate average risk/reward
        avg_rr = sum(s['risk_reward_ratio'] for s in self.signals) / total
        
        return {
            'total_signals': total,
            'by_direction': direction_counts,
            'by_pattern': pattern_counts,
            'avg_risk_reward': round(avg_rr, 1),
            'signals': self.signals
        }
    
    def export_signals_for_backtesting(self, filename: str = None) -> str:
        """Export signals with DATE information for manual validation"""
        if not filename:
            filename = f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        signal_data = []
        for signal in self.signals:
            signal_data.append({
                'signal_id': signal['signal_id'],
                'pair': signal['pair'],
                'timeframe': signal['timeframe'],
                'direction': signal['direction'],
                
                # CRITICAL: Date information for manual validation
                'zone_start_date': signal['zone_start_date'],
                'zone_end_date': signal['zone_end_date'],
                'base_start_date': signal['base_start_date'],
                'base_end_date': signal['base_end_date'],
                'zone_formation_period': signal['zone_formation_period'],
                
                # Trading parameters
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit_1': signal['take_profit_1'],
                'take_profit_2': signal['take_profit_2'],
                'position_size': signal['position_size'],
                'risk_amount': signal['risk_amount'],
                'stop_distance_pips': signal['stop_distance_pips'],
                
                # Zone details
                'zone_high': signal['zone_high'],
                'zone_low': signal['zone_low'],
                'zone_range': signal['zone_range'],
                'trend': signal['trend']
            })
        
        df = pd.DataFrame(signal_data)
        
        # Save to results directory
        import os
        os.makedirs('results', exist_ok=True)
        filepath = f"results/{filename}"
        df.to_csv(filepath, index=False)
        
        print(f"💾 Signals exported to: {filepath}")
        return filepath