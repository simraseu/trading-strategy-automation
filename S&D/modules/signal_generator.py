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
            
            # Step 3: Filter zones by trend alignment
            print("🎯 Step 3: Filtering by trend alignment...")
            aligned_zones = self.filter_zones_by_trend(zones, current_trend)
            print(f"   Trend-aligned zones: {len(aligned_zones)}")
            
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
            
            # Step 5: Score and rank signals
            print("📈 Step 5: Scoring and ranking...")
            final_signals = self.score_and_rank_signals(tradeable_signals)
            
            # Step 6: Apply signal filters
            filtered_signals = self.apply_signal_filters(final_signals)
            
            print(f"✅ Final signals generated: {len(filtered_signals)}")
            
            # Store signals
            self.signals = filtered_signals
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def filter_zones_by_trend(self, zones: Dict, current_trend: str) -> List[Dict]:
        """
        Filter zones that align with current trend direction WITH DEBUG
        
        Args:
            zones: Dictionary with zone patterns
            current_trend: Current trend classification
            
        Returns:
            List of trend-aligned zones
        """
        aligned_zones = []
        
        # Define trend alignment rules
        bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
        bearish_trends = ['strong_bearish', 'medium_bearish', 'weak_bearish']
        
        all_zones = zones['dbd_patterns'] + zones['rbr_patterns']
        
        print(f"   DEBUG: Current trend = {current_trend}")
        print(f"   DEBUG: Bullish trends = {bullish_trends}")
        print(f"   DEBUG: Total zones to check = {len(all_zones)}")
        
        recent_zone_count = 0
        old_zone_count = 0
        
        for zone in all_zones:
            zone_type = zone['type']
            
            # Calculate zone age for debugging
            if hasattr(self, 'data') and self.data is not None:
                zone_end_date = self.data.index[zone['end_idx']]
                days_ago = (self.data.index[-1] - zone_end_date).days
                is_recent = days_ago <= 365
            else:
                days_ago = 0
                is_recent = True
            
            # Trend alignment logic
            should_keep = False
            if current_trend in bullish_trends and zone_type == 'R-B-R':
                # Bullish trend + Demand zone = BUY signal
                should_keep = True
                aligned_zones.append(zone)
            elif current_trend in bearish_trends and zone_type == 'D-B-D':
                # Bearish trend + Supply zone = SELL signal
                should_keep = True
                aligned_zones.append(zone)
            # Ranging market = no signals (filtered out)
            
            # Debug output for recent zones
            if is_recent:
                recent_zone_count += 1
                status = "KEPT" if should_keep else "FILTERED"
                print(f"   DEBUG RECENT: {zone_type} zone {days_ago} days ago - {status}")
            elif should_keep:
                old_zone_count += 1
        
        print(f"   DEBUG: Recent zones processed: {recent_zone_count}")
        print(f"   DEBUG: Old zones kept: {old_zone_count}")
        print(f"   DEBUG: Total zones kept: {len(aligned_zones)}")
        
        return aligned_zones
    
    def create_signal(self, zone: Dict, risk_data: Dict, trend_data: pd.DataFrame, 
                 pair: str, zone_timeframe: str) -> Dict:
        """
        Create complete trading signal with all parameters INCLUDING DATES
        """
        signal_id = f"{pair}_{zone_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # GET ZONE FORMATION DATES from the stored data
        zone_start_date = self.data.index[zone['start_idx']]
        zone_end_date = self.data.index[zone['end_idx']]
        base_start_date = self.data.index[zone['base']['start_idx']]
        base_end_date = self.data.index[zone['base']['end_idx']]
        
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
            'zone_formation_period': f"{zone_start_date.strftime('%Y-%m-%d')} to {zone_end_date.strftime('%Y-%m-%d')}",
            
            # Trade Direction & Type
            'direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL',
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
            'zone_score': self.calculate_zone_score(zone),
            'zone_strength': zone['strength'],
            
            # Trend Context
            'trend': trend_data['trend_filtered'].iloc[-1],
            'trend_strength': trend_data['ema_separation'].iloc[-1],
            'ema_50': trend_data['ema_50'].iloc[-1],
            'ema_100': trend_data['ema_100'].iloc[-1],
            'ema_200': trend_data['ema_200'].iloc[-1],
            
            # Signal Quality
            'signal_score': 0,  # Will be calculated in scoring phase
            'confidence': self.calculate_confidence(zone, trend_data),
            'priority': 'medium'  # Will be updated in ranking
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
    
    def score_and_rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Calculate comprehensive signal scores and rank by quality
        
        Args:
            signals: List of signals to score
            
        Returns:
            Scored and ranked signals
        """
        for signal in signals:
            signal['signal_score'] = self.calculate_signal_score(signal)
            signal['priority'] = self.determine_priority(signal['signal_score'])
        
        # Sort by signal score (highest first)
        signals.sort(key=lambda x: x['signal_score'], reverse=True)
        
        return signals
    
    def calculate_signal_score(self, signal: Dict) -> float:
        """
        Calculate comprehensive signal score (0-100)
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Signal score
        """
        # Zone Quality (40% weight)
        zone_score = signal['zone_score'] * 0.4
        
        # Trend Alignment (25% weight)
        trend_score = self.calculate_trend_alignment_score(signal) * 0.25
        
        # Risk/Reward Quality (20% weight)
        rr_score = min(signal['risk_reward_ratio'] / 3.0, 1.0) * 20  # Cap at 3:1
        
        # Entry Method Quality (10% weight)
        entry_score = self.calculate_entry_score(signal) * 0.10
        
        # Trend Strength Bonus (5% weight)
        strength_score = signal['trend_strength'] * 5
        
        total_score = zone_score + (trend_score * 25) + rr_score + (entry_score * 10) + strength_score
        
        return min(total_score, 100)  # Cap at 100
    
    def calculate_trend_alignment_score(self, signal: Dict) -> float:
        """Calculate trend alignment quality score"""
        trend = signal['trend']
        direction = signal['direction']
        
        # Perfect alignment scores
        if ((direction == 'BUY' and 'bullish' in trend) or 
            (direction == 'SELL' and 'bearish' in trend)):
            
            if 'strong' in trend:
                return 1.0  # Perfect alignment with strong trend
            elif 'medium' in trend:
                return 0.8  # Good alignment with medium trend
            else:  # weak trend
                return 0.6  # Acceptable alignment with weak trend
        
        return 0.0  # No alignment (shouldn't happen due to filtering)
    
    def calculate_entry_score(self, signal: Dict) -> float:
        """Calculate entry method quality score"""
        entry_method = signal['entry_method']
        
        if entry_method == 'market_entry':
            return 1.0  # Best entry method
        elif entry_method == 'limit_entry':
            return 0.7  # Good entry method
        else:  # wait_retest
            return 0.4  # Lower quality entry
    
    def determine_priority(self, signal_score: float) -> str:
        """Determine signal priority based on score"""
        if signal_score >= 80:
            return 'high'
        elif signal_score >= 65:
            return 'medium'
        else:
            return 'low'
    
    def apply_signal_filters(self, signals: List[Dict]) -> List[Dict]:
        """
        Apply signal filters with RECENCY PRIORITY
        
        Args:
            signals: List of scored signals
            
        Returns:
            Filtered signals prioritizing recent zones
        """
        # First filter by minimum score
        min_score = self.config['quality_thresholds']['min_zone_score']
        
        score_filtered = []
        for signal in signals:
            if signal['signal_score'] >= min_score:
                score_filtered.append(signal)
        
        print(f"   Signals after score filter (≥{min_score}): {len(score_filtered)}")
        
        # CRITICAL: Add recency scoring to prioritize recent zones
        current_price = score_filtered[0]['zone_high'] if score_filtered else 1.18  # Approximate current price
        
        for signal in score_filtered:
            # Calculate distance from current price (closer = more relevant)
            zone_center = (signal['zone_high'] + signal['zone_low']) / 2
            distance_from_current = abs(zone_center - current_price)
            
            # Recency bonus: zones closer to current price get priority
            max_distance = 0.20  # 20 cents max relevant distance
            distance_factor = max(0, (max_distance - distance_from_current) / max_distance)
            
            # Boost signal score for recent/relevant zones
            signal['recency_bonus'] = distance_factor * 20  # Up to 20 point bonus
            signal['adjusted_score'] = signal['signal_score'] + signal['recency_bonus']
        
        # Sort by ADJUSTED score (includes recency)
        score_filtered.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        # Optional daily limit
        if 'max_signals_per_day' in self.config['risk_management']:
            max_signals = self.config['risk_management']['max_signals_per_day']
            return score_filtered[:max_signals]
        
        # Return top 5 most recent/relevant signals
        top_signals = score_filtered[:5]
        
        print(f"   Final signals after recency filter: {len(top_signals)}")
        for i, signal in enumerate(top_signals):
            zone_center = (signal['zone_high'] + signal['zone_low']) / 2
            distance = abs(zone_center - current_price)
            print(f"      Signal {i+1}: Score {signal['signal_score']:.1f} + Recency {signal['recency_bonus']:.1f} = {signal['adjusted_score']:.1f} (Distance: {distance:.5f})")
        
        return top_signals
    
    def calculate_zone_score(self, zone: Dict) -> float:
        """Calculate zone score (consistent with risk manager)"""
        return self.risk_manager.calculate_zone_score(zone)
    
    def calculate_confidence(self, zone: Dict, trend_data: pd.DataFrame) -> float:
        """
        Calculate signal confidence level
        
        Args:
            zone: Zone dictionary
            trend_data: Trend analysis data
            
        Returns:
            Confidence score (0-1)
        """
        zone_confidence = min(zone['strength'], 1.0)
        trend_confidence = trend_data['ema_separation'].iloc[-1]
        
        # Combined confidence
        overall_confidence = (zone_confidence * 0.6) + (trend_confidence * 0.4)
        
        return min(overall_confidence, 1.0)
    
    def get_signal_summary(self) -> Dict:
        """
        Get summary of generated signals
        
        Returns:
            Signal summary dictionary
        """
        if not self.signals:
            return {
                'total_signals': 0,
                'by_priority': {'high': 0, 'medium': 0, 'low': 0},
                'by_direction': {'BUY': 0, 'SELL': 0},
                'avg_score': 0,
                'avg_risk_reward': 0
            }
        
        total = len(self.signals)
        
        # Count by priority
        priority_counts = {'high': 0, 'medium': 0, 'low': 0}
        for signal in self.signals:
            priority_counts[signal['priority']] += 1
        
        # Count by direction
        direction_counts = {'BUY': 0, 'SELL': 0}
        for signal in self.signals:
            direction_counts[signal['direction']] += 1
        
        # Calculate averages
        avg_score = sum(s['signal_score'] for s in self.signals) / total
        avg_rr = sum(s['risk_reward_ratio'] for s in self.signals) / total
        
        return {
            'total_signals': total,
            'by_priority': priority_counts,
            'by_direction': direction_counts,
            'avg_score': round(avg_score, 1),
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
                'zone_score': signal['zone_score'],
                'signal_score': signal['signal_score'],
                'priority': signal['priority'],
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