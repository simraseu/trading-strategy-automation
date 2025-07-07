"""
Zone Detection Engine - Module 2 (BACK TO BASICS VERSION)
Simple, reliable D-B-D and R-B-R pattern detection
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import sys
import os

# Add path to find config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with fallback
try:
    from config.settings import ZONE_CONFIG
except ImportError:
    ZONE_CONFIG = {
        'min_base_candles': 1,
        'max_base_candles': 6,
        'min_legout_ratio': 1.5,
        'min_pattern_pips': 20,
        'pip_value': 0.0001
    }
    print("⚠️  Using fallback ZONE_CONFIG")

class ZoneDetector:
    """
    Simple, reliable zone detection system
    Back to basics: LEG-IN → BASE → LEG-OUT
    """
    
    def __init__(self, candle_classifier, config=None):
        """Initialize zone detector"""
        self.candle_classifier = candle_classifier
        self.config = config or ZONE_CONFIG
        self.logger = logging.getLogger(__name__)
        
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect all zone patterns"""
        try:
            # Validate input data
            self.validate_data(data)
            
            # Detect patterns
            dbd_patterns = self.detect_dbd_patterns(data)
            rbr_patterns = self.detect_rbr_patterns(data)
            
            print(f"✅ Zone detection complete:")
            print(f"   D-B-D patterns: {len(dbd_patterns)}")
            print(f"   R-B-R patterns: {len(rbr_patterns)}")
            
            return {
                'dbd_patterns': dbd_patterns,
                'rbr_patterns': rbr_patterns,
                'total_patterns': len(dbd_patterns) + len(rbr_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def detect_rbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect R-B-R patterns: Bullish LEG-IN → BASE → Bullish LEG-OUT
        """
        patterns = []
        
        for i in range(len(data) - 4):  # Need at least 5 candles
            try:
                # STEP 1: Find bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # STEP 2: Find base after leg-in
                base_start = leg_in['end_idx'] + 1
                base = self.identify_base(data, base_start)
                if not base:
                    continue
                
                # STEP 3: Find bullish leg-out after base
                leg_out_start = base['end_idx'] + 1
                leg_out = self.validate_leg_out_breakout(data, leg_out_start, base, 'bullish')
                if not leg_out:
                    continue
                
                # STEP 4: Calculate zone boundaries (BASE ONLY)
                zone_high = base['high']
                zone_low = base['low']
                
                # STEP 5: Calculate leg-out ratio
                zone_range = zone_high - zone_low
                leg_out['ratio_to_base'] = leg_out['range'] / base['range'] if base['range'] > 0 else 0
                
                # STEP 6: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < 1.5:
                    continue
                
                # Create R-B-R pattern
                pattern = {
                    'type': 'R-B-R',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': zone_high,
                    'zone_low': zone_low,
                    'zone_range': zone_range,
                    'strength': self.calculate_pattern_strength(leg_in, base, leg_out)
                }
                
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def detect_dbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect D-B-D patterns: Bearish LEG-IN → BASE → Bearish LEG-OUT
        """
        patterns = []
        
        for i in range(len(data) - 4):  # Need at least 5 candles
            try:
                # STEP 1: Find bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # STEP 2: Find base after leg-in
                base_start = leg_in['end_idx'] + 1
                base = self.identify_base(data, base_start)
                if not base:
                    continue
                
                # STEP 3: Find bearish leg-out after base
                leg_out_start = base['end_idx'] + 1
                leg_out = self.validate_leg_out_breakout(data, leg_out_start, base, 'bearish')
                if not leg_out:
                    continue
                
                # STEP 4: Calculate zone boundaries (BASE ONLY)
                zone_high = base['high']
                zone_low = base['low']
                
                # STEP 5: Calculate leg-out ratio
                zone_range = zone_high - zone_low
                leg_out['ratio_to_base'] = leg_out['range'] / base['range'] if base['range'] > 0 else 0
                
                # STEP 6: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < 1.5:
                    continue
                
                # Create D-B-D pattern
                pattern = {
                    'type': 'D-B-D',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': zone_high,
                    'zone_low': zone_low,
                    'zone_range': zone_range,
                    'strength': self.calculate_pattern_strength(leg_in, base, leg_out)
                }
                
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """
        Identify leg-in (1-3 candles)
        """
        if start_idx >= len(data):
            return None
            
        for leg_length in range(1, 4):  # Try 1, 2, or 3 candles
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            if self.is_valid_leg(leg_data, direction):
                leg_range = leg_data['high'].max() - leg_data['low'].min()
                
                # Minimum requirement (20 pips)
                min_range = 20 * 0.0001
                if leg_range < min_range:
                    continue
                
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': direction,
                    'range': leg_range,
                    'strength': self.calculate_leg_strength(leg_data, direction),
                    'candle_count': leg_length
                }
        
        return None
    
    def identify_base(self, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """
        Identify consecutive base candles (≤50% body ratio)
        """
        if start_idx >= len(data):
            return None
        
        max_base_length = self.config['max_base_candles']
        
        # Find consecutive base candles
        for base_length in range(1, max_base_length + 1):
            end_idx = start_idx + base_length - 1
            
            if end_idx >= len(data):
                break
            
            base_data = data.iloc[start_idx:end_idx + 1]
            
            # Check if ALL candles are base candles
            if self.is_valid_base(base_data):
                base_high = base_data['high'].max()
                base_low = base_data['low'].min()
                base_range = base_high - base_low
                
                # Minimum base range requirement
                if base_range < 10 * 0.0001:  # At least 10 pips
                    continue
                
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'high': base_high,
                    'low': base_low,
                    'range': base_range,
                    'candle_count': base_length,
                    'quality_score': self.calculate_base_quality(base_data)
                }
        
        return None
    
    def validate_leg_out_breakout(self, data: pd.DataFrame, start_idx: int, 
                                base: Dict, direction: str) -> Optional[Dict]:
        """
        Validate leg-out breaks out of base with sufficient distance
        """
        if start_idx >= len(data):
            return None
            
        for leg_length in range(1, 4):  # Try 1, 2, or 3 candles
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            # Check if leg is valid and breaks out of base
            if self.is_valid_leg(leg_data, direction):
                # Check breakout
                if direction == 'bullish':
                    leg_high = leg_data['high'].max()
                    if leg_high <= base['high']:
                        continue  # Didn't break out
                else:  # bearish
                    leg_low = leg_data['low'].min()
                    if leg_low >= base['low']:
                        continue  # Didn't break out
                
                leg_range = leg_data['high'].max() - leg_data['low'].min()
                
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': direction,
                    'range': leg_range,
                    'strength': self.calculate_leg_strength(leg_data, direction),
                    'candle_count': leg_length
                }
        
        return None
    
    def is_valid_leg(self, leg_data: pd.DataFrame, direction: str) -> bool:
        """
        Check if leg is valid in the specified direction
        """
        directional_candles = 0
        strong_candles = 0
        
        for idx in leg_data.index:
            candle = leg_data.loc[idx]
            
            # Check direction
            if direction == 'bullish' and candle['close'] > candle['open']:
                directional_candles += 1
            elif direction == 'bearish' and candle['close'] < candle['open']:
                directional_candles += 1
            
            # Check strength
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if candle_range > 0:
                body_ratio = body_size / candle_range
                if body_ratio > 0.50:
                    strong_candles += 1
        
        # Requirements: 70% directional, 50% strong
        direction_ratio = directional_candles / len(leg_data)
        strength_ratio = strong_candles / len(leg_data)
        
        return direction_ratio >= 0.7 and strength_ratio >= 0.5
    
    def is_valid_base(self, base_data: pd.DataFrame) -> bool:
        """
        Check if ALL candles are base candles (≤50% body ratio)
        """
        for idx in base_data.index:
            candle = base_data.loc[idx]
            
            # Get classification
            classification = self.candle_classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            # Must be base candle
            if classification != 'base':
                return False
        
        return True
    
    def calculate_leg_strength(self, leg_data: pd.DataFrame, direction: str) -> float:
        """Calculate leg strength score"""
        strong_candles = 0
        
        for idx in leg_data.index:
            candle = leg_data.loc[idx]
            classification = self.candle_classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            if classification in ['decisive', 'explosive']:
                strong_candles += 1
        
        return strong_candles / len(leg_data)
    
    def calculate_base_quality(self, base_data: pd.DataFrame) -> float:
        """Calculate base quality score"""
        candle_count = len(base_data)
        
        # Favor shorter bases
        if candle_count == 1:
            return 1.0
        elif candle_count == 2:
            return 0.9
        elif candle_count == 3:
            return 0.7
        else:
            return 0.5
    
    def calculate_pattern_strength(self, leg_in: Dict, base: Dict, leg_out: Dict) -> float:
        """Calculate overall pattern strength"""
        leg_in_strength = leg_in['strength']
        base_quality = base['quality_score']
        leg_out_strength = leg_out['strength']
        
        return (leg_in_strength + base_quality + leg_out_strength) / 3
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(data) < 10:
            raise ValueError("Insufficient data for pattern detection")