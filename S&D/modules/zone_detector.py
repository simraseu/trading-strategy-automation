"""
Zone Detection Engine - Module 2 (COMPLETELY REWRITTEN VERSION)
Perfect zone detection with proper base boundary calculation
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
    Perfect zone detection system with correct base boundary calculation
    Detects all 4 pattern types: D-B-D, R-B-R, D-B-R, R-B-D
    """
    
    def __init__(self, candle_classifier, config=None):
        """Initialize zone detector"""
        self.candle_classifier = candle_classifier
        self.config = config or ZONE_CONFIG
        self.logger = logging.getLogger(__name__)
        
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect all zone patterns including reversals"""
        try:
            # Validate input data
            self.validate_data(data)
            
            # Detect all 4 pattern types
            dbd_patterns = self.detect_dbd_patterns(data)
            rbr_patterns = self.detect_rbr_patterns(data)
            dbr_patterns = self.detect_dbr_patterns(data)
            rbd_patterns = self.detect_rbd_patterns(data)
            
            total_patterns = len(dbd_patterns) + len(rbr_patterns) + len(dbr_patterns) + len(rbd_patterns)
            
            print(f"✅ Zone detection complete:")
            print(f"   D-B-D patterns: {len(dbd_patterns)} (momentum)")
            print(f"   R-B-R patterns: {len(rbr_patterns)} (momentum)")
            print(f"   D-B-R patterns: {len(dbr_patterns)} (reversal)")
            print(f"   R-B-D patterns: {len(rbd_patterns)} (reversal)")
            
            return {
                'dbd_patterns': dbd_patterns,
                'rbr_patterns': rbr_patterns,
                'dbr_patterns': dbr_patterns,
                'rbd_patterns': rbd_patterns,
                'total_patterns': total_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def detect_rbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect R-B-R patterns: Bullish LEG-IN → ALL BASE CANDLES → Bullish LEG-OUT
        """
        patterns = []
        
        for i in range(len(data) - 4):  # Need at least 5 candles
            try:
                # STEP 1: Find bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # STEP 2: Find ALL consecutive base candles after leg-in
                base_sequence = self.find_all_consecutive_base_candles(data, leg_in['end_idx'] + 1)
                if not base_sequence:
                    continue
                
                # STEP 3: Find bullish leg-out immediately after base sequence
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bullish')
                if not leg_out:
                    continue
                
                # STEP 4: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # STEP 5: Create perfect R-B-R pattern
                pattern = self.create_pattern(
                    pattern_type='R-B-R',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def detect_dbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect D-B-D patterns: Bearish LEG-IN → ALL BASE CANDLES → Bearish LEG-OUT
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # STEP 1: Find bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # STEP 2: Find ALL consecutive base candles after leg-in
                base_sequence = self.find_all_consecutive_base_candles(data, leg_in['end_idx'] + 1)
                if not base_sequence:
                    continue
                
                # STEP 3: Find bearish leg-out immediately after base sequence
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bearish')
                if not leg_out:
                    continue
                
                # STEP 4: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # STEP 5: Create perfect D-B-D pattern
                pattern = self.create_pattern(
                    pattern_type='D-B-D',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def detect_dbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect D-B-R patterns: Bearish LEG-IN → ALL BASE CANDLES → Bullish LEG-OUT (Reversal)
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # STEP 1: Find bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # STEP 2: Find ALL consecutive base candles after leg-in
                base_sequence = self.find_all_consecutive_base_candles(data, leg_in['end_idx'] + 1)
                if not base_sequence:
                    continue
                
                # STEP 3: Find bullish leg-out immediately after base sequence (REVERSAL)
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bullish')
                if not leg_out:
                    continue
                
                # STEP 4: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # STEP 5: Create perfect D-B-R pattern
                pattern = self.create_pattern(
                    pattern_type='D-B-R',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def detect_rbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect R-B-D patterns: Bullish LEG-IN → ALL BASE CANDLES → Bearish LEG-OUT (Reversal)
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # STEP 1: Find bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # STEP 2: Find ALL consecutive base candles after leg-in
                base_sequence = self.find_all_consecutive_base_candles(data, leg_in['end_idx'] + 1)
                if not base_sequence:
                    continue
                
                # STEP 3: Find bearish leg-out immediately after base sequence (REVERSAL)
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bearish')
                if not leg_out:
                    continue
                
                # STEP 4: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # STEP 5: Create perfect R-B-D pattern
                pattern = self.create_pattern(
                    pattern_type='R-B-D',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def find_all_consecutive_base_candles(self, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """
        CRITICAL METHOD: Find ALL consecutive base candles (≤50% body ratio)
        This is the heart of correct zone boundary calculation
        
        Args:
            data: Full dataset
            start_idx: Starting index to search from
            
        Returns:
            Dictionary with all consecutive base candle information
        """
        if start_idx >= len(data):
            return None
        
        consecutive_base_indices = []
        
        # Scan forward for consecutive base candles
        for i in range(start_idx, min(start_idx + self.config['max_base_candles'], len(data))):
            candle = data.iloc[i]
            
            # Check if candle is base (≤50% body ratio)
            classification = self.candle_classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            if classification == 'base':
                consecutive_base_indices.append(i)
            else:
                # First non-base candle breaks the sequence
                break
        
        # Validate minimum requirements
        if len(consecutive_base_indices) < self.config['min_base_candles']:
            return None
        
        # Calculate boundaries from ALL consecutive base candles
        base_data = data.iloc[consecutive_base_indices]
        base_high = base_data['high'].max()
        base_low = base_data['low'].min()
        base_range = base_high - base_low
        
        # Minimum base range requirement
        if base_range < 10 * self.config['pip_value']:  # At least 10 pips
            return None
        
        return {
            'start_idx': consecutive_base_indices[0],
            'end_idx': consecutive_base_indices[-1],
            'high': base_high,
            'low': base_low,
            'range': base_range,
            'candle_count': len(consecutive_base_indices),
            'candle_indices': consecutive_base_indices,
            'quality_score': self.calculate_base_quality(len(consecutive_base_indices))
        }
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """
        Identify leg-in with flexible requirements
        
        Args:
            data: Full dataset
            start_idx: Starting index for search
            direction: 'bullish' or 'bearish'
            
        Returns:
            Leg-in dictionary or None if invalid
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
                min_range = 20 * self.config['pip_value']
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
    
    def identify_leg_out(self, data: pd.DataFrame, start_idx: int, 
                        base_sequence: Dict, direction: str) -> Optional[Dict]:
        """
        Identify leg-out that breaks out of ALL base candles with sufficient distance
        
        Args:
            data: Full dataset
            start_idx: Starting index for leg-out search
            base_sequence: Complete base sequence information
            direction: 'bullish' or 'bearish'
            
        Returns:
            Leg-out dictionary or None if invalid
        """
        if start_idx >= len(data):
            return None
            
        base_high = base_sequence['high']
        base_low = base_sequence['low']
        base_range = base_sequence['range']
        
        for leg_length in range(1, 4):  # Try 1, 2, or 3 candles
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            # Check if leg is valid and breaks out of ALL base candles
            if self.is_valid_leg(leg_data, direction):
                leg_range = leg_data['high'].max() - leg_data['low'].min()
                
                # Check breakout from complete base sequence
                if direction == 'bullish':
                    leg_high = leg_data['high'].max()
                    if leg_high <= base_high:
                        continue  # Didn't break out of full base range
                else:  # bearish
                    leg_low = leg_data['low'].min()
                    if leg_low >= base_low:
                        continue  # Didn't break out of full base range
                
                # Calculate ratio to complete base range
                ratio_to_base = leg_range / base_range if base_range > 0 else 0
                
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': direction,
                    'range': leg_range,
                    'ratio_to_base': ratio_to_base,
                    'strength': self.calculate_leg_strength(leg_data, direction),
                    'candle_count': leg_length
                }
        
        return None
    
    def create_pattern(self, pattern_type: str, leg_in: Dict, 
                      base_sequence: Dict, leg_out: Dict) -> Dict:
        """
        Create complete pattern with correct zone boundaries
        
        Args:
            pattern_type: 'D-B-D', 'R-B-R', 'D-B-R', or 'R-B-D'
            leg_in: Leg-in information
            base_sequence: Complete base sequence (CORRECT BOUNDARIES)
            leg_out: Leg-out information
            
        Returns:
            Complete pattern dictionary
        """
        # Zone boundaries = ALL consecutive base candles (CORRECT!)
        zone_high = base_sequence['high']
        zone_low = base_sequence['low']
        zone_range = base_sequence['range']
        
        return {
            'type': pattern_type,
            'start_idx': leg_in['start_idx'],
            'end_idx': leg_out['end_idx'],
            
            # Pattern structure
            'leg_in': leg_in,
            'base': {
                'start_idx': base_sequence['start_idx'],
                'end_idx': base_sequence['end_idx'],
                'high': base_sequence['high'],
                'low': base_sequence['low'],
                'range': base_sequence['range'],
                'candle_count': base_sequence['candle_count'],
                'quality_score': base_sequence['quality_score']
            },
            'leg_out': leg_out,
            
            # Zone boundaries (FROM ALL BASE CANDLES)
            'zone_high': zone_high,
            'zone_low': zone_low,
            'zone_range': zone_range,
            
            # Pattern quality
            'strength': self.calculate_pattern_strength(leg_in, base_sequence, leg_out)
        }
    
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
    
    def calculate_base_quality(self, candle_count: int) -> float:
        """Calculate base quality score (favor shorter bases)"""
        if candle_count == 1:
            return 1.0
        elif candle_count == 2:
            return 0.9
        elif candle_count == 3:
            return 0.7
        else:
            return 0.5
    
    def calculate_pattern_strength(self, leg_in: Dict, base_sequence: Dict, leg_out: Dict) -> float:
        """Calculate overall pattern strength"""
        leg_in_strength = leg_in['strength']
        base_quality = base_sequence['quality_score']
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