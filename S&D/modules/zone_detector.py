"""
Zone Detection Engine - Module 2 (COMPLETE CLEAN VERSION)
Detects all 4 pattern types: D-B-D, R-B-R, D-B-R, R-B-D
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
    Complete zone detection system for all 4 pattern types
    - D-B-D (momentum): Bearish → Base → Bearish
    - R-B-R (momentum): Bullish → Base → Bullish  
    - D-B-R (reversal): Bearish → Base → Bullish
    - R-B-D (reversal): Bullish → Base → Bearish
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
    
    def detect_dbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect D-B-D patterns: Bearish LEG-IN → BASE → Bearish LEG-OUT
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
                
                # STEP 5: Create D-B-D pattern
                pattern = self.create_pattern(
                    pattern_type='D-B-D',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                pattern['category'] = 'momentum'
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def detect_rbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect R-B-R patterns: Bullish LEG-IN → BASE → Bullish LEG-OUT
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
                
                # STEP 3: Find bullish leg-out immediately after base sequence
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bullish')
                if not leg_out:
                    continue
                
                # STEP 4: Validate minimum distance requirement
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # STEP 5: Create R-B-R pattern
                pattern = self.create_pattern(
                    pattern_type='R-B-R',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                pattern['category'] = 'momentum'
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def detect_dbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect D-B-R patterns: Bearish LEG-IN → BASE → Bullish LEG-OUT"""
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Find bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # Find base sequence
                base_sequence = self.find_all_consecutive_base_candles(data, leg_in['end_idx'] + 1)
                if not base_sequence:
                    continue
                
                # Find bullish leg-out (opposite direction)
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bullish')
                if not leg_out:
                    continue
                
                # Validate minimum distance
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # Create D-B-R pattern
                pattern = self.create_pattern(
                    pattern_type='D-B-R',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                pattern['category'] = 'reversal'
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns

    def detect_rbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect R-B-D patterns: Bullish LEG-IN → BASE → Bearish LEG-OUT"""
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Find bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # Find base sequence
                base_sequence = self.find_all_consecutive_base_candles(data, leg_in['end_idx'] + 1)
                if not base_sequence:
                    continue
                
                # Find bearish leg-out (opposite direction)
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bearish')
                if not leg_out:
                    continue
                
                # Validate minimum distance
                if leg_out['ratio_to_base'] < self.config['min_legout_ratio']:
                    continue
                
                # Create R-B-D pattern
                pattern = self.create_pattern(
                    pattern_type='R-B-D',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out
                )
                pattern['category'] = 'reversal'
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """Identify leg-in movement"""
        if start_idx >= len(data) - 2:
            return None
        
        for leg_length in range(1, 4):  # Try 1, 2, or 3 candles
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            if self.is_valid_leg(leg_data, direction):
                leg_range = leg_data['high'].max() - leg_data['low'].min()
                
                return {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'direction': direction,
                'range': leg_range,
                'candle_count': leg_length
                }
        
        return None
    
    def find_all_consecutive_base_candles(self, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """
        Find ALL consecutive base candles (≤50% body ratio)
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
            'candle_count': len(consecutive_base_indices)
        }
    
    def identify_leg_out(self, data: pd.DataFrame, start_idx: int, 
                        base_sequence: Dict, direction: str) -> Optional[Dict]:
        """Identify leg-out movement with proper breakout validation"""
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
            
            # Check if leg is valid and breaks out of base
            if self.is_valid_leg(leg_data, direction):
                leg_range = leg_data['high'].max() - leg_data['low'].min()
                
                # Check breakout from base
                if direction == 'bullish':
                    leg_high = leg_data['high'].max()
                    if leg_high <= base_high:
                        continue  # Didn't break out
                else:  # bearish
                    leg_low = leg_data['low'].min()
                    if leg_low >= base_low:
                        continue  # Didn't break out
                
                # Calculate ratio to base range
                ratio_to_base = leg_range / base_range if base_range > 0 else 0
                
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': direction,
                    'range': leg_range,
                    'ratio_to_base': ratio_to_base,
                    'candle_count': leg_length
                }
        
        return None
    
    def create_pattern(self, pattern_type: str, leg_in: Dict, 
                  base_sequence: Dict, leg_out: Dict) -> Dict:
        """Create complete pattern with zone boundaries"""
        # Zone boundaries = base candle boundaries
        zone_high = base_sequence['high']
        zone_low = base_sequence['low']
        zone_range = zone_high - zone_low
        
        return {
            'type': pattern_type,
            'start_idx': leg_in['start_idx'],
            'end_idx': leg_out['end_idx'],
            'leg_in': leg_in,
            'base': base_sequence,
            'leg_out': leg_out,
            'zone_high': zone_high,
            'zone_low': zone_low,
            'zone_range': zone_range,
            'formation_date': None  # Will be set by caller if needed
        }
    
    def is_valid_leg(self, leg_data: pd.DataFrame, direction: str) -> bool:
        """Check if leg movement is valid"""
        if len(leg_data) == 0:
            return False
        
        # Check directional movement
        start_price = leg_data.iloc[0]['open']
        end_price = leg_data.iloc[-1]['close']
        
        if direction == 'bullish':
            return end_price > start_price
        else:  # bearish
            return end_price < start_price
    
    
    def check_zone_testing(self, zone: Dict, data: pd.DataFrame, 
                          evaluation_date: Optional[pd.Timestamp] = None) -> Tuple[bool, str]:
        """
        Check if zone was valid at specific point in time with CORRECT approach direction logic
        
        Args:
            zone: Zone dictionary with zone boundaries and formation indices
            data: Full OHLC DataFrame with date index
            evaluation_date: Specific date to check validity up to. If None, uses current moment
            
        Returns:
            Tuple of (is_valid: bool, reason: str)
            - is_valid: True if zone was untested up to evaluation_date
            - reason: Explanation of validation result
            
        CORRECT LOGIC:
        - Demand zones: Only test when price approaches FROM BELOW and penetrates UP
        - Supply zones: Only test when price approaches FROM ABOVE and penetrates DOWN
        """
        try:
            zone_end_idx = zone['end_idx']
            zone_high = zone['zone_high']
            zone_low = zone['zone_low']
            zone_size = zone_high - zone_low
            zone_type = zone['type']
            
            # Edge case: Zero zone size
            if zone_size <= 0:
                return False, "Invalid zone - zero or negative size"
            
            # Determine evaluation cutoff point
            if evaluation_date is None:
                end_check_idx = len(data) - 1
                cutoff_date = data.index[-1]
            else:
                try:
                    end_check_idx = data.index.get_loc(evaluation_date)
                    cutoff_date = evaluation_date
                except KeyError:
                    return False, f"Evaluation date {evaluation_date} not found in data"
            
            # Edge case: No data after zone formation to check
            if zone_end_idx >= end_check_idx:
                return True, "Zone untested - no data between formation and evaluation point"
            
            # Check candles from zone end to evaluation cutoff
            start_check_idx = zone_end_idx + 1
            candles_to_check = data.iloc[start_check_idx:end_check_idx + 1]
            
            if len(candles_to_check) == 0:
                return True, "Zone untested - no candles in evaluation window"
            
            # CORRECTED LOGIC: Check approach direction before applying penetration rules
            for i, (date_idx, candle) in enumerate(candles_to_check.iterrows()):
                
                if zone_type in ['R-B-R', 'D-B-R']:  # Demand zones (expect bullish approach)
                    
                    # DEMAND ZONE LOGIC: Only test if price is approaching from below
                    # Price must be in or near the zone to trigger testing
                    
                    # Check if price is approaching the zone from below (bullish approach)
                    is_approaching_from_below = (
                        candle['low'] <= zone_high and  # Price reached zone level
                        candle['close'] >= zone_low     # Close is at or above zone bottom
                    )
                    
                    if is_approaching_from_below:
                        # Rule 1: 33% close penetration from TOP of zone
                        close_test_level = zone_high - (zone_size * 0.33)
                        if candle['close'] < close_test_level:
                            return False, f"Demand zone tested on {date_idx.strftime('%Y-%m-%d')} - close {candle['close']:.5f} below 33% level {close_test_level:.5f}"
                        
                        # Rule 2: 50% wick penetration from TOP of zone
                        wick_test_level = zone_high - (zone_size * 0.50)
                        if candle['low'] < wick_test_level:
                            return False, f"Demand zone deeply penetrated on {date_idx.strftime('%Y-%m-%d')} - low {candle['low']:.5f} below 50% level {wick_test_level:.5f}"
                    
                    # If price is completely below zone, ignore (not approaching)
                    # This fixes the bug where low prices were invalidating demand zones
                        
                elif zone_type in ['D-B-D', 'R-B-D']:  # Supply zones (expect bearish approach)
                    
                    # SUPPLY ZONE LOGIC: Only test if price is approaching from above
                    # Price must be in or near the zone to trigger testing
                    
                    # Check if price is approaching the zone from above (bearish approach)
                    is_approaching_from_above = (
                        candle['high'] >= zone_low and   # Price reached zone level
                        candle['close'] <= zone_high     # Close is at or below zone top
                    )
                    
                    if is_approaching_from_above:
                        # Rule 1: 33% close penetration from BOTTOM of zone
                        close_test_level = zone_low + (zone_size * 0.33)
                        if candle['close'] > close_test_level:
                            return False, f"Supply zone tested on {date_idx.strftime('%Y-%m-%d')} - close {candle['close']:.5f} above 33% level {close_test_level:.5f}"
                        
                        # Rule 2: 50% wick penetration from BOTTOM of zone
                        wick_test_level = zone_low + (zone_size * 0.50)
                        if candle['high'] > wick_test_level:
                            return False, f"Supply zone deeply penetrated on {date_idx.strftime('%Y-%m-%d')} - high {candle['high']:.5f} above 50% level {wick_test_level:.5f}"
                    
                    # If price is completely above zone, ignore (not approaching)
                
                else:
                    return False, f"Unknown zone type: {zone_type}"
            
            # Zone was valid throughout the evaluation period
            return True, f"Zone untested from formation to {cutoff_date.strftime('%Y-%m-%d')} ({len(candles_to_check)} candles checked)"
            
        except KeyError as e:
            return False, f"Missing zone data: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error in zone testing validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(data) < 10:
            raise ValueError("Insufficient data for pattern detection")