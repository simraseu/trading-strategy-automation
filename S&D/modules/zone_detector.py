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
        'min_legout_ratio': 0.5,
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
        """Initialize zone detector with uniqueness tracking"""
        self.candle_classifier = candle_classifier
        self.config = config or ZONE_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # CRITICAL FIX: Track used base sequences to prevent overlapping zones
        self.used_base_sequences = set()  # Track base sequence IDs
        self.created_zones = []  # Track all created zones for validation
        
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        FIXED: Detect all zone patterns with uniqueness validation and 2.5x target monitoring
        CRITICAL FIX: Added validation reporting and 2.5x target validation
        """
        try:
            # Validate input data
            self.validate_data(data)
            
            # CRITICAL FIX: Reset tracking for new dataset
            self.reset_zone_tracking()
            
            # Detect all 4 pattern types with uniqueness enforcement
            dbd_patterns = self.detect_dbd_patterns(data)
            rbr_patterns = self.detect_rbr_patterns(data)
            dbr_patterns = self.detect_dbr_patterns(data)
            rbd_patterns = self.detect_rbd_patterns(data)
            
            total_patterns = len(dbd_patterns) + len(rbr_patterns) + len(dbr_patterns) + len(rbd_patterns)
            
            print(f"✅ Zone detection complete:")
            print(f"   D-B-D: {len(dbd_patterns)}, R-B-R: {len(rbr_patterns)}, D-B-R: {len(dbr_patterns)}, R-B-D: {len(rbd_patterns)}")
            
            # MANDATORY: Filter zones to only return 2.5x validated ones
            if total_patterns > 0:
                print(f"🎯 Validating {total_patterns} zones for 2.5x achievement...")
                
                all_patterns = dbd_patterns + rbr_patterns + dbr_patterns + rbd_patterns
                validated_patterns = []
                trading_ready_patterns = []
                
                for pattern in all_patterns:
                    validated_pattern = self.validate_zone_2_5x_target(pattern, data)
                    validated_patterns.append(validated_pattern)
                    
                    # STRICT FILTER: Only zones that achieved 2.5x target
                    if validated_pattern.get('target_2_5x_hit') == True:
                        trading_ready_patterns.append(validated_pattern)
                
                # Split ONLY validated patterns into types
                validated_dbd = [p for p in trading_ready_patterns if p['type'] == 'D-B-D']
                validated_rbr = [p for p in trading_ready_patterns if p['type'] == 'R-B-R']
                validated_dbr = [p for p in trading_ready_patterns if p['type'] == 'D-B-R']
                validated_rbd = [p for p in trading_ready_patterns if p['type'] == 'R-B-D']
                
                # Report validation statistics
                validated_count = len(trading_ready_patterns)
                total_hit_target = len([p for p in validated_patterns if p['target_2_5x_hit']])
                
                print(f"   ✅ Valid zones for trading: {validated_count}/{total_patterns}")

                return {
                    'dbd_patterns': validated_dbd,
                    'rbr_patterns': validated_rbr,
                    'dbr_patterns': validated_dbr,
                    'rbd_patterns': validated_rbd,
                    'total_patterns': validated_count,  # Only return validated count
                    'validated_zones': total_hit_target,
                    'invalidated_zones': len(validated_patterns) - total_hit_target,
                    'pending_zones': 0
                }
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def detect_dbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        FIXED: Detect D-B-D patterns with uniqueness enforcement
        CRITICAL FIX: Each base sequence can only create ONE zone
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
                
                # CORRECTED FIX: Only check base sequence reuse (the original working approach)
                base_sequence_id = base_sequence['base_sequence_id']
                if base_sequence_id in self.used_base_sequences:
                    continue  # Skip - this base sequence already created a zone
                
                # STEP 3: Find bearish leg-out immediately after base sequence
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bearish')
                if not leg_out:
                    continue
                
                # STEP 5: Create D-B-D pattern and mark base sequence as used
                pattern = self.create_pattern(
                    pattern_type='D-B-D',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out,
                    data=data
                )
                pattern['category'] = 'momentum'
                
                # CRITICAL FIX: Mark this base sequence as used
                self.used_base_sequences.add(base_sequence_id)
                self.created_zones.append(pattern)
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
                
                # CORRECTED FIX: Only check base sequence reuse (the original working approach)
                base_sequence_id = base_sequence['base_sequence_id']
                if base_sequence_id in self.used_base_sequences:
                    continue  # Skip - this base sequence already created a zone
                
                # STEP 3: Find bullish leg-out immediately after base sequence
                leg_out_start = base_sequence['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base_sequence, 'bullish')
                if not leg_out:
                    continue
                
                # STEP 5: Create R-B-R pattern and mark base sequence as used
                pattern = self.create_pattern(
                    pattern_type='R-B-R',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out,
                    data=data
                )
                pattern['category'] = 'momentum'
                
                # CRITICAL FIX: Mark this base sequence as used
                self.used_base_sequences.add(base_sequence_id)
                self.created_zones.append(pattern)
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
                
                # CORRECTED FIX: Only check base sequence reuse (the original working approach)
                base_sequence_id = base_sequence['base_sequence_id']
                if base_sequence_id in self.used_base_sequences:
                    continue  # Skip - this base sequence already created a zone
                
                # STEP 5: Create D-B-R pattern and mark base sequence as used
                pattern = self.create_pattern(
                    pattern_type='D-B-R',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out,
                    data=data
                )
                pattern['category'] = 'reversal'
                
                # CRITICAL FIX: Mark this base sequence as used
                self.used_base_sequences.add(base_sequence_id)
                self.created_zones.append(pattern)
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
                
                # CORRECTED FIX: Only check base sequence reuse (the original working approach)
                base_sequence_id = base_sequence['base_sequence_id']
                if base_sequence_id in self.used_base_sequences:
                    continue  # Skip - this base sequence already created a zone
                
                # STEP 5: Create R-B-D pattern and mark base sequence as used
                pattern = self.create_pattern(
                    pattern_type='R-B-D',
                    leg_in=leg_in,
                    base_sequence=base_sequence,
                    leg_out=leg_out,
                    data=data
                )
                pattern['category'] = 'reversal'
                
                # CRITICAL FIX: Mark this base sequence as used
                self.used_base_sequences.add(base_sequence_id)
                self.created_zones.append(pattern)
                patterns.append(pattern)
                
            except Exception as e:
                continue
        
        return patterns
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """Identify leg-in movement with STRICT single-candle validation"""
        if start_idx >= len(data) - 2:
            return None
        
        # CORRECTED: Only check single candle for leg-in (most common pattern)
        # Multi-candle leg-ins are rare and often misclassified
        
        first_candle = data.iloc[start_idx]
        body_size = abs(first_candle['close'] - first_candle['open'])
        total_range = first_candle['high'] - first_candle['low']
        
        if total_range == 0:
            body_ratio = 0
        else:
            body_ratio = body_size / total_range
        
        # MANDATORY: Leg-in candle must be decisive or explosive (>50% body ratio)
        if body_ratio <= 0.50:
            return None  # Not a valid leg-in
        
        # Check directional validity
        is_bullish = first_candle['close'] > first_candle['open']
        is_bearish = first_candle['close'] < first_candle['open']
        
        if direction == 'bullish' and not is_bullish:
            return None
        elif direction == 'bearish' and not is_bearish:
            return None
        
        leg_range = first_candle['high'] - first_candle['low']
        
        return {
            'start_idx': start_idx,
            'end_idx': start_idx,  # Single candle
            'direction': direction,
            'range': leg_range,
            'candle_count': 1,  # Always 1 for corrected logic
            'first_candle_body_ratio': body_ratio
        }
    
    def find_all_consecutive_base_candles(self, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """
        FIXED: Find ALL consecutive base candles with STRICT validation
        CRITICAL FIX: Prevents multiple zones from same base sequence
        """
        if start_idx >= len(data):
            return None
        
        consecutive_base_indices = []
        
        # Scan forward for ALL consecutive base candles - no early stopping
        for i in range(start_idx, min(start_idx + self.config['max_base_candles'], len(data))):
            candle = data.iloc[i]
            
            # STRICT base candle validation - must be ≤50% body ratio
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            # Handle zero range edge case
            if total_range == 0:
                body_ratio = 0
            else:
                body_ratio = body_size / total_range
            
            # CRITICAL FIX: Strict 50% threshold enforcement
            if body_ratio <= 0.50:  # Base candle confirmed
                consecutive_base_indices.append(i)
            else:
                # First non-base candle breaks the sequence
                break
        
        # CRITICAL FIX: Validate minimum requirements
        if len(consecutive_base_indices) < self.config['min_base_candles']:
            return None
        
        # CRITICAL FIX: ALWAYS return the COMPLETE base sequence
        # This prevents multiple zones from subsets of the same base sequence
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
            'base_sequence_id': f"{consecutive_base_indices[0]}_{consecutive_base_indices[-1]}"  # UNIQUE ID
        }
    
    def identify_leg_out(self, data: pd.DataFrame, start_idx: int, 
                    base_sequence: Dict, direction: str) -> Optional[Dict]:
        """
        FIXED: Identify leg-out with STRICT validation AND pattern invalidation check
        CRITICAL FIX: Detect invalidating candles that break pattern integrity
        """
        if start_idx >= len(data):
            return None
        
        base_high = base_sequence['high']
        base_low = base_sequence['low']
        base_range = base_sequence['range']
        base_end_idx = base_sequence['end_idx']
        
        # CRITICAL FIX: Check for pattern invalidation BEFORE leg-out detection
        # Scan candles between base end and potential leg-out start
        # CRITICAL FIX: Check for pattern invalidation IMMEDIATELY after base sequence
        # ANY decisive/explosive candle after base sequence invalidates the pattern
        invalidation_check_idx = base_end_idx + 1
        
        # Check the FIRST candle after base sequence for invalidation
        if invalidation_check_idx < len(data):
            invalidation_candle = data.iloc[invalidation_check_idx]
            
            # Calculate body ratio for invalidation candle
            body_size = abs(invalidation_candle['close'] - invalidation_candle['open'])
            total_range = invalidation_candle['high'] - invalidation_candle['low']
            
            if total_range == 0:
                body_ratio = 0
            else:
                body_ratio = body_size / total_range
            
            # If candle is decisive/explosive (>50% body ratio), check for invalidation
            if body_ratio > 0.50:
                # Check if this candle moves opposite to expected leg-out direction
                is_bullish_candle = invalidation_candle['close'] > invalidation_candle['open']
                is_bearish_candle = invalidation_candle['close'] < invalidation_candle['open']
                
                # INVALIDATION RULES:
                if direction == 'bullish' and is_bearish_candle:
                    return None  # Bearish decisive candle invalidates bullish leg-out expectation
                elif direction == 'bearish' and is_bullish_candle:
                    return None  # Bullish decisive candle invalidates bearish leg-out expectation
        
        # If we reach here, no invalidation detected - proceed with leg-out detection
        # But ONLY accept leg-out if it starts immediately after base (no gaps allowed)
        if start_idx != base_end_idx + 1:
            return None  # Gap between base and leg-out not allowed
        
        # CORRECTED: Only check single candle for leg-out (most reliable pattern)
        if start_idx >= len(data):
            return None
            
        leg_data = data.iloc[start_idx:start_idx + 1]  # Single candle only
        end_idx = start_idx
        
        # CRITICAL FIX: STRICT first candle validation
        first_candle = data.iloc[start_idx]
        
        # Use direct body ratio calculation for consistency
        body_size = abs(first_candle['close'] - first_candle['open'])
        total_range = first_candle['high'] - first_candle['low']
        
        if total_range == 0:
            body_ratio = 0
        else:
            body_ratio = body_size / total_range
        
        # MANDATORY: First leg-out candle MUST NOT be base (>50% body ratio)
        if body_ratio <= 0.50:
            return None  # Invalid leg-out starting with base candle
        
        # Check if leg is valid and breaks out of base
        if self.is_valid_leg(leg_data, direction):
            leg_range = leg_data['high'].max() - leg_data['low'].min()
            
            # Check breakout from base
            if direction == 'bullish':
                leg_high = leg_data['high'].max()
                if leg_high <= base_high:
                    return None  # Didn't break out
            else:  # bearish
                leg_low = leg_data['low'].min()
                if leg_low >= base_low:
                    return None  # Didn't break out
            
            # Calculate ratio to base range
            ratio_to_base = leg_range / base_range if base_range > 0 else 0
            
            # Determine first candle classification for tracking
            if body_ratio <= 0.50:
                first_candle_type = 'base'
            elif body_ratio > 0.80:
                first_candle_type = 'explosive'
            else:
                first_candle_type = 'decisive'
            
            return {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'direction': direction,
                'range': leg_range,
                'ratio_to_base': ratio_to_base,
                'candle_count': 1,  # Always 1 for single candle
                'first_candle_type': first_candle_type
            }
        
        return None
    
    def create_pattern(self, pattern_type: str, leg_in: Dict, 
                  base_sequence: Dict, leg_out: Dict, data: pd.DataFrame) -> Dict:
        """Create complete pattern with CORRECT zone boundaries + leg-out wick extensions"""
        
        # Start with base candle boundaries
        zone_high = base_sequence['high']
        zone_low = base_sequence['low']
        
        # Get leg-out data for potential zone extension
        leg_out_data = data.iloc[leg_out['start_idx']:leg_out['end_idx'] + 1]
        leg_out_high = leg_out_data['high'].max()
        leg_out_low = leg_out_data['low'].min()
        
        # CORRECTED LOGIC: Only extend the "approach side" of zones
        if pattern_type in ['D-B-D', 'R-B-D']:  # Supply zones (price approaches from ABOVE)
            # Supply zones: ONLY extend zone_high if leg-out wick goes higher
            # Do NOT extend zone_low - keep base_low as zone_low
            if leg_out_high > zone_high:
                zone_high = leg_out_high
            # zone_low stays as base_low (no extension downward for supply zones)
                
        elif pattern_type in ['R-B-R', 'D-B-R']:  # Demand zones (price approaches from BELOW)
            # Demand zones: ONLY extend zone_low if leg-out wick goes lower
            # Do NOT extend zone_high - keep base_high as zone_high  
            if leg_out_low < zone_low:
                zone_low = leg_out_low
            # zone_high stays as base_high (no extension upward for demand zones)
        
        zone_range = zone_high - zone_low
        
        return {
            'type': pattern_type,
            'start_idx': leg_in['start_idx'],
            'end_idx': leg_out['end_idx'],
            'leg_in': leg_in,
            'base': base_sequence,
            'leg_out': leg_out,
            'zone_high': zone_high,  # Extended if leg-out exceeds base
            'zone_low': zone_low,    # Extended if leg-out exceeds base
            'zone_range': zone_range,
            'leg_out_high': leg_out_high,  # Track for analysis
            'leg_out_low': leg_out_low,    # Track for analysis
            'base_high': base_sequence['high'],  # Track original base boundaries
            'base_low': base_sequence['low'],    # Track original base boundaries
            'extended': leg_out_high > base_sequence['high'] or leg_out_low < base_sequence['low'],
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
                      evaluation_date: Optional[int] = None) -> Tuple[bool, str]:
        """
        WICK-BASED ZONE INVALIDATION - 50% penetration rule (NO CLOSE prices)
        
        Args:
            zone: Zone dictionary with zone boundaries and formation indices
            data: Full OHLC DataFrame with date index
            evaluation_date: Specific date to check validity up to. If None, uses current moment
            
        Returns:
            Tuple of (is_valid: bool, reason: str)
            - is_valid: True if zone was untested up to evaluation_date
            - reason: Explanation of validation result
            
        WICK-BASED LOGIC:
        - Demand zones: Invalidated if WICK penetrates 50% below zone_low
        - Supply zones: Invalidated if WICK penetrates 50% above zone_high
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
                cutoff_idx = data.index[-1]
            else:
                if evaluation_date in data.index:
                    end_check_idx = data.index.get_loc(evaluation_date)
                    cutoff_idx = evaluation_date
                else:
                    return False, f"Evaluation index {evaluation_date} not found in data"
            
            # Edge case: No data after zone formation to check
            if zone_end_idx >= end_check_idx:
                return True, "Zone untested - no data between formation and evaluation point"
            
            # Check candles from zone end to evaluation cutoff
            start_check_idx = zone_end_idx + 1
            candles_to_check = data.iloc[start_check_idx:end_check_idx + 1]

            if len(candles_to_check) == 0:
                return True, "Zone untested - no candles in evaluation window"

            # WICK-BASED INVALIDATION LOGIC (50% penetration)
            for i, candle in candles_to_check.iterrows():
                candle_idx = i  # Integer index
                
                if zone_type in ['R-B-R', 'D-B-R']:  # Demand zones (bullish leg-out)
                    # Invalidated if LOW WICK penetrates 50% below zone_low
                    invalidation_level = zone_low - (zone_size * 0.50)
                    if candle['low'] < invalidation_level:
                        return False, f"Demand zone invalidated at index {candle_idx} - wick penetrated 50% level at {invalidation_level:.5f} (low: {candle['low']:.5f})"
                        
                elif zone_type in ['D-B-D', 'R-B-D']:  # Supply zones (bearish leg-out)
                    # Invalidated if HIGH WICK penetrates 50% above zone_high
                    invalidation_level = zone_high + (zone_size * 0.50)
                    if candle['high'] > invalidation_level:
                        return False, f"Supply zone invalidated at index {candle_idx} - wick penetrated 50% level at {invalidation_level:.5f} (high: {candle['high']:.5f})"
                
                else:
                    return False, f"Unknown zone type: {zone_type}"
            
            # Zone was valid throughout the evaluation period
            return True, f"Zone untested from formation to index {cutoff_idx} ({len(candles_to_check)} candles checked)"
            
        except KeyError as e:
            return False, f"Missing zone data: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error in zone testing validation: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def validate_zone_2_5x_target(self, zone: Dict, data: pd.DataFrame) -> Dict:
        """
        Monitor price action after zone formation to validate 2.5x target
        
        Args:
            zone: Zone dictionary with formation data
            data: Full OHLC DataFrame with date index
            
        Returns:
            Updated zone with validation status and monitoring data
        """
        try:
            zone_high = zone['zone_high']
            zone_low = zone['zone_low'] 
            zone_range = zone_high - zone_low
            zone_type = zone['type']
            leg_out_end_idx = zone['leg_out']['end_idx']
            
            # Edge case: Zero zone range
            if zone_range <= 0:
                zone.update({
                    'immediate_leg_out_ratio': zone['leg_out']['ratio_to_base'],
                    'maximum_distance_ratio': zone['leg_out']['ratio_to_base'],
                    'target_2_5x_price': None,
                    'target_2_5x_hit': False,
                    'target_2_5x_date': None,
                    'zone_validation_status': 'INVALID_RANGE',
                    'invalidation_date': None,
                    'monitoring_candles_count': 0
                })
                return zone
            
            # Calculate 2.5x targets
            if zone_type in ['D-B-D', 'R-B-D']:  # Supply zones (bearish leg-out)
                target_2_5x = zone_low - (2.5 * zone_range)
                direction = 'bearish'
            else:  # Demand zones (bullish leg-out)
                target_2_5x = zone_high + (2.5 * zone_range)
                direction = 'bullish'
            
            # Initialize monitoring variables
            max_ratio = zone['leg_out']['ratio_to_base']  # Start with immediate leg-out
            target_hit = False
            target_hit_date = None
            invalidation_date = None
            monitoring_count = 0
            
            # Monitor all candles after leg-out formation
            start_monitor_idx = leg_out_end_idx + 1
            
            for i in range(start_monitor_idx, len(data)):
                candle = data.iloc[i]
                candle_date = data.index[i]
                monitoring_count += 1
                
                # Calculate current distance ratio
                if direction == 'bearish':
                    current_distance = zone_low - candle['low']  # How far below zone_low
                    current_ratio = current_distance / zone_range
                    
                    # Check for 2.5x target hit
                    if candle['low'] <= target_2_5x and not target_hit:
                        target_hit = True
                        target_hit_date = candle_date
                        
                else:  # bullish
                    current_distance = candle['high'] - zone_high  # How far above zone_high
                    current_ratio = current_distance / zone_range
                    
                    # Check for 2.5x target hit
                    if candle['high'] >= target_2_5x and not target_hit:
                        target_hit = True
                        target_hit_date = candle_date
                
                # Update maximum ratio achieved
                if current_ratio > max_ratio:
                    max_ratio = current_ratio
                
                # Check for zone invalidation (price returns to zone before hitting 2.5x)
                if not target_hit:
                    zone_invalidated, invalidation_reason = self.check_zone_testing(zone, data, candle_date)
                    if not zone_invalidated:  # check_zone_testing returns False when zone is tested/invalidated
                        invalidation_date = candle_date
                        break
                
                # Stop monitoring if target hit
                if target_hit:
                    break
            
            # Determine final validation status
            if target_hit:
                validation_status = 'VALIDATED'
            elif invalidation_date:
                validation_status = 'INVALIDATED'
            else:
                validation_status = 'PENDING'  # Never hit target, never invalidated
            
            # Update zone with validation data including completion index
            validation_completion_idx = leg_out_end_idx + monitoring_count if target_hit else len(data) - 1
            
            zone.update({
                'immediate_leg_out_ratio': zone['leg_out']['ratio_to_base'],
                'maximum_distance_ratio': max_ratio,
                'target_2_5x_price': target_2_5x,
                'target_2_5x_hit': target_hit,
                'target_2_5x_date': target_hit_date,
                'zone_validation_status': validation_status,
                'invalidation_date': invalidation_date,
                'monitoring_candles_count': monitoring_count,
                'validation_completion_idx': validation_completion_idx  # When validation completed
            })
            
            return zone
            
        except Exception as e:
            self.logger.error(f"Error in 2.5x validation: {str(e)}")
            # Add error status to zone
            zone.update({
                'immediate_leg_out_ratio': zone['leg_out']['ratio_to_base'],
                'maximum_distance_ratio': zone['leg_out']['ratio_to_base'],
                'target_2_5x_price': None,
                'target_2_5x_hit': False,
                'target_2_5x_date': None,
                'zone_validation_status': 'ERROR',
                'invalidation_date': None,
                'monitoring_candles_count': 0
            })
            return zone
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(data) < 5:
            raise ValueError("Insufficient data for pattern detection")

    def reset_zone_tracking(self):
        """Reset tracking for new dataset"""
        self.used_base_sequences.clear()
        self.created_zones.clear()