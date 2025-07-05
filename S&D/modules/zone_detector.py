"""
Zone Detection Engine - Module 2 (ENHANCED PROFESSIONAL VERSION)
Enhanced D-B-D and R-B-R pattern detection with strict validation
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import sys
import os

# CRITICAL FIX: Add path to find config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with fallback
try:
    from config.settings import ZONE_CONFIG
except ImportError:
    # Fallback configuration if import fails
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
    Enhanced professional zone detection system for forex trading automation
    Focuses on momentum patterns: D-B-D and R-B-R with strict validation
    """
    
    def __init__(self, candle_classifier, config=None):
        """
        Initialize zone detector with candle classifier dependency
        
        Args:
            candle_classifier: Instance of CandleClassifier
            config: Configuration dictionary (optional)
        """
        self.candle_classifier = candle_classifier
        self.config = config or ZONE_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Enhanced configuration
        self.min_legout_ratio = 1.5  # STRICT 1.5x requirement
        self.invalidation_window = 5  # Check 5 candles after base
        
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Detect all zone patterns in the dataset with enhanced validation
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with 'dbd_patterns' and 'rbr_patterns' keys
        """
        try:
            # Validate input data
            self.validate_data(data)
            
            # Detect patterns with enhanced methods
            dbd_patterns = self.detect_dbd_patterns_enhanced(data)
            rbr_patterns = self.detect_rbr_patterns_enhanced(data)
            
            self.logger.info(f"Detected {len(dbd_patterns)} D-B-D patterns")
            self.logger.info(f"Detected {len(rbr_patterns)} R-B-R patterns")
            
            return {
                'dbd_patterns': dbd_patterns,
                'rbr_patterns': rbr_patterns,
                'total_patterns': len(dbd_patterns) + len(rbr_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            raise
    
    def calculate_actual_zone_boundaries(self, data: pd.DataFrame, base: Dict, 
                                   leg_out: Dict, pattern_type: str) -> Tuple[float, float]:
        """
        Calculate actual measurement boundaries matching manual trading approach
        R-B-R: From base wick-low to highest open/close within base (per candle direction)
        D-B-D: From base wick-high to lowest open/close within base (per candle direction)
        
        Args:
            data: Full dataset
            base: Base pattern information
            leg_out: Leg-out pattern information  
            pattern_type: 'R-B-R' or 'D-B-D'
            
        Returns:
            Tuple of (zone_high, zone_low)
        """
        base_data = data.iloc[base['start_idx']:base['end_idx'] + 1]
        
        if pattern_type == 'R-B-R':
            # R-B-R: From base wick-low to highest open/close per candle
            zone_low = base_data['low'].min()        # Base wick-low (unchanged)
            
            # For each candle, take the higher of open/close
            highest_points = []
            for idx in base_data.index:
                candle = base_data.loc[idx]
                if candle['close'] >= candle['open']:  # Bullish candle
                    highest_points.append(candle['close'])  # Take close
                else:  # Bearish candle  
                    highest_points.append(candle['open'])   # Take open
            
            zone_high = max(highest_points)  # Highest among all candle tops
            
        else:  # D-B-D
            # D-B-D: From base wick-high to lowest open/close per candle
            zone_high = base_data['high'].max()      # Base wick-high (unchanged)
            
            # For each candle, take the lower of open/close
            lowest_points = []
            for idx in base_data.index:
                candle = base_data.loc[idx]
                if candle['close'] <= candle['open']:  # Bearish candle
                    lowest_points.append(candle['close'])   # Take close
                else:  # Bullish candle
                    lowest_points.append(candle['open'])    # Take open
            
            zone_low = min(lowest_points)   # Lowest among all candle bottoms
        
        return zone_high, zone_low

    def validate_base_boundaries(self, leg_in: Dict, base: Dict, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        CRITICAL: Validate base stays within leg-in boundaries
        
        Args:
            leg_in: Leg-in pattern information
            base: Base pattern information
            data: Full dataset
            
        Returns:
            Tuple of (is_valid, message)
        """
        leg_in_start = leg_in['start_idx']
        leg_in_end = leg_in['end_idx']
        direction = leg_in['direction']
        
        # Get leg-in boundary levels
        leg_in_data = data.iloc[leg_in_start:leg_in_end + 1]
        
        if direction == 'bearish':  # D-B-D pattern
            leg_in_high = leg_in_data['high'].max()
            base_high = base['high']
            
            # Base cannot exceed leg-in high
            if base_high > leg_in_high:
                return False, f"Base high {base_high:.5f} exceeds leg-in high {leg_in_high:.5f}"
                
        elif direction == 'bullish':  # R-B-R pattern
            leg_in_low = leg_in_data['low'].min()
            base_low = base['low']
            
            # Base cannot exceed leg-in low
            if base_low < leg_in_low:
                return False, f"Base low {base_low:.5f} exceeds leg-in low {leg_in_low:.5f}"
        
        return True, "Valid base boundaries"

    def validate_leg_out_breakout_distance(self, data: pd.DataFrame, start_idx: int, 
                                        base: Dict, direction: str) -> Optional[Dict]:
        """
        SIMPLIFIED: Validate leg-out breaks base boundaries with strength
        Distance calculation happens in pattern creation
        
        Args:
            data: Full dataset
            start_idx: Starting index for leg-out
            base: Base pattern information
            direction: 'bullish' or 'bearish'
            
        Returns:
            Leg-out dictionary or None if invalid
        """
        if start_idx >= len(data):
            return None
            
        max_leg_length = 5
        
        # Base boundaries for breakout validation
        base_high = base['high']
        base_low = base['low']
        
        for leg_length in range(1, max_leg_length + 1):
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            # Check basic leg validity and strength requirements
            if (self.is_valid_leg_in(leg_data, direction) and 
                self.is_valid_leg_out_strength(leg_data, direction)):
                
                # Check zone breakout
                if direction == 'bullish':
                    leg_high = leg_data['high'].max()
                    if leg_high <= base_high:
                        continue  # Didn't break out of base
                        
                else:  # bearish
                    leg_low = leg_data['low'].min()
                    if leg_low >= base_low:
                        continue  # Didn't break out of base
                
                # Return successful leg-out
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': direction,
                    'range': self.calculate_leg_range(leg_data, direction),
                    'ratio_to_base': 0,  # Will be calculated in pattern creation
                    'strength': self.calculate_leg_strength(leg_data, direction),
                    'candle_count': leg_length
                }
        
        return None

    def detect_dbd_patterns_enhanced(self, data: pd.DataFrame) -> List[Dict]:
        """
        Enhanced D-B-D detection with base boundary validation and proper zone calculation
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of D-B-D pattern dictionaries
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Phase 1: Identify bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation (start AFTER leg-in)
                base_start = leg_in['end_idx'] + 1
                base = self.identify_base(data, base_start)
                if not base:
                    continue
                
                # Phase 2.5: CRITICAL - Validate base boundaries
                base_valid, base_msg = self.validate_base_boundaries(leg_in, base, data)
                if not base_valid:
                    self.logger.debug(f"D-B-D at {i} rejected: {base_msg}")
                    continue
                
                # Phase 3: Validate bearish leg-out with BREAKOUT distance requirement
                leg_out_start = base['end_idx'] + 1
                leg_out = self.validate_leg_out_breakout_distance(
                    data, leg_out_start, base, direction='bearish'
                )
                if not leg_out:
                    continue
                
                # Phase 4: Calculate actual measurement boundaries
                zone_high, zone_low = self.calculate_actual_zone_boundaries(
                    data, base, leg_out, 'D-B-D'
                )
                
                # Phase 5: Check for zone invalidation BEFORE finalizing
                pattern_preview = {
                    'type': 'D-B-D',
                    'base': base,
                    'zone_high': zone_high,
                    'zone_low': zone_low
                }
                
                is_valid, validation_msg = self.enhanced_zone_invalidation(
                    pattern_preview, data, base['end_idx'] + 1
                )
                if not is_valid:
                    self.logger.debug(f"D-B-D pattern at {i} rejected: {validation_msg}")
                    continue
                
                # Phase 6: Calculate leg-out ratio using actual zone range
                zone_range = zone_high - zone_low
                leg_out['ratio_to_base'] = zone_range / base['range'] if base['range'] > 0 else 0
                
                # Create final pattern with actual measurement boundaries
                pattern = {
                    'type': 'D-B-D',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': zone_high,      # ← ACTUAL measurement boundary
                    'zone_low': zone_low,        # ← ACTUAL measurement boundary
                    'zone_range': zone_range
                }
                
                # Calculate strength
                pattern['strength'] = self.calculate_pattern_strength(leg_in, base, leg_out)
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing D-B-D at index {i}: {str(e)}")
                continue
        
        return patterns

    def detect_rbr_patterns_enhanced(self, data: pd.DataFrame) -> List[Dict]:
        """
        Enhanced R-B-R detection with base boundary validation and proper zone calculation
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of R-B-R pattern dictionaries
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Phase 1: Identify bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation (start AFTER leg-in)
                base_start = leg_in['end_idx'] + 1
                base = self.identify_base(data, base_start)
                if not base:
                    continue
                
                # Phase 2.5: CRITICAL - Validate base boundaries
                base_valid, base_msg = self.validate_base_boundaries(leg_in, base, data)
                if not base_valid:
                    self.logger.debug(f"R-B-R at {i} rejected: {base_msg}")
                    continue
                
                # Phase 3: Validate bullish leg-out with BREAKOUT distance requirement
                leg_out_start = base['end_idx'] + 1
                leg_out = self.validate_leg_out_breakout_distance(
                    data, leg_out_start, base, direction='bullish'
                )
                if not leg_out:
                    continue
                
                # Phase 4: Calculate actual measurement boundaries
                zone_high, zone_low = self.calculate_actual_zone_boundaries(
                    data, base, leg_out, 'R-B-R'
                )
                
                # Phase 5: Check for zone invalidation BEFORE finalizing
                pattern_preview = {
                    'type': 'R-B-R',
                    'base': base,
                    'zone_high': zone_high,
                    'zone_low': zone_low
                }
                
                is_valid, validation_msg = self.enhanced_zone_invalidation(
                    pattern_preview, data, base['end_idx'] + 1
                )
                if not is_valid:
                    self.logger.debug(f"R-B-R pattern at {i} rejected: {validation_msg}")
                    continue
                
                # Phase 6: Calculate leg-out ratio using actual zone range
                zone_range = zone_high - zone_low
                leg_out['ratio_to_base'] = zone_range / base['range'] if base['range'] > 0 else 0
                
                # Create final pattern with actual measurement boundaries
                pattern = {
                    'type': 'R-B-R',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': zone_high,      # ← ACTUAL measurement boundary
                    'zone_low': zone_low,        # ← ACTUAL measurement boundary
                    'zone_range': zone_range
                }
                
                # Calculate strength
                pattern['strength'] = self.calculate_pattern_strength(leg_in, base, leg_out)
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing R-B-R at index {i}: {str(e)}")
                continue
        
        return patterns

    def enhanced_zone_invalidation(self, pattern: Dict, data: pd.DataFrame, start_check_idx: int) -> Tuple[bool, str]:
        """
        Enhanced invalidation - check for invalidating candles immediately after base
        
        Args:
            pattern: Pattern dictionary for validation
            data: Full dataset
            start_check_idx: Index to start checking from
            
        Returns:
            Tuple of (is_valid, message)
        """
        if start_check_idx >= len(data):
            return True, "Valid"
            
        pattern_type = pattern['type']
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        
        # Check next 3-5 candles immediately after base (before leg-out)
        invalidation_window = min(self.invalidation_window, len(data) - start_check_idx)
        
        if invalidation_window > 0:
            check_candles = data.iloc[start_check_idx:start_check_idx + invalidation_window]
            
            for relative_idx, (idx, candle) in enumerate(check_candles.iterrows()):
                actual_idx = start_check_idx + relative_idx
                
                # Get candle classification
                classification = self.candle_classifier.classify_single_candle(
                    candle['open'], candle['high'], candle['low'], candle['close']
                )
                
                # CRITICAL: Check for zone-breaking decisive/explosive candles
                if pattern_type == 'R-B-R':  # Demand zone
                    is_bearish = candle['close'] < candle['open']
                    
                    # Invalidated by decisive/explosive bearish candle that closes below zone
                    if (is_bearish and 
                        classification in ['decisive', 'explosive'] and
                        candle['close'] < zone_low):
                        return False, f"Zone invalidated by bearish {classification} at candle {actual_idx}"
                
                elif pattern_type == 'D-B-D':  # Supply zone
                    is_bullish = candle['close'] > candle['open']
                    
                    # Invalidated by decisive/explosive bullish candle that closes above zone
                    if (is_bullish and 
                        classification in ['decisive', 'explosive'] and
                        candle['close'] > zone_high):
                        return False, f"Zone invalidated by bullish {classification} at candle {actual_idx}"
        
        return True, "Valid"
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """
        Identify leg-in with flexible requirements (can be >50%)
        
        Args:
            data: Full dataset
            start_idx: Starting index for search
            direction: 'bullish' or 'bearish'
            
        Returns:
            Leg-in dictionary or None if invalid
        """
        if start_idx >= len(data):
            return None
            
        max_leg_length = 3
        
        for leg_length in range(1, max_leg_length + 1):
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            if self.is_valid_leg_in(leg_data, direction):
                leg_range = self.calculate_leg_range(leg_data, direction)
                
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
        Identify base consolidation - ALL candles must be ≤50% (base classification)
        
        Args:
            data: Full dataset
            start_idx: Starting index for search
            
        Returns:
            Base dictionary or None if invalid
        """
        if start_idx >= len(data):
            return None
            
        max_base_length = self.config['max_base_candles']
        
        for base_length in range(1, max_base_length + 1):
            end_idx = start_idx + base_length - 1
            
            if end_idx >= len(data):
                break
            
            base_data = data.iloc[start_idx:end_idx + 1]
            
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
    
    def is_valid_leg_in(self, leg_data: pd.DataFrame, direction: str) -> bool:
        """
        Leg-in validation - flexible requirements (can be >50%)
        
        Args:
            leg_data: DataFrame containing leg candles
            direction: 'bullish' or 'bearish'
            
        Returns:
            Boolean indicating validity
        """
        # For single candles
        if len(leg_data) == 1:
            candle = leg_data.iloc[0]
            
            # Check direction
            if direction == 'bullish':
                correct_direction = candle['close'] > candle['open']
            else:
                correct_direction = candle['close'] < candle['open']
            
            if not correct_direction:
                return False
            
            # Calculate body ratio
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if candle_range == 0:
                return False
            
            body_ratio = body_size / candle_range
            return body_ratio > 0.30  # Minimum threshold
        
        # For multi-candle legs
        directional_candles = 0
        strong_candles = 0
        
        for idx in leg_data.index:
            candle_data = leg_data.loc[idx]
            
            # Check direction
            if direction == 'bullish' and candle_data['close'] > candle_data['open']:
                directional_candles += 1
            elif direction == 'bearish' and candle_data['close'] < candle_data['open']:
                directional_candles += 1
            
            # Check body ratio
            body_size = abs(candle_data['close'] - candle_data['open'])
            candle_range = candle_data['high'] - candle_data['low']
            
            if candle_range > 0:
                body_ratio = body_size / candle_range
                if body_ratio > 0.50:
                    strong_candles += 1
        
        direction_ratio = directional_candles / len(leg_data)
        strength_ratio = strong_candles / len(leg_data)
        
        return direction_ratio >= 0.7 and strength_ratio >= 0.5
    
    def is_valid_base(self, base_data: pd.DataFrame) -> bool:
        """
        STRICT: ALL candles must be ≤50% body-to-range (base classification)
        
        Args:
            base_data: DataFrame containing base candles
            
        Returns:
            Boolean indicating validity
        """
        for idx in base_data.index:
            candle = base_data.loc[idx]
            
            # Get classification using candle classifier
            classification = self.candle_classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            # STRICT REQUIREMENT: Must be base classification (≤50%)
            if classification != 'base':
                return False
            
            # Double-check with manual calculation
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if candle_range > 0:
                body_ratio = body_size / candle_range
                if body_ratio > 0.50:  # Must be ≤50%
                    return False
        
        # Additional consolidation checks
        base_high = base_data['high'].max()
        base_low = base_data['low'].min()
        base_range = base_high - base_low
        
        # Check overall movement is minimal
        total_close_movement = abs(base_data['close'].iloc[-1] - base_data['close'].iloc[0])
        if base_range > 0 and total_close_movement > base_range * 0.5:
            return False
        
        return True
    
    def is_valid_leg_out_strength(self, leg_data: pd.DataFrame, direction: str) -> bool:
        """
        Check leg-out specific >70% body-to-range requirement
        
        Args:
            leg_data: DataFrame containing leg-out candles
            direction: 'bullish' or 'bearish'
            
        Returns:
            Boolean indicating if strength requirement is met
        """
        for idx in leg_data.index:
            candle = leg_data.loc[idx]
            
            # Check direction
            if direction == 'bullish':
                correct_direction = candle['close'] > candle['open']
            else:
                correct_direction = candle['close'] < candle['open']
            
            if not correct_direction:
                continue  # Skip counter-trend candles
            
            # Calculate body ratio
            body_size = abs(candle['close'] - candle['open'])
            candle_range = candle['high'] - candle['low']
            
            if candle_range > 0:
                body_ratio = body_size / candle_range
                
                # REQUIREMENT: At least one candle must be >70%
                if body_ratio > 0.70:
                    return True
        
        return False  # No candle met the >70% requirement
    
    def calculate_leg_range(self, leg_data: pd.DataFrame, direction: str) -> float:
        """
        Calculate the range of a leg
        
        Args:
            leg_data: DataFrame containing leg candles
            direction: 'bullish' or 'bearish' (unused but kept for consistency)
            
        Returns:
            Range as float
        """
        return leg_data['high'].max() - leg_data['low'].min()
    
    def calculate_leg_strength(self, leg_data: pd.DataFrame, direction: str) -> float:
        """
        Calculate the strength score of a leg
        
        Args:
            leg_data: DataFrame containing leg candles
            direction: 'bullish' or 'bearish' (unused but kept for consistency)
            
        Returns:
            Strength score as float
        """
        strong_candles = 0
        
        for idx in leg_data.index:
            candle_data = leg_data.loc[idx]
            classification = self.candle_classifier.classify_single_candle(
                candle_data['open'], candle_data['high'], 
                candle_data['low'], candle_data['close']
            )
            
            if classification in ['decisive', 'explosive']:
                strong_candles += 1
        
        return strong_candles / len(leg_data)
    
    def calculate_base_quality(self, base_data: pd.DataFrame) -> float:
        """
        Calculate quality score for base consolidation
        
        Args:
            base_data: DataFrame containing base candles
            
        Returns:
            Quality score as float
        """
        # Favor shorter bases (1-2 candles optimal)
        candle_count_score = 1.0 / len(base_data)
        
        base_range = base_data['high'].max() - base_data['low'].min()
        avg_candle_range = base_data.apply(lambda x: x['high'] - x['low'], axis=1).mean()
        range_score = avg_candle_range / base_range if base_range > 0 else 0
        
        return (candle_count_score + range_score) / 2
    
    def calculate_pattern_strength(self, leg_in: Dict, base: Dict, leg_out: Dict) -> float:
        """
        Calculate overall pattern strength score
        
        Args:
            leg_in: Leg-in pattern information
            base: Base pattern information
            leg_out: Leg-out pattern information
            
        Returns:
            Overall strength score as float
        """
        leg_in_strength = leg_in['strength']
        base_quality = base['quality_score']
        leg_out_strength = leg_out['strength']
        leg_out_ratio = leg_out['ratio_to_base']
        
        # Enhanced scoring with distance priority
        base_bonus = 1.0 if base['candle_count'] <= 2 else 0.8 if base['candle_count'] <= 3 else 0.6
        ratio_bonus = min(leg_out_ratio / 2.0, 1.0)  # Cap at 2x for bonus
        
        strength = (leg_in_strength * 0.25 + 
                   base_quality * 0.25 + 
                   leg_out_strength * 0.25 + 
                   ratio_bonus * 0.25) * base_bonus
        
        return round(strength, 3)
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data format
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(data) < 10:
            raise ValueError("Insufficient data for pattern detection")
    
    def debug_pattern_detection(self, data: pd.DataFrame, start_idx: int, end_idx: int):
        """
        Debug specific candle range to understand pattern detection
        
        Args:
            data: Full dataset
            start_idx: Start index for debugging
            end_idx: End index for debugging
        """
        print(f"\n🔍 DEBUGGING CANDLES {start_idx}-{end_idx}")
        print("=" * 70)
        
        for i in range(start_idx, min(end_idx + 1, len(data))):
            candle = data.iloc[i]
            classification = self.candle_classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            direction = "Bullish" if candle['close'] > candle['open'] else "Bearish"
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            ratio = body_size / total_range if total_range > 0 else 0
            
            print(f"Candle {i:2d}: {classification:9s} | {direction:7s} | Ratio: {ratio:.3f} | "
                  f"OHLC: {candle['open']:.5f}/{candle['high']:.5f}/{candle['low']:.5f}/{candle['close']:.5f}")
    
    def get_pattern_summary(self, patterns: Dict) -> Dict:
       """
       Generate summary statistics for detected patterns
       
       Args:
           patterns: Dictionary containing pattern lists
           
       Returns:
           Summary statistics dictionary
       """
       dbd_count = len(patterns['dbd_patterns'])
       rbr_count = len(patterns['rbr_patterns'])
       total_count = patterns['total_patterns']
       
       if total_count == 0:
           return {
               'total_patterns': 0,
               'dbd_patterns': 0,
               'rbr_patterns': 0,
               'avg_strength': 0.0,
               'avg_legout_ratio': 0.0,
               'strength_distribution': {'high': 0, 'medium': 0, 'low': 0}
           }
       
       all_strengths = []
       all_ratios = []
       
       for pattern in patterns['dbd_patterns'] + patterns['rbr_patterns']:
           all_strengths.append(pattern['strength'])
           all_ratios.append(pattern['leg_out']['ratio_to_base'])
       
       avg_strength = sum(all_strengths) / len(all_strengths)
       avg_ratio = sum(all_ratios) / len(all_ratios)
       
       high_strength = sum(1 for s in all_strengths if s >= 0.8)
       medium_strength = sum(1 for s in all_strengths if 0.5 <= s < 0.8)
       low_strength = sum(1 for s in all_strengths if s < 0.5)
       
       return {
           'total_patterns': total_count,
           'dbd_patterns': dbd_count,
           'rbr_patterns': rbr_count,
           'avg_strength': round(avg_strength, 3),
           'avg_legout_ratio': round(avg_ratio, 2),
           'strength_distribution': {
               'high': high_strength,
               'medium': medium_strength,
               'low': low_strength
           }
       }
   
   # Legacy methods for backwards compatibility
    def detect_dbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Legacy method - redirects to enhanced version
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of D-B-D patterns
        """
        return self.detect_dbd_patterns_enhanced(data)
    
    def detect_rbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Legacy method - redirects to enhanced version
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of R-B-R patterns
        """
        return self.detect_rbr_patterns_enhanced(data)
    
    def validate_zone_integrity(self, pattern: Dict, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Legacy method - redirects to enhanced version
        
        Args:
            pattern: Pattern dictionary to validate
            data: Full dataset
            
        Returns:
            Tuple of (is_valid, message)
        """
        return self.enhanced_zone_invalidation(pattern, data, pattern['base']['end_idx'] + 1)