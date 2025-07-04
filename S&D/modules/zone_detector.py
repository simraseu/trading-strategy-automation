"""
Zone Detection Engine - Module 2 (PROFESSIONAL COMPLETE)
Detects Drop-Base-Drop (D-B-D) and Rally-Base-Rally (R-B-R) patterns
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
    Professional zone detection system for forex trading automation
    Focuses on momentum patterns: D-B-D and R-B-R
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
    
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Detect all zone patterns in the dataset
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with 'dbd_patterns' and 'rbr_patterns' keys
        """
        try:
            # Validate input data
            self.validate_data(data)
            
            # Detect patterns
            dbd_patterns = self.detect_dbd_patterns(data)
            rbr_patterns = self.detect_rbr_patterns(data)
            
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
    
    def detect_dbd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Enhanced D-B-D detection with zone invalidation
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected D-B-D patterns
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Phase 1: Identify bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation (start AFTER leg-in)
                base = self.identify_base(data, leg_in['end_idx'] + 1)
                if not base:
                    continue
                
                # Phase 3: Validate bearish leg-out (start AFTER base)
                leg_out = self.validate_leg_out(data, base['end_idx'] + 1, 
                                            base['range'], direction='bearish')
                if not leg_out:
                    continue
                
                # Create pattern object
                pattern = {
                    'type': 'D-B-D',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': base['high'],
                    'zone_low': base['low'],
                    'zone_range': base['range']
                }
                
                # CRITICAL: Validate pattern integrity (check for invalidating candles)
                is_valid, validation_msg = self.validate_zone_integrity(pattern, data)
                if not is_valid:
                    self.logger.debug(f"D-B-D pattern at {i} rejected: {validation_msg}")
                    continue
                
                # Calculate strength
                pattern['strength'] = self.calculate_pattern_strength(leg_in, base, leg_out)
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing D-B-D at index {i}: {str(e)}")
                continue
        
        return patterns
    
    def detect_rbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Enhanced R-B-R detection with zone invalidation
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected R-B-R patterns
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Phase 1: Identify bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation (start AFTER leg-in)
                base = self.identify_base(data, leg_in['end_idx'] + 1)
                if not base:
                    continue
                
                # Phase 3: Validate bullish leg-out (start AFTER base)
                leg_out = self.validate_leg_out(data, base['end_idx'] + 1, 
                                            base['range'], direction='bullish')
                if not leg_out:
                    continue
                
                # Create pattern object
                pattern = {
                    'type': 'R-B-R',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': base['high'],
                    'zone_low': base['low'],
                    'zone_range': base['range']
                }
                
                # CRITICAL: Validate pattern integrity (check for invalidating candles)
                is_valid, validation_msg = self.validate_zone_integrity(pattern, data)
                if not is_valid:
                    self.logger.debug(f"R-B-R pattern at {i} rejected: {validation_msg}")
                    continue
                
                # Calculate strength
                pattern['strength'] = self.calculate_pattern_strength(leg_in, base, leg_out)
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing R-B-R at index {i}: {str(e)}")
                continue
        
        return patterns
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """
        Identify leg-in with flexible requirements (can be >50%)
        """
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
        """
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
    
    def validate_leg_out(self, data: pd.DataFrame, start_idx: int, 
                         base_range: float, direction: str) -> Optional[Dict]:
        """
        Validate leg-out - must have >70% body-to-range requirement
        """
        max_leg_length = 5
        min_leg_out_range = base_range * 1.5
        
        for leg_length in range(1, max_leg_length + 1):
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            # Check basic leg validity and >70% requirement
            if (self.is_valid_leg_in(leg_data, direction) and 
                self.is_valid_leg_out_strength(leg_data, direction)):
                
                leg_range = self.calculate_leg_range(leg_data, direction)
                
                if leg_range >= min_leg_out_range:
                    return {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'direction': direction,
                        'range': leg_range,
                        'ratio_to_base': leg_range / base_range,
                        'strength': self.calculate_leg_strength(leg_data, direction),
                        'candle_count': leg_length
                    }
        
        return None
    
    def is_valid_leg_in(self, leg_data: pd.DataFrame, direction: str) -> bool:
        """
        Leg-in validation - flexible requirements (can be >50%)
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
    
    def validate_zone_integrity(self, pattern: Dict, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        ENHANCED: Check for zone invalidation by opposite-direction decisive/explosive candles
        """
        pattern_end = pattern['end_idx']
        pattern_type = pattern['type']
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        
        # Check next 3-10 candles for invalidation
        validation_window = min(10, len(data) - pattern_end - 1)
        
        if validation_window > 0:
            next_candles = data.iloc[pattern_end + 1:pattern_end + 1 + validation_window]
            
            for relative_idx, (idx, candle) in enumerate(next_candles.iterrows()):
                actual_idx = pattern_end + 1 + relative_idx
                
                # Get candle classification
                classification = self.candle_classifier.classify_single_candle(
                    candle['open'], candle['high'], candle['low'], candle['close']
                )
                
                # CRITICAL: Check for zone-breaking candles
                if pattern_type == 'D-B-D':  # Supply zone
                    is_bullish = candle['close'] > candle['open']
                    
                    # Invalidated by decisive/explosive bullish candle that breaks above zone
                    if (is_bullish and 
                        classification in ['decisive', 'explosive'] and 
                        candle['close'] > zone_high):
                        return False, f"Zone invalidated by bullish {classification} breaking above zone at candle {actual_idx}"
                
                elif pattern_type == 'R-B-R':  # Demand zone
                    is_bearish = candle['close'] < candle['open']
                    
                    # Invalidated by decisive/explosive bearish candle that breaks below zone
                    if (is_bearish and 
                        classification in ['decisive', 'explosive'] and 
                        candle['close'] < zone_low):
                        return False, f"Zone invalidated by bearish {classification} breaking below zone at candle {actual_idx}"
        
        return True, "Valid"
    
    def calculate_leg_range(self, leg_data: pd.DataFrame, direction: str) -> float:
        """Calculate the range of a leg"""
        return leg_data['high'].max() - leg_data['low'].min()
    
    def calculate_leg_strength(self, leg_data: pd.DataFrame, direction: str) -> float:
        """Calculate the strength score of a leg"""
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
        """Calculate quality score for base consolidation"""
        candle_count_score = 1.0 / len(base_data)
        
        base_range = base_data['high'].max() - base_data['low'].min()
        avg_candle_range = base_data.apply(lambda x: x['high'] - x['low'], axis=1).mean()
        range_score = avg_candle_range / base_range if base_range > 0 else 0
        
        return (candle_count_score + range_score) / 2
    
    def calculate_pattern_strength(self, leg_in: Dict, base: Dict, leg_out: Dict) -> float:
        """Calculate overall pattern strength score"""
        leg_in_strength = leg_in['strength']
        base_quality = base['quality_score']
        leg_out_strength = leg_out['strength']
        leg_out_ratio = leg_out['ratio_to_base']
        
        base_bonus = 1.0 if base['candle_count'] <= 3 else 0.7
        ratio_bonus = min(leg_out_ratio / 1.5, 1.0)
        
        strength = (leg_in_strength * 0.3 + 
                   base_quality * 0.2 + 
                   leg_out_strength * 0.3 + 
                   ratio_bonus * 0.2) * base_bonus
        
        return round(strength, 3)
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format"""
        required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(data) < 10:
            raise ValueError("Insufficient data for pattern detection")
    
    def get_pattern_summary(self, patterns: Dict) -> Dict:
        """Generate summary statistics for detected patterns"""
        dbd_count = len(patterns['dbd_patterns'])
        rbr_count = len(patterns['rbr_patterns'])
        total_count = patterns['total_patterns']
        
        if total_count == 0:
            return {
                'total_patterns': 0,
                'dbd_patterns': 0,
                'rbr_patterns': 0,
                'avg_strength': 0.0,
                'strength_distribution': {}
            }
        
        all_strengths = []
        for pattern in patterns['dbd_patterns'] + patterns['rbr_patterns']:
            all_strengths.append(pattern['strength'])
        
        avg_strength = sum(all_strengths) / len(all_strengths)
        
        high_strength = sum(1 for s in all_strengths if s >= 0.8)
        medium_strength = sum(1 for s in all_strengths if 0.5 <= s < 0.8)
        low_strength = sum(1 for s in all_strengths if s < 0.5)
        
        return {
            'total_patterns': total_count,
            'dbd_patterns': dbd_count,
            'rbr_patterns': rbr_count,
            'avg_strength': round(avg_strength, 3),
            'strength_distribution': {
                'high': high_strength,
                'medium': medium_strength,
                'low': low_strength
            }
        }