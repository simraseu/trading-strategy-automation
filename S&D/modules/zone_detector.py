"""
Zone Detection Engine - Module 2 
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
        'min_legout_ratio': 2.0,  # FIXED: Now 2.0x requirement
        'min_pattern_pips': 20,
        'pip_value': 0.0001
    }
    print("⚠️  Using fallback ZONE_CONFIG")

class ZoneDetector:
    """
    Zone detection system with CRITICAL FIXES:
    1. Leg-out ratio now measures breakout distance correctly
    2. 2.0x minimum ratio requirement (not 1.5x)
    """
    
    def __init__(self, candle_classifier, config=None):
        """Initialize zone detector"""
        self.candle_classifier = candle_classifier
        self.config = config or ZONE_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # FIXED: 2.0x minimum ratio requirement
        self.min_legout_ratio = 2.0
        
    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect all zone patterns with corrected ratio calculation"""
        try:
            # Validate input data
            self.validate_data(data)
            
            # Detect patterns with CORRECTED ratio calculation
            dbd_patterns = self.detect_dbd_patterns_fixed(data)
            rbr_patterns = self.detect_rbr_patterns_fixed(data)
            
            total_patterns = len(dbd_patterns) + len(rbr_patterns)
            
            print(f"✅ Zone detection complete (FIXED ratios):")
            print(f"   D-B-D patterns: {len(dbd_patterns)}")
            print(f"   R-B-R patterns: {len(rbr_patterns)}")
            print(f"   Total: {total_patterns}")
            
            return {
                'dbd_patterns': dbd_patterns,
                'rbr_patterns': rbr_patterns,
                'total_patterns': total_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            raise

    def detect_dbd_patterns_fixed(self, data: pd.DataFrame) -> List[Dict]:
        """
        FIXED D-B-D detection with correct breakout distance calculation
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Phase 1: Identify bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation
                base_start = leg_in['end_idx'] + 1
                base = self.identify_base(data, base_start)
                if not base:
                    continue
                
                # Phase 3: Identify leg-out
                leg_out_start = base['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base, direction='bearish')
                if not leg_out:
                    continue
                
                # Phase 4: FIXED - Calculate breakout distance correctly
                base_high = base['high']  # Base boundary
                leg_out_data = data.iloc[leg_out['start_idx']:leg_out['end_idx'] + 1]
                breakout_low = leg_out_data['low'].min()  # Breakout price
                
                # FIXED: Measure breakout distance from base boundary
                breakout_distance = base_high - breakout_low
                base_range = base['range']
                
                # FIXED: Calculate ratio as breakout distance / base range
                breakout_ratio = breakout_distance / base_range if base_range > 0 else 0
                
                # FIXED: Check 2.0x minimum requirement
                if breakout_ratio < 2.0:
                    continue
                
                # Create pattern with CORRECTED ratio
                pattern = {
                    'type': 'D-B-D',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': base_high,
                    'zone_low': breakout_low,
                    'zone_range': base_high - breakout_low,
                    'strength': self.calculate_pattern_strength(leg_in, base, leg_out)
                }
                
                # FIXED: Store correct ratio
                pattern['leg_out']['ratio_to_base'] = breakout_ratio
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing D-B-D at index {i}: {str(e)}")
                continue
        
        return patterns

    def detect_rbr_patterns_fixed(self, data: pd.DataFrame) -> List[Dict]:
        """
        FIXED R-B-R detection with correct breakout distance calculation
        """
        patterns = []
        
        for i in range(len(data) - 4):
            try:
                # Phase 1: Identify bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation
                base_start = leg_in['end_idx'] + 1
                base = self.identify_base(data, base_start)
                if not base:
                    continue
                
                # Phase 3: Identify leg-out
                leg_out_start = base['end_idx'] + 1
                leg_out = self.identify_leg_out(data, leg_out_start, base, direction='bullish')
                if not leg_out:
                    continue
                
                # Phase 4: FIXED - Calculate breakout distance correctly
                base_low = base['low']  # Base boundary
                leg_out_data = data.iloc[leg_out['start_idx']:leg_out['end_idx'] + 1]
                breakout_high = leg_out_data['high'].max()  # Breakout price
                
                # FIXED: Measure breakout distance from base boundary
                breakout_distance = breakout_high - base_low
                base_range = base['range']
                
                # FIXED: Calculate ratio as breakout distance / base range
                breakout_ratio = breakout_distance / base_range if base_range > 0 else 0
                
                # FIXED: Check 2.0x minimum requirement
                if breakout_ratio < 2.0:
                    continue
                
                # Create pattern with CORRECTED ratio
                pattern = {
                    'type': 'R-B-R',
                    'start_idx': leg_in['start_idx'],
                    'end_idx': leg_out['end_idx'],
                    'leg_in': leg_in,
                    'base': base,
                    'leg_out': leg_out,
                    'zone_high': breakout_high,
                    'zone_low': base_low,
                    'zone_range': breakout_high - base_low,
                    'strength': self.calculate_pattern_strength(leg_in, base, leg_out)
                }
                
                # FIXED: Store correct ratio
                pattern['leg_out']['ratio_to_base'] = breakout_ratio
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing R-B-R at index {i}: {str(e)}")
                continue
        
        return patterns

    # [Rest of the original methods remain the same - identify_leg_in, identify_base, etc.]
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(data) < 10:
            raise ValueError("Insufficient data for pattern detection")
        
    

    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, direction: str) -> Optional[Dict]:
        """Identify leg-in pattern"""
        if start_idx >= len(data):
            return None
            
        for leg_length in range(1, 4):
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

    def identify_base(self, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """Identify base consolidation pattern"""
        if start_idx >= len(data):
            return None
            
        max_base_candles = self.config['max_base_candles']
        
        for base_length in range(1, max_base_candles + 1):
            end_idx = start_idx + base_length - 1
            
            if end_idx >= len(data):
                break
            
            base_data = data.iloc[start_idx:end_idx + 1]
            
            if self.is_valid_base(base_data):
                base_range = base_data['high'].max() - base_data['low'].min()
                
                return {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'range': base_range,
                    'high': base_data['high'].max(),
                    'low': base_data['low'].min(),
                    'candle_count': base_length,
                    'quality_score': self.calculate_base_quality(base_length)
                }
        
        return None

    def identify_leg_out(self, data: pd.DataFrame, start_idx: int, base: Dict, direction: str) -> Optional[Dict]:
        """Identify leg-out pattern"""
        if start_idx >= len(data):
            return None
            
        base_high = base['high']
        base_low = base['low']
        
        for leg_length in range(1, 4):
            end_idx = start_idx + leg_length - 1
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            if self.is_valid_leg(leg_data, direction):
                # Check breakout from base
                if direction == 'bullish':
                    leg_high = leg_data['high'].max()
                    if leg_high <= base_high:
                        continue
                else:  # bearish
                    leg_low = leg_data['low'].min()
                    if leg_low >= base_low:
                        continue
                
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
        """Check if leg is valid for the direction"""
        if direction == 'bullish':
            return leg_data['close'].iloc[-1] > leg_data['open'].iloc[0]
        else:
            return leg_data['close'].iloc[-1] < leg_data['open'].iloc[0]

    def is_valid_base(self, base_data: pd.DataFrame) -> bool:
        """Check if base consolidation is valid"""
        base_range = base_data['high'].max() - base_data['low'].min()
        return base_range > 0

    def calculate_leg_strength(self, leg_data: pd.DataFrame, direction: str) -> float:
        """Calculate leg strength (0-1)"""
        # Simple strength calculation based on candle classification
        strength = 0.0
        for idx in leg_data.index:
            candle_type = self.candle_classifier.classify_single_candle(leg_data.loc[idx])
            if candle_type == 'explosive':
                strength += 0.5
            elif candle_type == 'decisive':
                strength += 0.3
            else:
                strength += 0.1
        
        return min(strength, 1.0)

    def calculate_base_quality(self, candle_count: int) -> float:
        """Calculate base quality score (0-1)"""
        if candle_count <= 2:
            return 0.9
        elif candle_count == 3:
            return 0.8
        elif candle_count <= 4:
            return 0.6
        else:
            return 0.4

    def calculate_pattern_strength(self, leg_in: Dict, base: Dict, leg_out: Dict) -> float:
        """Calculate overall pattern strength"""
        leg_in_strength = leg_in['strength']
        base_quality = base['quality_score']
        leg_out_strength = leg_out['strength']
        
        return (leg_in_strength + base_quality + leg_out_strength) / 3.0