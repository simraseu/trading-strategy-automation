"""
Zone Detection Engine - Module 2
Detects Drop-Base-Drop (D-B-D) and Rally-Base-Rally (R-B-R) patterns
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from config.settings import ZONE_CONFIG

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
        Detect Drop-Base-Drop patterns
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected D-B-D patterns
        """
        patterns = []
        
        # Need minimum 7 candles for pattern (2 leg-in + 1 base + 2 leg-out + buffer)
        for i in range(len(data) - 7):
            try:
                # Phase 1: Identify bearish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bearish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation
                base = self.identify_base(data, leg_in['end_idx'])
                if not base:
                    continue
                
                # Phase 3: Validate bearish leg-out
                leg_out = self.validate_leg_out(data, base['end_idx'], 
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
                    'strength': self.calculate_pattern_strength(leg_in, base, leg_out),
                    'zone_high': base['high'],
                    'zone_low': base['low'],
                    'zone_range': base['range']
                }
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing D-B-D at index {i}: {str(e)}")
                continue
        
        return patterns
    
    def detect_rbr_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect Rally-Base-Rally patterns
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected R-B-R patterns
        """
        patterns = []
        
        for i in range(len(data) - 7):
            try:
                # Phase 1: Identify bullish leg-in
                leg_in = self.identify_leg_in(data, i, direction='bullish')
                if not leg_in:
                    continue
                
                # Phase 2: Identify base consolidation
                base = self.identify_base(data, leg_in['end_idx'])
                if not base:
                    continue
                
                # Phase 3: Validate bullish leg-out
                leg_out = self.validate_leg_out(data, base['end_idx'], 
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
                    'strength': self.calculate_pattern_strength(leg_in, base, leg_out),
                    'zone_high': base['high'],
                    'zone_low': base['low'],
                    'zone_range': base['range']
                }
                
                patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Error processing R-B-R at index {i}: {str(e)}")
                continue
        
        return patterns
    
    def identify_leg_in(self, data: pd.DataFrame, start_idx: int, 
                        direction: str) -> Optional[Dict]:
        """
        Identify strong momentum leg into base
        
        Args:
            data: DataFrame with OHLC data
            start_idx: Starting index for leg search
            direction: 'bullish' or 'bearish'
            
        Returns:
            Leg-in information or None if not found
        """
        max_leg_length = 5  # Maximum candles to check for leg-in
        
        for leg_length in range(2, max_leg_length + 1):
            end_idx = start_idx + leg_length
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            # Check if this forms a valid leg-in
            if self.is_valid_leg(leg_data, direction):
                leg_range = self.calculate_leg_range(leg_data, direction)
                
                # Check minimum strength requirement
                if leg_range < self.config['min_pattern_pips'] * self.config['pip_value']:
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
        Identify base consolidation after leg-in
        
        Args:
            data: DataFrame with OHLC data
            start_idx: Starting index for base search
            
        Returns:
            Base information or None if not found
        """
        max_base_length = self.config['max_base_candles']
        
        for base_length in range(self.config['min_base_candles'], max_base_length + 1):
            end_idx = start_idx + base_length
            
            if end_idx >= len(data):
                break
            
            base_data = data.iloc[start_idx:end_idx + 1]
            
            # Check if this forms a valid base
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
        Validate leg-out meets minimum requirements
        
        Args:
            data: DataFrame with OHLC data
            start_idx: Starting index for leg-out search
            base_range: Range of the base consolidation
            direction: 'bullish' or 'bearish'
            
        Returns:
            Leg-out information or None if not valid
        """
        max_leg_length = 5  # Maximum candles to check for leg-out
        min_leg_out_range = base_range * self.config['min_legout_ratio']
        
        for leg_length in range(1, max_leg_length + 1):
            end_idx = start_idx + leg_length
            
            if end_idx >= len(data):
                break
            
            leg_data = data.iloc[start_idx:end_idx + 1]
            
            # Check if this forms a valid leg-out
            if self.is_valid_leg(leg_data, direction):
                leg_range = self.calculate_leg_range(leg_data, direction)
                
                # Check minimum leg-out requirement (2x base range)
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
    
    def is_valid_leg(self, leg_data: pd.DataFrame, direction: str) -> bool:
        """
        Check if candle sequence forms a valid leg
        
        Args:
            leg_data: DataFrame with leg candles
            direction: 'bullish' or 'bearish'
            
        Returns:
            True if valid leg, False otherwise
        """
        # Check candle strength (need decisive/explosive candles)
        strong_candles = 0
        
        for idx in leg_data.index:
            candle_data = leg_data.loc[idx]
            classification = self.candle_classifier.classify_single_candle(
                candle_data['open'], candle_data['high'], 
                candle_data['low'], candle_data['close']
            )
            
            # Count decisive and explosive candles
            if classification in ['decisive', 'explosive']:
                strong_candles += 1
        
        # Need at least 50% strong candles
        strength_ratio = strong_candles / len(leg_data)
        
        if strength_ratio < self.config['min_leg_strength']:
            return False
        
        # Check directional consistency
        if direction == 'bullish':
            return leg_data['close'].iloc[-1] > leg_data['open'].iloc[0]
        else:
            return leg_data['close'].iloc[-1] < leg_data['open'].iloc[0]
    
    def is_valid_base(self, base_data: pd.DataFrame) -> bool:
        """
        Check if candle sequence forms a valid base
        
        Args:
            base_data: DataFrame with base candles
            
        Returns:
            True if valid base, False otherwise
        """
        # Check range containment (no excessive breakouts)
        base_high = base_data['high'].max()
        base_low = base_data['low'].min()
        base_range = base_high - base_low
        
        # Check for excessive wicks or breakouts
        for idx in base_data.index:
            candle = base_data.loc[idx]
            
            # Check if any candle breaks significantly beyond base range
            if (candle['high'] > base_high * 1.1 or 
                candle['low'] < base_low * 0.9):
                return False
        
        return True
    
    def calculate_leg_range(self, leg_data: pd.DataFrame, direction: str) -> float:
        """Calculate the range of a leg in the specified direction"""
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
        # Higher score for fewer candles (more compressed)
        candle_count_score = 1.0 / len(base_data)
        
        # Higher score for tighter range
        base_range = base_data['high'].max() - base_data['low'].min()
        avg_candle_range = base_data.apply(lambda x: x['high'] - x['low'], axis=1).mean()
        range_score = avg_candle_range / base_range if base_range > 0 else 0
        
        return (candle_count_score + range_score) / 2
    
    def calculate_pattern_strength(self, leg_in: Dict, base: Dict, leg_out: Dict) -> float:
        """Calculate overall pattern strength score"""
        # Combine multiple factors
        leg_in_strength = leg_in['strength']
        base_quality = base['quality_score']
        leg_out_strength = leg_out['strength']
        leg_out_ratio = leg_out['ratio_to_base']
        
        # Base candle count bonus (fewer is better)
        base_bonus = 1.0 if base['candle_count'] <= 3 else 0.5
        
        # Leg-out ratio bonus
        ratio_bonus = min(leg_out_ratio / 2.0, 1.0)  # Cap at 1.0
        
        strength = (leg_in_strength * 0.3 + 
                   base_quality * 0.2 + 
                   leg_out_strength * 0.3 + 
                   ratio_bonus * 0.2) * base_bonus
        
        return round(strength, 3)
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format - FIXED FOR LOWERCASE COLUMNS"""
        required_columns = ['open', 'high', 'low', 'close']
        
        print(f"DEBUG: Looking for columns: {required_columns}")
        print(f"DEBUG: Available columns: {list(data.columns)}")
        
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
        
        # Calculate average strength
        all_strengths = []
        for pattern in patterns['dbd_patterns'] + patterns['rbr_patterns']:
            all_strengths.append(pattern['strength'])
        
        avg_strength = sum(all_strengths) / len(all_strengths)
        
        # Strength distribution
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