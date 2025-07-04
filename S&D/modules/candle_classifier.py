"""
Candle Classification Engine
Classifies candles as Base, Decisive, or Explosive based on body-to-range ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class CandleClassifier:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLC data
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close']
        """
        self.data = data.copy()
        # Your exact thresholds
        self.thresholds = {
            'base_max_ratio': 0.50,      # Base: ≤ 50%
            'decisive_max_ratio': 0.80,   # Decisive: 50% < ratio ≤ 80%
            'explosive_min_ratio': 0.80   # Explosive: > 80%
        }
        self.classifications = None
        
    def classify_single_candle(self, open_price: float, high: float, 
                              low: float, close: float) -> str:
        """
        Classify a single candle based on body-to-range ratio
        
        Args:
            open_price, high, low, close: OHLC values
            
        Returns:
            'base', 'decisive', or 'explosive'
        """
        # Calculate body and total range
        body_size = abs(close - open_price)
        total_range = high - low
        
        # Handle edge case of zero range
        if total_range == 0:
            return 'base'
        
        # Calculate body-to-range ratio
        body_ratio = body_size / total_range

        # Round to avoid floating point precision issues
        body_ratio = round(body_ratio, 6)  # Round to 6 decimal places
        
        # CORRECTED CLASSIFICATION LOGIC WITH PRECISION FIX
        if body_ratio <= self.thresholds['base_max_ratio']:  # ≤ 50%
            return 'base'
        elif body_ratio > self.thresholds['explosive_min_ratio']:  # > 80%
            return 'explosive'
        else:  # 50% < ratio ≤ 80%
            return 'decisive'
    
    def classify_all_candles(self) -> pd.DataFrame:
        """
        Classify all candles in the dataset
        
        Returns:
            DataFrame with additional 'candle_type' column
        """
        print("🔍 Classifying candles...")
        
        # Apply classification to each row
        self.data['candle_type'] = self.data.apply(
            lambda row: self.classify_single_candle(
                row['open'], row['high'], row['low'], row['close']
            ), axis=1
        )
        
        # Calculate body ratios for analysis
        self.data['body_ratio'] = self.get_body_ratios()
        
        # Store classifications for analysis
        self.classifications = self.data['candle_type'].value_counts()
        
        print(f"✅ Classification complete:")
        print(f"   Base candles: {self.classifications.get('base', 0)}")
        print(f"   Decisive candles: {self.classifications.get('decisive', 0)}")
        print(f"   Explosive candles: {self.classifications.get('explosive', 0)}")
        
        return self.data.copy()
    
    def get_classification_stats(self) -> Dict:
        """
        Get detailed statistics about candle classifications
        
        Returns:
            Dictionary with classification statistics
        """
        if self.classifications is None:
            self.classify_all_candles()
        
        total_candles = len(self.data)
        
        stats = {
            'total_candles': total_candles,
            'base_count': self.classifications.get('base', 0),
            'decisive_count': self.classifications.get('decisive', 0),
            'explosive_count': self.classifications.get('explosive', 0),
            'base_percentage': (self.classifications.get('base', 0) / total_candles) * 100,
            'decisive_percentage': (self.classifications.get('decisive', 0) / total_candles) * 100,
            'explosive_percentage': (self.classifications.get('explosive', 0) / total_candles) * 100
        }
        
        return stats
    
    def get_body_ratios(self) -> pd.Series:
        """
        Calculate body-to-range ratios for all candles
        
        Returns:
            Series of body ratios
        """
        body_sizes = abs(self.data['close'] - self.data['open'])
        total_ranges = self.data['high'] - self.data['low']
        
        # Handle division by zero
        ratios = np.where(total_ranges == 0, 0, body_sizes / total_ranges)
        
        return pd.Series(ratios, index=self.data.index)
    
    def validate_classification(self, manual_labels: List[str]) -> float:
        """
        Validate classification against manual labels
        
        Args:
            manual_labels: List of manual classifications for comparison
            
        Returns:
            Accuracy percentage
        """
        if self.classifications is None:
            self.classify_all_candles()
        
        if len(manual_labels) != len(self.data):
            raise ValueError("Manual labels length must match data length")
        
        correct = sum(1 for auto, manual in zip(self.data['candle_type'], manual_labels) 
                     if auto == manual)
        
        accuracy = correct / len(manual_labels)
        
        print(f"🎯 Validation Results:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Correct: {correct}/{len(manual_labels)}")
        
        return accuracy
    
    def analyze_threshold_sensitivity(self) -> Dict:
        """
        Analyze how sensitive classifications are to threshold changes
        
        Returns:
            Dictionary with threshold sensitivity analysis
        """
        # Test different thresholds
        sensitivity_results = {}
        
        for base_threshold in [0.45, 0.50, 0.55]:
            for explosive_threshold in [0.75, 0.80, 0.85]:
                if explosive_threshold > base_threshold:
                    # Temporarily change thresholds
                    original_thresholds = self.thresholds.copy()
                    self.thresholds['base_max_ratio'] = base_threshold
                    self.thresholds['explosive_min_ratio'] = explosive_threshold
                    
                    # Classify with new thresholds
                    temp_data = self.data.copy()
                    temp_data['candle_type'] = temp_data.apply(
                        lambda row: self.classify_single_candle(
                            row['open'], row['high'], row['low'], row['close']
                        ), axis=1
                    )
                    
                    # Store results
                    counts = temp_data['candle_type'].value_counts()
                    sensitivity_results[f"base_{base_threshold}_explosive_{explosive_threshold}"] = {
                        'base_count': counts.get('base', 0),
                        'decisive_count': counts.get('decisive', 0),
                        'explosive_count': counts.get('explosive', 0),
                        'base_percentage': (counts.get('base', 0) / len(temp_data)) * 100,
                        'decisive_percentage': (counts.get('decisive', 0) / len(temp_data)) * 100,
                        'explosive_percentage': (counts.get('explosive', 0) / len(temp_data)) * 100
                    }
                    
                    # Restore original thresholds
                    self.thresholds = original_thresholds
        
        return sensitivity_results
    
    def get_sample_candles(self, n_samples: int = 10) -> pd.DataFrame:
        """
        Get sample candles for manual validation
        
        Args:
            n_samples: Number of samples per candle type
            
        Returns:
            DataFrame with sample candles
        """
        if self.classifications is None:
            self.classify_all_candles()
        
        samples = []
        
        for candle_type in ['base', 'decisive', 'explosive']:
            type_data = self.data[self.data['candle_type'] == candle_type]
            if len(type_data) >= n_samples:
                sample = type_data.sample(n=n_samples, random_state=42)
                samples.append(sample)
        
        if samples:
            return pd.concat(samples).sort_index()
        else:
            return pd.DataFrame()