"""
Trend Classification Engine - Module 3 (TRIPLE EMA VERSION)
Three-EMA trend detection: Fast(50), Medium(100), Slow(200)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config.settings import TREND_CONFIG

class TrendClassifier:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLC data
        
        Args:
            data: DataFrame with OHLC data
        """
        self.data = data.copy()
        self.config = TREND_CONFIG
        self.trends = None
        
    def calculate_emas(self) -> pd.DataFrame:
        """
        Calculate EMA 50, 100, and 200
        
        Returns:
            DataFrame with EMA columns added
        """
        print("📊 Calculating Triple EMAs...")
        
        # Calculate all three EMAs
        self.data['ema_50'] = self.data['close'].ewm(span=50, adjust=False).mean()
        self.data['ema_100'] = self.data['close'].ewm(span=100, adjust=False).mean()
        self.data['ema_200'] = self.data['close'].ewm(span=200, adjust=False).mean()
        
        print(f"✅ Triple EMAs calculated for {len(self.data)} candles")
        return self.data.copy()
    
    def classify_trend(self) -> pd.DataFrame:
        """
        Classify trend based on EMA hierarchy
        
        Returns:
            DataFrame with trend classification
        """
        print("🔍 Classifying trends with Triple EMA system...")
        
        # Ensure EMAs are calculated
        if 'ema_50' not in self.data.columns:
            self.calculate_emas()
        
        # Apply triple EMA classification
        conditions = [
            # Strong Bullish: 50 > 100 > 200
            (self.data['ema_50'] > self.data['ema_100']) & 
            (self.data['ema_100'] > self.data['ema_200']),
            
            # Medium Bullish: 50 > 100, but 100 < 200
            (self.data['ema_50'] > self.data['ema_100']) & 
            (self.data['ema_100'] < self.data['ema_200']),
            
            # Weak Bullish: 50 > 100 & 200, but 100 < 200
            (self.data['ema_50'] > self.data['ema_100']) & 
            (self.data['ema_50'] > self.data['ema_200']) & 
            (self.data['ema_100'] < self.data['ema_200']),
            
            # Strong Bearish: 50 < 100 < 200
            (self.data['ema_50'] < self.data['ema_100']) & 
            (self.data['ema_100'] < self.data['ema_200']),
            
            # Medium Bearish: 50 < 100, but 100 > 200
            (self.data['ema_50'] < self.data['ema_100']) & 
            (self.data['ema_100'] > self.data['ema_200']),
            
            # Weak Bearish: 50 < 100 & 200, but 100 > 200
            (self.data['ema_50'] < self.data['ema_100']) & 
            (self.data['ema_50'] < self.data['ema_200']) & 
            (self.data['ema_100'] > self.data['ema_200'])
        ]
        
        choices = [
            'strong_bullish',
            'medium_bullish', 
            'weak_bullish',
            'strong_bearish',
            'medium_bearish',
            'weak_bearish'
        ]
        
        # Apply classification
        self.data['trend'] = np.select(conditions, choices, default='neutral')
        
        # Store trend statistics
        self.trends = self.data['trend'].value_counts()
        
        print(f"✅ Triple EMA trend classification complete:")
        for trend_type, count in self.trends.items():
            print(f"   {trend_type}: {count}")
        
        return self.data.copy()
    
    def get_current_trend(self) -> Dict:
        """
        Get current trend status
        
        Returns:
            Dictionary with current trend information
        """
        if self.trends is None:
            self.classify_trend()
        
        latest_data = self.data.iloc[-1]
        
        return {
            'trend': latest_data['trend'],
            'ema_50': latest_data['ema_50'],
            'ema_100': latest_data['ema_100'],
            'ema_200': latest_data['ema_200'],
            'close': latest_data['close'],
            'ema_order': self.get_ema_order(latest_data)
        }
    
    def get_ema_order(self, row) -> str:
        """
        Get the current EMA ordering
        
        Args:
            row: Data row with EMA values
            
        Returns:
            String describing EMA order
        """
        ema_50 = row['ema_50']
        ema_100 = row['ema_100'] 
        ema_200 = row['ema_200']
        
        if ema_50 > ema_100 > ema_200:
            return "50 > 100 > 200 (Perfect Bull)"
        elif ema_50 < ema_100 < ema_200:
            return "50 < 100 < 200 (Perfect Bear)"
        elif ema_50 > ema_100 and ema_100 < ema_200:
            return "50 > 100 < 200 (Mixed Bull)"
        elif ema_50 < ema_100 and ema_100 > ema_200:
            return "50 < 100 > 200 (Mixed Bear)"
        else:
            return "Complex EMA Structure"
    
    def get_trend_statistics(self) -> Dict:
        """
        Get comprehensive trend statistics
        
        Returns:
            Dictionary with trend statistics
        """
        if self.trends is None:
            self.classify_trend()
        
        total_periods = len(self.data)
        
        stats = {
            'total_periods': total_periods,
            'trend_distribution': {}
        }
        
        # Calculate percentages for each trend type
        for trend_type, count in self.trends.items():
            stats['trend_distribution'][trend_type] = {
                'count': count,
                'percentage': (count / total_periods) * 100
            }
        
        return stats
    
    def detect_trend_changes(self) -> List[Dict]:
        """
        Detect trend change points
        
        Returns:
            List of trend change dictionaries
        """
        if self.trends is None:
            self.classify_trend()
        
        changes = []
        
        # Simple approach: compare each row with previous
        for i in range(1, len(self.data)):
            current_trend = self.data.iloc[i]['trend']
            previous_trend = self.data.iloc[i-1]['trend']
            
            if current_trend != previous_trend:
                changes.append({
                    'index': i,
                    'from_trend': previous_trend,
                    'to_trend': current_trend,
                    'price': self.data.iloc[i]['close'],
                    'ema_order': self.get_ema_order(self.data.iloc[i])
                })
        
        return changes
    
    def validate_trend_classification(self, manual_labels: List[str]) -> float:
        """
        Validate trend classification against manual analysis
        
        Args:
            manual_labels: List of manual trend classifications
            
        Returns:
            Accuracy percentage
        """
        if self.trends is None:
            self.classify_trend()
        
        if len(manual_labels) != len(self.data):
            raise ValueError("Manual labels length must match data length")
        
        correct = sum(1 for auto, manual in zip(self.data['trend'], manual_labels) 
                     if auto == manual)
        
        accuracy = correct / len(manual_labels)
        
        print(f"🎯 Triple EMA Trend Classification Validation:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Correct: {correct}/{len(manual_labels)}")
        
        return accuracy
    
    def calculate_ema_separation(self) -> pd.Series:
        """
        Calculate EMA separation to identify ranging vs trending markets
        
        Returns:
            Series with separation scores (0-1, where 1 = strong trend)
        """
        # Calculate distances between EMAs
        fast_medium_gap = abs(self.data['ema_50'] - self.data['ema_100'])
        medium_slow_gap = abs(self.data['ema_100'] - self.data['ema_200'])
        fast_slow_gap = abs(self.data['ema_50'] - self.data['ema_200'])
        
        # Total EMA spread
        total_spread = fast_slow_gap
        
        # Normalize by current price (percentage separation)
        spread_percentage = total_spread / self.data['close']
        
        # Convert to 0-1 scale (0.5% = minimum trending, 2% = strong trending)
        separation_score = np.clip(spread_percentage * 200, 0, 1)
        
        return separation_score

    def classify_trend_with_filter(self, min_separation=0.0) -> pd.DataFrame:
        """
        SIMPLIFIED: Just return EMA50 vs EMA200 comparison (no ranging filter)
        
        Returns:
            DataFrame with simplified trend classification
        """
        print(f"🔍 SIMPLIFIED trend classification (EMA50 vs EMA200 only)...")
        
        # Calculate EMAs
        self.calculate_emas()
        
        # Simple trend classification
        self.data['trend_filtered'] = np.where(
            self.data['ema_50'] > self.data['ema_200'], 
            'bullish', 
            'bearish'
        )
        
        # Keep separation calculation for reference but don't use for filtering
        self.data['ema_separation'] = self.calculate_ema_separation()
        
        print(f"✅ Simplified trend classification complete:")
        trend_counts = self.data['trend_filtered'].value_counts()
        for trend_type, count in trend_counts.items():
            print(f"   {trend_type}: {count}")
        
        return self.data.copy()