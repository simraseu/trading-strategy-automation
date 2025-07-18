"""
Trend Classification Engine - Module 3 (CRITICAL SIMPLIFICATION)
SIMPLIFIED: Only EMA50 vs EMA200 comparison (no complex classifications)
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

class TrendClassifier:
    """
    SIMPLIFIED trend classification system:
    - EMA50 > EMA200 = bullish
    - EMA50 < EMA200 = bearish
    - No ranging filters, no complex classifications
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with OHLC data"""
        self.data = data.copy()
        self.logger = logging.getLogger(__name__)
        
    def calculate_emas(self) -> pd.DataFrame:
        """
        Calculate ONLY EMA50 and EMA200 (removed EMA100)
        """
        print("📊 Calculating SIMPLIFIED EMAs (50 and 200 only)...")
        
        # Calculate only the two EMAs we need
        self.data['ema_50'] = self.data['close'].ewm(span=50, adjust=False).mean()
        self.data['ema_200'] = self.data['close'].ewm(span=200, adjust=False).mean()
        
        print(f"✅ Simplified EMAs calculated for {len(self.data)} candles")
        return self.data.copy()
    
    def classify_trend_simplified(self) -> pd.DataFrame:
        """
        SIMPLIFIED trend classification using ONLY EMA50 vs EMA200
        """
        print("🔍 SIMPLIFIED trend classification (EMA50 vs EMA200 only)...")
        
        # Ensure EMAs are calculated
        if 'ema_50' not in self.data.columns or 'ema_200' not in self.data.columns:
            self.calculate_emas()
        
        # SIMPLIFIED: Just compare EMA50 vs EMA200
        self.data['trend'] = np.where(
            self.data['ema_50'] > self.data['ema_200'], 
            'bullish', 
            'bearish'
        )
        
        # Count trends
        trend_counts = self.data['trend'].value_counts()
        
        print(f"✅ Simplified trend classification complete:")
        for trend_type, count in trend_counts.items():
            print(f"   {trend_type}: {count}")
        
        return self.data.copy()
    
    def get_current_trend(self) -> Dict:
        """
        Get current trend status (simplified)
        """
        if 'trend' not in self.data.columns:
            self.classify_trend_simplified()
        
        latest_data = self.data.iloc[-1]
        
        return {
            'trend': latest_data['trend'],
            'ema_50': latest_data['ema_50'],
            'ema_200': latest_data['ema_200'],
            'close': latest_data['close'],
            'is_bullish': latest_data['ema_50'] > latest_data['ema_200']
        }
    
    def get_trend_statistics(self) -> Dict:
        """
        Get simplified trend statistics
        """
        if 'trend' not in self.data.columns:
            self.classify_trend_simplified()
        
        total_periods = len(self.data)
        trend_counts = self.data['trend'].value_counts()
        
        stats = {
            'total_periods': total_periods,
            'bullish_periods': trend_counts.get('bullish', 0),
            'bearish_periods': trend_counts.get('bearish', 0),
            'bullish_percentage': (trend_counts.get('bullish', 0) / total_periods) * 100,
            'bearish_percentage': (trend_counts.get('bearish', 0) / total_periods) * 100
        }
        
        return stats