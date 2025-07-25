"""
Trend Classification Engine - Module 3 (FLEXIBLE BACKTESTING READY)
Configurable EMA system with 50/200 default for manual trading consistency
Expandable for systematic EMA optimization and multi-EMA research
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class TrendEngine:
    """
    Flexible trend classification system:
    - DEFAULT: EMA50 vs EMA200 (matches manual trading)
    - RESEARCH: Any EMA combination via config
    - FUTURE: Triple EMA, SMA crossovers, custom indicators
    """
    
    def __init__(self, data: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialize with OHLC data and optional configuration
        
        Args:
            data: OHLC DataFrame
            config: Optional config dict with 'fast_ema', 'slow_ema', 'method'
                   If None, defaults to manual trading setup (50/200)
        """
        self.data = data.copy()
        self.logger = logging.getLogger(__name__)
        
        # DEFAULT: Manual trading setup (EMA50 vs EMA200)
        if config is None:
            self.config = {
                'fast_ema': 50,
                'slow_ema': 200,
                'method': 'dual_ema'
            }
            self.mode = 'default'
        else:
            self.config = config
            self.mode = 'research'
        
        # Cache for performance optimization
        self.ema_cache = {}
        
        print(f"🎯 TrendEngine initialized:")
        print(f"   Mode: {self.mode}")
        print(f"   Fast EMA: {self.config['fast_ema']}")
        print(f"   Slow EMA: {self.config['slow_ema']}")
        print(f"   Method: {self.config['method']}")
        
    def _calculate_ema(self, period: int) -> pd.Series:
        """
        Calculate EMA for given period with caching for performance
        
        Args:
            period: EMA period (e.g., 50, 200)
            
        Returns:
            EMA series
        """
        if period not in self.ema_cache:
            self.ema_cache[period] = self.data['close'].ewm(span=period, adjust=False).mean()
        return self.ema_cache[period]
    
    def calculate_emas(self) -> pd.DataFrame:
        """
        Calculate configured EMAs (default: 50 and 200)
        """
        fast_period = self.config['fast_ema']
        slow_period = self.config['slow_ema']
        
        print(f"📊 Calculating EMAs: {fast_period} and {slow_period}...")
        
        # Calculate the configured EMAs
        fast_ema = self._calculate_ema(fast_period)
        slow_ema = self._calculate_ema(slow_period)
        
        # Store with standard names for compatibility
        self.data['ema_50'] = fast_ema  # Always named ema_50 for signal generator compatibility
        self.data['ema_200'] = slow_ema  # Always named ema_200 for signal generator compatibility
        
        # Also store with dynamic names for flexibility
        self.data[f'ema_{fast_period}'] = fast_ema
        self.data[f'ema_{slow_period}'] = slow_ema
        
        print(f"✅ EMAs calculated for {len(self.data)} candles")
        return self.data.copy()
    
    def classify_trend_with_filter(self) -> pd.DataFrame:
        """
        Main trend classification method (REQUIRED by signal generator)
        Uses configured EMA periods, defaults to 50/200 for manual trading consistency
        """
        fast_period = self.config['fast_ema']
        slow_period = self.config['slow_ema']
        
        print(f"🔍 Trend classification: EMA{fast_period} vs EMA{slow_period}...")
        
        # Ensure EMAs are calculated
        if 'ema_50' not in self.data.columns or 'ema_200' not in self.data.columns:
            self.calculate_emas()
        
        # Get the configured EMAs (using standard column names)
        fast_ema = self.data['ema_50']  # This contains the configured fast EMA
        slow_ema = self.data['ema_200']  # This contains the configured slow EMA
        
        # Trend classification
        self.data['trend_filtered'] = np.where(
            fast_ema > slow_ema, 
            'bullish', 
            'bearish'
        )
        
        # Calculate EMA separation for trend strength
        self.data['ema_separation'] = fast_ema - slow_ema
        
        # Legacy compatibility
        self.data['trend'] = self.data['trend_filtered']
        
        # Count trends
        trend_counts = self.data['trend_filtered'].value_counts()
        
        print(f"✅ Trend classification complete:")
        for trend_type, count in trend_counts.items():
            print(f"   {trend_type}: {count}")
        
        return self.data.copy()
    
    def classify_trend_simplified(self) -> pd.DataFrame:
        """Legacy method - calls classify_trend_with_filter for compatibility"""
        return self.classify_trend_with_filter()
    
    def get_current_trend(self) -> Dict:
        """
        Get current trend status with configured EMAs
        """
        if 'trend_filtered' not in self.data.columns:
            self.classify_trend_with_filter()
        
        latest_data = self.data.iloc[-1]
        fast_period = self.config['fast_ema']
        slow_period = self.config['slow_ema']
        
        return {
            'trend': latest_data['trend_filtered'],
            'ema_fast': latest_data['ema_50'],  # Contains configured fast EMA
            'ema_slow': latest_data['ema_200'],  # Contains configured slow EMA
            'ema_separation': latest_data['ema_separation'],
            'close': latest_data['close'],
            'is_bullish': latest_data['ema_50'] > latest_data['ema_200'],
            'config': self.config,
            'fast_period': fast_period,
            'slow_period': slow_period
        }
    
    def get_trend_statistics(self) -> Dict:
        """
        Get comprehensive trend statistics with EMA configuration details
        """
        if 'trend_filtered' not in self.data.columns:
            self.classify_trend_with_filter()
        
        total_periods = len(self.data)
        trend_counts = self.data['trend_filtered'].value_counts()
        
        # Calculate trend strength statistics
        ema_sep_stats = self.data['ema_separation'].describe()
        
        stats = {
            'total_periods': total_periods,
            'bullish_periods': trend_counts.get('bullish', 0),
            'bearish_periods': trend_counts.get('bearish', 0),
            'bullish_percentage': (trend_counts.get('bullish', 0) / total_periods) * 100,
            'bearish_percentage': (trend_counts.get('bearish', 0) / total_periods) * 100,
            'config': self.config,
            'ema_separation_stats': {
                'mean': ema_sep_stats['mean'],
                'std': ema_sep_stats['std'],
                'min': ema_sep_stats['min'],
                'max': ema_sep_stats['max']
            }
        }
        
        return stats
    
    # RESEARCH METHODS FOR FUTURE EMA BACKTESTING
    
    def test_ema_combination(self, fast_period: int, slow_period: int) -> pd.DataFrame:
        """
        Test specific EMA combination without changing instance config
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            DataFrame with trend analysis for this combination
        """
        fast_ema = self._calculate_ema(fast_period)
        slow_ema = self._calculate_ema(slow_period)
        
        result = self.data.copy()
        result[f'ema_{fast_period}'] = fast_ema
        result[f'ema_{slow_period}'] = slow_ema
        result['trend'] = np.where(fast_ema > slow_ema, 'bullish', 'bearish')
        result['ema_separation'] = fast_ema - slow_ema
        
        return result
    
    def batch_test_emas(self, ema_combinations: List[Tuple[int, int]]) -> Dict:
        """
        Test multiple EMA combinations efficiently
        
        Args:
            ema_combinations: List of (fast_period, slow_period) tuples
            
        Returns:
            Dictionary mapping combinations to trend DataFrames
        """
        results = {}
        
        print(f"🔬 Testing {len(ema_combinations)} EMA combinations...")
        
        for fast, slow in ema_combinations:
            if fast >= slow:
                print(f"   ⚠️  Skipping invalid combination: EMA{fast} >= EMA{slow}")
                continue
                
            trend_data = self.test_ema_combination(fast, slow)
            results[(fast, slow)] = trend_data
            
            # Quick stats
            bullish_pct = (trend_data['trend'] == 'bullish').mean() * 100
            print(f"   ✅ EMA{fast}/EMA{slow}: {bullish_pct:.1f}% bullish")
        
        print(f"✅ Batch testing complete: {len(results)} combinations processed")
        return results
    
    def get_optimal_ema_periods(self, ema_combinations: List[Tuple[int, int]], 
                               metric: str = 'trend_consistency') -> Dict:
        """
        Find optimal EMA combination based on specified metric
        
        Args:
            ema_combinations: EMA combinations to test
            metric: Optimization metric ('trend_consistency', 'separation_strength', etc.)
            
        Returns:
            Dictionary with optimal combination and analysis
        """
        batch_results = self.batch_test_emas(ema_combinations)
        
        if metric == 'trend_consistency':
            # Find combination with most stable trends (fewer whipsaws)
            best_score = float('inf')
            best_combination = None
            
            for (fast, slow), trend_data in batch_results.items():
                # Count trend changes (whipsaws)
                trend_changes = (trend_data['trend'] != trend_data['trend'].shift(1)).sum()
                whipsaw_ratio = trend_changes / len(trend_data)
                
                if whipsaw_ratio < best_score:
                    best_score = whipsaw_ratio
                    best_combination = (fast, slow)
        
        return {
            'optimal_combination': best_combination,
            'metric_value': best_score,
            'metric_name': metric,
            'all_results': batch_results
        }

# BACKWARD COMPATIBILITY ALIAS
TrendClassifier = TrendEngine