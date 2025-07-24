"""
ZONE QUALITY BACKTESTER - EXTENDED FROM PROVEN ARCHITECTURE
Built from proven backtest_zone_age_marketcond.py logic with zone quality analysis
Implements 5-factor quality scoring system with age + quality strategy combinations
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import psutil
import warnings
import glob
warnings.filterwarnings('ignore')

# Import your proven modules
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager


def check_system_requirements():
    """Check system resources before starting analysis"""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = cpu_count()
    memory_percent = psutil.virtual_memory().percent
    
    print(f"💻 SYSTEM RESOURCES CHECK:")
    print(f"   RAM: {memory_gb:.1f} GB available")
    print(f"   CPU: {cpu_cores} cores")
    print(f"   Current memory usage: {memory_percent:.1f}%")
    
    if memory_gb < 8:
        print("⚠️  WARNING: Less than 8GB RAM. Consider reducing scope.")
    
    if memory_percent > 60:
        print("⚠️  WARNING: High memory usage. Close other applications.")
    
    return memory_gb >= 4  # Minimum 4GB required


class ZoneQualityBacktester:
    """
    ZONE QUALITY BACKTESTER - Extended from proven architecture
    Adds 5-factor quality scoring with comprehensive strategy combinations
    """
    
    # Age categories (inherited from proven system)
    ZONE_AGE_CATEGORIES = {
        'Ultra_Fresh': (0, 7),      # 0-7 days
        'Fresh': (8, 30),           # 8-30 days  
        'Recent': (31, 90),         # 31-90 days
        'Aged': (91, 180),          # 91-180 days
        'Stale': (181, 365),        # 181-365 days
        'Ancient': (365, 99999)     # 365+ days
    }
    
    # Quality scoring factors (NEW)
    QUALITY_FACTORS = {
        'base_candle_count': {
            'weight': 0.25,
            'scores': {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.5, 5: 0.3, 6: 0.1}
        },
        'leg_in_strength': {
            'weight': 0.20,
            'threshold': 0.5  # Minimum leg strength
        },
        'leg_out_distance': {
            'weight': 0.30,
            'base_threshold': 2.0  # Base 2.0x requirement
        },
        'zone_range_pips': {
            'weight': 0.15,
            'pip_scores': {
                (0, 10): 1.0,     # <10 pips = perfect
                (10, 25): 0.8,    # 10-25 pips = good
                (25, 50): 0.6,    # 25-50 pips = fair
                (50, 100): 0.4,   # 50-100 pips = poor
                (100, 999): 0.2   # >100 pips = very poor
            }
        },
        'pattern_strength': {
            'weight': 0.10,
            'base_value': 0.5  # Normalize pattern strength
        }
    }
    
    # Complete strategy matrix (Age + Quality combinations)
    STRATEGIES = {
        # Baseline strategies
        'Baseline': {
            'age_filter': None,
            'quality_filter': None,
            'description': 'Baseline - no filters'
        },
        
        # Age-only strategies (from proven system)
        'Ultra_Fresh_Only': {
            'age_filter': 'Ultra_Fresh',
            'quality_filter': None,
            'description': 'Ultra fresh zones only (0-7 days)'
        },
        'Fresh_Only': {
            'age_filter': 'Fresh',
            'quality_filter': None,
            'description': 'Fresh zones only (8-30 days)'
        },
        'Combined_Fresh': {
            'age_filter': ['Ultra_Fresh', 'Fresh'],
            'quality_filter': None,
            'description': 'Ultra fresh + fresh zones (0-30 days)'
        },
        
        # Quality-only strategies (NEW)
        'High_Quality_Only': {
            'age_filter': None,
            'quality_filter': {'min_score': 0.7},
            'description': 'High quality zones only (score ≥ 0.7)'
        },
        'Premium_Quality': {
            'age_filter': None,
            'quality_filter': {'min_score': 0.8},
            'description': 'Premium quality zones (score ≥ 0.8)'
        },
        'Base_1_Only': {
            'age_filter': None,
            'quality_filter': {'base_candles': 1},
            'description': 'Single candle bases only'
        },
        'Strong_LegOut_Only': {
            'age_filter': None,
            'quality_filter': {'min_legout_ratio': 3.0},
            'description': 'Strong leg-out only (≥3x distance)'
        },
        
        # Combined age + quality strategies (NEW)
        'Fresh_HighQuality': {
            'age_filter': ['Ultra_Fresh', 'Fresh'],
            'quality_filter': {'min_score': 0.7},
            'description': 'Fresh + high quality zones'
        },
        'Fresh_Premium': {
            'age_filter': ['Ultra_Fresh', 'Fresh'],
            'quality_filter': {'min_score': 0.8},
            'description': 'Fresh + premium quality zones'
        },
        'UltraFresh_Base1': {
            'age_filter': 'Ultra_Fresh',
            'quality_filter': {'base_candles': 1},
            'description': 'Ultra fresh single-candle bases'
        },
        'Fresh_Strong_LegOut': {
            'age_filter': ['Ultra_Fresh', 'Fresh'],
            'quality_filter': {'min_legout_ratio': 3.0},
            'description': 'Fresh zones with strong leg-out'
        }
    }
    
    def __init__(self, max_workers=None):
        """Initialize with system optimization"""
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.data_loader = DataLoader()
        
        print(f"🎯 ZONE QUALITY BACKTESTER INITIALIZED")
        print(f"   💡 5-factor quality scoring system")
        print(f"   🔄 {len(self.STRATEGIES)} quality + age strategies")
        print(f"   ⚡ {self.max_workers} parallel workers")
    
    def calculate_zone_quality_score(self, pattern: Dict) -> float:
        """
        Calculate comprehensive 5-factor quality score
        """
        try:
            score = 0.0
            
            # Factor 1: Base Candle Count (25% weight)
            base_count = pattern['base']['candle_count']
            base_score = self.QUALITY_FACTORS['base_candle_count']['scores'].get(base_count, 0.1)
            score += base_score * self.QUALITY_FACTORS['base_candle_count']['weight']
            
            # Factor 2: Leg-In Strength (20% weight)
            leg_in_strength = pattern['leg_in']['strength']
            # Normalize to 0-1 range
            normalized_leg_in = min(leg_in_strength, 1.0)
            score += normalized_leg_in * self.QUALITY_FACTORS['leg_in_strength']['weight']
            
            # Factor 3: Leg-Out Distance (30% weight) - Most important
            leg_out_ratio = pattern['leg_out']['ratio_to_base']
            # Scale relative to 2.0x base requirement
            distance_score = min(leg_out_ratio / 4.0, 1.0)  # Cap at 4x for perfect score
            score += distance_score * self.QUALITY_FACTORS['leg_out_distance']['weight']
            
            # Factor 4: Zone Range in Pips (15% weight)
            zone_range = pattern['zone_range']
            pip_value = 0.0001  # Assuming EURUSD pip value
            zone_pips = zone_range / pip_value
            
            pip_score = 0.2  # Default poor score
            for (min_pips, max_pips), pip_score_val in self.QUALITY_FACTORS['zone_range_pips']['pip_scores'].items():
                if min_pips <= zone_pips < max_pips:
                    pip_score = pip_score_val
                    break
            
            score += pip_score * self.QUALITY_FACTORS['zone_range_pips']['weight']
            
            # Factor 5: Pattern Strength (10% weight)
            pattern_strength = pattern.get('strength', 0.5)
            normalized_pattern = min(pattern_strength, 1.0)
            score += normalized_pattern * self.QUALITY_FACTORS['pattern_strength']['weight']
            
            return round(score, 3)
            
        except Exception as e:
            print(f"⚠️  Error calculating quality score: {str(e)}")
            return 0.5  # Default moderate score
    
    def passes_quality_filter(self, pattern: Dict, quality_score: float, quality_filter: Dict) -> bool:
        """Check if pattern passes quality filter requirements"""
        if quality_filter is None:
            return True
        
        # Check minimum score requirement
        min_score = quality_filter.get('min_score')
        if min_score is not None and quality_score < min_score:
            return False
        
        # Check base candle count requirement
        base_candles = quality_filter.get('base_candles')
        if base_candles is not None and pattern['base']['candle_count'] != base_candles:
            return False
        
        # Check minimum leg-out ratio requirement
        min_legout_ratio = quality_filter.get('min_legout_ratio')
        if min_legout_ratio is not None and pattern['leg_out']['ratio_to_base'] < min_legout_ratio:
            return False
        
        return True
    
    def get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get timeframe multiplier for age calculations"""
        timeframe_map = {
            '1D': 1.0, '2D': 2.0, '3D': 3.0, '4D': 4.0, '5D': 5.0,
            'H4': 1/6, 'H8': 1/3, 'H12': 0.5, 'Weekly': 7.0
        }
        return timeframe_map.get(timeframe, 1.0)
    
    def load_data_clean(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Load data using proven data loader - FIXED METHOD NAME"""
        try:
            # Map your timeframe format to DataLoader format
            if timeframe == '1D':
                return self.data_loader.load_pair_data(pair, 'Daily')
            elif timeframe == '2D':
                return self.data_loader.load_pair_data(pair, '2Daily')
            elif timeframe == '3D':
                return self.data_loader.load_pair_data(pair, '3Daily')
            elif timeframe == '4D':
                return self.data_loader.load_pair_data(pair, '4Daily')
            elif timeframe == '5D':
                return self.data_loader.load_pair_data(pair, '5Daily')
            else:
                return self.data_loader.load_pair_data(pair, timeframe)
        except Exception as e:
            print(f"❌ Error loading {pair} {timeframe}: {str(e)}")
            return None
    
    def create_empty_result(self, pair: str, timeframe: str, strategy: str, reason: str) -> Dict:
        """Create empty result structure"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy,
            'description': reason,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0,
            'avg_zone_age_days': 0.0,
            'avg_quality_score': 0.0,
            'quality_distribution': {}
        }
    
    def execute_trades_with_quality_filtering(self, patterns: List[Dict], data: pd.DataFrame,
                                            trend_data: pd.DataFrame, risk_manager: RiskManager,
                                            strategy_config: Dict, timeframe: str) -> List[Dict]:
        """
        Execute trades with both age and quality filtering
        """
        trades = []
        used_zones = set()
        timeframe_multiplier = self.get_timeframe_multiplier(timeframe)
        
        # Calculate quality scores for all patterns
        pattern_quality_scores = {}
        for i, pattern in enumerate(patterns):
            quality_score = self.calculate_zone_quality_score(pattern)
            pattern_quality_scores[i] = quality_score
        
        # Build activation schedule
        zone_activation_schedule = []
        for i, pattern in enumerate(patterns):
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(data):
                zone_activation_schedule.append({
                    'date': data.index[zone_end_idx],
                    'pattern': pattern,
                    'pattern_idx': i,
                    'quality_score': pattern_quality_scores[i],
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}",
                    'zone_end_idx': zone_end_idx
                })
        
        zone_activation_schedule.sort(key=lambda x: x['date'])
        
        # Simulate through time with quality + age filtering
        total_iterations = len(data) - 200
        
        for current_idx in range(200, len(data)):
            current_date = data.index[current_idx]
            
            # Memory check every 1000 iterations
            if current_idx % 1000 == 0:
                progress = ((current_idx - 200) / total_iterations) * 100
            
            # Check each zone for trading opportunities
            for zone_info in zone_activation_schedule:
                pattern = zone_info['pattern']
                zone_id = zone_info['zone_id']
                zone_end_idx = zone_info['zone_end_idx']
                quality_score = zone_info['quality_score']
                
                # Skip if already used or zone hasn't formed yet
                if zone_id in used_zones or zone_end_idx >= current_idx:
                    continue
                
                # Calculate age at this point in time
                zone_formation_date = data.index[zone_end_idx]
                try:
                    age_days = (current_date - zone_formation_date).total_seconds() / (24 * 3600)
                except AttributeError:
                    # Handle case where dates might be integers or other types
                    if isinstance(current_date, (int, float)) and isinstance(zone_formation_date, (int, float)):
                        age_days = abs(current_date - zone_formation_date)  # Assume already in days
                    else:
                        age_days = 30  # Default to 30 days if calculation fails
                
                # Determine age category
                age_category = 'Ancient'
                if age_days <= 7:
                    age_category = 'Ultra_Fresh'
                elif age_days <= 30:
                    age_category = 'Fresh'
                elif age_days <= 90:
                    age_category = 'Recent'
                elif age_days <= 180:
                    age_category = 'Aged'
                elif age_days <= 365:
                    age_category = 'Stale'
                
                zone_age_info = {'age_days': age_days, 'age_category': age_category}
                
                # Apply age filter
                age_filter = strategy_config.get('age_filter')
                if age_filter is not None:
                    if isinstance(age_filter, str):
                        if zone_age_info['age_category'] != age_filter:
                            continue
                    elif isinstance(age_filter, list):
                        if zone_age_info['age_category'] not in age_filter:
                            continue
                
                # Apply quality filter (NEW)
                quality_filter = strategy_config.get('quality_filter')
                if not self.passes_quality_filter(pattern, quality_score, quality_filter):
                    continue
                
                # Check trend alignment
                if current_idx >= len(trend_data):
                    continue
                    
                current_trend = trend_data['trend'].iloc[current_idx]
                is_aligned = (
                    (pattern['type'] in ['R-B-R'] and current_trend == 'bullish') or
                    (pattern['type'] in ['D-B-D'] and current_trend == 'bearish')
                )
                
                if not is_aligned:
                    continue
                
                # Try to execute trade
                trade_result = self.execute_single_trade_proven(
                    pattern, data, current_idx, timeframe_multiplier
                )
                
                if trade_result:
                    # Add quality and age info to trade result
                    trade_result['zone_age_days'] = zone_age_info['age_days']
                    trade_result['zone_age_category'] = zone_age_info['age_category']
                    trade_result['quality_score'] = quality_score
                    trade_result['base_candle_count'] = pattern['base']['candle_count']
                    trade_result['leg_out_ratio'] = pattern['leg_out']['ratio_to_base']
                    trade_result['pattern_strength'] = pattern.get('strength', 0.5)
                    
                    trades.append(trade_result)
                    used_zones.add(zone_id)
                    
                    print(f"   ✅ Trade executed: {pattern['type']} age {zone_age_info['age_days']:.1f}d, quality {quality_score:.2f}")
        
        return trades
    
    def execute_single_trade_proven(self, pattern: Dict, data: pd.DataFrame,
                                   current_idx: int, timeframe_multiplier: float) -> Optional[Dict]:
        """
        Execute single trade using PROVEN entry/exit logic
        """
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        zone_range = zone_high - zone_low
        
        # PROVEN entry and stop logic
        if pattern['type'] == 'R-B-R':  # Demand zone
            entry_price = zone_low + (zone_range * 0.05)  # 5% front-run
            direction = 'BUY'
            initial_stop = zone_low - (zone_range * 0.33)  # 33% buffer
        else:  # D-B-D - Supply zone
            entry_price = zone_high - (zone_range * 0.05)  # 5% front-run
            direction = 'SELL'
            initial_stop = zone_high + (zone_range * 0.33)  # 33% buffer
        
        # Check if current price can trigger entry
        current_candle = data.iloc[current_idx]
        current_low = current_candle['low']
        current_high = current_candle['high']
        
        can_enter = False
        if direction == 'BUY' and current_low <= entry_price:
            can_enter = True
        elif direction == 'SELL' and current_high >= entry_price:
            can_enter = True
        
        if not can_enter:
            return None
        
        # Calculate position size (5% risk)
        risk_amount = 10000 * 0.05  # 5% of $10,000
        pip_value = 0.0001
        stop_distance_pips = abs(entry_price - initial_stop) / pip_value
        
        if stop_distance_pips <= 0:
            return None
        
        position_size = risk_amount / stop_distance_pips
        
        # Set targets (1:2.5 risk reward)
        risk_distance = abs(entry_price - initial_stop)
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * 2.5)
        else:
            target_price = entry_price - (risk_distance * 2.5)
        
        # Simulate trade outcome
        entry_time = data.index[current_idx]
        
        # Look ahead for exit (simplified simulation)
        for exit_idx in range(current_idx + 1, min(current_idx + 100, len(data))):
            exit_candle = data.iloc[exit_idx]
            exit_time = data.index[exit_idx]
            
            # Check stops and targets
            if direction == 'BUY':
                if exit_candle['low'] <= initial_stop:
                    # Stopped out
                    return {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': initial_stop,
                        'result': 'LOSS',
                        'pips': -stop_distance_pips,
                        'position_size': position_size
                    }
                elif exit_candle['high'] >= target_price:
                    # Target hit
                    return {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'result': 'WIN',
                        'pips': stop_distance_pips * 2.5,
                        'position_size': position_size
                    }
            else:  # SELL
                if exit_candle['high'] >= initial_stop:
                    # Stopped out
                    return {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': initial_stop,
                        'result': 'LOSS',
                        'pips': -stop_distance_pips,
                        'position_size': position_size
                    }
                elif exit_candle['low'] <= target_price:
                    # Target hit
                    return {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': target_price,
                        'result': 'WIN',
                        'pips': stop_distance_pips * 2.5,
                        'position_size': position_size
                    }
        
        # Trade still open at end of simulation (treat as neutral)
        return None
    
    def calculate_performance_with_quality(self, trades: List[Dict], pair: str, timeframe: str,
                                         strategy_name: str, strategy_config: Dict) -> Dict:
        """
        Calculate performance metrics including quality analysis
        """
        if not trades:
            return self.create_empty_result(pair, timeframe, strategy_name, "No trades executed")
        
        # Basic performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['result'] == 'WIN'])
        losing_trades = len([t for t in trades if t['result'] == 'LOSS'])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L calculation
        total_pips = sum([t['pips'] for t in trades])
        gross_profit = sum([t['pips'] for t in trades if t['pips'] > 0])
        gross_loss = abs(sum([t['pips'] for t in trades if t['pips'] < 0]))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        total_return = (total_pips * 10)  # Assuming $10 per pip
        
        # Quality-specific metrics
        avg_zone_age = np.mean([t.get('zone_age_days', 0) for t in trades])
        avg_quality_score = np.mean([t.get('quality_score', 0.5) for t in trades])
        
        # Quality distribution
        quality_ranges = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
        quality_distribution = {}
        for min_q, max_q in quality_ranges:
            count = len([t for t in trades if min_q <= t.get('quality_score', 0.5) < max_q])
            quality_distribution[f"{min_q:.1f}-{max_q:.1f}"] = count
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': strategy_config.get('description', ''),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_return': round(total_return, 2),
            'total_pips': round(total_pips, 1),
            'avg_zone_age_days': round(avg_zone_age, 1),
            'avg_quality_score': round(avg_quality_score, 3),
            'quality_distribution': quality_distribution,
            'age_filter': strategy_config.get('age_filter', 'None'),
            'quality_filter': str(strategy_config.get('quality_filter', 'None'))
        }
    
    def run_backtest_with_quality_filters(self, data: pd.DataFrame, patterns: Dict,
                                        trend_data: pd.DataFrame, risk_manager: RiskManager,
                                        strategy_config: Dict, pair: str, timeframe: str,
                                        strategy_name: str) -> Dict:
        """
        Run backtest with quality and age filtering
        """
        # Combine momentum patterns (PROVEN logic)
        momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
        
        # Apply distance filter (PROVEN 2.0x threshold)
        valid_patterns = [
            pattern for pattern in momentum_patterns
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
            and pattern['leg_out']['ratio_to_base'] >= 2.0
        ]
        
        if not valid_patterns:
            return self.create_empty_result(pair, timeframe, strategy_name, "No patterns meet 2.0x distance")
        
        print(f"   📊 Found {len(valid_patterns)} patterns after distance filter")
        
        # Execute trades with quality filtering
        trades = self.execute_trades_with_quality_filtering(
            valid_patterns, data, trend_data, risk_manager, strategy_config, timeframe
        )
        
        # Calculate performance with quality metrics
        return self.calculate_performance_with_quality(
            trades, pair, timeframe, strategy_name, strategy_config
        )
    
    def run_single_test(self, pair: str, timeframe: str, strategy_name: str, days_back: int = 730) -> Dict:
        """
        Run single test with quality analysis
        """
        try:
            print(f"\n🧪 Testing {pair} {timeframe} - {strategy_name}")
            
            # Load data
            data = self.load_data_clean(pair, timeframe)
            if data is None or len(data) < 100:
                return self.create_empty_result(pair, timeframe, strategy_name, "Insufficient data")
            
            # Limit data if needed
            if days_back < 9999:
                max_candles = min(days_back + 365, len(data))
                data = data.iloc[-max_candles:]
            
            # Initialize components using proven logic
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()
            
            risk_manager = RiskManager(account_balance=10000)
            
            # Get strategy configuration
            strategy_config = self.STRATEGIES[strategy_name]
            
            # Run backtest with quality filtering
            result = self.run_backtest_with_quality_filters(
                data, patterns, trend_data, risk_manager, 
                strategy_config, pair, timeframe, strategy_name
            )
            
            return result
            
        except Exception as e:
            return self.create_empty_result(pair, timeframe, strategy_name, f"Error: {str(e)}")


def main():
    """Main function with quality analysis options"""
    
    print("🎯 ZONE QUALITY BACKTESTER")
    print("⚡ Extended from proven backtest_zone_age_marketcond.py")
    print("🔬 5-factor quality scoring + comprehensive strategy combinations")
    
    # System requirements check
    if not check_system_requirements():
        print("❌ Insufficient system resources. Minimum 4GB RAM required.")
        return
    
    print("🏗️  Built from proven trading logic")
    print("🔧 Supports: 1D, 2D, 3D, 4D, 5D, H4, H12, Weekly")
    print("=" * 60)
    
    backtester = ZoneQualityBacktester()
    
    print("\nSelect analysis type:")
    print("1. Quick Quality Test (EURUSD 3D, key strategies)")
    print("2. Quality vs Baseline Comparison")
    print("3. Age vs Quality Strategy Comparison")
    print("4. Complete Quality Analysis (all 12 strategies)")
    print("5. Multi-Timeframe Quality Test")
    print("6. Custom Quality Configuration")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        # Quick quality test
        print("🧪 QUICK QUALITY TEST - Key Strategies:")
        
        key_strategies = ['Baseline', 'High_Quality_Only', 'Premium_Quality', 'Base_1_Only', 'Fresh_HighQuality']
        
        for strategy in key_strategies:
            result = backtester.run_single_test('EURUSD', '3D', strategy, 730)
            
            print(f"\n📊 {strategy}:")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Avg Quality: {result.get('avg_quality_score', 0):.3f}")
            print(f"   Avg Age: {result.get('avg_zone_age_days', 0):.1f} days")
            
            if result['total_trades'] == 0:
                print(f"   ❌ Issue: {result['description']}")
    
    elif choice == '2':
        # Quality vs baseline comparison
        print("\n🔬 QUALITY vs BASELINE COMPARISON")
        
        quality_strategies = ['Baseline', 'High_Quality_Only', 'Premium_Quality', 'Base_1_Only', 'Strong_LegOut_Only']
        results = []
        
        for strategy in quality_strategies:
            result = backtester.run_single_test('EURUSD', '3D', strategy, 730)
            results.append(result)
            
            print(f"   {strategy}: {result['total_trades']} trades, "
                    f"PF {result['profit_factor']:.2f}, Quality {result.get('avg_quality_score', 0):.3f}")
        
        # Find best quality strategy
        successful = [r for r in results if r['total_trades'] > 0]
        if successful:
            baseline = next((r for r in successful if r['strategy'] == 'Baseline'), None)
            quality_only = [r for r in successful if r['strategy'] != 'Baseline']
            
            if baseline and quality_only:
                best_quality = max(quality_only, key=lambda x: x['profit_factor'])
                improvement = ((best_quality['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor'] * 100) if baseline['profit_factor'] > 0 else 0
                print(f"\n🏆 Best Quality Strategy: {best_quality['strategy']}")
                print(f"   PF: {best_quality['profit_factor']:.2f} ({improvement:+.1f}% vs baseline)")
    
    elif choice == '3':
        # Age vs quality comparison
        print("\n⚖️  AGE vs QUALITY STRATEGY COMPARISON")
        
        age_strategies = ['Ultra_Fresh_Only', 'Fresh_Only', 'Combined_Fresh']
        quality_strategies = ['High_Quality_Only', 'Premium_Quality', 'Base_1_Only']
        combined_strategies = ['Fresh_HighQuality', 'Fresh_Premium', 'UltraFresh_Base1']
        
        all_strategies = age_strategies + quality_strategies + combined_strategies
        results = []
        
        for strategy in all_strategies:
            result = backtester.run_single_test('EURUSD', '3D', strategy, 730)
            results.append(result)
            
            category = "AGE" if strategy in age_strategies else "QUALITY" if strategy in quality_strategies else "COMBINED"
            print(f"   [{category}] {strategy}: {result['total_trades']} trades, PF {result['profit_factor']:.2f}")
        
        # Find best in each category
        successful = [r for r in results if r['total_trades'] > 0]
        if successful:
            age_results = [r for r in successful if r['strategy'] in age_strategies]
            quality_results = [r for r in successful if r['strategy'] in quality_strategies]
            combined_results = [r for r in successful if r['strategy'] in combined_strategies]
            
            print(f"\n🏆 CATEGORY WINNERS:")
            if age_results:
                best_age = max(age_results, key=lambda x: x['profit_factor'])
                print(f"   Age: {best_age['strategy']} (PF: {best_age['profit_factor']:.2f})")
            
            if quality_results:
                best_quality = max(quality_results, key=lambda x: x['profit_factor'])
                print(f"   Quality: {best_quality['strategy']} (PF: {best_quality['profit_factor']:.2f})")
            
            if combined_results:
                best_combined = max(combined_results, key=lambda x: x['profit_factor'])
                print(f"   Combined: {best_combined['strategy']} (PF: {best_combined['profit_factor']:.2f})")
    
    elif choice == '4':
        # Complete quality analysis
        print("\n📊 COMPLETE QUALITY ANALYSIS (all 12 strategies)")
        
        all_results = []
        total_strategies = len(backtester.STRATEGIES)
        current_strategy = 0
        
        for strategy_name in backtester.STRATEGIES.keys():
            current_strategy += 1
            print(f"\n🔄 Testing strategy {current_strategy}/{total_strategies}: {strategy_name}")
            
            result = backtester.run_single_test('EURUSD', '3D', strategy_name, 730)
            all_results.append(result)
            
            if result['total_trades'] > 0:
                print(f"   ✅ {result['total_trades']} trades, PF {result['profit_factor']:.2f}, "
                        f"Quality {result.get('avg_quality_score', 0):.3f}")
            else:
                print(f"   ❌ No trades: {result['description']}")
        
        # Create results DataFrame
        df = pd.DataFrame(all_results)
        successful_df = df[df['total_trades'] > 0]
        
        if len(successful_df) > 0:
            # Sort by profit factor
            successful_df = successful_df.sort_values('profit_factor', ascending=False)
            
            print(f"\n🎯 QUALITY ANALYSIS RESULTS:")
            print(f"   Total strategies tested: {len(df)}")
            print(f"   Successful strategies: {len(successful_df)}")
            
            # Top 5 strategies
            print(f"\n🏆 TOP 5 STRATEGIES:")
            for i, (_, row) in enumerate(successful_df.head(5).iterrows()):
                print(f"   {i+1}. {row['strategy']}: PF {row['profit_factor']:.2f}, "
                        f"WR {row['win_rate']:.1f}%, Quality {row.get('avg_quality_score', 0):.3f}")
            
            # Save to Excel
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/zone_quality_analysis_{timestamp}.xlsx"
            os.makedirs('results', exist_ok=True)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All_Results', index=False)
                successful_df.to_excel(writer, sheet_name='Successful_Strategies', index=False)
                
                # Quality analysis
                quality_summary = []
                baseline_pf = df[df['strategy'] == 'Baseline']['profit_factor'].iloc[0] if len(df[df['strategy'] == 'Baseline']) > 0 else 0
                
                for _, row in successful_df.iterrows():
                    if row['strategy'] != 'Baseline':
                        improvement = ((row['profit_factor'] - baseline_pf) / baseline_pf * 100) if baseline_pf > 0 else 0
                        quality_summary.append({
                            'Strategy': row['strategy'],
                            'Profit_Factor': row['profit_factor'],
                            'Improvement_vs_Baseline': f"{improvement:+.1f}%",
                            'Win_Rate': row['win_rate'],
                            'Total_Trades': row['total_trades'],
                            'Avg_Quality_Score': row.get('avg_quality_score', 0),
                            'Avg_Zone_Age_Days': row.get('avg_zone_age_days', 0)
                        })
                
                quality_df = pd.DataFrame(quality_summary)
                quality_df.to_excel(writer, sheet_name='Quality_Analysis', index=False)
            
            print(f"\n📁 EXCEL REPORT SAVED: {filename}")
        else:
            print("❌ No successful strategies found")
    
    elif choice == '5':
        # Multi-timeframe quality test
        print("\n📊 MULTI-TIMEFRAME QUALITY TEST")
        
        timeframes = ['1D', '2D', '3D', '4D', '5D']
        strategy = 'Fresh_HighQuality'  # Best combined strategy
        
        for tf in timeframes:
            print(f"\n🕒 Testing {tf} timeframe...")
            result = backtester.run_single_test('EURUSD', tf, strategy, 730)
            
            if result['total_trades'] > 0:
                print(f"   ✅ {tf}: {result['total_trades']} trades, "
                        f"PF {result['profit_factor']:.2f}, Quality {result.get('avg_quality_score', 0):.3f}")
            else:
                print(f"   ❌ {tf}: {result['description']}")
    
    elif choice == '6':
        # Custom quality configuration
        print("\n🔧 CUSTOM QUALITY CONFIGURATION")
        
        pairs_input = input("Enter pairs (comma-separated, e.g., EURUSD,GBPUSD): ").strip().upper()
        pairs = [p.strip() for p in pairs_input.split(',')] if pairs_input else ['EURUSD']
        
        tf_input = input("Enter timeframes (comma-separated, e.g., 1D,3D,5D): ").strip()
        timeframes = [tf.strip() for tf in tf_input.split(',')] if tf_input else ['3D']
        
        days_input = input("Enter days back (default 730): ").strip()
        days_back = int(days_input) if days_input.isdigit() else 730
        
        strategies_input = input("Enter strategies (comma-separated, default: key strategies): ").strip()
        if strategies_input:
            strategies = [s.strip() for s in strategies_input.split(',')]
        else:
            strategies = ['Baseline', 'High_Quality_Only', 'Premium_Quality', 'Fresh_HighQuality']
        
        print(f"\n📊 Running custom quality analysis:")
        print(f"   Pairs: {pairs}")
        print(f"   Timeframes: {timeframes}")
        print(f"   Strategies: {strategies}")
        print(f"   Days back: {days_back}")
        
        all_results = []
        total_tests = len(pairs) * len(timeframes) * len(strategies)
        current_test = 0
        
        for pair in pairs:
            for timeframe in timeframes:
                for strategy in strategies:
                    current_test += 1
                    print(f"\n🔄 Test {current_test}/{total_tests}: {pair} {timeframe} {strategy}")
                    
                    result = backtester.run_single_test(pair, timeframe, strategy, days_back)
                    all_results.append(result)
                    
                    if result['total_trades'] > 0:
                        print(f"   ✅ {result['total_trades']} trades, PF {result['profit_factor']:.2f}")
                    else:
                        print(f"   ❌ {result['description']}")
        
        # Save custom results
        df = pd.DataFrame(all_results)
        successful_df = df[df['total_trades'] > 0]
        
        if len(successful_df) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/custom_quality_analysis_{timestamp}.xlsx"
            os.makedirs('results', exist_ok=True)
            df.to_excel(filename, index=False)
            print(f"\n📁 CUSTOM RESULTS SAVED: {filename}")
            
            # Print summary
            print(f"\n🎯 CUSTOM ANALYSIS SUMMARY:")
            print(f"   Total tests: {len(df)}")
            print(f"   Successful tests: {len(successful_df)}")
            
            if len(successful_df) > 0:
                best = successful_df.loc[successful_df['profit_factor'].idxmax()]
                print(f"   Best result: {best['pair']} {best['timeframe']} {best['strategy']}")
                print(f"   Performance: PF {best['profit_factor']:.2f}, WR {best['win_rate']:.1f}%")
        else:
            print("❌ No successful tests in custom configuration")
    
    print("\n✅ ZONE QUALITY ANALYSIS COMPLETE!")
    print("🔬 5-factor quality scoring implemented")
    print("📊 Comprehensive strategy comparison available")
    print("📁 Detailed Excel reports saved in results/ folder")


if __name__ == "__main__":
   main()