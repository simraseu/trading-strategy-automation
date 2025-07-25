"""
Enhanced Trade Management Backtesting System - Optimized for i5-10400F
Systematic testing of 55+ advanced exit strategies for optimal trading edge discovery
Memory and CPU optimized for 16GB RAM, 6-core/12-thread processor
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
import os
import glob
import time
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Set process priority to high for maximum CPU usage
try:
    if os.name == 'nt':  # Windows
        import ctypes
        # Set high priority
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
except:
    pass

def get_optimized_days_back_input() -> int:
        """Optimized days input for 16GB RAM system"""
        
        print("\nSelect optimized data period (16GB RAM):")
        print("1. Last 730 days (2 years) - Optimal for memory")
        print("2. Last 1095 days (3 years) - Balanced analysis")  
        print("3. Last 365 days (1 year) - Fast processing")
        print("4. All available data (memory permitting)")
        print("5. Custom days")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        memory_percent = psutil.virtual_memory().percent
        
        if choice == '1':
            return 730
        elif choice == '2':
            if memory_percent > 70:
                print("⚠️  High memory usage detected, recommending 730 days instead")
                return 730
            return 1095
        elif choice == '3':
            return 365
        elif choice == '4':
            if memory_percent > 60:
                print("⚠️  Memory usage high, limiting to 2 years for stability")
                return 730
            print("   📊 Using all available data (memory optimized)")
            return 99999
        elif choice == '5':
            days = int(input("Enter custom days: "))
            if days > 1500 and memory_percent > 60:
                print(f"⚠️  Large dataset + high memory usage, limiting to 1095 days")
                return 1095
            return days
        else:
            print("Invalid choice, using optimal 730 days")
            return 730

class OptimizedEnhancedTradeManagementBacktester:
    """Enhanced backtester optimized for i5-10400F (6C/12T) with 16GB RAM"""
    
    # COMPREHENSIVE STRATEGY CONFIGURATIONS (55 strategies)
    ENHANCED_MANAGEMENT_STRATEGIES = {
        
        # ===== BREAK-EVEN VARIATIONS (18 strategies) =====
        'BE_0.5R_TP_1R': {
            'type': 'breakeven',
            'breakeven_at': 0.5,
            'target': 1.0,
            'description': 'Break-even at 0.5R, target at 1R'
        },
        'BE_0.5R_TP_2R': {
            'type': 'breakeven',
            'breakeven_at': 0.5,
            'target': 2.0,
            'description': 'Break-even at 0.5R, target at 2R'
        },
        'BE_0.5R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 0.5,
            'target': 3.0,
            'description': 'Break-even at 0.5R, target at 3R'
        },
        'BE_1.0R_TP_2R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 2.0,
            'description': 'Break-even at 1R, target at 2R (baseline)'
        },
        'BE_1.0R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 3.0,
            'description': 'Break-even at 1R, target at 3R'
        },
        'BE_1.0R_TP_4R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 4.0,
            'description': 'Break-even at 1R, target at 4R'
        },
        'BE_1.0R_TP_5R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 5.0,
            'description': 'Break-even at 1R, target at 5R'
        },
        'BE_1.5R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 1.5,
            'target': 3.0,
            'description': 'Break-even at 1.5R, target at 3R'
        },
        'BE_1.5R_TP_4R': {
            'type': 'breakeven',
            'breakeven_at': 1.5,
            'target': 4.0,
            'description': 'Break-even at 1.5R, target at 4R'
        },
        'BE_2.0R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 2.0,
            'target': 3.0,
            'description': 'Break-even at 2R, target at 3R'
        },
        'BE_2.0R_TP_4R': {
            'type': 'breakeven',
            'breakeven_at': 2.0,
            'target': 4.0,
            'description': 'Break-even at 2R, target at 4R'
        },
        'BE_2.5R_TP_5R': {
            'type': 'breakeven',
            'breakeven_at': 2.5,
            'target': 5.0,
            'description': 'Break-even at 2.5R, target at 5R'
        },
        'BE_3.0R_TP_5R': {
            'type': 'breakeven',
            'breakeven_at': 3.0,
            'target': 5.0,
            'description': 'Break-even at 3R, target at 5R'
        },
        
        # Profit Break-Even Variations
        'ProfitBE_0.25R_TP_2R': {
            'type': 'profit_breakeven',
            'breakeven_trigger': 1.0,
            'profit_be_level': 0.25,
            'target': 2.0,
            'description': 'Move stop to +0.25R profit at 1R, target at 2R'
        },
        'ProfitBE_0.5R_TP_2R': {
            'type': 'profit_breakeven',
            'breakeven_trigger': 1.0,
            'profit_be_level': 0.5,
            'target': 2.0,
            'description': 'Move stop to +0.5R profit at 1R, target at 2R'
        },
        'ProfitBE_0.5R_TP_3R': {
            'type': 'profit_breakeven',
            'breakeven_trigger': 1.0,
            'profit_be_level': 0.5,
            'target': 3.0,
            'description': 'Move stop to +0.5R profit at 1R, target at 3R'
        },
        'ProfitBE_0.75R_TP_3R': {
            'type': 'profit_breakeven',
            'breakeven_trigger': 1.0,
            'profit_be_level': 0.75,
            'target': 3.0,
            'description': 'Move stop to +0.75R profit at 1R, target at 3R'
        },
        'ProfitBE_1.0R_TP_4R': {
            'type': 'profit_breakeven',
            'breakeven_trigger': 1.5,
            'profit_be_level': 1.0,
            'target': 4.0,
            'description': 'Move stop to +1R profit at 1.5R, target at 4R'
        },
        
        # ===== PARTIAL EXIT STRATEGIES (22 strategies) =====
        'Partial_50at1R_Trail_Mom_2R': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 50}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum'],
            'min_trail_distance': 2.0,
            'description': '50% exit at 1R, trail remainder with momentum zones (2R)'
        },
        'Partial_50at1R_Trail_Rev_2R': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 50}],
            'trail_activation': 1.0,
            'trail_zone_types': ['reversal'],
            'min_trail_distance': 2.0,
            'description': '50% exit at 1R, trail remainder with reversal zones (2R)'
        },
        'Partial_50at1R_Trail_Both_2R': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 50}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.0,
            'description': '50% exit at 1R, trail remainder with both zones (2R)'
        },
        'Partial_50at1R_Trail_Both_2.5R': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 50}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': '50% exit at 1R, trail remainder with both zones (2.5R)'
        },
        'Partial_50at1R_Trail_Both_3R': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 50}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.0,
            'description': '50% exit at 1R, trail remainder with both zones (3R)'
        },
        'Partial_50at2R_Trail_Both_2.5R': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 2.0, 'percentage': 50}],
            'trail_activation': 2.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': '50% exit at 2R, trail remainder with both zones (2.5R)'
        },
        
        # Triple Partial Exits
        'Partial_33at1R_33at2R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 1.0, 'percentage': 33},
                {'at_level': 2.0, 'percentage': 33}
            ],
            'trail_activation': 2.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': '33% at 1R, 33% at 2R, trail final 34%'
        },
        'Partial_33at1R_33at3R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 1.0, 'percentage': 33},
                {'at_level': 3.0, 'percentage': 33}
            ],
            'trail_activation': 3.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.0,
            'description': '33% at 1R, 33% at 3R, trail final 34%'
        },
        'Partial_33at1.5R_33at2.5R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 1.5, 'percentage': 33},
                {'at_level': 2.5, 'percentage': 33}
            ],
            'trail_activation': 2.5,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.0,
            'description': '33% at 1.5R, 33% at 2.5R, trail final 34%'
        },
        
        # Quadruple Partial Exits
        'Partial_25at1R_25at2R_25at3R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 1.0, 'percentage': 25},
                {'at_level': 2.0, 'percentage': 25},
                {'at_level': 3.0, 'percentage': 25}
            ],
            'trail_activation': 3.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.0,
            'description': '25% at 1R, 25% at 2R, 25% at 3R, trail final 25%'
        },
        
        # Different Percentage Splits
        'Partial_75at1R_Trail_25': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 75}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.0,
            'description': '75% exit at 1R, trail remaining 25%'
        },
        'Partial_25at1R_Trail_75': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 25}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.0,
            'description': '25% exit at 1R, trail remaining 75%'
        },
        'Partial_60at1.5R_Trail_40': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.5, 'percentage': 60}],
            'trail_activation': 1.5,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': '60% exit at 1.5R, trail remaining 40%'
        },
        'Partial_40at1R_Trail_60': {
            'type': 'partial_trail',
            'partial_exits': [{'at_level': 1.0, 'percentage': 40}],
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.0,
            'description': '40% exit at 1R, trail remaining 60%'
        },
        
        # Partial + Break-Even Combinations
        'Partial_50at1R_BE_1.5R_TP_3R': {
            'type': 'partial_breakeven',
            'partial_exits': [{'at_level': 1.0, 'percentage': 50}],
            'breakeven_at': 1.5,
            'target': 3.0,
            'description': '50% exit at 1R, BE at 1.5R, target at 3R'
        },
        'Partial_33at1R_BE_2R_TP_4R': {
            'type': 'partial_breakeven',
            'partial_exits': [{'at_level': 1.0, 'percentage': 33}],
            'breakeven_at': 2.0,
            'target': 4.0,
            'description': '33% exit at 1R, BE at 2R, target at 4R'
        },
        'Partial_25at1R_BE_1.5R_TP_3R': {
            'type': 'partial_breakeven',
            'partial_exits': [{'at_level': 1.0, 'percentage': 25}],
            'breakeven_at': 1.5,
            'target': 3.0,
            'description': '25% exit at 1R, BE at 1.5R, target at 3R'
        },
        'Partial_60at2R_BE_2.5R_TP_5R': {
            'type': 'partial_breakeven',
            'partial_exits': [{'at_level': 2.0, 'percentage': 60}],
            'breakeven_at': 2.5,
            'target': 5.0,
            'description': '60% exit at 2R, BE at 2.5R, target at 5R'
        },
        
        # Staggered Partials
        'Partial_40at0.5R_40at1.5R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 0.5, 'percentage': 40},
                {'at_level': 1.5, 'percentage': 40}
            ],
            'trail_activation': 1.5,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': '40% at 0.5R, 40% at 1.5R, trail final 20%'
        },
        'Partial_30at1R_30at2R_30at3R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 1.0, 'percentage': 30},
                {'at_level': 2.0, 'percentage': 30},
                {'at_level': 3.0, 'percentage': 30}
            ],
            'trail_activation': 3.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.0,
            'description': '30% at 1R, 30% at 2R, 30% at 3R, trail final 10%'
        },
        'Partial_20at0.5R_50at1.5R_Trail': {
            'type': 'partial_trail',
            'partial_exits': [
                {'at_level': 0.5, 'percentage': 20},
                {'at_level': 1.5, 'percentage': 50}
            ],
            'trail_activation': 1.5,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.0,
            'description': '20% at 0.5R, 50% at 1.5R, trail final 30%'
        },
        
        # ===== ZONE TRAILING REFINEMENTS (10 strategies) =====
        'Trail_1R_1.5R_Both': {
            'type': 'zone_trailing',
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 1.5,
            'description': 'Trail after 1R with 1.5R distance, both zones'
        },
        'Trail_1R_2.0R_Both': {
            'type': 'zone_trailing',
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.0,
            'description': 'Trail after 1R with 2.0R distance, both zones'
        },
        'Trail_1R_2.5R_Both': {
            'type': 'zone_trailing',
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail after 1R with 2.5R distance, both zones'
        },
        'Trail_1R_3.0R_Both': {
            'type': 'zone_trailing',
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.0,
            'description': 'Trail after 1R with 3.0R distance, both zones'
        },
        'Trail_1R_3.5R_Both': {
            'type': 'zone_trailing',
            'trail_activation': 1.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 3.5,
            'description': 'Trail after 1R with 3.5R distance, both zones'
        },
        'Trail_0.5R_2.5R_Mom': {
            'type': 'zone_trailing',
            'trail_activation': 0.5,
            'trail_zone_types': ['momentum'],
            'min_trail_distance': 2.5,
            'description': 'Trail after 0.5R with momentum zones only'
        },
        'Trail_1.5R_2.5R_Rev': {
            'type': 'zone_trailing',
            'trail_activation': 1.5,
            'trail_zone_types': ['reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail after 1.5R with reversal zones only'
        },
        'Trail_2R_3R_Mom': {
            'type': 'zone_trailing',
            'trail_activation': 2.0,
            'trail_zone_types': ['momentum'],
            'min_trail_distance': 3.0,
            'description': 'Trail after 2R with momentum zones, 3R distance'
        },
        'Trail_2R_3R_Rev': {
            'type': 'zone_trailing',
            'trail_activation': 2.0,
            'trail_zone_types': ['reversal'],
            'min_trail_distance': 3.0,
            'description': 'Trail after 2R with reversal zones, 3R distance'
        },
        'Trail_Immediate_2.5R_Both': {
            'type': 'zone_trailing',
            'trail_activation': 0.0,
            'trail_zone_types': ['momentum', 'reversal'],
            'min_trail_distance': 2.5,
            'description': 'Trail immediately with 2.5R distance, both zones'
        },
        
        # ===== SIMPLE TARGETS (5 strategies) =====
        'Simple_1R': {
            'type': 'simple',
            'target': 1.0,
            'description': 'Simple 1R target'
        },
        'Simple_2R': {
            'type': 'simple',
            'target': 2.0,
            'description': 'Simple 2R target'
        },
        'Simple_3R': {
            'type': 'simple',
            'target': 3.0,
            'description': 'Simple 3R target'
        },
        'Simple_4R': {
            'type': 'simple',
            'target': 4.0,
            'description': 'Simple 4R target'
        },
        'Simple_5R': {
            'type': 'simple',
            'target': 5.0,
            'description': 'Simple 5R target'
        }
    }
    
    def __init__(self, max_workers: int = None):
        """Initialize with optimized settings for i5-10400F"""
        self.data_loader = DataLoader()
        
        # Optimal worker count for i5-10400F (6C/12T) with 16GB RAM
        if max_workers is None:
            # Use 10 workers (leave 2 threads for system + GUI)
            self.max_workers = 11
        else:
            self.max_workers = min(max_workers, 11)  # Cap at 11 max
        
        self.results = []
        
        # Memory optimization settings
        self.chunk_size = 1000  # Process in smaller chunks
        self.memory_threshold = 0.85  # Trigger cleanup at 85% RAM usage
        
        print(f"🚀 Optimized for i5-10400F System:")
        print(f"   CPU Cores: 6 (12 threads)")
        print(f"   RAM: 16GB available")
        print(f"   Optimal workers: {self.max_workers}")
        print(f"   Memory threshold: {self.memory_threshold*100:.0f}%")
    
    def check_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed"""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            print(f"⚠️  Memory usage high ({memory_percent*100:.1f}%), triggering cleanup...")
            gc.collect()
            return True
        return False
    
    def detect_zones_after_activation(self, data: pd.DataFrame, activation_idx: int,
                                    trail_zone_types: List[str], min_distance: float,
                                    trade_direction: str, entry_price: float,
                                    risk_distance: float) -> List[Dict]:
        """Memory-optimized zone detection after trail activation"""
        
        if activation_idx >= len(data) - 10:
            return []
        
        # Limit data size for memory efficiency
        post_activation_data = data.iloc[activation_idx:activation_idx + min(500, len(data) - activation_idx)]
        
        if len(post_activation_data) < 10:
            return []
        
        try:
            # Initialize components with memory cleanup
            candle_classifier = CandleClassifier(post_activation_data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            # Clean up intermediate objects
            del candle_classifier, classified_data
            
            valid_zones = []
            
            # Get patterns by type
            if 'momentum' in trail_zone_types:
                momentum_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
                valid_zones.extend(momentum_zones)
            
            if 'reversal' in trail_zone_types:
                reversal_zones = patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', [])
                valid_zones.extend(reversal_zones)
            
            # Filter by distance requirement
            distance_filtered = [
                zone for zone in valid_zones 
                if 'leg_out' in zone and 'ratio_to_base' in zone['leg_out']
                and zone['leg_out']['ratio_to_base'] >= min_distance
            ]
            
            # Proximity filtering with memory cleanup
            if distance_filtered and len(post_activation_data) > 0:
                current_price = post_activation_data['close'].iloc[-1]
                max_trail_distance = risk_distance * 5.0
                
                proximity_filtered = []
                for zone in distance_filtered:
                    zone_center = (zone['zone_high'] + zone['zone_low']) / 2
                    distance_from_price = abs(zone_center - current_price)
                    
                    if distance_from_price <= max_trail_distance:
                        # Adjust indices for original data
                        zone['start_idx'] += activation_idx
                        zone['end_idx'] += activation_idx
                        if 'base' in zone:
                            zone['base']['start_idx'] += activation_idx
                            zone['base']['end_idx'] += activation_idx
                        proximity_filtered.append(zone)
                
                # Memory cleanup
                del distance_filtered
                gc.collect()
                
                return proximity_filtered
            
            return []
            
        except Exception as e:
            # Clean up on error
            gc.collect()
            return []
    
    def select_best_trailing_zone(self, zones: List[Dict], current_price: float, direction: str) -> Optional[Dict]:
        """Optimized zone selection with minimal memory overhead"""
        if not zones:
            return None
        
        best_zone = None
        best_score = -1
        
        for zone in zones:
            zone_center = (zone['zone_high'] + zone['zone_low']) / 2
            distance = abs(zone_center - current_price)
            strength = zone.get('strength', 0.5)
            
            # Simplified scoring for speed
            score = strength / (1 + distance * 10000)
            
            if score > best_score:
                best_score = score
                best_zone = zone
        
        return best_zone
    
    def calculate_zone_trail_stop(self, zone: Dict, direction: str, min_distance: float, risk_distance: float) -> float:
        """Fast zone trailing stop calculation"""
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        if direction == 'BUY':
            return zone_low - (zone_range * 0.33)
        else:
            return zone_high + (zone_range * 0.33)
    
    def is_valid_trail_stop(self, new_stop: float, current_stop: float, direction: str) -> bool:
        """Fast trailing stop validation"""
        if direction == 'BUY':
            return new_stop > current_stop
        else:
            return new_stop < current_stop
    
    def simulate_enhanced_trade_with_partials(self, zone: Dict, entry_price: float,
                                            initial_stop: float, entry_idx: int,
                                            data: pd.DataFrame, strategy_config: Dict,
                                            direction: str, position_size: float,
                                            risk_amount: float) -> Dict:
        """Memory-optimized trade simulation with partial exits"""
        
        # Initialize state variables
        remaining_position = 1.0
        total_pnl = 0.0
        current_stop = initial_stop
        risk_distance = abs(entry_price - initial_stop)
        
        partial_exits_executed = []
        breakeven_moved = False
        trailing_active = False
        
        strategy_type = strategy_config['type']
        
        # Limit simulation length for memory efficiency
        max_simulation_length = min(1000, len(data) - entry_idx - 1)
        
        for i in range(entry_idx + 1, entry_idx + 1 + max_simulation_length):
            if i >= len(data):
                break
                
            candle = data.iloc[i]
            current_date = data.index[i]
            days_held = i - entry_idx
            current_price = candle['close']
            
            # Calculate R:R efficiently
            if direction == 'BUY':
                current_rr = (current_price - entry_price) / risk_distance
            else:
                current_rr = (entry_price - current_price) / risk_distance
            
            # Process partial exits
            if strategy_type in ['partial_trail', 'partial_breakeven'] and 'partial_exits' in strategy_config:
                for exit_config in strategy_config['partial_exits']:
                    exit_level = exit_config['at_level']
                    exit_percentage = exit_config['percentage'] / 100.0
                    
                    exit_key = f"{exit_level}R"
                    already_executed = any(pe['level'] == exit_key for pe in partial_exits_executed)
                    
                    if not already_executed and current_rr >= exit_level:
                        # Execute partial exit
                        exit_amount = remaining_position * exit_percentage
                        remaining_position -= exit_amount
                        
                        if direction == 'BUY':
                            exit_price = entry_price + (risk_distance * exit_level)
                            partial_pnl = (exit_price - entry_price) * position_size * exit_amount
                        else:
                            exit_price = entry_price - (risk_distance * exit_level)
                            partial_pnl = (entry_price - exit_price) * position_size * exit_amount
                       
                        total_pnl += partial_pnl
                        
                        partial_exits_executed.append({
                            'level': exit_key,
                            'percentage': exit_percentage * 100,
                            'amount': exit_amount,
                            'pnl': partial_pnl,
                            'date': current_date,
                            'exit_price': exit_price
                        })
            
            # Check stop loss
            if direction == 'BUY' and candle['low'] <= current_stop:
                final_pnl = (current_stop - entry_price) * position_size * remaining_position
                total_pnl += final_pnl
                
                return self.create_enhanced_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    total_pnl, 'stop_loss', days_held, strategy_config,
                    partial_exits_executed, remaining_position
                )
            
            elif direction == 'SELL' and candle['high'] >= current_stop:
                final_pnl = (entry_price - current_stop) * position_size * remaining_position
                total_pnl += final_pnl
                
                return self.create_enhanced_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    total_pnl, 'stop_loss', days_held, strategy_config,
                    partial_exits_executed, remaining_position
                )
            
            # Break-even management
            if not breakeven_moved:
                if strategy_type == 'profit_breakeven' and 'breakeven_trigger' in strategy_config:
                    be_trigger = strategy_config['breakeven_trigger']
                    if current_rr >= be_trigger:
                        profit_level = strategy_config.get('profit_be_level', 0.5)
                        if direction == 'BUY':
                            current_stop = entry_price + (risk_distance * profit_level)
                        else:
                            current_stop = entry_price - (risk_distance * profit_level)
                        breakeven_moved = True
                
                elif 'breakeven_at' in strategy_config and strategy_config['breakeven_at']:
                    be_level = strategy_config['breakeven_at']
                    if current_rr >= be_level:
                        current_stop = entry_price
                        breakeven_moved = True
            
            # Zone trailing activation
            if strategy_type in ['partial_trail', 'zone_trailing'] and 'trail_activation' in strategy_config:
                trail_activation = strategy_config.get('trail_activation', 1.0)
                
                if not trailing_active and current_rr >= trail_activation:
                    trailing_active = True
                
                if trailing_active:
                    new_zones = self.detect_zones_after_activation(
                        data, i, strategy_config['trail_zone_types'],
                        strategy_config['min_trail_distance'],
                        direction, entry_price, risk_distance
                    )
                    
                    if new_zones:
                        best_zone = self.select_best_trailing_zone(new_zones, current_price, direction)
                        if best_zone:
                            new_stop = self.calculate_zone_trail_stop(
                                best_zone, direction, strategy_config['min_trail_distance'], risk_distance
                            )
                            if self.is_valid_trail_stop(new_stop, current_stop, direction):
                                current_stop = new_stop
            
            # Target check
            if 'target' in strategy_config and strategy_config['target']:
                target_level = strategy_config['target']
                if current_rr >= target_level:
                    if direction == 'BUY':
                        target_price = entry_price + (risk_distance * target_level)
                    else:
                        target_price = entry_price - (risk_distance * target_level)
                    
                    final_pnl = (target_price - entry_price) * position_size * remaining_position if direction == 'BUY' else (entry_price - target_price) * position_size * remaining_position
                    total_pnl += final_pnl
                    
                    return self.create_enhanced_trade_result(
                        entry_idx, data, entry_price, target_price, direction,
                        total_pnl, 'take_profit', days_held, strategy_config,
                        partial_exits_executed, remaining_position
                    )
            
            # Memory check every 100 iterations
            if i % 100 == 0:
                self.check_memory_usage()
        
        # End of data or max length reached
        final_price = data.iloc[min(entry_idx + max_simulation_length, len(data) - 1)]['close']
        final_pnl = (final_price - entry_price) * position_size * remaining_position if direction == 'BUY' else (entry_price - final_price) * position_size * remaining_position
        total_pnl += final_pnl
        
        return self.create_enhanced_trade_result(
            entry_idx, data, entry_price, final_price, direction,
            total_pnl, 'end_of_data', min(max_simulation_length, len(data) - entry_idx - 1), strategy_config,
            partial_exits_executed, remaining_position
        )
    
    def create_enhanced_trade_result(self, entry_idx: int, data: pd.DataFrame,
                                    entry_price: float, exit_price: float, direction: str,
                                    total_pnl: float, exit_reason: str, days_held: int,
                                    strategy_config: Dict, partial_exits: List[Dict],
                                    remaining_position: float) -> Dict:
        """Memory-efficient trade result creation"""
        
        return {
            'entry_date': data.index[entry_idx],
            'entry_price': entry_price,
            'exit_date': data.index[min(entry_idx + days_held, len(data) - 1)],
            'exit_price': exit_price,
            'direction': direction,
            'total_pnl': total_pnl,
            'exit_reason': exit_reason,
            'days_held': days_held,
            'strategy': strategy_config['description'],
            'strategy_type': strategy_config['type'],
            'partial_exits': partial_exits,
            'remaining_position_pct': remaining_position * 100,
            'partial_exit_count': len(partial_exits),
            'partial_exit_pnl': sum(pe['pnl'] for pe in partial_exits),
            'remainder_pnl': total_pnl - sum(pe['pnl'] for pe in partial_exits)
        }
    
    def backtest_with_enhanced_management(self, data: pd.DataFrame, patterns: Dict,
                                        trend_data: pd.DataFrame, risk_manager: RiskManager,
                                        strategy_config: Dict, pair: str, timeframe: str,
                                        strategy_name: str) -> Dict:
        """Memory-optimized backtesting with enhanced management"""

        # Combine patterns efficiently
        momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
        reversal_patterns = patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', [])
        all_patterns = momentum_patterns + reversal_patterns

        # Filter by distance (2.0R threshold)
        valid_patterns = [
            pattern for pattern in all_patterns
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
            and pattern['leg_out']['ratio_to_base'] >= 2.0
        ]

        if not valid_patterns:
            return self.empty_result(pair, timeframe, strategy_name, "No valid patterns")

        # Build activation schedule efficiently
        zone_activation_schedule = []
        for pattern in valid_patterns:
            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is not None and zone_end_idx < len(data):
                zone_activation_schedule.append({
                    'date': data.index[zone_end_idx],
                    'pattern': pattern,
                    'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}"
                })

        zone_activation_schedule.sort(key=lambda x: x['date'])

        # Initialize tracking
        trades = []
        account_balance = 10000
        used_zones = set()

        # Process zones with memory monitoring
        for i, zone_info in enumerate(zone_activation_schedule):
            pattern = zone_info['pattern']
            zone_id = zone_info['zone_id']

            if zone_id in used_zones:
                continue

            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is None or zone_end_idx >= len(trend_data):
                continue

            # Trend alignment check
            current_trend = trend_data['trend'].iloc[zone_end_idx]
            is_aligned = (
                (pattern['type'] in ['R-B-R', 'D-B-R'] and current_trend == 'bullish') or
                (pattern['type'] in ['D-B-D', 'R-B-D'] and current_trend == 'bearish')
            )

            if not is_aligned:
                continue

            # Execute trade
            trade_result = self.execute_enhanced_trade(
                pattern, data, strategy_config, zone_end_idx
            )

            if trade_result:
                trades.append(trade_result)
                account_balance += trade_result['total_pnl']
                used_zones.add(zone_id)

            # Memory check every 10 zones
            if i % 10 == 0:
                self.check_memory_usage()

        return self.calculate_enhanced_performance_metrics(
            trades, account_balance, pair, timeframe, strategy_name, strategy_config
        )
    
    def execute_enhanced_trade(self, pattern: Dict, data: pd.DataFrame,
                                strategy_config: Dict, zone_end_idx: int) -> Optional[Dict]:
        """Memory-optimized trade execution"""
        
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        zone_range = zone_high - zone_low

        # Entry and stop logic
        if pattern['type'] in ['R-B-R', 'D-B-R']:
            entry_price = zone_low + (zone_range * 0.05)
            direction = 'BUY'
            initial_stop = zone_low - (zone_range * 0.33)
        else:
            entry_price = zone_high - (zone_range * 0.05)
            direction = 'SELL'
            initial_stop = zone_high + (zone_range * 0.33)

        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            return None

        risk_amount = 500
        position_size = risk_amount / stop_distance

        # Find entry efficiently
        entry_idx = None
        search_limit = min(100, len(data) - zone_end_idx - 1)  # Limit search for memory
        
        for i in range(zone_end_idx + 1, zone_end_idx + 1 + search_limit):
            if i >= len(data):
                break
                
            candle = data.iloc[i]

            if direction == 'BUY' and candle['low'] <= entry_price:
                entry_idx = i
                break
            elif direction == 'SELL' and candle['high'] >= entry_price:
                entry_idx = i
                break

        if entry_idx is None:
            return None

        return self.simulate_enhanced_trade_with_partials(
            pattern, entry_price, initial_stop, entry_idx, data,
            strategy_config, direction, position_size, risk_amount
        )
    
    def calculate_enhanced_performance_metrics(self, trades: List[Dict], final_balance: float,
                                                pair: str, timeframe: str, strategy_name: str,
                                                strategy_config: Dict) -> Dict:
        """Optimized performance calculation"""
        
        if not trades:
            return self.empty_result(pair, timeframe, strategy_name, "No trades executed")
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['total_pnl'] > 0)
        breakeven_trades = sum(1 for t in trades if t['total_pnl'] == 0)
        losing_trades = total_trades - winning_trades - breakeven_trades
        
        win_rate = (winning_trades / total_trades) * 100
        
        # Partial exit analysis
        trades_with_partials = sum(1 for t in trades if t['partial_exits'])
        total_partial_exits = sum(len(t['partial_exits']) for t in trades)
        
        partial_exit_success_rate = 0
        if total_partial_exits > 0:
            successful_partials = sum(1 for t in trades for pe in t['partial_exits'] if pe['pnl'] > 0)
            partial_exit_success_rate = (successful_partials / total_partial_exits) * 100
        
        # P&L calculations
        total_pnl = sum(t['total_pnl'] for t in trades)
        partial_pnl = sum(t['partial_exit_pnl'] for t in trades)
        remainder_pnl = sum(t['remainder_pnl'] for t in trades)
        
        gross_profit = sum(t['total_pnl'] for t in trades if t['total_pnl'] > 0)
        gross_loss = abs(sum(t['total_pnl'] for t in trades if t['total_pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Return and risk metrics
        total_return = ((final_balance / 10000) - 1) * 100
        expectancy = total_pnl / total_trades
        max_drawdown = self.calculate_max_drawdown_fast(trades)
        
        # Duration analysis (optimized)
        durations = [t['days_held'] for t in trades]
        winning_durations = [t['days_held'] for t in trades if t['total_pnl'] > 0]
        losing_durations = [t['days_held'] for t in trades if t['total_pnl'] < 0]
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': strategy_config['description'],
            'strategy_type': strategy_config['type'],
            
            # Core Performance
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'breakeven_trades': breakeven_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_return': round(total_return, 1),
            'expectancy': round(expectancy, 2),
            'final_balance': round(final_balance, 2),
            
            # P&L Breakdown
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'partial_exit_pnl': round(partial_pnl, 2),
            'remainder_pnl': round(remainder_pnl, 2),
            'partial_pnl_contribution': round((partial_pnl / total_pnl) * 100, 1) if total_pnl != 0 else 0,
            
            # Partial Exit Analysis
            'trades_with_partials': trades_with_partials,
            'trades_with_partials_pct': round((trades_with_partials / total_trades) * 100, 1),
            'total_partial_exits': total_partial_exits,
            'avg_partials_per_trade': round(total_partial_exits / total_trades, 1),
            'partial_exit_success_rate': round(partial_exit_success_rate, 1),
            
            # Risk & Duration
            'max_drawdown': round(max_drawdown, 2),
            'avg_duration_days': round(np.mean(durations), 1) if durations else 0,
            'avg_winner_duration': round(np.mean(winning_durations), 1) if winning_durations else 0,
            'avg_loser_duration': round(np.mean(losing_durations), 1) if losing_durations else 0,
            'median_duration': round(np.median(durations), 1) if durations else 0,
            
            # Raw Data (minimal for memory)
            'trades_data': trades if len(trades) <= 50 else trades[:50]  # Limit for memory
        }
    
    def calculate_max_drawdown_fast(self, trades: List[Dict]) -> float:
        """Fast drawdown calculation"""
        if not trades:
            return 0
        
        cumulative_pnl = 0
        peak = 10000
        max_dd = 0
        
        for trade in trades:
            cumulative_pnl += trade['total_pnl']
            current_balance = 10000 + cumulative_pnl
            
            if current_balance > peak:
                peak = current_balance
            
            drawdown = ((peak - current_balance) / peak) * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def empty_result(self, pair: str, timeframe: str, strategy_name: str, reason: str) -> Dict:
        """Minimal empty result for memory efficiency"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'description': reason,
            'strategy_type': 'failed',
            'total_trades': 0,
            'winning_trades': 0,
            'profit_factor': 0,
            'total_return': 0,
            'final_balance': 10000,
            'trades_with_partials': 0,
            'total_partial_exits': 0,
            'max_drawdown': 0,
            'trades_data': []
        }
    
    def run_single_backtest(self, pair: str, timeframe: str, strategy_name: str,
                            days_back: int = 730) -> Dict:
        """Memory-optimized single backtest"""
        
        try:
            # Load data with memory monitoring
            if timeframe == '1D':
                data = self.data_loader.load_pair_data(pair, 'Daily')
            elif timeframe == '2D':
                data = self.data_loader.load_pair_data(pair, '2Daily')
            elif timeframe == '3D':
                data = self.data_loader.load_pair_data(pair, '3Daily')
            elif timeframe == '4D':
                data = self.data_loader.load_pair_data(pair, '4Daily')
            else:
                data = self.data_loader.load_pair_data(pair, timeframe)
            
            if len(data) < 100:
                return self.empty_result(pair, timeframe, strategy_name, "Insufficient data")
            
            # Limit data size for memory efficiency
            if days_back < 99999:
                max_candles = min(days_back + 365, len(data))  # Add lookback buffer
                data = data.iloc[-max_candles:]
            elif len(data) > 5000:  # Limit very large datasets
                data = data.iloc[-5000:]
            
            # Initialize components with cleanup
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            # Clean up large objects
            del candle_classifier, classified_data
            gc.collect()
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()
            risk_manager = RiskManager(account_balance=10000)
            
            strategy_config = self.ENHANCED_MANAGEMENT_STRATEGIES[strategy_name]
            
            # Run backtest with cleanup
            results = self.backtest_with_enhanced_management(
                data, patterns, trend_data, risk_manager,
                strategy_config, pair, timeframe, strategy_name
            )
            
            # Final cleanup
            del data, patterns, trend_data, risk_manager
            gc.collect()
            
            return results
            
        except Exception as e:
            gc.collect()  # Cleanup on error
            return self.empty_result(pair, timeframe, strategy_name, f"Error: {str(e)}")
    
    def get_all_available_data_files(self) -> List[Dict]:
        """Fast file scanning with memory optimization"""
        
        data_path = self.data_loader.raw_path
        if not os.path.exists(data_path):
            return []
        
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        available_data = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            pair = None
            timeframe = None
            
            # Fast pattern matching
            if 'OANDA_' in filename and ', ' in filename:
                parts = filename.replace('OANDA_', '').replace('.csv', '').split(', ')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1].split('_')[0]
            elif '.raw_' in filename:
                parts = filename.split('.raw_')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1].split('_')[0]
            elif '_' in filename:
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 2:
                    pair = parts[0]
                    timeframe = parts[1]
            
            if pair and timeframe:
                timeframe_map = {
                    '1D': '1D', 'Daily': '1D', '2D': '2D', '2Daily': '2D',
                    '3D': '3D', '3Daily': '3D', '4D': '4D', '4Daily': '4D',
                    'H4': 'H4', '4H': 'H4', 'H12': 'H12', '12H': 'H12',
                    'Weekly': 'Weekly', '1W': 'Weekly'
                }
                
                normalized_tf = timeframe_map.get(timeframe, timeframe)
                
                available_data.append({
                    'pair': pair,
                    'timeframe': normalized_tf,
                    'filename': filename,
                    'filepath': file_path
                })
        
        # Remove duplicates efficiently
        unique_data = []
        seen = set()
        
        for item in available_data:
            key = (item['pair'], item['timeframe'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        unique_data.sort(key=lambda x: (x['pair'], x['timeframe']))
        return unique_data
    
    def run_optimized_parallel_analysis(self, test_all: bool = False, 
                                        pairs: List[str] = None, 
                                        timeframes: List[str] = None,
                                        strategies: List[str] = None,
                                        days_back: int = 730) -> pd.DataFrame:
        """Optimized parallel analysis for i5-10400F"""
        
        print("🚀 OPTIMIZED ENHANCED TRADE MANAGEMENT ANALYSIS")
        print("💻 Optimized for i5-10400F (6C/12T) + 16GB RAM")
        print("🎯 55+ Advanced Exit Strategies")
        print("=" * 70)
        
        # Memory and CPU monitoring
        print(f"💾 Current RAM usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"🖥️  CPU cores available: {psutil.cpu_count()}")
        
        test_combinations = self._prepare_test_combinations_optimized(
            test_all, pairs, timeframes, strategies, days_back
        )
        
        if not test_combinations:
            print("❌ No test combinations found!")
            return pd.DataFrame()
        
        total_tests = len(test_combinations)
        
        # Dynamic worker adjustment based on system load
        current_memory = psutil.virtual_memory().percent
        if current_memory > 70:
            adjusted_workers = max(6, self.max_workers - 2)
            print(f"⚠️  High memory usage detected, reducing workers to {adjusted_workers}")
            self.max_workers = adjusted_workers
        
        estimated_time = (total_tests * 2.5) / self.max_workers  # Optimized timing
        
        print(f"\n📋 OPTIMIZED PERFORMANCE PROJECTION:")
        print(f"   Total tests: {total_tests}")
        print(f"   Strategy count: {len(self.ENHANCED_MANAGEMENT_STRATEGIES)}")
        print(f"   Optimized workers: {self.max_workers}")
        print(f"   Estimated time: {estimated_time/60:.1f} minutes")
        print(f"   Memory threshold: {self.memory_threshold*100:.0f}%")
        
        confirm = input(f"\n🚀 Start optimized analysis? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled.")
            return pd.DataFrame()
        
        # Run optimized parallel processing
        results = self._run_optimized_parallel_tests(test_combinations)
        
        # Convert to DataFrame with memory optimization
        df = pd.DataFrame(results)
        
        # Save and report with cleanup
        self.save_optimized_results_to_excel(df)
        self.generate_optimized_summary_report(df)
        
        # Final cleanup
        gc.collect()
        
        return df
    
    def _prepare_test_combinations_optimized(self, test_all: bool, pairs: List[str], 
                                            timeframes: List[str], strategies: List[str],
                                            days_back: int) -> List[Dict]:
        """Memory-optimized test combination preparation"""
        
        if test_all:
            available_data = self.get_all_available_data_files()
            if not available_data:
                return []
            
            pairs = list(set([item['pair'] for item in available_data]))
            timeframes = list(set([item['timeframe'] for item in available_data]))
        else:
            if pairs is None:
                pairs = ['EURUSD']
            if timeframes is None:
                timeframes = ['3D']
        
        if strategies is None:
            strategies = list(self.ENHANCED_MANAGEMENT_STRATEGIES.keys())
        
        # Create combinations with memory consideration
        test_combinations = []
        for pair in pairs:
            for timeframe in timeframes:
                for strategy in strategies:
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'days_back': days_back
                    })
        
        # Limit total combinations if memory constrained
        max_combinations = 2000  # Reasonable limit for 16GB RAM
        if len(test_combinations) > max_combinations:
            print(f"⚠️  Limiting combinations to {max_combinations} for memory efficiency")
            test_combinations = test_combinations[:max_combinations]
        
        return test_combinations
    
    def _run_optimized_parallel_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """Optimized parallel execution with memory monitoring"""
        
        print(f"\n🔄 Starting optimized parallel execution...")
        start_time = time.time()
        results = []
        
        # Process in chunks for memory management
        chunk_size = min(self.chunk_size, len(test_combinations))
        total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(test_combinations))
            chunk_combinations = test_combinations[chunk_start:chunk_end]
            
            print(f"\n📦 Processing chunk {chunk_idx + 1}/{total_chunks} "
                    f"({len(chunk_combinations)} tests)")
            
            # Memory check before chunk
            memory_percent = psutil.virtual_memory().percent
            print(f"💾 Memory usage: {memory_percent:.1f}%")
            
            if memory_percent > self.memory_threshold * 100:
                print("⚠️  High memory usage, triggering cleanup...")
                gc.collect()
            
            # Process chunk with parallel execution
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_test = {
                    executor.submit(self._run_single_test_static_optimized, test_config): test_config 
                    for test_config in chunk_combinations
                }
                
                chunk_completed = 0
                for future in as_completed(future_to_test):
                    try:
                        result = future.result()
                        results.append(result)
                        chunk_completed += 1
                        
                        # Progress for chunk
                        if chunk_completed % 10 == 0:
                            chunk_progress = (chunk_completed / len(chunk_combinations)) * 100
                            overall_progress = ((chunk_idx * chunk_size + chunk_completed) / len(test_combinations)) * 100
                            print(f"   Chunk progress: {chunk_progress:.1f}% | Overall: {overall_progress:.1f}%", end='\r')
                    
                    except Exception as e:
                        test_config = future_to_test[future]
                        results.append(self.empty_result(
                            test_config['pair'], test_config['timeframe'], 
                            test_config['strategy'], f"Error: {str(e)}"
                        ))
                        chunk_completed += 1
            
            # Cleanup after chunk
            gc.collect()
        
        total_time = time.time() - start_time
        actual_speedup = (len(test_combinations) * 2.5) / total_time
        
        print(f"\n\n✅ Optimized execution complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Tests per second: {len(test_combinations)/total_time:.1f}")
        print(f"   Actual speedup: {actual_speedup:.1f}x")
        print(f"   Final memory usage: {psutil.virtual_memory().percent:.1f}%")
        
        return results
    
    @staticmethod
    def _run_single_test_static_optimized(test_config: Dict) -> Dict:
        """Optimized static method for parallel execution"""
        try:
            # Create fresh instance with limited workers for memory
            backtester = OptimizedEnhancedTradeManagementBacktester(max_workers=1)
            result = backtester.run_single_backtest(
                test_config['pair'],
                test_config['timeframe'], 
                test_config['strategy'],
                test_config['days_back']
            )
            
            # Clean up after test
            del backtester
            gc.collect()
            
            return result
            
        except Exception as e:
            gc.collect()
            return {
                'pair': test_config['pair'],
                'timeframe': test_config['timeframe'],
                'strategy': test_config['strategy'],
                'description': f"Error: {str(e)}",
                'strategy_type': 'failed',
                'total_trades': 0,
                'winning_trades': 0,
                'profit_factor': 0,
                'total_return': 0,
                'final_balance': 10000,
                'error': str(e)
            }
    
    def save_optimized_results_to_excel(self, df: pd.DataFrame):
        """Memory-optimized Excel export"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = r"C:\Users\sim\Desktop\Quant\OTC\trading-strategy-automation\results\enhanced_trademanagement"
        os.makedirs(export_path, exist_ok=True)
        filename = os.path.join(export_path, f"optimized_enhanced_TM_{timestamp}.xlsx")
        
        print(f"\n💾 Saving optimized results to Excel...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Main results sheet (essential columns only)
            essential_columns = [
                'pair', 'timeframe', 'strategy', 'description', 'strategy_type',
                'total_trades', 'win_rate', 'profit_factor', 'total_return', 
                'expectancy', 'max_drawdown', 'trades_with_partials', 
                'partial_pnl_contribution', 'avg_duration_days'
            ]
            
            df_main = df[essential_columns].copy()
            df_main.to_excel(writer, sheet_name='Main_Results', index=False)
            
            # Top performers only (memory efficient)
            successful_df = df[df['total_trades'] > 0].copy()
            if len(successful_df) > 0:
                top_20 = successful_df.nlargest(20, 'profit_factor')
                top_20[essential_columns].to_excel(writer, sheet_name='Top_20', index=False)
                
                # Strategy type summary (aggregated, not raw data)
                strategy_summary = successful_df.groupby('strategy_type').agg({
                    'profit_factor': ['mean', 'std', 'count', 'max'],
                    'win_rate': 'mean',
                    'total_return': 'mean',
                    'partial_pnl_contribution': 'mean'
                }).round(2)
                strategy_summary.to_excel(writer, sheet_name='Strategy_Summary')
                
                # Partial exit summary
                partial_df = successful_df[successful_df['trades_with_partials'] > 0]
                if len(partial_df) > 0:
                    partial_summary = partial_df.groupby('strategy_type').agg({
                        'partial_pnl_contribution': ['mean', 'max'],
                        'trades_with_partials': 'sum',
                        'total_partial_exits': 'sum'
                    }).round(2)
                    partial_summary.to_excel(writer, sheet_name='Partial_Analysis')
        
        print(f"✅ Optimized results saved: {filename}")
        print(f"📊 File size optimized for 16GB RAM system")
        return filename
    
    def generate_optimized_summary_report(self, df: pd.DataFrame):
        """Memory-optimized summary report generation"""
        
        print(f"\n📊 OPTIMIZED ENHANCED ANALYSIS REPORT")
        print("💻 i5-10400F Performance Optimized")
        print("=" * 70)
        
        total_tests = len(df)
        successful_tests = len(df[df['total_trades'] > 0])
        
        print(f"📋 SYSTEM PERFORMANCE:")
        print(f"   Total tests completed: {total_tests}")
        print(f"   Successful strategies: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"   Memory usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Strategy configurations: {len(self.ENHANCED_MANAGEMENT_STRATEGIES)}")
        
        if successful_tests == 0:
            print("❌ No successful strategies found!")
            return
        
        successful_df = df[df['total_trades'] > 0].copy()
        
        # Top 5 performers (memory efficient)
        print(f"\n🏆 TOP 5 STRATEGIES (Optimized Analysis):")
        top_5 = successful_df.nlargest(5, 'profit_factor')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            partial_info = f", Partials: {row['partial_pnl_contribution']:.1f}%" if row.get('total_partial_exits', 0) > 0 else ""
            print(f"   {i}. {row['strategy']}")
            print(f"      {row['pair']} {row['timeframe']} → PF: {row['profit_factor']:.2f}, "
                    f"WR: {row['win_rate']:.1f}%, Return: {row['total_return']:.1f}%{partial_info}")
        
        # Strategy type performance (aggregated)
        print(f"\n🎯 STRATEGY TYPE PERFORMANCE:")
        strategy_types = successful_df.groupby('strategy_type').agg({
            'profit_factor': ['mean', 'count'],
            'win_rate': 'mean',
            'partial_pnl_contribution': 'mean'
        }).round(2)
        
        for strategy_type in strategy_types.index:
            pf_mean = strategy_types.loc[strategy_type, ('profit_factor', 'mean')]
            count = strategy_types.loc[strategy_type, ('profit_factor', 'count')]
            wr_mean = strategy_types.loc[strategy_type, ('win_rate', 'mean')]
            partial_contrib = strategy_types.loc[strategy_type, ('partial_pnl_contribution', 'mean')]
            
            print(f"   {strategy_type.replace('_', ' ').title()}: "
                    f"PF {pf_mean:.2f}, WR {wr_mean:.1f}% ({count} tests)")
            if partial_contrib > 0:
                print(f"      Avg partial contribution: {partial_contrib:.1f}%")
        
        # Key insights (memory efficient calculations)
        best_overall = successful_df.loc[successful_df['profit_factor'].idxmax()]
        best_return = successful_df.loc[successful_df['total_return'].idxmax()]
        
        print(f"\n📈 KEY INSIGHTS:")
        print(f"   🥇 Best profit factor: {best_overall['strategy']} → {best_overall['profit_factor']:.2f}")
        print(f"   💰 Best return: {best_return['strategy']} → {best_return['total_return']:.1f}%")
        
        # Partial exit insights (if applicable)
        partial_strategies = successful_df[successful_df['trades_with_partials'] > 0]
        if len(partial_strategies) > 0:
            avg_partial_contrib = partial_strategies['partial_pnl_contribution'].mean()
            best_partial = partial_strategies.loc[partial_strategies['partial_pnl_contribution'].idxmax()]
            
            print(f"\n🔄 PARTIAL EXIT INSIGHTS:")
            print(f"   Strategies using partials: {len(partial_strategies)} ({len(partial_strategies)/successful_tests*100:.1f}%)")
            print(f"   Average contribution: {avg_partial_contrib:.1f}%")
            print(f"   Best partial strategy: {best_partial['strategy']} → {best_partial['partial_pnl_contribution']:.1f}%")
        
        # Final optimization stats
        print(f"\n💻 OPTIMIZATION PERFORMANCE:")
        print(f"   Peak memory usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Tests per GB RAM: {successful_tests/16:.1f}")
        print(f"   CPU efficiency: Optimized for 6C/12T")
        
        # Cleanup
        gc.collect()
def main_optimized():
    """Optimized main function for i5-10400F system"""
    
    print("🚀 OPTIMIZED ENHANCED TRADE MANAGEMENT SYSTEM")
    print("💻 Optimized for Intel i5-10400F (6C/12T) + 16GB RAM")
    print("🎯 55+ Advanced Exit Strategies")
    print("⚡ Memory & CPU Optimized")
    print("=" * 70)
    
    # System check
    memory_percent = psutil.virtual_memory().percent
    cpu_count = psutil.cpu_count()
    
    print(f"\n💻 SYSTEM STATUS:")
    print(f"   RAM usage: {memory_percent:.1f}% of 16GB")
    print(f"   CPU cores: {cpu_count} available")
    print(f"   Recommended workers: 10 (6C/12T optimized)")
    
    if memory_percent > 80:
        print("⚠️  WARNING: High memory usage detected!")
        print("   Consider closing other applications for optimal performance")
    
    backtester = OptimizedEnhancedTradeManagementBacktester()
    
    print(f"\n📊 OPTIMIZED STRATEGIES: {len(backtester.ENHANCED_MANAGEMENT_STRATEGIES)}")
    
    # Count strategy types efficiently
    strategy_counts = {}
    for config in backtester.ENHANCED_MANAGEMENT_STRATEGIES.values():
        strategy_type = config['type']
        strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
    
    print("\nStrategy Distribution:")
    for strategy_type, count in strategy_counts.items():
        print(f"   {strategy_type.replace('_', ' ').title()}: {count}")
    
    print("\nSelect optimized testing mode:")
    print("1. Single strategy test (minimal memory)")
    print("2. Parallel: Specific pairs/timeframes (balanced)")
    print("3. Parallel: ALL AVAILABLE DATA (memory optimized)")
    print("4. Parallel: Strategy type focus (efficient)")
    print("5. Quick comparison test (fast)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        # Single test with memory monitoring
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (1D, 2D, 3D, 4D): ").strip()
        
        strategy_names = list(backtester.ENHANCED_MANAGEMENT_STRATEGIES.keys())
        print(f"\nSelect strategy (1-{len(strategy_names)}):")
        for i, strategy in enumerate(strategy_names[:10], 1):
            print(f"   {i:2d}. {strategy}")
        if len(strategy_names) > 10:
            print(f"   ... and {len(strategy_names) - 10} more")
        
        strategy_idx = int(input(f"\nStrategy number: ")) - 1
        strategy_name = strategy_names[strategy_idx]
        days_back = get_optimized_days_back_input()
        
        print(f"\n🧪 Testing {strategy_name} on {pair} {timeframe}...")
        start_time = time.time()
        
        result = backtester.run_single_backtest(pair, timeframe, strategy_name, days_back)
        
        elapsed = time.time() - start_time
        final_memory = psutil.virtual_memory().percent
        
        print(f"\n✅ Test completed in {elapsed:.1f} seconds")
        print(f"💾 Final memory usage: {final_memory:.1f}%")
        print(f"📊 Result: {result['total_trades']} trades, {result['win_rate']}% WR, PF: {result['profit_factor']:.2f}")
        
        if result.get('total_partial_exits', 0) > 0:
            print(f"🔄 Partials: {result['total_partial_exits']} exits, {result['partial_pnl_contribution']:.1f}% contribution")
    
    elif choice == '2':
        # Parallel: Specific configuration
        pairs = input("Enter pairs (comma-separated): ").strip().upper().split(',')
        pairs = [p.strip() for p in pairs]
        timeframes = input("Enter timeframes (comma-separated): ").strip().split(',')
        timeframes = [tf.strip() for tf in timeframes]
        days_back = get_optimized_days_back_input()
        
        print(f"\n🔧 Configuration: {len(pairs)} pairs × {len(timeframes)} timeframes × {len(backtester.ENHANCED_MANAGEMENT_STRATEGIES)} strategies")
        estimated_tests = len(pairs) * len(timeframes) * len(backtester.ENHANCED_MANAGEMENT_STRATEGIES)
        print(f"📊 Estimated tests: {estimated_tests}")
        
        df = backtester.run_optimized_parallel_analysis(
            test_all=False, pairs=pairs, timeframes=timeframes, days_back=days_back
        )
    
    elif choice == '3':
        # All available data (memory optimized)
        days_back = get_optimized_days_back_input()
        
        df = backtester.run_optimized_parallel_analysis(
            test_all=True, days_back=days_back
        )
    
    elif choice == '4':
        # Strategy type focus
        print("\nSelect strategy types to focus on:")
        available_types = list(set(config['type'] for config in backtester.ENHANCED_MANAGEMENT_STRATEGIES.values()))
        for i, stype in enumerate(available_types, 1):
            count = sum(1 for config in backtester.ENHANCED_MANAGEMENT_STRATEGIES.values() if config['type'] == stype)
            print(f"   {i}. {stype.replace('_', ' ').title()} ({count} strategies)")
        
        type_indices = input("\nEnter type numbers (comma-separated): ").strip().split(',')
        selected_types = [available_types[int(i)-1] for i in type_indices]
        
        selected_strategies = [
            name for name, config in backtester.ENHANCED_MANAGEMENT_STRATEGIES.items() 
            if config['type'] in selected_types
        ]
        
        pairs = ['EURUSD']  # Focus on main pair
        timeframes = ['3D']  # Optimal timeframe
        days_back = get_optimized_days_back_input()
        
        print(f"\n🎯 Testing {len(selected_strategies)} strategies from {len(selected_types)} types")
        
        df = backtester.run_optimized_parallel_analysis(
            test_all=False, pairs=pairs, timeframes=timeframes, 
            strategies=selected_strategies, days_back=days_back
        )
    
    elif choice == '5':
        # Quick comparison test
        print("\n⚡ Quick comparison: Top strategy types on EURUSD 3D")
        
        # Select representative strategies from each type
        quick_strategies = [
            'BE_1.0R_TP_2R',  # Baseline
            'Partial_50at1R_Trail_Both_2.5R',  # Partial trail
            'Trail_1R_2.5R_Both',  # Zone trailing
            'Simple_3R'  # Simple target
        ]
        
        days_back = 730  # Fixed for speed
        
        df = backtester.run_optimized_parallel_analysis(
            test_all=False, pairs=['EURUSD'], timeframes=['3D'], 
            strategies=quick_strategies, days_back=days_back
        )
    
    print("\n✅ Optimized analysis complete!")
    print("💾 Results saved with memory optimization")
    print("🎯 Check Excel file for comprehensive analysis")
    print(f"💻 Final system status: {psutil.virtual_memory().percent:.1f}% RAM usage")

if __name__ == "__main__":  
    main_optimized()

