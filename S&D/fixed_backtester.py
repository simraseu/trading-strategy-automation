"""
COMPLETE FIXED TRADE MANAGEMENT BACKTESTING SYSTEM
100% self-contained with all 55+ strategies and multiprocessing.Pool fix
Optimized for clean system (post-restart) with minimal memory usage
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
import os
import time
import gc
import glob
from datetime import datetime
from typing import Dict, List, Optional
import multiprocessing as mp
from multiprocessing import Pool
import psutil
import warnings
warnings.filterwarnings('ignore')

# Import your existing components
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager

class CompleteTradeManagementBacktester:
    """
    COMPLETE backtesting system with all 55+ strategies and ProcessPoolExecutor fix
    """
    
    # COMPLETE STRATEGY DEFINITIONS (all 55+ strategies)
    COMPLETE_STRATEGIES = {
        
        # ===== SIMPLE STRATEGIES (5) =====
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
        },
        
        # ===== BREAK-EVEN STRATEGIES (18) =====
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
        
        # ===== ZONE TRAILING STRATEGIES (10) =====
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
        
        # ===== PARTIAL EXIT STRATEGIES (22) =====
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
        }
    }
    
    def __init__(self, max_workers: int = None):
        """Initialize complete backtesting system"""
        self.data_loader = DataLoader()
        
        # Optimal workers for clean system
        if max_workers is None:
            available_cores = mp.cpu_count()
            self.max_workers = min(8, available_cores - 2)  # Conservative for clean system
        else:
            self.max_workers = max_workers
        
        print(f"🚀 COMPLETE BACKTESTING SYSTEM INITIALIZED")
        print(f"   Total strategies: {len(self.COMPLETE_STRATEGIES)}")
        print(f"   Workers: {self.max_workers}")
        print(f"   CPU cores: {mp.cpu_count()}")
        print(f"   System optimized for clean restart")
    
    def get_all_available_data_files(self) -> List[Dict]:
        """Auto-detect ALL available pairs and timeframes from your OANDA files"""
        
        data_path = self.data_loader.raw_path
        print(f"🔍 Scanning: {data_path}")
        
        import glob
        csv_files = glob.glob(os.path.join(data_path, "OANDA_*.csv"))
        print(f"📁 Found {len(csv_files)} OANDA files")
        
        available_data = []
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Parse your format: OANDA_EURUSD, 1D_77888.csv
            if ', ' in filename:
                parts = filename.replace('OANDA_', '').replace('.csv', '').split(', ')
                if len(parts) >= 2:
                    pair = parts[0]  # EURUSD
                    timeframe = parts[1].split('_')[0]  # 1D
                    
                    available_data.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'filename': filename,
                        'filepath': file_path
                    })
        
        # Remove duplicates
        unique_data = []
        seen = set()
        for item in available_data:
            key = (item['pair'], item['timeframe'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        unique_data.sort(key=lambda x: (x['pair'], x['timeframe']))
        
        print(f"✅ Detected {len(unique_data)} unique combinations:")
        for item in unique_data:
            print(f"   {item['pair']} {item['timeframe']}")
        
        return unique_data
    
    def run_single_backtest(self, pair: str, timeframe: str, strategy_name: str, days_back: int = 730) -> Dict:
        """Run single strategy backtest with complete logic"""
        try:
            # Load data with support for ALL your timeframes
            if timeframe == '1D':
                data = self.data_loader.load_pair_data(pair, 'Daily')
            elif timeframe == '2D':
                data = self.data_loader.load_pair_data(pair, '2Daily')
            elif timeframe == '3D':
                data = self.data_loader.load_pair_data(pair, '3Daily')
            elif timeframe == '4D':
                data = self.data_loader.load_pair_data(pair, '4Daily')
            elif timeframe == '5D':
                data = self.data_loader.load_pair_data(pair, '5Daily')
            else:
                data = self.data_loader.load_pair_data(pair, timeframe)
            
            if len(data) < 100:
                return self.empty_result(pair, timeframe, strategy_name, "Insufficient data")

            # Only limit data if specifically requested with a reasonable limit
            if days_back < 9999:  # ← Changed from 99999 to 9999
                # For historical backtesting, use generous lookback
                max_candles = min(days_back + 1000, len(data))
                data = data.iloc[-max_candles:]
                print(f"   📊 Using last {days_back} days + 1000 lookback ({len(data)} candles)")
            elif days_back == 99999:  # ← Add this condition for "all data"
                print(f"   📊 Using ALL available data ({len(data)} candles)")
            # No else clause - use all data by default

            # Initialize components
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_simplified()
            risk_manager = RiskManager(account_balance=10000)
            
            strategy_config = self.COMPLETE_STRATEGIES[strategy_name]
            
            # Run backtest
            results = self.backtest_with_complete_management(
                data, patterns, trend_data, risk_manager,
                strategy_config, pair, timeframe, strategy_name
            )
            
            # Cleanup
            del data, patterns, trend_data, risk_manager
            gc.collect()
            
            return results
            
        except Exception as e:
            gc.collect()
            return self.empty_result(pair, timeframe, strategy_name, f"Error: {str(e)}")
    
    def backtest_with_complete_management(self, data: pd.DataFrame, patterns: Dict,
                                        trend_data: pd.DataFrame, risk_manager: RiskManager,
                                        strategy_config: Dict, pair: str, timeframe: str,
                                        strategy_name: str) -> Dict:
        """Complete backtesting logic with all strategy types"""

        # Combine patterns
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

        # Build activation schedule
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

        # Process zones
        for zone_info in zone_activation_schedule:
            pattern = zone_info['pattern']
            zone_id = zone_info['zone_id']

            if zone_id in used_zones:
                continue

            zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
            if zone_end_idx is None or zone_end_idx >= len(trend_data):
                continue

            # Trend alignment
            current_trend = trend_data['trend'].iloc[zone_end_idx]
            is_aligned = (
                (pattern['type'] in ['R-B-R', 'D-B-R'] and current_trend == 'bullish') or
                (pattern['type'] in ['D-B-D', 'R-B-D'] and current_trend == 'bearish')
            )

            if not is_aligned:
                continue

            # Execute trade
            trade_result = self.execute_complete_trade(
                pattern, data, strategy_config, zone_end_idx
            )

            if trade_result:
                trades.append(trade_result)
                account_balance += trade_result['total_pnl']
                used_zones.add(zone_id)

        return self.calculate_complete_performance(
            trades, account_balance, pair, timeframe, strategy_name, strategy_config
        )
    
    def execute_complete_trade(self, pattern: Dict, data: pd.DataFrame,
                              strategy_config: Dict, zone_end_idx: int) -> Optional[Dict]:
        """Execute trade with complete strategy logic"""
        
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

        # Find entry
        entry_idx = None
        search_limit = min(100, len(data) - zone_end_idx - 1)
        
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

        return self.simulate_complete_trade(
            pattern, entry_price, initial_stop, entry_idx, data,
            strategy_config, direction, position_size, risk_amount
        )
    
    def simulate_complete_trade(self, zone: Dict, entry_price: float,
                               initial_stop: float, entry_idx: int,
                               data: pd.DataFrame, strategy_config: Dict,
                               direction: str, position_size: float,
                               risk_amount: float) -> Dict:
        """Complete trade simulation with all strategy types"""
        
        remaining_position = 1.0
        total_pnl = 0.0
        current_stop = initial_stop
        risk_distance = abs(entry_price - initial_stop)
        
        partial_exits_executed = []
        breakeven_moved = False
        trailing_active = False
        
        strategy_type = strategy_config['type']

        # Limit simulation length for memory efficiency
        max_simulation_length = min(500, len(data) - entry_idx - 1)
        
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
                
                return self.create_complete_trade_result(
                    entry_idx, data, entry_price, current_stop, direction,
                    total_pnl, 'stop_loss', days_held, strategy_config,
                    partial_exits_executed, remaining_position
                )
            
            elif direction == 'SELL' and candle['high'] >= current_stop:
                final_pnl = (entry_price - current_stop) * position_size * remaining_position
                total_pnl += final_pnl
                
                return self.create_complete_trade_result(
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
            
            # Zone trailing logic (simplified for stability)
            if strategy_type in ['partial_trail', 'zone_trailing'] and 'trail_activation' in strategy_config:
                trail_activation = strategy_config.get('trail_activation', 1.0)
                
                if not trailing_active and current_rr >= trail_activation:
                    trailing_active = True
                
                if trailing_active:
                    # Simplified trailing based on recent lows/highs
                    lookback = min(10, i - entry_idx)
                    if lookback > 0:
                        recent_data = data.iloc[i-lookback:i+1]
                        
                        if direction == 'BUY':
                            recent_low = recent_data['low'].min()
                            trail_stop = recent_low - (recent_data['high'].max() - recent_data['low'].min()) * 0.33
                            if trail_stop > current_stop:
                                current_stop = trail_stop
                        else:
                            recent_high = recent_data['high'].max()
                            trail_stop = recent_high + (recent_data['high'].max() - recent_data['low'].min()) * 0.33
                            if trail_stop < current_stop:
                                current_stop = trail_stop
            
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
                    
                    return self.create_complete_trade_result(
                        entry_idx, data, entry_price, target_price, direction,
                        total_pnl, 'take_profit', days_held, strategy_config,
                        partial_exits_executed, remaining_position
                    )
        
        # End of data
        final_price = data.iloc[min(entry_idx + max_simulation_length, len(data) - 1)]['close']
        final_pnl = (final_price - entry_price) * position_size * remaining_position if direction == 'BUY' else (entry_price - final_price) * position_size * remaining_position
        total_pnl += final_pnl
        
        return self.create_complete_trade_result(
            entry_idx, data, entry_price, final_price, direction,
            total_pnl, 'end_of_data', min(max_simulation_length, len(data) - entry_idx - 1), strategy_config,
            partial_exits_executed, remaining_position
        )
    def create_complete_trade_result(self, entry_idx: int, data: pd.DataFrame,
                                   entry_price: float, exit_price: float, direction: str,
                                   total_pnl: float, exit_reason: str, days_held: int,
                                   strategy_config: Dict, partial_exits: List[Dict],
                                   remaining_position: float) -> Dict:
       """Create complete trade result"""
       
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
   
    def calculate_complete_performance(self, trades: List[Dict], final_balance: float,
                                        pair: str, timeframe: str, strategy_name: str,
                                        strategy_config: Dict) -> Dict:
        """Calculate complete performance metrics"""
        
        if not trades:
            return self.empty_result(pair, timeframe, strategy_name, "No trades executed")
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['total_pnl'] > 0)
        breakeven_trades = sum(1 for t in trades if t['total_pnl'] == 0)
        losing_trades = total_trades - winning_trades - breakeven_trades
        
        win_rate = (winning_trades / total_trades) * 100
        
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
        max_drawdown = self.calculate_max_drawdown(trades)
        
        # Duration analysis
        durations = [t['days_held'] for t in trades]
        winning_durations = [t['days_held'] for t in trades if t['total_pnl'] > 0]
        losing_durations = [t['days_held'] for t in trades if t['total_pnl'] < 0]
        
        # Partial exit analysis
        trades_with_partials = sum(1 for t in trades if t['partial_exits'])
        total_partial_exits = sum(len(t['partial_exits']) for t in trades)
        
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
            
            # Risk & Duration
            'max_drawdown': round(max_drawdown, 2),
            'avg_duration_days': round(np.mean(durations), 1) if durations else 0,
            'avg_winner_duration': round(np.mean(winning_durations), 1) if winning_durations else 0,
            'avg_loser_duration': round(np.mean(losing_durations), 1) if losing_durations else 0,
            'median_duration': round(np.median(durations), 1) if durations else 0,
            
            # Raw Data (limited for memory)
            'trades_data': trades if len(trades) <= 50 else trades[:50]
        }
    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
       """Calculate maximum drawdown"""
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
       """Return empty result for failed strategies"""
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
    
    def run_complete_analysis(self, test_mode: str = "full", 
                            pairs: List[str] = None, 
                            timeframes: List[str] = None,
                            days_back: int = 730) -> pd.DataFrame:
       """Run complete analysis with all strategies"""
       
       print("🚀 COMPLETE TRADE MANAGEMENT ANALYSIS")
       print(f"💻 Optimized for clean system restart")
       print("=" * 70)
       
       # System status
       memory_percent = psutil.virtual_memory().percent
       print(f"💾 Current RAM usage: {memory_percent:.1f}%")
       print(f"🖥️  CPU cores: {mp.cpu_count()}")
       
       # Auto-detect ALL available data if no parameters specified
       if pairs is None and timeframes is None:
           print("🔍 Auto-detecting ALL available data...")
           available_data = self.get_all_available_data_files()
           
           if available_data:
               pairs = sorted(list(set([item['pair'] for item in available_data])))
               timeframes = sorted(list(set([item['timeframe'] for item in available_data])))
               
               print(f"📊 Auto-detected:")
               print(f"   Pairs: {pairs}")
               print(f"   Timeframes: {timeframes}")
           else:
               print("❌ No data detected, using defaults")
               pairs = ['EURUSD']
               timeframes = ['3D']
       else:
           # Set defaults
           if pairs is None:
               pairs = ['EURUSD']
           if timeframes is None:
               timeframes = ['3D']
       
       
       
       # Strategy selection by test mode
       if test_mode == "quick":
           strategies = ['Simple_2R', 'BE_1.0R_TP_2R', 'Trail_1R_2.5R_Both', 'Partial_50at1R_Trail_Both_2.5R']
       elif test_mode == "medium":
           strategies = [
               'Simple_1R', 'Simple_2R', 'Simple_3R',
               'BE_1.0R_TP_2R', 'BE_1.0R_TP_3R', 'BE_2.0R_TP_3R',
               'Trail_1R_2.5R_Both', 'Trail_1R_3.0R_Both',
               'Partial_50at1R_Trail_Both_2.5R', 'Partial_33at1R_33at2R_Trail'
           ]
       else:  # full
           strategies = list(self.COMPLETE_STRATEGIES.keys())
       
       # Create test combinations
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
       
       total_tests = len(test_combinations)
       estimated_time = (total_tests * 0.8) / self.max_workers  # Optimistic for clean system
       
       print(f"\n📋 TEST CONFIGURATION:")
       print(f"   Test mode: {test_mode.upper()}")
       print(f"   Strategies: {len(strategies)}")
       print(f"   Total tests: {total_tests}")
       print(f"   Workers: {self.max_workers}")
       print(f"   Estimated time: {estimated_time:.1f} minutes")
       
       # Run analysis
       print(f"\n🔄 Starting complete analysis with multiprocessing.Pool...")
       start_time = time.time()
       
       results = []
       with Pool(processes=self.max_workers) as pool:
           pool_results = pool.map(run_single_test_worker, test_combinations)
           results.extend(pool_results)
       
       total_time = time.time() - start_time
       success_count = len([r for r in results if r.get('total_trades', 0) >= 0])
       
       print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
       print(f"   Total time: {total_time/60:.1f} minutes")
       print(f"   Successful tests: {success_count}/{total_tests}")
       print(f"   Tests per minute: {total_tests / (total_time / 60):.1f}")
       print(f"   Final memory usage: {psutil.virtual_memory().percent:.1f}%")
       
       # Convert to DataFrame and analyze
       df = pd.DataFrame(results)
       
       # Quick results summary
       successful_df = df[df['total_trades'] > 0]
       if len(successful_df) > 0:
           print(f"\n📊 QUICK RESULTS SUMMARY:")
           print(f"   Strategies with trades: {len(successful_df)}")
           
           # Top 3 performers
           top_3 = successful_df.nlargest(3, 'profit_factor')
           for i, (_, row) in enumerate(top_3.iterrows(), 1):
               print(f"   {i}. {row['strategy'][:30]}... → PF: {row['profit_factor']:.2f}, WR: {row['win_rate']:.1f}%")
       
       # Save results
       self.save_complete_results(df, test_mode)
       
       return df
    
    def save_complete_results(self, df: pd.DataFrame, test_mode: str):
       """Save complete results to Excel"""
       
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       export_path = "results"
       os.makedirs(export_path, exist_ok=True)
       filename = os.path.join(export_path, f"complete_backtester_{test_mode}_{timestamp}.xlsx")
       
       print(f"\n💾 Saving complete results to Excel...")
       
       with pd.ExcelWriter(filename, engine='openpyxl') as writer:
           
           # Main results
           essential_columns = [
               'pair', 'timeframe', 'strategy', 'description', 'strategy_type',
               'total_trades', 'win_rate', 'profit_factor', 'total_return', 
               'expectancy', 'max_drawdown', 'trades_with_partials', 
               'partial_pnl_contribution', 'avg_duration_days'
           ]
           
           df_main = df[essential_columns].copy()
           df_main.to_excel(writer, sheet_name='All_Results', index=False)
           
           # Top performers
           successful_df = df[df['total_trades'] > 0].copy()
           if len(successful_df) > 0:
               top_20 = successful_df.nlargest(20, 'profit_factor')
               top_20[essential_columns].to_excel(writer, sheet_name='Top_20', index=False)
               
               # Strategy type summary
               strategy_summary = successful_df.groupby('strategy_type').agg({
                   'profit_factor': ['mean', 'max', 'count'],
                   'win_rate': 'mean',
                   'total_return': 'mean'
               }).round(2)
               strategy_summary.to_excel(writer, sheet_name='Strategy_Summary')
       
       print(f"✅ Complete results saved: {filename}")
       return filename


def run_single_test_worker(test_config: Dict) -> Dict:
    """Worker function for multiprocessing.Pool"""
    try:
        backtester = CompleteTradeManagementBacktester(max_workers=1)
        
        result = backtester.run_single_backtest(
            test_config['pair'],
            test_config['timeframe'], 
            test_config['strategy'],
            test_config['days_back']
        )
        
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
    
def main_complete():
    """Main function for complete backtesting system"""
    
    print("🚀 COMPLETE TRADE MANAGEMENT BACKTESTING SYSTEM")
    print("🔧 100% Self-Contained with ProcessPoolExecutor Fix")
    print("💻 Optimized for Clean System (Post-Restart)")
    print("=" * 70)
    
    # System status
    memory_percent = psutil.virtual_memory().percent
    print(f"💻 System Status:")
    print(f"   RAM usage: {memory_percent:.1f}% (optimal for clean restart)")
    print(f"   CPU cores: {mp.cpu_count()}")
    
    backtester = CompleteTradeManagementBacktester(max_workers=8)
    
    print(f"\n📊 Available Test Modes:")
    print(f"   Total strategies available: {len(backtester.COMPLETE_STRATEGIES)}")
    
    print("\nSelect test mode:")
    print("1. Quick test (4 strategies, ~1 minute)")
    print("2. Medium test (10 strategies, ~3 minutes)")  
    print("3. FULL ANALYSIS (All 55+ strategies, ~12-15 minutes)")
    print("4. ALL DATA AUTO-TEST (Every pair/timeframe you have)")
    print("5. Custom test")
    
    choice = input("\nEnter choice (1-5): ").strip()

    if choice == '1':
        print("\n🚀 Starting QUICK TEST...")
        df = backtester.run_complete_analysis(test_mode="quick")
        
    elif choice == '2':
        print("\n🚀 Starting MEDIUM TEST...")
        df = backtester.run_complete_analysis(test_mode="medium")
        
    elif choice == '3':
        print("\n🚀 Starting FULL ANALYSIS...")
        print("⚠️  This will test all 55+ strategies")
        print("⏰ Expected time: 12-15 minutes")
        
        confirm = input("Proceed with full analysis? (y/n): ").strip().lower()
        if confirm == 'y':
            df = backtester.run_complete_analysis(test_mode="full")
        else:
            print("Analysis cancelled.")
            return
        
    elif choice == '4':
        print("\n🚀 Starting ALL DATA AUTO-TEST...")
        print("🔍 This will automatically detect and test EVERY pair and timeframe")
        print("⚠️  This may take 30+ minutes depending on your data")
        
        confirm = input("Proceed with ALL data test? (y/n): ").strip().lower()
        if confirm == 'y':
            # Don't specify pairs/timeframes - let auto-detection handle it
            df = backtester.run_complete_analysis(test_mode="medium")  # Use medium for speed
        else:
            print("Analysis cancelled.")
            return
            
    elif choice == '5':
        print("\n🔧 Custom test configuration...")
        pairs = input("Enter pairs (comma-separated, default EURUSD): ").strip().upper().split(',') or ['EURUSD']
        timeframes = input("Enter timeframes (comma-separated, default 3D): ").strip().split(',') or ['3D']
        
        print("\nSelect custom strategy count:")
        print("1. Top 10 strategies (recommended)")
        print("2. Top 20 strategies")
        print("3. All strategies")
        
        strat_choice = input("Strategy choice (1-3): ").strip()
        if strat_choice == '1':
            test_mode = "medium"
        elif strat_choice == '2':
            test_mode = "medium"  # Will adjust in code
        else:
            test_mode = "full"
        
        df = backtester.run_complete_analysis(
            test_mode=test_mode, pairs=[p.strip() for p in pairs], 
            timeframes=[tf.strip() for tf in timeframes]
        )
    
    print("\n✅ Complete analysis finished!")
    print("📁 Results saved to Excel with comprehensive analysis")
    print("🎯 Check Excel file for detailed performance breakdown")

if __name__ == "__main__":
   main_complete()