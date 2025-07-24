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

# Set process priority for maximum CPU usage
try:
    if os.name == 'nt':  # Windows
        import ctypes
        # Set high priority
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
except:
    pass

# UPDATED BACKTESTING FRAMEWORK - MODULE 6
ANALYSIS_PERIODS = {
    'priority_1': {
        'name': '2015-2025 (10 years) - PRIMARY ANALYSIS',
        'days_back': 3847,
        'description': 'Modern market structure validation'
    },
    'priority_2': {
        'name': '2020-2025 (4 years) - RECENT VALIDATION', 
        'days_back': 2021,
        'description': 'Post-COVID market performance'
    },
    'priority_3': {
        'name': '2018-2025 (6 years) - MEDIUM-TERM VIEW',
        'days_back': 2751,
        'description': 'Mid-term consistency check'
    },
    'priority_4': {
        'name': '2010-2025 (14 years) - FULL MODERN ERA',
        'days_back': 5673,
        'description': 'Complete post-financial crisis era'
    }
}

# LIVE BASELINE PERFORMANCE (Reference Standard)
LIVE_BASELINE = {
    'profit_factor': 2.5,
    'win_rate': 40.0,
    'tolerance': 0.15  # 15% tolerance
}


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

def discover_all_pairs():
    """Discover all available currency pairs from data files using DataLoader path"""
    from modules.data_loader import DataLoader
    
    # Use DataLoader to get the correct path
    data_loader = DataLoader()
    data_path = data_loader.raw_path
    
    print(f"🔍 Searching in: {data_path}")
    
    data_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"📁 Found {len(data_files)} CSV files")
    
    pairs = []
    
    for file in data_files:
        filename = os.path.basename(file)
        
        # Handle your OANDA format: OANDA_AUDCAD, 1D_0d7fe.csv
        if 'OANDA_' in filename and ', ' in filename:
            # Remove OANDA_ prefix and .csv suffix
            clean_name = filename.replace('OANDA_', '').replace('.csv', '')
            # Split on comma-space to get pair
            parts = clean_name.split(', ')
            if len(parts) >= 1:
                pair = parts[0]  # AUDCAD
                if pair not in pairs and len(pair) == 6:  # Valid forex pair format
                    pairs.append(pair)
    
    pairs.sort()
    print(f"📊 Found {len(pairs)} currency pairs: {', '.join(pairs)}")
    
    return pairs


def run_comprehensive_multi_analysis(backtester, days_back, analysis_name):
    """Run comprehensive analysis with parallel processing optimization"""
    print(f"\n🔄 RUNNING {analysis_name}")
    print(f"📊 Days back: {days_back:,}")
    print("=" * 60)
    
    # Discover all pairs
    pairs = discover_all_pairs()
    if not pairs:
        print("❌ No currency pairs found in data/ folder")
        return
    
    # All timeframes and strategies
    timeframes = ['1D', '2D', '3D', '4D', '5D', 'H4', 'H12', 'Weekly']
    strategies = list(backtester.STRATEGIES.keys())
    
    # Create test combinations
    test_combinations = []
    for pair in pairs:
        for timeframe in timeframes:
            for strategy in strategies:
                test_combinations.append({
                    'pair': pair,
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'days_back': days_back,
                    'analysis_name': analysis_name
                })
    
    print(f"📊 Testing {len(pairs)} pairs × {len(timeframes)} timeframes × {len(strategies)} strategies")
    print(f"📊 Total combinations: {len(test_combinations):,}")
    
    # Run optimized parallel processing
    all_results = run_parallel_tests_optimized(backtester, test_combinations)
    
    # Create comprehensive Excel file with multiple tabs
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # IMMEDIATE EMERGENCY SAVE - preserve your 40-minute analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        emergency_filename = f"results/{analysis_name.lower().replace(' ', '_')}_EMERGENCY_SAVE_{timestamp}.xlsx"
        print(f"\n🚨 EMERGENCY SAVE: Preserving your 40-minute analysis...")
        df_all.to_excel(emergency_filename, index=False)
        print(f"✅ EMERGENCY SAVE COMPLETE: {emergency_filename}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{analysis_name.lower().replace(' ', '_')}_comprehensive_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Tab 1: All Results
            df_all.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Tab 2: Quality Analysis (aggregated)
            successful_df = df_all[df_all['total_trades'] > 0]
            if len(successful_df) > 0:
                quality_analysis = create_quality_analysis_tab(successful_df)
                quality_analysis.to_excel(writer, sheet_name='Quality_Analysis', index=False)
                
                # Tab 3: Quality Analysis by Timeframe
                try:
                    quality_by_tf = create_quality_analysis_by_timeframe(successful_df)
                    quality_by_tf.to_excel(writer, sheet_name='Quality_Analysis_by_Timeframe', index=False)
                except Exception as e:
                    print(f"⚠️  Timeframe analysis failed: {e}")
                    # Create basic timeframe summary as fallback
                    tf_summary = successful_df.groupby(['timeframe', 'strategy']).agg({
                        'profit_factor': 'mean',
                        'win_rate': 'mean',
                        'total_trades': 'sum'
                    }).round(2).reset_index()
                    tf_summary.to_excel(writer, sheet_name='Quality_Analysis_by_Timeframe', index=False)
                
                # Tab 4: Quality Analysis by Pair
                quality_by_pair = create_quality_analysis_by_pair(successful_df)
                quality_by_pair.to_excel(writer, sheet_name='Quality_Analysis_by_Pair', index=False)
        
        print(f"\n📁 COMPREHENSIVE RESULTS SAVED: {filename}")
        print(f"📊 4 Excel tabs created with full analysis breakdown")
    
    # Summary
    successful_results = [r for r in all_results if r['total_trades'] > 0]
    print(f"\n🎯 {analysis_name} COMPREHENSIVE SUMMARY:")
    print(f"   Total combinations tested: {len(all_results):,}")
    print(f"   Successful combinations: {len(successful_results):,}")
    print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
    
    if successful_results:
        avg_pf = sum(r['profit_factor'] for r in successful_results) / len(successful_results)
        avg_wr = sum(r['win_rate'] for r in successful_results) / len(successful_results)
        print(f"   Average PF: {avg_pf:.2f}")
        print(f"   Average WR: {avg_wr:.1f}%")
        
        # Best overall performer
        best = max(successful_results, key=lambda x: x['profit_factor'])
        print(f"   Best overall: {best['pair']} {best['timeframe']} {best['strategy']}")
        print(f"   Best performance: PF {best['profit_factor']:.2f}, WR {best['win_rate']:.1f}%")
    
    return successful_results

def run_parallel_tests_optimized(backtester, test_combinations):
    """Optimized parallel processing with memory management"""
    from multiprocessing import Pool
    import time
    
    print(f"\n🔄 Starting optimized parallel execution...")
    start_time = time.time()
    results = []
    
    # Process in chunks for memory management
    chunk_size = backtester.chunk_size
    total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(test_combinations))
        chunk_tests = test_combinations[chunk_start:chunk_end]
        
        print(f"\n📦 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_tests)} tests)")
        
        # Memory check before chunk
        memory_percent = psutil.virtual_memory().percent
        print(f"💾 Memory usage: {memory_percent:.1f}%")
        
        if memory_percent > backtester.memory_threshold * 100:
            print("⚠️  High memory usage, triggering cleanup...")
            gc.collect()
        
        # Process chunk with multiprocessing
        with Pool(processes=backtester.max_workers) as pool:
            chunk_results = pool.map(run_single_test_worker, chunk_tests)
            results.extend(chunk_results)
        
        # Progress tracking
        completed = chunk_end
        progress = (completed / len(test_combinations)) * 100
        print(f"✅ Chunk complete. Progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
        
        # Memory cleanup after each chunk
        gc.collect()
    
    total_time = time.time() - start_time
    success_count = len([r for r in results if r.get('total_trades', 0) > 0])
    
    print(f"\n✅ OPTIMIZED PARALLEL EXECUTION COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"🎯 Success rate: {success_count}/{len(test_combinations)} ({success_count/len(test_combinations)*100:.1f}%)")
    print(f"⚡ Speed: {len(test_combinations)/total_time:.1f} tests/second")
    
    return results

def run_single_test_worker(test_config):
    """Worker function for parallel processing"""
    try:
        # Create fresh backtester instance with proper error handling
        backtester = ZoneQualityBacktester(max_workers=1)
        
        # Verify DataLoader is working
        if not hasattr(backtester, 'data_loader') or backtester.data_loader is None:
            raise Exception(f"DataLoader initialization failed")
        
        result = backtester.run_single_test(
            test_config['pair'],
            test_config['timeframe'],
            test_config['strategy'],
            test_config['days_back']
        )
        
        result['analysis_period'] = test_config['analysis_name']
        
        # Clean up
        del backtester
        gc.collect()
        
        return result
        
    except Exception as e:
        gc.collect()
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'],
            'strategy': test_config['strategy'],
            'analysis_period': test_config['analysis_name'],
            'total_trades': 0,
            'description': f"Worker error: {str(e)}"
        }

def create_quality_analysis_tab(df):
    """Create aggregated quality analysis across all pairs/timeframes"""
    baseline_pf = df[df['strategy'] == 'Baseline']['profit_factor'].mean() if len(df[df['strategy'] == 'Baseline']) > 0 else 1.0
    
    analysis = df.groupby('strategy').agg({
        'profit_factor': 'mean',
        'win_rate': 'mean', 
        'total_trades': 'sum',
        'avg_quality_score': 'mean',
        'avg_zone_age_days': 'mean'
    }).round(3)
    
    analysis['improvement_vs_baseline'] = ((analysis['profit_factor'] - baseline_pf) / baseline_pf * 100).round(1)
    analysis = analysis.sort_values('profit_factor', ascending=False)
    
    # Rename columns to match your format
    analysis.columns = ['Profit_Factor', 'Win_Rate', 'Total_Trades', 'Avg_Quality_Score', 'Avg_Zone_Age_Days', 'Improvement_vs_Baseline']
    analysis['Improvement_vs_Baseline'] = analysis['Improvement_vs_Baseline'].apply(lambda x: f"{x:+.1f}%")
    
    return analysis.reset_index()

def create_quality_analysis_by_timeframe(df):
    """Create quality analysis broken down by timeframe"""
    results = []
    
    for timeframe in df['timeframe'].unique():
        tf_df = df[df['timeframe'] == timeframe]
        baseline_pf = tf_df[tf_df['strategy'] == 'Baseline']['profit_factor'].mean() if len(tf_df[tf_df['strategy'] == 'Baseline']) > 0 else 1.0
        
        tf_analysis = tf_df.groupby('strategy').agg({
            'profit_factor': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum', 
            'avg_quality_score': 'mean',
            'avg_zone_age_days': 'mean'
        }).round(3)
        
        tf_analysis['improvement_vs_baseline'] = ((tf_analysis['profit_factor'] - baseline_pf) / baseline_pf * 100).round(1)
        tf_analysis['timeframe'] = timeframe
        
        results.append(tf_analysis.reset_index())
    
    combined = pd.concat(results, ignore_index=True)
    combined['Improvement_vs_Baseline'] = combined['improvement_vs_baseline'].apply(lambda x: f"{x:+.1f}%")
    combined = combined.sort_values(['timeframe', 'profit_factor'], ascending=[True, False])
    
    # Rename columns
    combined.columns = ['Strategy', 'Profit_Factor', 'Win_Rate', 'Total_Trades', 'Avg_Quality_Score', 'Avg_Zone_Age_Days', 'Improvement_vs_Baseline', 'Timeframe']
    return combined[['Timeframe', 'Strategy', 'Profit_Factor', 'Win_Rate', 'Total_Trades', 'Avg_Quality_Score', 'Avg_Zone_Age_Days', 'Improvement_vs_Baseline']]

def create_quality_analysis_by_pair(df):
    """Create quality analysis broken down by pair"""
    results = []
    
    for pair in df['pair'].unique():
        pair_df = df[df['pair'] == pair]
        baseline_pf = pair_df[pair_df['strategy'] == 'Baseline']['profit_factor'].mean() if len(pair_df[pair_df['strategy'] == 'Baseline']) > 0 else 1.0
        
        pair_analysis = pair_df.groupby('strategy').agg({
            'profit_factor': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum',
            'avg_quality_score': 'mean', 
            'avg_zone_age_days': 'mean'
        }).round(3)
        
        pair_analysis['improvement_vs_baseline'] = ((pair_analysis['profit_factor'] - baseline_pf) / baseline_pf * 100).round(1)
        pair_analysis['pair'] = pair
        
        results.append(pair_analysis.reset_index())
    
    combined = pd.concat(results, ignore_index=True)
    combined['Improvement_vs_Baseline'] = combined['improvement_vs_baseline'].apply(lambda x: f"{x:+.1f}%")
    combined = combined.sort_values(['pair', 'profit_factor'], ascending=[True, False])
    
    # Rename columns
    combined.columns = ['Strategy', 'Profit_Factor', 'Win_Rate', 'Total_Trades', 'Avg_Quality_Score', 'Avg_Zone_Age_Days', 'Improvement_vs_Baseline', 'Pair']
    return combined[['Pair', 'Strategy', 'Profit_Factor', 'Win_Rate', 'Total_Trades', 'Avg_Quality_Score', 'Avg_Zone_Age_Days', 'Improvement_vs_Baseline']]

def results_consistent(primary, recent, tolerance=0.15):
    """Check if results are within 15% tolerance"""
    
    pf_diff = abs(primary['profit_factor'] - recent['profit_factor']) / primary['profit_factor']
    wr_diff = abs(primary['win_rate'] - recent['win_rate']) / primary['win_rate']
    
    print(f"\n📋 CONSISTENCY ANALYSIS:")
    print(f"   PF Difference: {pf_diff*100:.1f}% (tolerance: {tolerance*100:.0f}%)")
    print(f"   WR Difference: {wr_diff*100:.1f}% (tolerance: {tolerance*100:.0f}%)")
    
    return pf_diff <= tolerance and wr_diff <= tolerance

def evaluate_results(results):
    """Evaluate if results meet baseline standards"""
    pf = results.get('profit_factor', 0)
    wr = results.get('win_rate', 0)
    trades = results.get('total_trades', 0)
    
    baseline_pf = LIVE_BASELINE['profit_factor']
    baseline_wr = LIVE_BASELINE['win_rate']
    tolerance = LIVE_BASELINE['tolerance']
    
    pf_ok = pf >= baseline_pf * (1 - tolerance)
    wr_ok = wr >= baseline_wr * (1 - tolerance)
    trades_ok = trades >= 5
    
    print(f"\n📊 BASELINE EVALUATION:")
    print(f"   Profit Factor: {pf:.2f} (min: {baseline_pf * (1 - tolerance):.2f}) {'✅' if pf_ok else '❌'}")
    print(f"   Win Rate: {wr:.1f}% (min: {baseline_wr * (1 - tolerance):.1f}%) {'✅' if wr_ok else '❌'}")
    print(f"   Trade Count: {trades} (min: 5) {'✅' if trades_ok else '❌'}")
    
    return pf_ok and wr_ok and trades_ok

def run_decision_framework(backtester):
    """Implement 4-step decision framework"""
    print("\n🎯 BACKTESTING DECISION FRAMEWORK")
    print("=" * 50)
    
    # Step 1: Primary Analysis (2015-2025)
    print("\n📊 STEP 1: PRIMARY ANALYSIS (2015-2025)")
    primary_results = backtester.run_single_test('EURUSD', '3D', 'Fresh_HighQuality', 3847)
    
    if evaluate_results(primary_results):
        print("✅ Primary results GOOD - Proceeding to validation")
        
        # Step 2: Recent Validation (2020-2025)  
        print("\n📊 STEP 2: RECENT VALIDATION (2020-2025)")
        recent_results = backtester.run_single_test('EURUSD', '3D', 'Fresh_HighQuality', 2021)
        
        if results_consistent(primary_results, recent_results):
            print("✅ CONSISTENT RESULTS - IMPLEMENT STRATEGY")
            return 'IMPLEMENT', primary_results, recent_results
        else:
            print("⚠️  DIVERGENT RESULTS - Investigating...")
            
            # Step 4: Medium-term Investigation
            print("\n📊 STEP 4: MEDIUM-TERM INVESTIGATION (2018-2025)")
            medium_results = backtester.run_single_test('EURUSD', '3D', 'Fresh_HighQuality', 2751)
            
            return 'INVESTIGATE', primary_results, recent_results, medium_results
    else:
        print("❌ Primary results insufficient")
        return 'REJECT', primary_results


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
        # Import here to avoid circular imports
        from modules.data_loader import DataLoader
        
        # Initialize DataLoader with correct path
        self.data_loader = DataLoader()

        # CPU optimization for i5-10400F (6C/12T)
        available_cores = cpu_count()
        if available_cores >= 12:  # Hyperthreaded 6-core
            self.max_workers = 10  # Leave 2 threads for system
        elif available_cores >= 6:
            self.max_workers = available_cores - 1
        else:
            self.max_workers = max(1, available_cores - 1)
        
        # Memory optimization settings
        self.chunk_size = 100  # Process in chunks
        self.memory_threshold = 0.75  # 75% memory trigger cleanup
                
        print(f"🎯 ZONE QUALITY BACKTESTER INITIALIZED")
        print(f"   💡 5-factor quality scoring system")
        print(f"   🔄 {len(self.STRATEGIES)} quality + age strategies")
        print(f"   ⚡ {self.max_workers} parallel workers (optimized)")
        print(f"   💾 Memory threshold: {self.memory_threshold*100:.0f}%")
    
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
    def save_results_emergency(all_results, analysis_name):
        """Emergency save function to preserve your 40-minute analysis"""
        if all_results:
            df_all = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/{analysis_name.lower().replace(' ', '_')}_EMERGENCY_SAVE_{timestamp}.xlsx"
            os.makedirs('results', exist_ok=True)
            
            print(f"\n🚨 EMERGENCY SAVE: Preserving your 40-minute analysis...")
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Tab 1: All Results (this will definitely work)
                df_all.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Tab 2: Successful only
                successful_df = df_all[df_all['total_trades'] > 0]
                if len(successful_df) > 0:
                    successful_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    
                    # Tab 3: Basic analysis without complex column operations
                    basic_analysis = successful_df.groupby('strategy').agg({
                        'profit_factor': 'mean',
                        'win_rate': 'mean',
                        'total_trades': 'sum'
                    }).round(2).reset_index()
                    basic_analysis.to_excel(writer, sheet_name='Basic_Strategy_Analysis', index=False)
            
            print(f"✅ EMERGENCY SAVE COMPLETE: {filename}")
            return filename
        return None

def main():
    """Updated main function with simplified 4-option framework"""
    
    print("🎯 ZONE QUALITY BACKTESTING - MODULE 6")
    print("🏗️  4-Priority Decision System - All Pairs Analysis")
    print("=" * 60)
    
    # System requirements check
    if not check_system_requirements():
        print("❌ Insufficient system resources. Minimum 4GB RAM required.")
        return
    
    # Initialize backtester
    backtester = ZoneQualityBacktester()
    
    print("\nSelect analysis mode:")
    print("1. Quick Validation (EURUSD only - test functionality)")
    print("2. Priority 1: 2015-2025 (10 years) - PRIMARY ANALYSIS (Days back: 3,847)")
    print("3. Priority 2: 2020-2025 (4 years) - RECENT VALIDATION (Days back: 2,021)")
    print("4. Priority 3: 2018-2025 (6 years) - MEDIUM-TERM VIEW (Days back: 2,751)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Quick validation - EURUSD only with 5 key strategies (same as original)
        print("\n🧪 QUICK VALIDATION - Key Strategies Test:")
        
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
        
        print(f"\n✅ QUICK VALIDATION COMPLETE - All 5 strategies tested")
        
    elif choice == '2':
        # Priority 1: Primary Analysis
        run_comprehensive_multi_analysis(backtester, 3847, "Priority_1_Primary_Analysis")
        
    elif choice == '3':
        # Priority 2: Recent Validation
        run_comprehensive_multi_analysis(backtester, 2021, "Priority_2_Recent_Validation")
        
    elif choice == '4':
        # Priority 3: Medium-term View
        run_comprehensive_multi_analysis(backtester, 2751, "Priority_3_Medium_Term_View")
        
    else:
        print("❌ Invalid choice. Please select 1-4.")
        return
        # Decision Framework Summary
    print(f"\n🎯 YOUR DECISION FRAMEWORK:")
    print(f"   Step 1: Run Priority 2 (3,847 days) → Review results")
    print(f"   Step 2: If good → Run Priority 3 (2,021 days) validation")
    print(f"   Step 3: If consistent → ✅ IMPLEMENT")
    print(f"   Step 4: If divergent → Investigate with Priority 4 (2,751 days)")
    
    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()