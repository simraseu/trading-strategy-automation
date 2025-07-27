"""
Baseline Validation Engine - Module 6 Extension
Multi-period validation of current system configuration against live trading baseline
Extends CoreBacktestEngine with period comparison and baseline documentation
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your existing engine
from core_backtest_engine import CoreBacktestEngine

class BaselineValidationEngine(CoreBacktestEngine):
    """
    Extends CoreBacktestEngine with multi-period baseline validation
    Tests current system configuration against live trading performance baseline
    """
    
    def __init__(self):
        """Initialize with baseline validation configuration"""
        super().__init__()
        
        # Live Trading Baseline (Reference Standard)
        self.live_baseline = {
            'profit_factor': 2.5,
            'win_rate': 40.0,
            'trade_management': 'manual_discretionary',
            'sample_period': '3_month_forward_test',
            'trade_count': 20,
            'status': 'proven_live_performance'
        }
        
        # Current System Configuration
        self.current_config = {
            'stop_buffer': 0.33,        # 33% buffer beyond zone boundary
            'entry_front_run': 0.05,    # 5% front-run beyond zone
            'trade_management': '1R_breakeven_to_2.5R_target',
            'penetration_rules': '33%_close_50%_wick',
            'risk_per_trade': 0.05,     # 5% risk per trade
            'development_status': 'parameter_optimization'
        }
        
        # Validation Periods
        self.validation_periods = {
            'primary': {
                'name': '2015-2025 (Modern Market Structure)',
                'days_back': 3847,
                'description': 'Primary validation across modern market era',
                'weight': 0.5  # 50% weight in final assessment
            },
            'validation': {
                'name': '2020-2025 (Recent Performance)',
                'days_back': 2021,
                'description': 'Recent market validation (post-COVID)',
                'weight': 0.35  # 35% weight in final assessment
            },
            'reference': {
                'name': '2018-2025 (Medium-term Reference)',
                'days_back': 2751,
                'description': 'Medium-term consistency check',
                'weight': 0.15  # 15% weight in final assessment
            }
        }
        
        # Performance Tolerance (15% from live baseline)
        self.tolerance = 0.15
        self.target_ranges = {
            'profit_factor': {
                'min': self.live_baseline['profit_factor'] * (1 - self.tolerance),  # 2.125
                'max': self.live_baseline['profit_factor'] * (1 + self.tolerance),  # 2.875
                'target': self.live_baseline['profit_factor']  # 2.5
            },
            'win_rate': {
                'min': self.live_baseline['win_rate'] * (1 - self.tolerance),  # 34.0%
                'max': self.live_baseline['win_rate'] * (1 + self.tolerance),  # 46.0%
                'target': self.live_baseline['win_rate']  # 40.0%
            }
        }
        
        print(f"🎯 BASELINE VALIDATION ENGINE INITIALIZED")
        print(f"📊 Live Baseline: PF {self.live_baseline['profit_factor']:.1f}, WR {self.live_baseline['win_rate']:.1f}%")
        print(f"🎯 Target Ranges: PF {self.target_ranges['profit_factor']['min']:.2f}-{self.target_ranges['profit_factor']['max']:.2f}, WR {self.target_ranges['win_rate']['min']:.1f}%-{self.target_ranges['win_rate']['max']:.1f}%")
    
    def run_multi_period_baseline_validation(self, pairs: List[str] = None, 
                                           timeframes: List[str] = None) -> Dict:
        """
        Run multi-period validation using existing CoreBacktestEngine functionality
        
        Args:
            pairs: Optional list of pairs to test (if None, discovers all)
            timeframes: Optional list of timeframes to test (if None, discovers all)
            
        Returns:
            Complete baseline validation report
        """
        print(f"\n🎯 MULTI-PERIOD BASELINE VALIDATION")
        print(f"🔧 Current System Config: {self.current_config['stop_buffer']:.0%} stop, {self.current_config['entry_front_run']:.0%} entry, {self.current_config['trade_management']}")
        print(f"📊 Live Baseline: PF {self.live_baseline['profit_factor']:.1f}, WR {self.live_baseline['win_rate']:.1f}% ({self.live_baseline['trade_count']} trades)")
        print("=" * 80)
        
        # Auto-discover pairs and timeframes if not provided
        if pairs is None or timeframes is None:
            valid_combinations = self.discover_valid_data_combinations()
            if pairs is None:
                pairs = list(set([combo[0] for combo in valid_combinations]))
            if timeframes is None:
                timeframes = list(set([combo[1] for combo in valid_combinations]))
        
        print(f"📊 Testing {len(pairs)} pairs across {len(timeframes)} timeframes")
        print(f"🔍 Pairs: {', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''}")
        print(f"⏰ Timeframes: {', '.join(timeframes)}")
        
        # Run tests for each validation period
        period_results = {}
        
        for period_key, period_config in self.validation_periods.items():
            print(f"\n📅 TESTING PERIOD: {period_config['name']}")
            print(f"   Days back: {period_config['days_back']:,}")
            print(f"   Weight: {period_config['weight']:.0%}")
            
            # Create test combinations for this period
            test_combinations = []
            for pair in pairs:
                for timeframe in timeframes:
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'days_back': period_config['days_back'],
                        'analysis_period': period_key
                    })
            
            # Run parallel tests using existing engine functionality
            period_results[period_key] = self.run_optimized_parallel_tests(test_combinations)
            
            # Quick period summary
            successful = [r for r in period_results[period_key] if r['total_trades'] > 0]
            if successful:
                avg_pf = np.mean([r['profit_factor'] for r in successful])
                avg_wr = np.mean([r['win_rate'] for r in successful])
                print(f"   ✅ Period Summary: {len(successful)} successful tests, Avg PF {avg_pf:.2f}, Avg WR {avg_wr:.1f}%")
            else:
                print(f"   ❌ Period Summary: No successful tests")
        
        # Generate comprehensive baseline validation report
        validation_report = self.generate_baseline_validation_report(period_results)
        
        return validation_report
    
    def generate_baseline_validation_report(self, period_results: Dict) -> Dict:
        """
        Generate comprehensive baseline validation report with period comparison
        
        Args:
            period_results: Results from each validation period
            
        Returns:
            Complete validation report with recommendations
        """
        print(f"\n📊 GENERATING BASELINE VALIDATION REPORT...")
        
        # Combine all results
        all_results = []
        for period_key, results in period_results.items():
            for result in results:
                result['period'] = period_key
                all_results.append(result)
        
        # Filter successful results only
        successful_results = [r for r in all_results if r['total_trades'] > 0]
        
        if not successful_results:
            return {
                'validation_status': 'FAILED',
                'reason': 'No successful trades across any period',
                'recommendation': 'Review system configuration and data quality'
            }
        
        # Calculate period-specific performance
        period_performance = {}
        for period_key, period_config in self.validation_periods.items():
            period_data = [r for r in successful_results if r['period'] == period_key]
            
            if period_data:
                period_performance[period_key] = {
                    'name': period_config['name'],
                    'weight': period_config['weight'],
                    'test_count': len(period_data),
                    'avg_profit_factor': np.mean([r['profit_factor'] for r in period_data]),
                    'avg_win_rate': np.mean([r['win_rate'] for r in period_data]),
                    'total_trades': sum([r['total_trades'] for r in period_data]),
                    'avg_return': np.mean([r['total_return'] for r in period_data])
                }
            else:
                period_performance[period_key] = {
                    'name': period_config['name'],
                    'weight': period_config['weight'],
                    'test_count': 0,
                    'avg_profit_factor': 0.0,
                    'avg_win_rate': 0.0,
                    'total_trades': 0,
                    'avg_return': 0.0
                }
        
        # Calculate weighted performance (based on period weights)
        weighted_pf = sum([
            period_performance[key]['avg_profit_factor'] * period_performance[key]['weight']
            for key in period_performance.keys()
            if period_performance[key]['test_count'] > 0
        ])
        
        weighted_wr = sum([
            period_performance[key]['avg_win_rate'] * period_performance[key]['weight']
            for key in period_performance.keys()
            if period_performance[key]['test_count'] > 0
        ])
        
        # Baseline validation assessment
        pf_within_range = (self.target_ranges['profit_factor']['min'] <= weighted_pf <= 
                          self.target_ranges['profit_factor']['max'])
        wr_within_range = (self.target_ranges['win_rate']['min'] <= weighted_wr <= 
                          self.target_ranges['win_rate']['max'])
        
        # Overall validation status
        if pf_within_range and wr_within_range:
            validation_status = 'PASSED'
            recommendation = 'System validates against live baseline. Ready for live trading preparation.'
        elif pf_within_range or wr_within_range:
            validation_status = 'PARTIAL'
            recommendation = 'Partial validation. Consider parameter optimization before live trading.'
        else:
            validation_status = 'FAILED'
            recommendation = 'System does not validate against live baseline. Requires optimization.'
        
        # Create comprehensive report
        validation_report = {
            'validation_status': validation_status,
            'recommendation': recommendation,
            'system_configuration': self.current_config,
            'live_baseline': self.live_baseline,
            'target_ranges': self.target_ranges,
            'weighted_performance': {
                'profit_factor': round(weighted_pf, 2),
                'win_rate': round(weighted_wr, 1),
                'pf_vs_target': round(((weighted_pf / self.live_baseline['profit_factor']) - 1) * 100, 1),
                'wr_vs_target': round(weighted_wr - self.live_baseline['win_rate'], 1)
            },
            'period_performance': period_performance,
            'validation_flags': {
                'profit_factor_within_range': pf_within_range,
                'win_rate_within_range': wr_within_range,
                'sufficient_sample_size': sum([p['total_trades'] for p in period_performance.values()]) >= 30
            },
            'all_results': all_results,
            'successful_results_count': len(successful_results),
            'total_results_count': len(all_results)
        }
        
        # Generate Excel report using existing functionality
        self.export_baseline_validation_excel(validation_report)
        
        # Print summary
        self.print_validation_summary(validation_report)
        
        return validation_report
    
    def export_baseline_validation_excel(self, validation_report: Dict):
        """
        Export baseline validation report to Excel using existing framework
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/baseline_validation_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: Validation Summary
                summary_data = []
                summary_data.append(['BASELINE VALIDATION SUMMARY', ''])
                summary_data.append(['Validation Status', validation_report['validation_status']])
                summary_data.append(['Recommendation', validation_report['recommendation']])
                summary_data.append(['', ''])
                summary_data.append(['LIVE BASELINE (Reference)', ''])
                summary_data.append(['Profit Factor', validation_report['live_baseline']['profit_factor']])
                summary_data.append(['Win Rate', f"{validation_report['live_baseline']['win_rate']:.1f}%"])
                summary_data.append(['Trade Count', validation_report['live_baseline']['trade_count']])
                summary_data.append(['', ''])
                summary_data.append(['WEIGHTED PERFORMANCE', ''])
                summary_data.append(['Profit Factor', validation_report['weighted_performance']['profit_factor']])
                summary_data.append(['Win Rate', f"{validation_report['weighted_performance']['win_rate']:.1f}%"])
                summary_data.append(['PF vs Target', f"{validation_report['weighted_performance']['pf_vs_target']:+.1f}%"])
                summary_data.append(['WR vs Target', f"{validation_report['weighted_performance']['wr_vs_target']:+.1f}pp"])
                
                pd.DataFrame(summary_data, columns=['Metric', 'Value']).to_excel(
                    writer, sheet_name='Validation_Summary', index=False)
                
                # SHEET 2: Period Performance
                period_df = pd.DataFrame.from_dict(validation_report['period_performance'], orient='index')
                period_df.to_excel(writer, sheet_name='Period_Performance', index=True)
                
                # SHEET 3: All Results (using existing framework)
                all_results_df = pd.DataFrame(validation_report['all_results'])
                all_results_df.to_excel(writer, sheet_name='All_Results', index=False)
                
                # SHEET 4: Successful Results Only
                successful_results = [r for r in validation_report['all_results'] if r['total_trades'] > 0]
                if successful_results:
                    successful_df = pd.DataFrame(successful_results)
                    successful_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    
                    # SHEET 5: Strategy Analysis
                    strategy_analysis = self.create_strategy_analysis(successful_df)
                    strategy_analysis.to_excel(writer, sheet_name='Strategy_Analysis', index=False)
                    
                    # SHEET 6: Timeframe Analysis
                    timeframe_analysis = self.create_timeframe_analysis(successful_df)
                    timeframe_analysis.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    
                    # SHEET 7: Pair Analysis
                    pair_analysis = self.create_pair_analysis(successful_df)
                    pair_analysis.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    
                    print("   ✅ Strategy Analysis")
                    print("   ✅ Timeframe Analysis") 
                    print("   ✅ Pair Analysis")
                else:
                    # Create empty analysis sheets
                    empty_df = pd.DataFrame({'Note': ['No successful results to analyze']})
                    empty_df.to_excel(writer, sheet_name='Strategy_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ⚠️  Empty analysis sheets (no successful results)")
            
            print(f"📁 BASELINE VALIDATION REPORT: {filename}")
            print(f"📊 7 comprehensive analysis sheets created")
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {str(e)}")
    
    def print_validation_summary(self, validation_report: Dict):
        """Print comprehensive validation summary"""
        print(f"\n🎯 BASELINE VALIDATION SUMMARY")
        print("=" * 60)
        print(f"🏆 VALIDATION STATUS: {validation_report['validation_status']}")
        print(f"💡 RECOMMENDATION: {validation_report['recommendation']}")
        
        print(f"\n📊 PERFORMANCE vs LIVE BASELINE:")
        wp = validation_report['weighted_performance']
        live = validation_report['live_baseline']
        print(f"   Profit Factor: {wp['profit_factor']:.2f} vs {live['profit_factor']:.1f} target ({wp['pf_vs_target']:+.1f}%)")
        print(f"   Win Rate: {wp['win_rate']:.1f}% vs {live['win_rate']:.1f}% target ({wp['wr_vs_target']:+.1f}pp)")
        
        print(f"\n📅 PERIOD BREAKDOWN:")
        for period_key, period_data in validation_report['period_performance'].items():
            if period_data['test_count'] > 0:
                print(f"   {period_data['name']} (Weight: {period_data['weight']:.0%}):")
                print(f"      PF {period_data['avg_profit_factor']:.2f}, WR {period_data['avg_win_rate']:.1f}% ({period_data['total_trades']} trades)")
            else:
                print(f"   {period_data['name']}: No successful tests")
        
        print(f"\n✅ VALIDATION FLAGS:")
        flags = validation_report['validation_flags']
        print(f"   Profit Factor in Range: {'✅' if flags['profit_factor_within_range'] else '❌'}")
        print(f"   Win Rate in Range: {'✅' if flags['win_rate_within_range'] else '❌'}")
        print(f"   Sufficient Sample Size: {'✅' if flags['sufficient_sample_size'] else '❌'}")
        
        print(f"\n📊 TEST STATISTICS:")
        print(f"   Successful Tests: {validation_report['successful_results_count']}/{validation_report['total_results_count']}")
        success_rate = (validation_report['successful_results_count'] / validation_report['total_results_count']) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
    
    def create_strategy_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create overall strategy performance analysis across all dimensions"""
        try:
            # Overall strategy performance
            total_trades = df['total_trades'].sum()
            total_winning = df['winning_trades'].sum()
            total_losing = df['losing_trades'].sum()
            
            # Weighted averages based on trade count
            weights = df['total_trades'] / total_trades
            weighted_pf = (df['profit_factor'] * weights).sum()
            weighted_wr = (df['win_rate'] * weights).sum()
            weighted_return = (df['total_return'] * weights).sum()
            
            # Performance distribution
            pf_above_target = len(df[df['profit_factor'] >= 2.125])  # Within target range
            wr_above_target = len(df[df['win_rate'] >= 34.0])  # Within target range
            
            strategy_data = {
                'Metric': [
                    'Total Strategy Combinations Tested',
                    'Successful Combinations',
                    'Success Rate (%)',
                    'Total Trades Executed',
                    'Total Winning Trades',
                    'Total Losing Trades',
                    'Weighted Average Profit Factor',
                    'Weighted Average Win Rate (%)',
                    'Weighted Average Return (%)',
                    'Combinations Meeting PF Target (≥2.125)',
                    'Combinations Meeting WR Target (≥34%)',
                    'Best Profit Factor',
                    'Best Win Rate (%)',
                    'Best Return (%)',
                    'Worst Profit Factor',
                    'Strategy Consistency Score'
                ],
                'Value': [
                    len(df),
                    len(df),
                    100.0,  # All results here are successful
                    total_trades,
                    total_winning,
                    total_losing,
                    round(weighted_pf, 2),
                    round(weighted_wr, 1),
                    round(weighted_return, 1),
                    pf_above_target,
                    wr_above_target,
                    round(df['profit_factor'].max(), 2),
                    round(df['win_rate'].max(), 1),
                    round(df['total_return'].max(), 1),
                    round(df['profit_factor'].min(), 2),
                    round((pf_above_target / len(df)) * 100, 1)  # % meeting targets
                ]
            }
            
            return pd.DataFrame(strategy_data)
            
        except Exception as e:
            print(f"   ⚠️  Strategy analysis error: {str(e)}")
            return pd.DataFrame({'Metric': ['Error'], 'Value': [str(e)]})

def main():
    """Main function for baseline validation"""
    print("🎯 BASELINE VALIDATION ENGINE")
    print("Validates current system against live trading baseline")
    print("=" * 60)
    
    # Initialize validation engine
    validator = BaselineValidationEngine()
    
    # Check system resources
    if not validator.check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    print("\n🎯 VALIDATION OPTIONS:")
    print("1. Quick Validation (EURUSD + GBPUSD, 3D + 1W only)")
    print("2. Focused Validation (Major pairs, Daily + Weekly)")
    print("3. Comprehensive Validation (All available pairs/timeframes)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Quick validation
        pairs = ['EURUSD', 'GBPUSD']
        timeframes = ['3D', '1W']
        print(f"\n🧪 QUICK VALIDATION: {len(pairs)} pairs, {len(timeframes)} timeframes")
        
    elif choice == '2':
        # Focused validation
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']
        timeframes = ['1D', '3D', '1W']
        print(f"\n🎯 FOCUSED VALIDATION: {len(pairs)} pairs, {len(timeframes)} timeframes")
        
    elif choice == '3':
        # Comprehensive validation
        pairs = None  # Auto-discover all
        timeframes = None  # Auto-discover all
        print(f"\n🚀 COMPREHENSIVE VALIDATION: All available pairs and timeframes")
        print("⚠️  This may take 30+ minutes depending on data availability")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Validation cancelled")
            return
    else:
        print("❌ Invalid choice")
        return
    
    # Run validation
    validation_report = validator.run_multi_period_baseline_validation(pairs, timeframes)
    
    print(f"\n🎯 BASELINE VALIDATION COMPLETE!")
    print(f"📁 Results exported to Excel")
    print(f"🏆 Status: {validation_report['validation_status']}")

if __name__ == "__main__":
    main()