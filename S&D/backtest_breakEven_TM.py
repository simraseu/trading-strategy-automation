"""
BREAK-EVEN STRATEGY COMPREHENSIVE ANALYSIS TOOL
Analyzes all break-even combinations from backtesting data with detailed win/break-even/loss rate breakdowns
Built on top of fixed_backtester.py system
Author: Trading Strategy Automation Project
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your existing components
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from fixed_backtester_TM import CompleteTradeManagementBacktester, run_single_test_worker

class BreakEvenAnalysisEngine:
    """
    COMPREHENSIVE BREAK-EVEN STRATEGY ANALYSIS ENGINE
    Focus: Triple-rate analysis (Win/Break-Even/Loss rates) for all break-even combinations
    """
    
    # BREAK-EVEN STRATEGY DEFINITIONS (Filtered from complete set)
    BREAKEVEN_STRATEGIES = {
        # Basic Break-Even Strategies
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
        'BE_1.0R_TP_1.5R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 1.5,
            'description': 'Break-even at 1R, target at 1.5R'
        },
        'BE_1.0R_TP_2R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 2.0,
            'description': 'Break-even at 1R, target at 2R (baseline)'
        },
        'BE_1.0R_TP_2.5R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 2.5,
            'description': 'Break-even at 1R, target at 2.5R'
        },
        'BE_1.0R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 3.0,
            'description': 'Break-even at 1R, target at 3R'
        },
        'BE_1.0R_TP_3.5R': {
            'type': 'breakeven',
            'breakeven_at': 1.0,
            'target': 3.5,
            'description': 'Break-even at 1R, target at 3.5R'
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
        'BE_1.5R_TP_2.5R': {
            'type': 'breakeven',
            'breakeven_at': 1.5,
            'target': 2.5,
            'description': 'Break-even at 1.5R, target at 2.5R'
        },
        'BE_1.5R_TP_3R': {
            'type': 'breakeven',
            'breakeven_at': 1.5,
            'target': 3.0,
            'description': 'Break-even at 1.5R, target at 3R'
        },
        'BE_1.5R_TP_3.5R': {
            'type': 'breakeven',
            'breakeven_at': 1.5,
            'target': 3.5,
            'description': 'Break-even at 1.5R, target at 3.5R'
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
        'BE_2.0R_TP_3.5R': {
            'type': 'breakeven',
            'breakeven_at': 2.0,
            'target': 3.5,
            'description': 'Break-even at 2R, target at 3.5R'
        },
        'BE_2.0R_TP_4R': {
            'type': 'breakeven',
            'breakeven_at': 2.0,
            'target': 4.0,
            'description': 'Break-even at 2R, target at 4R'
        },
        'BE_2.0R_TP_5R': {
            'type': 'breakeven',
            'breakeven_at': 2.0,
            'target': 5.0,
            'description': 'Break-even at 2R, target at 5R'
        },
        'BE_2.5R_TP_4R': {
            'type': 'breakeven',
            'breakeven_at': 2.5,
            'target': 4.0,
            'description': 'Break-even at 2.5R, target at 4R'
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
        }
    }

    def __init__(self):
        """Initialize the Break-Even Analysis Engine"""
        self.data_loader = DataLoader()
        self.results_data = []
        
        print("🎯 BREAK-EVEN STRATEGY ANALYSIS ENGINE")
        print("=" * 60)
        print(f"📊 Break-even strategies to analyze: {len(self.BREAKEVEN_STRATEGIES)}")
        print("🎲 Triple-rate focus: Win/Break-Even/Loss breakdown")
        
    def get_forex_pairs_only(self) -> List[str]:
        """Get only forex pairs from available data"""
        backtester = CompleteTradeManagementBacktester()
        available_data = backtester.get_all_available_data_files()
        
        # Get all pairs without exclusion
        forex_pairs = [item['pair'] for item in available_data]
        
        # Remove duplicates and sort
        forex_pairs = sorted(list(set(forex_pairs)))
        
        print(f"📈 Detected forex pairs: {forex_pairs}")
        return forex_pairs
    
    def run_comprehensive_breakeven_analysis(self, 
                                           pairs: List[str] = None, 
                                           timeframes: List[str] = None,
                                           days_back: int = 730) -> pd.DataFrame:
        """
        Run comprehensive break-even analysis with triple-rate breakdown
        """
        
        print(f"\n🚀 COMPREHENSIVE BREAK-EVEN ANALYSIS")
        print("🎯 Focus: Win/Break-Even/Loss Rate Breakdown")
        print("=" * 70)
        
        # Auto-detect forex pairs if not specified
        if pairs is None:
            pairs = self.get_forex_pairs_only()  # <- get all pairs, no slicing
            print(f"📊 Auto-selected pairs: {pairs}")


        if timeframes is None:
            timeframes = ['3D']  # Optimal timeframe from your testing
            print(f"⏰ Using timeframe: {timeframes}")
        
        # Create test combinations (BREAKEVEN ONLY)
        test_combinations = []
        for pair in pairs:
            for timeframe in timeframes:
                for strategy_name in self.BREAKEVEN_STRATEGIES.keys():
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'strategy': strategy_name,
                        'days_back': days_back
                    })
        
        total_tests = len(test_combinations)
        estimated_time = (total_tests * 0.8) / 8  # Optimistic timing
        
        print(f"\n📋 BREAK-EVEN TEST CONFIGURATION:")
        print(f"   Strategies: {len(self.BREAKEVEN_STRATEGIES)} break-even only")
        print(f"   Pairs: {len(pairs)} forex pairs")
        print(f"   Total tests: {total_tests}")
        print(f"   Estimated time: {estimated_time:.1f} minutes")
        
        # Run analysis using your existing multiprocessing system
        print(f"\n🔄 Starting break-even analysis...")
        
        from multiprocessing import Pool
        import time
        
        start_time = time.time()
        results = []
        
        # Use multiprocessing Pool (same as your fixed_backtester.py)
        with Pool(processes=8) as pool:
            pool_results = pool.map(self.run_breakeven_test_worker, test_combinations)
            results.extend(pool_results)
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ BREAK-EVEN ANALYSIS COMPLETE!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Successful tests: {success_count}/{total_tests}")
        
        # Store results for analysis
        self.results_data = results
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Generate comprehensive analysis
        self.generate_comprehensive_breakeven_report(df)
        
        return df
    
    def run_breakeven_test_worker(self, test_config: Dict) -> Dict:
        """Worker function for break-even testing (same pattern as your system)"""
        try:
            # Create backtester with modified strategies (BREAKEVEN ONLY)
            backtester = CompleteTradeManagementBacktester(max_workers=1)
            
            # CRITICAL: Replace the strategy set with BREAKEVEN ONLY
            backtester.COMPLETE_STRATEGIES = self.BREAKEVEN_STRATEGIES
            
            result = backtester.run_single_backtest(
                test_config['pair'],
                test_config['timeframe'], 
                test_config['strategy'],
                test_config['days_back']
            )
            
            # Enhanced break-even analysis
            result = self.enhance_result_with_breakeven_analysis(result, test_config)
            
            del backtester
            import gc
            gc.collect()
            
            return result
            
        except Exception as e:
            import gc
            gc.collect()
            return {
                'pair': test_config['pair'],
                'timeframe': test_config['timeframe'],
                'strategy': test_config['strategy'],
                'description': f"Error: {str(e)}",
                'strategy_type': 'failed',
                'total_trades': 0,
                'error': str(e)
            }
    
    def enhance_result_with_breakeven_analysis(self, result: Dict, test_config: Dict) -> Dict:
        """
        Enhance result with detailed break-even rate analysis
        CRITICAL: Calculate Win/Break-Even/Loss rates
        """
        
        if result.get('total_trades', 0) == 0:
            # No trades - return empty enhanced result
            result.update({
                'be_trigger': 0,
                'profit_target': 0,
                'win_rate_detailed': 0,
                'breakeven_rate': 0,
                'loss_rate': 0,
                'effective_win_rate': 0,
                'breakeven_success_rate': 0,
                'loss_prevention_score': 0
            })
            return result
        
        # Extract strategy parameters
        strategy_name = test_config['strategy']
        strategy_config = self.BREAKEVEN_STRATEGIES[strategy_name]
        
        be_trigger = strategy_config['breakeven_at']
        profit_target = strategy_config['target']
        
        # Analyze trades data for break-even behavior
        trades_data = result.get('trades_data', [])
        
        if trades_data:
            # Calculate detailed rates
            win_trades = [t for t in trades_data if t.get('total_pnl', 0) > 50]  # Significant profit
            breakeven_trades = [t for t in trades_data if -50 <= t.get('total_pnl', 0) <= 50]  # Near breakeven
            loss_trades = [t for t in trades_data if t.get('total_pnl', 0) < -50]  # Significant loss
            
            total_trades = len(trades_data)
            
            if total_trades > 0:
                win_rate_detailed = (len(win_trades) / total_trades) * 100
                breakeven_rate = (len(breakeven_trades) / total_trades) * 100
                loss_rate = (len(loss_trades) / total_trades) * 100
                effective_win_rate = win_rate_detailed + breakeven_rate
                
                # Break-even effectiveness metrics
                trades_reaching_be = [t for t in trades_data 
                                    if t.get('exit_reason') in ['take_profit', 'stop_loss'] 
                                    and t.get('days_held', 0) > 1]  # Proxy for reaching BE level
                
                breakeven_success_rate = (len(trades_reaching_be) / total_trades) * 100 if total_trades > 0 else 0
                
                # Loss prevention score (how much BE feature reduces losses)
                loss_prevention_score = 100 - loss_rate  # Higher is better
            else:
                win_rate_detailed = breakeven_rate = loss_rate = 0
                effective_win_rate = breakeven_success_rate = loss_prevention_score = 0
        else:
            win_rate_detailed = breakeven_rate = loss_rate = 0
            effective_win_rate = breakeven_success_rate = loss_prevention_score = 0
        
        # Add enhanced metrics to result
        result.update({
            'be_trigger': be_trigger,
            'profit_target': profit_target,
            'win_rate_detailed': round(win_rate_detailed, 1),
            'breakeven_rate': round(breakeven_rate, 1),
            'loss_rate': round(loss_rate, 1),
            'effective_win_rate': round(effective_win_rate, 1),
            'breakeven_success_rate': round(breakeven_success_rate, 1),
            'loss_prevention_score': round(loss_prevention_score, 1),
            'rate_validation': round(win_rate_detailed + breakeven_rate + loss_rate, 1)  # Should equal ~100%
        })
        
        return result
    
    def generate_comprehensive_breakeven_report(self, df: pd.DataFrame):
        """
        Generate comprehensive break-even analysis report
        PHASE 1-7 IMPLEMENTATION
        """
        
        print(f"\n📊 COMPREHENSIVE BREAK-EVEN ANALYSIS REPORT")
        print("🎯 Triple-Rate Analysis: Win/Break-Even/Loss Breakdown")
        print("=" * 80)
        
        # Filter to successful tests only
        successful_df = df[df['total_trades'] > 0].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful break-even strategies found!")
            return
        
        print(f"📈 Successful break-even strategies: {len(successful_df)}")
        
        # PHASE 1: Break-Even Combination Matrix
        print(f"\n🔥 PHASE 1: BREAK-EVEN COMBINATION MATRIX")
        print("=" * 50)
        
        be_triggers = sorted(successful_df['be_trigger'].unique())
        profit_targets = sorted(successful_df['profit_target'].unique())
        
        print(f"Break-even triggers tested: {be_triggers}R")
        print(f"Profit targets tested: {profit_targets}R")
        print(f"Total combinations analyzed: {len(successful_df)} combinations")
        
        # PHASE 2: Triple-Rate Analysis (KEY FOCUS)
        print(f"\n🎲 PHASE 2: TRIPLE-RATE ANALYSIS")
        print("=" * 50)
        
        # Create comprehensive rate table
        rate_table = successful_df[['strategy', 'be_trigger', 'profit_target', 
                                   'win_rate_detailed', 'breakeven_rate', 'loss_rate', 
                                   'effective_win_rate', 'total_trades', 'profit_factor']].copy()
        
        rate_table = rate_table.sort_values(['be_trigger', 'profit_target'])
        
        print("COMPREHENSIVE WIN/BREAK-EVEN/LOSS RATE TABLE:")
        print("-" * 100)
        print(f"{'Strategy':<25} {'BE':<4} {'TP':<4} {'Win%':<6} {'BE%':<6} {'Loss%':<7} {'Eff%':<6} {'Trades':<7} {'PF':<5}")
        print("-" * 100)
        
        for _, row in rate_table.iterrows():
            strategy_short = row['strategy'].replace('BE_', '').replace('R_TP_', 'R→')
            print(f"{strategy_short:<25} {row['be_trigger']:<4} {row['profit_target']:<4} "
                  f"{row['win_rate_detailed']:<6.1f} {row['breakeven_rate']:<6.1f} {row['loss_rate']:<7.1f} "
                  f"{row['effective_win_rate']:<6.1f} {row['total_trades']:<7} {row['profit_factor']:<5.2f}")
        
        # PHASE 3: Performance Ranking with Rate Analysis
        print(f"\n🏆 PHASE 3: PERFORMANCE RANKING WITH RATE ANALYSIS")
        print("=" * 60)
        
        # Multiple ranking criteria
        rankings = {
            'Highest Profit Factor': successful_df.nlargest(3, 'profit_factor'),
            'Highest Effective Win Rate': successful_df.nlargest(3, 'effective_win_rate'),
            'Lowest Loss Rate': successful_df.nsmallest(3, 'loss_rate'),
            'Best Loss Prevention': successful_df.nlargest(3, 'loss_prevention_score')
        }
        
        for ranking_name, top_3 in rankings.items():
            print(f"\n🥇 {ranking_name}:")
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                strategy_clean = row['strategy'].replace('BE_', '').replace('R_TP_', 'R→')
                print(f"   {i}. {strategy_clean}: "
                      f"Win {row['win_rate_detailed']:.1f}% | BE {row['breakeven_rate']:.1f}% | "
                      f"Loss {row['loss_rate']:.1f}% | PF {row['profit_factor']:.2f}")
        
        # PHASE 4: Break-Even Effectiveness Analysis
        print(f"\n🛡️  PHASE 4: BREAK-EVEN EFFECTIVENESS ANALYSIS")
        print("=" * 60)
        
        be_effectiveness = successful_df.groupby('be_trigger').agg({
            'breakeven_rate': 'mean',
            'loss_rate': 'mean',
            'loss_prevention_score': 'mean',
            'effective_win_rate': 'mean',
            'total_trades': 'sum'
        }).round(1)
        
        print("BREAK-EVEN TRIGGER EFFECTIVENESS:")
        print(f"{'BE Trigger':<12} {'Avg BE%':<8} {'Avg Loss%':<10} {'Loss Prev':<10} {'Eff Win%':<10} {'Total Trades':<12}")
        print("-" * 70)
        
        for be_trigger, row in be_effectiveness.iterrows():
            print(f"{be_trigger}R{'':<9} {row['breakeven_rate']:<8.1f} {row['loss_rate']:<10.1f} "
                  f"{row['loss_prevention_score']:<10.1f} {row['effective_win_rate']:<10.1f} {row['total_trades']:<12.0f}")
        
        # PHASE 5: Optimization Matrix with Rate Focus
        print(f"\n🎯 PHASE 5: OPTIMIZATION MATRIX")
        print("=" * 50)
        
        # Top combinations by different criteria
        optimization_criteria = {
            'Lowest Loss Rate': successful_df.nsmallest(5, 'loss_rate'),
            'Highest Effective Win Rate': successful_df.nlargest(5, 'effective_win_rate'),
            'Best Risk-Adjusted Performance': successful_df.nlargest(5, 'loss_prevention_score'),
            'Maximum Profitability': successful_df.nlargest(5, 'profit_factor')
        }
        
        for criteria_name, top_5 in optimization_criteria.items():
            print(f"\n📊 {criteria_name}:")
            print(f"{'Strategy':<20} {'Win%':<6} {'BE%':<6} {'Loss%':<7} {'PF':<5} {'Eff%':<6}")
            print("-" * 50)
            
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                strategy_short = row['strategy'].replace('BE_', '').replace('R_TP_', 'R→')[:18]
                print(f"{strategy_short:<20} {row['win_rate_detailed']:<6.1f} {row['breakeven_rate']:<6.1f} "
                      f"{row['loss_rate']:<7.1f} {row['profit_factor']:<5.2f} {row['effective_win_rate']:<6.1f}")
        
        # PHASE 6: Break-Even Strategy Effectiveness
        print(f"\n🔬 PHASE 6: BREAK-EVEN STRATEGY EFFECTIVENESS")
        print("=" * 60)
        
        # Calculate theoretical vs actual performance
        avg_loss_rate = successful_df['loss_rate'].mean()
        avg_breakeven_rate = successful_df['breakeven_rate'].mean()
        avg_effective_win = successful_df['effective_win_rate'].mean()
        
        print(f"BREAK-EVEN FEATURE IMPACT ANALYSIS:")
        print(f"   Average Loss Rate: {avg_loss_rate:.1f}% (Lower is better)")
        print(f"   Average Break-Even Rate: {avg_breakeven_rate:.1f}% (Trades saved from loss)")
        print(f"   Average Effective Win Rate: {avg_effective_win:.1f}% (Win + BE combined)")
        print(f"   Loss Prevention Effectiveness: {100 - avg_loss_rate:.1f}%")
        
        # PHASE 7: Optimal Combinations with Rate Focus
        print(f"\n🌟 PHASE 7: OPTIMAL COMBINATIONS RECOMMENDATION")
        print("=" * 60)
        
        # Find the best overall combination using weighted scoring
        successful_df['composite_score'] = (
            successful_df['effective_win_rate'] * 0.3 +  # 30% weight on effective win rate
            successful_df['loss_prevention_score'] * 0.3 +  # 30% weight on loss prevention
            (successful_df['profit_factor'] * 20).clip(upper=100) * 0.2 +  # 20% weight on profit factor (capped)
            (100 - successful_df['loss_rate']) * 0.2  # 20% weight on avoiding losses
        )
        
        top_recommendation = successful_df.loc[successful_df['composite_score'].idxmax()]
        
        print(f"🏆 TOP RECOMMENDATION:")
        print(f"   Strategy: {top_recommendation['strategy']}")
        print(f"   Win Rate: {top_recommendation['win_rate_detailed']:.1f}%")
        print(f"   Break-Even Rate: {top_recommendation['breakeven_rate']:.1f}%")
        print(f"   Loss Rate: {top_recommendation['loss_rate']:.1f}%")
        print(f"   Effective Win Rate: {top_recommendation['effective_win_rate']:.1f}%")
        print(f"   Profit Factor: {top_recommendation['profit_factor']:.2f}")
        print(f"   Composite Score: {top_recommendation['composite_score']:.1f}/100")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        best_loss_prevention = successful_df.loc[successful_df['loss_rate'].idxmin()]
        print(f"   💎 Best Loss Prevention: {best_loss_prevention['strategy']} "
              f"(Only {best_loss_prevention['loss_rate']:.1f}% loss rate)")
        
        best_effective_win = successful_df.loc[successful_df['effective_win_rate'].idxmax()]
        print(f"   🎯 Best Effective Win Rate: {best_effective_win['strategy']} "
              f"({best_effective_win['effective_win_rate']:.1f}% effective wins)")
        
        most_profitable = successful_df.loc[successful_df['profit_factor'].idxmax()]
        print(f"   💰 Most Profitable: {most_profitable['strategy']} "
              f"(PF: {most_profitable['profit_factor']:.2f})")
        
        # Save comprehensive results
        self.save_breakeven_analysis_results(successful_df)
        
        print(f"\n✅ COMPREHENSIVE BREAK-EVEN ANALYSIS COMPLETE!")
        print(f"📁 Detailed results saved to Excel with all rate breakdowns")
    
    def save_breakeven_analysis_results(self, df: pd.DataFrame):
        """Save comprehensive break-even analysis results to Excel"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = "results"
        os.makedirs(export_path, exist_ok=True)
        filename = os.path.join(export_path, f"breakeven_comprehensive_analysis_{timestamp}.xlsx")
        
        print(f"\n💾 Saving comprehensive break-even analysis...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Sheet 1: Complete Triple-Rate Analysis
            triple_rate_columns = [
                'strategy', 'be_trigger', 'profit_target', 'pair', 'timeframe',
                'total_trades', 'win_rate_detailed', 'breakeven_rate', 'loss_rate',
                'effective_win_rate', 'profit_factor', 'total_return', 'max_drawdown',
                'breakeven_success_rate', 'loss_prevention_score', 'rate_validation'
            ]
            
            df_rates = df[triple_rate_columns].copy()
            df_rates.to_excel(writer, sheet_name='Triple_Rate_Analysis', index=False)
            
            # Sheet 2: Break-Even Trigger Effectiveness
            be_effectiveness = df.groupby('be_trigger').agg({
                'breakeven_rate': ['mean', 'std', 'count'],
                'loss_rate': ['mean', 'min'],
                'effective_win_rate': ['mean', 'max'],
                'loss_prevention_score': ['mean', 'max'],
                'profit_factor': ['mean', 'max']
            }).round(2)
            
            be_effectiveness.to_excel(writer, sheet_name='BE_Trigger_Effectiveness')
            
            # Sheet 3: Profit Target Analysis
            pt_effectiveness = df.groupby('profit_target').agg({
                'win_rate_detailed': ['mean', 'std'],
                'breakeven_rate': 'mean',
                'loss_rate': ['mean', 'std'],
                'profit_factor': ['mean', 'max'],
                'total_trades': 'sum'
            }).round(2)
            
            pt_effectiveness.to_excel(writer, sheet_name='Profit_Target_Analysis')
            
            # Sheet 4: Combination Matrix (Pivot Table)
            pivot_pf = df.pivot_table(
                values='profit_factor', 
                index='be_trigger', 
                columns='profit_target', 
                aggfunc='mean'
            ).round(2)
            pivot_pf.to_excel(writer, sheet_name='Profit_Factor_Matrix')
            
            pivot_loss = df.pivot_table(
                values='loss_rate', 
                index='be_trigger', 
                columns='profit_target',
                aggfunc='mean'
            ).round(1)
            pivot_loss.to_excel(writer, sheet_name='Loss_Rate_Matrix')
            pivot_effective = df.pivot_table(
               values='effective_win_rate', 
               index='be_trigger', 
               columns='profit_target', 
               aggfunc='mean'
            ).round(1)
            pivot_effective.to_excel(writer, sheet_name='Effective_Win_Rate_Matrix')
           
            # Sheet 5: Top Performers by Category
            rankings = {}
            
            rankings['Top_Profit_Factor'] = df.nlargest(10, 'profit_factor')[triple_rate_columns]
            rankings['Top_Effective_Win_Rate'] = df.nlargest(10, 'effective_win_rate')[triple_rate_columns]
            rankings['Lowest_Loss_Rate'] = df.nsmallest(10, 'loss_rate')[triple_rate_columns]
            rankings['Best_Loss_Prevention'] = df.nlargest(10, 'loss_prevention_score')[triple_rate_columns]
            
            for sheet_name, ranking_df in rankings.items():
                ranking_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet 6: Psychological Comfort Analysis
            df['psychological_score'] = (
                df['effective_win_rate'] * 0.4 +  # High effective win rate
                (100 - df['loss_rate']) * 0.4 +   # Low loss rate
                df['breakeven_rate'] * 0.2         # Reasonable break-even frequency
            )
            
            psychological_analysis = df[triple_rate_columns + ['psychological_score']].copy()
            psychological_analysis = psychological_analysis.sort_values('psychological_score', ascending=False)
            psychological_analysis.to_excel(writer, sheet_name='Psychological_Comfort', index=False)
            
            # Sheet 7: Risk-Comfort Analysis
            risk_comfort = df.copy()
            risk_comfort['full_loss_rate'] = risk_comfort['loss_rate']
            risk_comfort['breakeven_comfort'] = risk_comfort['breakeven_rate']
            risk_comfort['psychological_score_10'] = (risk_comfort['psychological_score'] / 10).round(1)
            
            risk_columns = ['strategy', 'be_trigger', 'profit_target', 
                            'full_loss_rate', 'breakeven_comfort', 'psychological_score_10',
                            'effective_win_rate', 'profit_factor']
            
            risk_comfort[risk_columns].to_excel(writer, sheet_name='Risk_Comfort_Analysis', index=False)
        
        print(f"✅ Comprehensive analysis saved: {filename}")
        print(f"📊 7 analysis sheets created with complete rate breakdowns")
        return filename
    
def main_breakeven_analysis():
   """
   Main function for comprehensive break-even strategy analysis
   """
   
   print("🎯 COMPREHENSIVE BREAK-EVEN STRATEGY ANALYSIS")
   print("🎲 Triple-Rate Focus: Win/Break-Even/Loss Breakdown")
   print("=" * 70)
   
   # Initialize analysis engine
   analyzer = BreakEvenAnalysisEngine()
   
   print("\nSelect analysis scope:")
   print("1. Quick Analysis (EURUSD only, 3D timeframe)")
   print("2. Comprehensive Analysis (All forex pairs)")
   print("3. Custom Analysis (Specify pairs and timeframes)")
   print("4. Deep Analysis (Extended period, all combinations)")
   
   choice = input("\nEnter choice (1-4): ").strip()
   
   if choice == '1':
       print("\n🚀 Starting QUICK BREAK-EVEN ANALYSIS...")
       print("📊 Pair: EURUSD | Timeframe: 3D | Period: 9999 days")
       
       df = analyzer.run_comprehensive_breakeven_analysis(
           pairs=['EURUSD'], 
           timeframes=['3D'], 
           days_back=9999
       )
       
   elif choice == '2':
       print("\n🚀 Starting COMPREHENSIVE BREAK-EVEN ANALYSIS...")
       print("📊 All detected forex pairs | Timeframe: 3D | Period: 730 days")
       
       df = analyzer.run_comprehensive_breakeven_analysis(
           pairs=None,  # Auto-detect all forex pairs
           timeframes=['3D'], 
           days_back=730
       )
       
   elif choice == '3':
       print("\n🔧 CUSTOM BREAK-EVEN ANALYSIS...")
       pairs_input = input("Enter forex pairs (comma-separated, e.g., EURUSD,GBPUSD): ").strip().upper()
       pairs = [p.strip() for p in pairs_input.split(',')] if pairs_input else ['EURUSD']
       
       tf_input = input("Enter timeframes (comma-separated, e.g., 1D,3D): ").strip()
       timeframes = [tf.strip() for tf in tf_input.split(',')] if tf_input else ['3D']
       
       days_input = input("Enter days back (default 730): ").strip()
       days_back = int(days_input) if days_input.isdigit() else 730
       
       print(f"\n📊 Custom analysis: {pairs} | {timeframes} | {days_back} days")
       
       df = analyzer.run_comprehensive_breakeven_analysis(
           pairs=pairs, 
           timeframes=timeframes, 
           days_back=days_back
       )
       
   elif choice == '4':
       print("\n🚀 Starting DEEP BREAK-EVEN ANALYSIS...")
       print("📊 All forex pairs | Multiple timeframes | Extended period")
       print("⚠️  This may take 20-30 minutes")
       
       confirm = input("Proceed with deep analysis? (y/n): ").strip().lower()
       if confirm == 'y':
           df = analyzer.run_comprehensive_breakeven_analysis(
               pairs=None,  # All forex pairs
               timeframes=['1D', '2D', '3D', '4D', '5D'], 
               days_back=9999 #~27.4 years, covers full dataset
           )
       else:
           print("Analysis cancelled.")
           return
   
   print("\n✅ BREAK-EVEN ANALYSIS COMPLETE!")
   print("🎯 Key Questions Answered:")
   print("   ✓ Win/Break-Even/Loss rates for each combination")
   print("   ✓ Break-even effectiveness at different trigger levels")
   print("   ✓ Loss prevention analysis and psychological comfort")
   print("   ✓ Optimal combinations for funded trading")
   print("📁 Detailed Excel report with 7 analysis sheets created")


if __name__ == "__main__":
   main_breakeven_analysis()