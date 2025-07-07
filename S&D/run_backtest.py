"""
Complete Backtesting Execution Script - Module 6 (WITH AUTO EXPORT)
Professional historical validation with automatic result export
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator
from modules.backtester import TradingBacktester

def run_complete_backtest():
    """Execute comprehensive historical backtest with automatic export"""
    print("🚀 COMPLETE HISTORICAL BACKTEST WITH AUTO EXPORT")
    print("=" * 60)
    
    try:
        # Load and prepare data
        print("📊 Loading EURUSD data...")
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        print(f"✅ Data loaded: {len(data)} candles")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        
        # Initialize all components
        print("\n🔧 Initializing trading system components...")
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(data)
        risk_manager = RiskManager(account_balance=10000)
        signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
        
        print("✅ All components initialized")
        
        # Setup backtest parameters
        print("\n📋 Configuring backtest parameters...")
        
        # Use substantial historical period
        end_date = data.index[-1]
        start_date = data.index[365]  # Start after 365 days for lookback
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        backtest_days = (end_date - start_date).days
        
        print(f"   Start Date: {start_date_str}")
        print(f"   End Date: {end_date_str}")
        print(f"   Backtest Period: {backtest_days} days")
        print(f"   Lookback Window: 365 days")
        print(f"   Initial Balance: $10,000")
        
        # Create backtester
        backtester = TradingBacktester(
            signal_generator=signal_generator,
            initial_balance=10000,
            config={
                'max_concurrent_trades': 3,
                'slippage_pips': 2,
                'commission_per_lot': 7.0,
                'signal_generation_frequency': 'daily'
            }
        )
        
        # Execute backtest
        print(f"\n🔄 Executing historical backtest...")
        print(f"   This may take several minutes for {backtest_days} days...")
        
        results = backtester.run_walk_forward_backtest(
            data=classified_data,
            start_date=start_date_str,
            end_date=end_date_str,
            lookback_days=365,
            pair='EURUSD'
        )
        
        # Display results
        print_backtest_results(results)
        
        # AUTOMATIC EXPORT (NEW)
        export_backtest_results(results)
        
        # Create visualizations (NEW)
        create_equity_curve(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_backtest_results(results):
    """Print comprehensive backtest results"""
    print(f"\n📊 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    # Test period info
    print(f"📅 Test Period: {results['start_date']} to {results['end_date']}")
    print(f"💰 Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"💰 Final Balance: ${results['final_balance']:,.2f}")
    print(f"📈 Total Return: {results['total_return_pct']:.1f}%")
    
    # Trade statistics
    print(f"\n🎯 TRADE STATISTICS:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Winning Trades: {results['winning_trades']}")
    print(f"   Losing Trades: {results['losing_trades']}")
    print(f"   Win Rate: {results['win_rate']}%")
    print(f"   Average Trade Duration: {results['avg_trade_duration_days']:.1f} days")
    
    # Performance metrics
    print(f"\n💹 PERFORMANCE METRICS:")
    print(f"   Net Profit: ${results['net_profit']:,.2f}")
    print(f"   Gross Profit: ${results['gross_profit']:,.2f}")
    print(f"   Gross Loss: ${results['gross_loss']:,.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Expectancy: ${results['expectancy']:.2f} ({results['expectancy_pct']:.3f}%)")
    print(f"   Avg Return/Trade: ${results['avg_return_per_trade']:.2f}")
    
    # Risk metrics
    print(f"\n🛡️  RISK METRICS:")
    print(f"   Maximum Drawdown: ${results['max_drawdown']:,.2f} ({results['max_drawdown_pct']:.1f}%)")
    print(f"   Max Concurrent Trades: {results['max_concurrent_trades']}")
    print(f"   Avg Winning Trade: ${results['avg_winning_trade']:.2f}")
    print(f"   Avg Losing Trade: ${results['avg_losing_trade']:.2f}")
    
    # Manual strategy comparison
    print(f"\n🎯 MANUAL STRATEGY COMPARISON:")
    print(f"   Profit Factor: {results['pf_vs_manual']}")
    print(f"   Win Rate: {results['wr_vs_manual']}")
    print(f"   Manual Baseline: PF={results['manual_pf_baseline']}, WR={results['manual_wr_baseline']}%")
    
    if results['within_15pct_tolerance']:
        print(f"   ✅ WITHIN 15% TOLERANCE - STRATEGY VALIDATED!")
    else:
        print(f"   ⚠️  OUTSIDE 15% TOLERANCE - REVIEW REQUIRED")

def export_backtest_results(results):
    """AUTOMATIC EXPORT: Export backtest results to files"""
    print(f"\n💾 EXPORTING BACKTEST RESULTS...")
    
    # Create results directory
    os.makedirs('results/backtest', exist_ok=True)
    
    # Generate timestamp for files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Export individual trades for manual verification
    if results['closed_trades']:
        trades_df = pd.DataFrame(results['closed_trades'])
        trades_filename = f"results/backtest/trades_{timestamp}.csv"
        trades_df.to_csv(trades_filename, index=False)
        print(f"   ✅ Individual trades: {trades_filename}")
        print(f"      Contains: {len(trades_df)} trades with entry/exit details")
    
    # 2. Export equity curve for performance analysis
    if results['equity_curve']:
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_filename = f"results/backtest/equity_curve_{timestamp}.csv"
        equity_df.to_csv(equity_filename, index=False)
        print(f"   ✅ Daily equity curve: {equity_filename}")
        print(f"      Contains: {len(equity_df)} daily performance records")
    
    # 3. Export summary metrics
    summary_data = {key: value for key, value in results.items() 
                   if key not in ['equity_curve', 'closed_trades', 'trade_pnls']}
    
    summary_df = pd.DataFrame([summary_data])
    summary_filename = f"results/backtest/summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"   ✅ Performance summary: {summary_filename}")
    print(f"      Contains: All key metrics for comparison")
    
    # 4. Export trade-by-trade P&L list
    if results['trade_pnls']:
        pnl_df = pd.DataFrame({
            'trade_number': range(1, len(results['trade_pnls']) + 1),
            'pnl': results['trade_pnls']
        })
        pnl_filename = f"results/backtest/trade_pnls_{timestamp}.csv"
        pnl_df.to_csv(pnl_filename, index=False)
        print(f"   ✅ Trade P&L sequence: {pnl_filename}")
        print(f"      Contains: {len(results['trade_pnls'])} trade results")
    
    print(f"\n📁 All results exported to: results/backtest/")
    print(f"📋 Open these CSV files in Excel for manual verification")

def create_equity_curve(results):
    """Create and save equity curve visualization"""
    if not results['equity_curve']:
        print("⚠️  No equity data to plot")
        return
    
    print(f"\n📈 Creating equity curve visualization...")
    
    # Prepare data
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot equity curve
    ax1.plot(equity_df['date'], equity_df['total_equity'], 
             linewidth=2, color='#2E86AB', label='Total Equity')
    ax1.plot(equity_df['date'], equity_df['balance'], 
             linewidth=1, color='#A23B72', alpha=0.7, label='Realized P&L')
    
    ax1.set_title(f'EURUSD Strategy - Equity Curve ({results["start_date"]} to {results["end_date"]})', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Account Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot drawdown
    ax2.fill_between(equity_df['date'], 0, -equity_df['drawdown'], 
                     color='#F18F01', alpha=0.7, label='Drawdown')
    ax2.plot(equity_df['date'], -equity_df['drawdown'], 
             linewidth=1, color='#C73E1D')
    
    ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format y-axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_filename = f"results/backtest/equity_curve_{timestamp}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Equity chart saved: {chart_filename}")
    
    plt.show()

def quick_test_backtest():
    """Quick test with recent data only"""
    print("🧪 QUICK BACKTEST TEST (Recent 6 months)")
    print("=" * 50)
    
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # Initialize components
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(data)
        risk_manager = RiskManager(account_balance=10000)
        signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
        
        # Quick test: recent 6 months
        end_date = data.index[-1]
        start_date = end_date - pd.Timedelta(days=180)
        
        # Find closest available date
        available_start = data.index[data.index >= start_date][0]
        
        start_date_str = available_start.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"   Testing: {start_date_str} to {end_date_str}")
        
        # Run quick backtest
        backtester = TradingBacktester(signal_generator, initial_balance=10000)
        results = backtester.run_walk_forward_backtest(
            classified_data, start_date_str, end_date_str, 365, 'EURUSD'
        )
        
        print(f"\n📊 Quick Test Results:")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']}%")
        print(f"   Profit Factor: {results['profit_factor']}")
        print(f"   Final Balance: ${results['final_balance']:,.2f}")
        
        # Export quick test results
        if results['total_trades'] > 0:
            export_backtest_results(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("Select backtest option:")
    print("1. Quick Test (6 months)")
    print("2. Complete Backtest (Full history)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = quick_test_backtest()
    elif choice == "2":
        results = run_complete_backtest()
        success = results is not None
    else:
        print("Invalid choice")
        success = False
    
    if success:
        print("\n🎉 Backtesting completed successfully!")
        print("📋 Check the results/backtest/ folder for:")
        print("   • trades_YYYYMMDD_HHMMSS.csv - Individual trade details")
        print("   • equity_curve_YYYYMMDD_HHMMSS.csv - Daily performance")
        print("   • summary_YYYYMMDD_HHMMSS.csv - Overall metrics")
        print("   • equity_curve_YYYYMMDD_HHMMSS.png - Performance chart")
        print("\n📈 Open CSV files in Excel for manual verification!")
    else:
        print("\n⚠️  Backtesting requires fixes")