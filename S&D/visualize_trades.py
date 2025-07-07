"""
Comprehensive Trade Validation Visualization
Shows zones, entries, exits, and trend context for manual validation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator
from modules.backtester import TradingBacktester

plt.style.use('default')

class TradeValidationVisualizer:
    """Professional trade validation with comprehensive chart analysis"""
    
    def __init__(self):
        self.colors = {
            'bullish_candle': '#2ECC71',
            'bearish_candle': '#E74C3C',
            'doji_candle': '#95A5A6',
            'demand_zone': '#4ECDC4',
            'supply_zone': '#FF6B6B',
            'ema_50': '#3498DB',
            'ema_100': '#E67E22',
            'ema_200': '#9B59B6',
            'buy_entry': '#27AE60',
            'sell_entry': '#C0392B',
            'stop_loss': '#E74C3C',
            'take_profit': '#2ECC71',
            'trend_bullish': '#D5F4E6',
            'trend_bearish': '#FADBD8',
            'trend_ranging': '#F8F9FA'
        }
    
    def validate_recent_trades(self, days_back=60):
        """
        Create comprehensive validation chart for recent trades
        
        Args:
            days_back: Days back from latest data to analyze
        """
        print("📊 COMPREHENSIVE TRADE VALIDATION")
        print("=" * 50)
        
        try:
            # Load and prepare data
            data_loader = DataLoader()
            data = data_loader.load_pair_data('EURUSD', 'Daily')
            
            # Get recent period for visualization
            end_date = data.index[-1]
            start_date = end_date - pd.Timedelta(days=days_back)
            
            # Find actual start in data
            start_idx = max(0, len(data) - days_back)
            recent_data = data.iloc[start_idx:].copy()
            recent_data = recent_data.reset_index()
            
            print(f"📅 Analyzing period: {recent_data['date'].iloc[0]} to {recent_data['date'].iloc[-1]}")
            print(f"📈 Candles: {len(recent_data)}")
            
            # Run mini-backtest to get actual trades
            trades = self.run_validation_backtest(data, days_back)
            
            if not trades:
                print("⚠️  No trades found in recent period")
                return
            
            print(f"🎯 Trades found: {len(trades)}")
            
            # Create comprehensive validation chart
            self.create_validation_chart(recent_data, trades, days_back)
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_validation_backtest(self, data, days_back):
        """Run mini-backtest to get actual trade data"""
        try:
            # Initialize components
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            trend_classifier = TrendClassifier(data)
            risk_manager = RiskManager(account_balance=10000)
            signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
            
            # Run mini-backtest
            backtester = TradingBacktester(signal_generator, initial_balance=10000)
            
            # Get recent period dates
            end_date = data.index[-1]
            start_date = end_date - pd.Timedelta(days=days_back)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Run backtest
            results = backtester.run_walk_forward_backtest(
                classified_data, start_date_str, end_date_str, 365, 'EURUSD'
            )
            
            return results.get('closed_trades', [])
            
        except Exception as e:
            print(f"❌ Mini-backtest failed: {e}")
            return []
    
    def create_validation_chart(self, data, trades, days_back):
        """Create comprehensive validation chart"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0])
        # EMA chart  
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        # Trend classification
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        # Volume/trade info
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        
        # Plot main price action with trades
        self.plot_price_and_trades(ax1, data, trades)
        
        # Plot EMAs
        self.plot_emas(ax2, data)
        
        # Plot trend classification
        self.plot_trend_background(ax3, data)
        
        # Plot trade information
        self.plot_trade_info(ax4, data, trades)
        
        # Format all axes
        self.format_axes(ax1, ax2, ax3, ax4, data, days_back)
        
        # Save chart
        self.save_validation_chart(fig, days_back)
        
        plt.tight_layout()
        plt.show()
    
    def plot_price_and_trades(self, ax, data, trades):
        """Plot candlesticks, zones, and trade markers"""
        
        # Plot candlesticks
        self.plot_candlesticks(ax, data)
        
        # Plot zones for each trade
        for i, trade in enumerate(trades):
            self.plot_trade_zone(ax, trade, data, i)
        
        # Plot trade entries and exits
        for i, trade in enumerate(trades):
            self.plot_trade_markers(ax, trade, data, i)
    
    def plot_candlesticks(self, ax, data):
        """Plot professional candlesticks"""
        
        for i, row in data.iterrows():
            x = i
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Candle color
            if close_price > open_price:
                color = self.colors['bullish_candle']
                edge_color = '#27AE60'
                body_bottom = open_price
                body_top = close_price
            elif close_price < open_price:
                color = self.colors['bearish_candle']
                edge_color = '#C0392B'
                body_bottom = close_price
                body_top = open_price
            else:
                color = self.colors['doji_candle']
                edge_color = '#7F8C8D'
                body_bottom = open_price
                body_top = close_price
            
            # Draw wick
            ax.plot([x, x], [low_price, high_price], 
                   color=edge_color, linewidth=1.5, alpha=0.8)
            
            # Draw body
            body_height = abs(close_price - open_price)
            if body_height > 0:
                rect = patches.Rectangle((x - 0.4, body_bottom), 0.8, body_height, 
                                       facecolor=color, edgecolor=edge_color, 
                                       linewidth=0.8, alpha=0.8)
                ax.add_patch(rect)
            else:
                # Doji
                ax.plot([x - 0.4, x + 0.4], [open_price, open_price], 
                       color=edge_color, linewidth=2.5)
    
    def plot_trade_zone(self, ax, trade, data, trade_num):
        """Plot the zone that generated this trade"""
        
        zone_high = trade['zone_high']
        zone_low = trade['zone_low']
        direction = trade['direction']
        
        # Zone color
        if direction == 'BUY':
            zone_color = self.colors['demand_zone']
            edge_color = '#006666'
        else:
            zone_color = self.colors['supply_zone']
            edge_color = '#CC0000'
        
        # Draw zone rectangle across entire chart
        rect = patches.Rectangle((0, zone_low), len(data), zone_high - zone_low,
                               facecolor=zone_color, alpha=0.15,
                               edgecolor=edge_color, linewidth=1.5,
                               label=f'Trade {trade_num+1} Zone')
        ax.add_patch(rect)
        
        # Zone boundary lines
        ax.hlines(zone_high, 0, len(data), colors=edge_color, linewidth=2, alpha=0.7)
        ax.hlines(zone_low, 0, len(data), colors=edge_color, linewidth=2, alpha=0.7)
        
        # Zone label
        ax.text(len(data) * 0.02, (zone_high + zone_low) / 2, 
               f'T{trade_num+1} Zone\n{zone_low:.5f}-{zone_high:.5f}',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=zone_color, alpha=0.8),
               fontsize=8, fontweight='bold')
    
    def plot_trade_markers(self, ax, trade, data, trade_num):
        """Plot entry, stops, and targets for trade"""
        
        # Find entry date in data
        entry_date = pd.to_datetime(trade['entry_date'])
        entry_idx = None
        
        for i, row in data.iterrows():
            if pd.to_datetime(row['date']).date() == entry_date.date():
                entry_idx = i
                break
        
        if entry_idx is None:
            return
        
        entry_price = trade['entry_price']
        stop_price = trade['stop_loss']
        tp1_price = trade['take_profit_1']
        tp2_price = trade['take_profit_2']
        direction = trade['direction']
        
        # Entry marker
        if direction == 'BUY':
            marker = '^'
            color = self.colors['buy_entry']
        else:
            marker = 'v'
            color = self.colors['sell_entry']
        
        ax.scatter(entry_idx, entry_price, marker=marker, s=200, 
                  color=color, edgecolor='white', linewidth=2, 
                  zorder=10, label=f'T{trade_num+1} Entry')
        
        # Entry price line
        ax.hlines(entry_price, entry_idx, len(data), 
                 colors=color, linewidth=2, alpha=0.8, linestyle='-')
        
        # Stop loss line
        ax.hlines(stop_price, entry_idx, len(data), 
                 colors=self.colors['stop_loss'], linewidth=2, 
                 alpha=0.8, linestyle='--')
        
        # Take profit lines
        ax.hlines(tp1_price, entry_idx, len(data), 
                 colors=self.colors['take_profit'], linewidth=1.5, 
                 alpha=0.7, linestyle=':')
        ax.hlines(tp2_price, entry_idx, len(data), 
                 colors=self.colors['take_profit'], linewidth=2, 
                 alpha=0.8, linestyle=':')
        
        # Exit marker if trade is closed
        if 'exit_date' in trade and trade['exit_date']:
            exit_date = pd.to_datetime(trade['exit_date'])
            exit_idx = None
            
            for i, row in data.iterrows():
                if pd.to_datetime(row['date']).date() >= exit_date.date():
                    exit_idx = i
                    break
            
            if exit_idx is not None:
                exit_price = trade['exit_price']
                pnl = trade['pnl']
                
                if pnl > 0:
                    exit_color = self.colors['take_profit']
                    exit_marker = 'o'
                else:
                    exit_color = self.colors['stop_loss']
                    exit_marker = 'x'
                
                ax.scatter(exit_idx, exit_price, marker=exit_marker, s=150,
                          color=exit_color, edgecolor='white', linewidth=2,
                          zorder=10)
        
        # Trade info text
        pnl = trade.get('pnl', 0)
        pnl_text = f"+${pnl:.0f}" if pnl > 0 else f"${pnl:.0f}"
        position_size = trade.get('position_size', 0)
        
        ax.text(entry_idx + 2, entry_price, 
               f'T{trade_num+1}: {direction}\n{position_size:.2f} lots\n{pnl_text}',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
               fontsize=8, fontweight='bold', color='white')
    
    def plot_emas(self, ax, data):
        """Plot EMAs with trend context"""
        
        # Calculate EMAs for the period
        data_copy = data.copy()
        data_copy['ema_50'] = data_copy['close'].ewm(span=50).mean()
        data_copy['ema_100'] = data_copy['close'].ewm(span=100).mean()
        data_copy['ema_200'] = data_copy['close'].ewm(span=200).mean()
        
        x = range(len(data_copy))
        
        # Plot EMAs
        ax.plot(x, data_copy['ema_50'], color=self.colors['ema_50'], 
               linewidth=2, label='EMA 50', alpha=0.9)
        ax.plot(x, data_copy['ema_100'], color=self.colors['ema_100'], 
               linewidth=2, label='EMA 100', alpha=0.9)
        ax.plot(x, data_copy['ema_200'], color=self.colors['ema_200'], 
               linewidth=2, label='EMA 200', alpha=0.9)
        
        ax.set_ylabel('EMAs', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_trend_background(self, ax, data):
        """Plot trend classification over time"""
        
        # This would need trend classification data
        # For now, show placeholder
        ax.text(0.5, 0.5, 'Trend Classification\n(Strong Bullish/Bearish/Ranging)', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, fontweight='bold')
        ax.set_ylabel('Trend', fontsize=10)
    
    def plot_trade_info(self, ax, data, trades):
        """Plot trade information timeline"""
        
        # Show trade entries as bars
        trade_data = []
        for i, trade in enumerate(trades):
            entry_date = pd.to_datetime(trade['entry_date'])
            
            # Find index in data
            for j, row in data.iterrows():
                if pd.to_datetime(row['date']).date() == entry_date.date():
                    trade_data.append(j)
                    break
        
        if trade_data:
            ax.bar(trade_data, [1] * len(trade_data), 
                  color=self.colors['buy_entry'], alpha=0.7,
                  label='Trade Entries')
        
        ax.set_ylabel('Trades', fontsize=10)
        ax.set_ylim(0, 2)
        ax.legend(loc='upper left', fontsize=8)
    
    def format_axes(self, ax1, ax2, ax3, ax4, data, days_back):
        """Format all chart axes"""
        
        # Main chart
        ax1.set_title(f'EURUSD Trade Validation - Last {days_back} Days', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)
        
        # X-axis formatting for bottom chart only
        ax4.set_xlabel('Date', fontsize=12)
        
        # Set x-ticks to show dates
        tick_positions = range(0, len(data), max(1, len(data)//10))
        tick_labels = [data.iloc[i]['date'].strftime('%m-%d') for i in tick_positions]
        
        ax4.set_xticks(tick_positions)
        ax4.set_xticklabels(tick_labels, rotation=45)
        
        # Hide x-labels for upper charts
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
    
    def save_validation_chart(self, fig, days_back):
        """Save validation chart"""
        import os
        os.makedirs('results/validation', exist_ok=True)
        
        filename = f"results/validation/trade_validation_{days_back}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Validation chart saved: {filename}")

def validate_trades():
    """Main function to run trade validation"""
    visualizer = TradeValidationVisualizer()
    
    print("Trade Validation Options:")
    print("1. Last 30 days")
    print("2. Last 60 days") 
    print("3. Last 90 days")
    
    choice = input("Enter choice (1-3): ").strip()
    
    days_map = {'1': 30, '2': 60, '3': 90}
    days_back = days_map.get(choice, 60)
    
    visualizer.validate_recent_trades(days_back)

if __name__ == "__main__":
    validate_trades()