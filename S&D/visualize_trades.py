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

class VisiblePeriodSignalGenerator:
    """Signal generator that uses zones from visible period only"""
    
    def __init__(self, visible_patterns, trend_classifier, risk_manager, extended_data, visible_data):
        self.visible_patterns = visible_patterns
        self.trend_classifier = trend_classifier
        self.risk_manager = risk_manager
        self.extended_data = extended_data
        self.visible_data = visible_data
    
    def generate_signals(self, data, timeframe, pair):
        """Generate signals using ONLY zones from visible period"""
        current_price = self.visible_data['close'].iloc[-1]
        
        # Get trend from extended data (has proper EMAs)
        trend_data = self.trend_classifier.classify_trend_with_filter()
        current_trend = trend_data['trend_filtered'].iloc[-1]
        
        if current_trend == 'ranging':
            return []
        
        # Filter zones by trend direction (zones are already from visible period only)
        bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
        
        if current_trend in bullish_trends:
            valid_zones = self.visible_patterns['rbr_patterns']
        else:
            valid_zones = self.visible_patterns['dbd_patterns']
        
        print(f"   Signal generation: {len(valid_zones)} zones available for {current_trend} trend")
        
        signals = []
        for i, zone in enumerate(valid_zones):
            print(f"   Testing zone {i+1}: {zone['type']} at {zone['zone_low']:.5f}-{zone['zone_high']:.5f}")
            
            # Validate with risk manager using extended data (for proper risk calculation)
            risk_validation = self.risk_manager.validate_zone_for_trading(
                zone, current_price, pair, self.extended_data
            )
            
            if risk_validation['is_tradeable']:
                signal = {
                    'signal_id': f"{pair}_{timeframe}_{len(signals)}",
                    'pair': pair,
                    'direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL',
                    'entry_price': risk_validation['entry_price'],
                    'stop_loss': risk_validation['stop_loss_price'],
                    'take_profit_1': risk_validation['take_profit_1'],
                    'take_profit_2': risk_validation['take_profit_2'],
                    'position_size': risk_validation['position_size'],
                    'risk_amount': risk_validation['risk_amount'],
                    'zone_high': zone['zone_high'],
                    'zone_low': zone['zone_low'],
                    'timeframe': timeframe
                }
                signals.append(signal)
                print(f"   ✅ Zone {i+1} created signal: {signal['direction']}")
            else:
                print(f"   ❌ Zone {i+1} rejected: {risk_validation['reason']}")
        
        return signals

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
        """FIXED: Only detect zones from the EXACT visualization period"""
        try:
            # Get the EXACT same period we're visualizing
            end_date = data.index[-1]
            start_date = end_date - pd.Timedelta(days=days_back)
            
            # STEP 1: Get ONLY the visible period data for zone detection
            start_idx = max(0, len(data) - days_back)
            visible_only_data = data.iloc[start_idx:].copy()
            
            # STEP 2: Get extended data ONLY for EMA calculation (but don't detect zones in it)
            min_lookback_for_emas = 200
            total_periods_needed = days_back + min_lookback_for_emas
            extended_start_idx = max(0, len(data) - total_periods_needed)
            extended_data = data.iloc[extended_start_idx:].copy()
            
            print(f"📊 FIXED: Zone detection from EXACT visualization period only:")
            print(f"   Visible period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"   Visible data for zones: {len(visible_only_data)} candles")
            print(f"   Extended data for EMAs: {len(extended_data)} candles")
            
            # STEP 3: Detect zones ONLY in visible period
            visible_candle_classifier = CandleClassifier(visible_only_data)
            visible_classified_data = visible_candle_classifier.classify_all_candles()
            
            visible_zone_detector = ZoneDetector(visible_candle_classifier)
            visible_patterns = visible_zone_detector.detect_all_patterns(visible_classified_data)
            
            print(f"   Zones found in visible period: {visible_patterns['total_patterns']}")
            print(f"   D-B-D zones: {len(visible_patterns['dbd_patterns'])}")
            print(f"   R-B-R zones: {len(visible_patterns['rbr_patterns'])}")
            
            # STEP 4: Use extended data ONLY for trend classification (needs EMAs)
            trend_classifier = TrendClassifier(extended_data)
            risk_manager = RiskManager(account_balance=10000)
            
            # STEP 5: Create signal generator with visible zones + extended trend data
            signal_generator = VisiblePeriodSignalGenerator(
                visible_patterns, trend_classifier, risk_manager, extended_data, visible_only_data
            )
            
            # STEP 6: Run backtest
            backtester = TradingBacktester(signal_generator, initial_balance=10000)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Use the visible classified data for backtesting
            results = backtester.run_walk_forward_backtest(
                visible_classified_data, start_date_str, end_date_str, min_lookback_for_emas, 'EURUSD'
            )
            
            trades = results.get('closed_trades', [])
            print(f"   Trades from visible-period zones only: {len(trades)}")
            
            return trades
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_validation_chart(self, data, trades, days_back):
        """Create streamlined validation chart - Price + EMAs only"""
        
        # Create figure with just 2 subplots - more space for price and EMAs
        fig = plt.figure(figsize=(24, 12))  # Wider and taller
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.2)  # More space for price
        
        # Main price chart (larger)
        ax1 = fig.add_subplot(gs[0])
        # EMA chart (smaller but visible)
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Plot main price action with trades
        self.plot_price_and_trades(ax1, data, trades)
        
        # Plot EMAs
        self.plot_emas(ax2, data)
        
        # Format axes
        self.format_axes_simplified(ax1, ax2, data, days_back)
        
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
        """Plot the zone that generated this trade - only if valid"""
        
        zone_high = trade['zone_high']
        zone_low = trade['zone_low']
        direction = trade['direction']
        
        # CRITICAL: Validate zone before plotting
        zone_start_idx = self.find_zone_formation_index(trade, data)
        
        if zone_start_idx is None:
            print(f"   ⚠️  Skipping invalid Trade {trade_num+1} zone - no candle overlap")
            return
        
        # Zone color
        if direction == 'BUY':
            zone_color = self.colors['demand_zone']
            edge_color = '#006666'
        else:
            zone_color = self.colors['supply_zone']
            edge_color = '#CC0000'
        
        # Find zone formation date (when base ended)
        zone_start_idx = self.find_zone_formation_index(trade, data)
        
        if zone_start_idx is not None:
            # Find when trade was entered (zone becomes inactive after entry)
            entry_date = pd.to_datetime(trade['entry_date'])
            entry_idx = None
            
            for i, row in data.iterrows():
                if pd.to_datetime(row['date']).date() == entry_date.date():
                    entry_idx = i
                    break
            
            # Zone should only show from formation to entry (not beyond)
            zone_end_idx = entry_idx if entry_idx is not None else len(data)
            zone_width = zone_end_idx - zone_start_idx
            
            if zone_width > 0:
                rect = patches.Rectangle((zone_start_idx, zone_low), zone_width, zone_high - zone_low,
                                    facecolor=zone_color, alpha=0.15,
                                    edgecolor=edge_color, linewidth=1.5,
                                    label=f'Trade {trade_num+1} Zone')
                ax.add_patch(rect)
                
                # Zone boundary lines - only from formation to entry
                ax.hlines(zone_high, zone_start_idx, zone_end_idx, colors=edge_color, linewidth=2, alpha=0.7)
                ax.hlines(zone_low, zone_start_idx, zone_end_idx, colors=edge_color, linewidth=2, alpha=0.7)
            
            # Zone label positioned at formation point
            label_x = zone_start_idx + (zone_width * 0.05)  # 5% into the zone
            ax.text(label_x, (zone_high + zone_low) / 2, 
                f'T{trade_num+1} Zone\n{zone_low:.5f}-{zone_high:.5f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=zone_color, alpha=0.8),
                fontsize=8, fontweight='bold')
    
    def find_zone_formation_index(self, trade, data):
        """
        Find zone formation index within VISIBLE period only
        """
        try:
            zone_high = trade['zone_high']
            zone_low = trade['zone_low']
            
            # Find FIRST candle in VISIBLE data that overlaps with zone
            for i, row in data.iterrows():
                candle_high = row['high']
                candle_low = row['low']
                
                # Check if this candle overlaps with the zone
                if (candle_low <= zone_high and candle_high >= zone_low):
                    return i  # Start zone exactly where first overlap occurs
            
            # If no overlap in visible period, don't show zone
            return None
            
        except Exception as e:
            return None
        
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
        
        # Find exit date to stop lines when trade closes
        exit_idx = len(data)  # Default to end of chart
        
        if 'exit_date' in trade and trade['exit_date']:
            exit_date = pd.to_datetime(trade['exit_date'])
            
            for i, row in data.iterrows():
                if pd.to_datetime(row['date']).date() >= exit_date.date():
                    exit_idx = i
                    break
        
        # Entry price line - only from entry to exit
        ax.hlines(entry_price, entry_idx, exit_idx, 
                 colors=color, linewidth=2, alpha=0.8, linestyle='-')
        
        # Stop loss line - only from entry to exit
        ax.hlines(stop_price, entry_idx, exit_idx, 
                 colors=self.colors['stop_loss'], linewidth=2, 
                 alpha=0.8, linestyle='--')
        
        # Take profit lines - only from entry to exit
        ax.hlines(tp1_price, entry_idx, exit_idx, 
                 colors=self.colors['take_profit'], linewidth=1.5, 
                 alpha=0.7, linestyle=':')
        ax.hlines(tp2_price, entry_idx, exit_idx, 
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
        
    def format_axes_simplified(self, ax1, ax2, data, days_back):
        """Format simplified chart axes"""
        
        # Main price chart
        ax1.set_title(f'EURUSD Trade Validation - Last {days_back} Days', 
                    fontsize=18, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)
        
        # EMA chart
        ax2.set_ylabel('EMAs', fontsize=12)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Set x-ticks to show dates
        tick_positions = range(0, len(data), max(1, len(data)//12))  # More date labels
        tick_labels = [data.iloc[i]['date'].strftime('%m-%d') for i in tick_positions]
        
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45)
        
        # Hide x-labels for upper chart
        ax1.set_xticklabels([])
    
    def save_validation_chart(self, fig, days_back):
        """Save streamlined validation chart"""
        import os
        os.makedirs('results/validation', exist_ok=True)
        
        filename = f"results/validation/trade_validation_streamlined_{days_back}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Streamlined validation chart saved: {filename}")

def validate_trades():
    """Main function to run trade validation"""
    visualizer = TradeValidationVisualizer()
    
    print("Trade Validation Options:")
    print("1. Last 180 days")
    print("2. Last 365 days") 
    print("3. Last 700 days")
    
    choice = input("Enter choice (1-3): ").strip()
    
    days_map = {'1': 180, '2': 365, '3': 700}
    days_back = days_map.get(choice, 60)
    
    visualizer.validate_recent_trades(days_back)

if __name__ == "__main__":
    validate_trades()