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
    """Signal generator using zones EXCLUSIVELY from visible period with validation"""
    
    def __init__(self, restricted_patterns, trend_classifier, risk_manager, extended_data, visible_data):
        self.restricted_patterns = restricted_patterns
        self.trend_classifier = trend_classifier
        self.risk_manager = risk_manager
        self.extended_data = extended_data
        self.visible_data = visible_data
        
        # VALIDATION: Confirm zone restriction worked
        self.validate_zone_restriction()
    
    def validate_zone_restriction(self):
        """Verify all zones are within visible data boundaries"""
        visible_range = len(self.visible_data)
        all_zones = self.restricted_patterns['dbd_patterns'] + self.restricted_patterns['rbr_patterns']
        
        violations = 0
        for i, zone in enumerate(all_zones):
            if zone['start_idx'] < 0 or zone['end_idx'] >= visible_range:
                print(f"⚠️  Zone {i+1} boundary violation: indices {zone['start_idx']}-{zone['end_idx']} outside 0-{visible_range-1}")
                violations += 1
        
        if violations == 0:
            print(f"✅ Zone restriction validated: All {len(all_zones)} zones within visible boundaries")
        else:
            print(f"❌ Zone restriction failed: {violations} violations detected")
    
    def generate_signals(self, data, timeframe, pair):
        """Generate signals using EXCLUSIVELY restricted zones"""
        current_price = self.visible_data['close'].iloc[-1]
        
        # Get trend classification from extended data (needs EMA history)
        trend_data = self.trend_classifier.classify_trend_with_filter()
        current_trend = trend_data['trend_filtered'].iloc[-1]
        
        if current_trend == 'ranging':
            print(f"   Ranging market detected - no signals generated")
            return []
        
        # Apply trend filter to restricted zones
        bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
        
        if current_trend in bullish_trends:
            filtered_zones = self.restricted_patterns['rbr_patterns']
        else:
            filtered_zones = self.restricted_patterns['dbd_patterns']
        
        print(f"   Restricted zones for {current_trend}: {len(filtered_zones)}")
        
        signals = []
        for i, zone in enumerate(filtered_zones):
            zone_info = f"{zone['type']} at {zone['zone_low']:.5f}-{zone['zone_high']:.5f}"
            print(f"   Validating restricted zone {i+1}: {zone_info}")
            
            # Risk validation using extended data for proper calculations
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
                print(f"   ✅ Restricted zone {i+1} generated signal: {signal['direction']}")
            else:
                print(f"   ❌ Restricted zone {i+1} rejected: {risk_validation['reason']}")
        
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
    
    def validate_recent_trades(self, days_back=180):
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
        """FIXED: Use proper date boundaries for backtesting"""
        try:
            # CORRECTED: Use actual data indices instead of calculated dates
            visible_start_idx = max(0, len(data) - days_back)
            visible_data = data.iloc[visible_start_idx:].copy()
            
            # Extended data for EMAs (use more history from available data)
            ema_minimum = 200
            extended_start_idx = max(0, len(data) - (days_back + ema_minimum))
            extended_data = data.iloc[extended_start_idx:].copy()
            
            # Get ACTUAL dates from the data indices
            actual_start_date = visible_data.index[0]
            actual_end_date = visible_data.index[-1]
            
            print(f"🎯 CORRECTED SCOPE RESTRICTION:")
            print(f"   Actual visualization period: {actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}")
            print(f"   Zone detection data: {len(visible_data)} candles (RESTRICTED)")
            print(f"   EMA calculation data: {len(extended_data)} candles (for trends only)")
            
            # STEP 1: Detect zones EXCLUSIVELY from visible period
            visible_classifier = CandleClassifier(visible_data)
            visible_classified = visible_classifier.classify_all_candles()
            
            visible_zone_detector = ZoneDetector(visible_classifier)
            restricted_patterns = visible_zone_detector.detect_all_patterns(visible_classified)
            
            print(f"   Zones found in visible period ONLY: {restricted_patterns['total_patterns']}")
            print(f"   D-B-D zones: {len(restricted_patterns['dbd_patterns'])}")
            print(f"   R-B-R zones: {len(restricted_patterns['rbr_patterns'])}")
            
            # STEP 2: Calculate trends from extended data (EMAs need history)
            trend_classifier = TrendClassifier(extended_data)
            risk_manager = RiskManager(account_balance=10000)
            
            # STEP 3: CREATE VISUALIZATION TRADES (BYPASS RISK VALIDATION FOR VISUALIZATION)
            print(f"🎯 CREATING VISUALIZATION TRADES FROM ALL DETECTED ZONES:")

            # Get current trend for filtering
            trend_data = trend_classifier.classify_trend_with_filter()
            current_trend = trend_data['trend_filtered'].iloc[-1]
            current_price = visible_data['close'].iloc[-1]

            print(f"   Current trend: {current_trend}")
            print(f"   Current price: {current_price:.5f}")

            # Filter zones by trend direction for visualization
            bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']

            if current_trend in bullish_trends:
                visualization_zones = restricted_patterns['rbr_patterns']
                trade_direction = 'BUY'
            elif current_trend in ['strong_bearish', 'medium_bearish', 'weak_bearish']:
                visualization_zones = restricted_patterns['dbd_patterns'] 
                trade_direction = 'SELL'
            else:
                print(f"   Ranging market - showing all zones for visualization")
                visualization_zones = restricted_patterns['rbr_patterns'] + restricted_patterns['dbd_patterns']
                trade_direction = 'MIXED'

            print(f"   Creating visualization for {len(visualization_zones)} zones")

            # Create visualization trades for ALL detected zones (ignore testing)
            trades = []
            for i, zone in enumerate(visualization_zones):
                print(f"   📊 Creating visualization trade {i+1}: {zone['type']} at {zone['zone_low']:.5f}-{zone['zone_high']:.5f}")
                
                # Calculate basic trade parameters for visualization
                zone_center = (zone['zone_high'] + zone['zone_low']) / 2
                zone_size = zone['zone_high'] - zone['zone_low']
                
                if zone['type'] == 'R-B-R':
                    entry_price = zone['zone_high'] + (zone_size * 0.05)  # 5% above zone
                    stop_loss = zone['zone_low'] - (zone_size * 0.33)    # 33% below zone
                    direction = 'BUY'
                else:  # D-B-D
                    entry_price = zone['zone_low'] - (zone_size * 0.05)   # 5% below zone  
                    stop_loss = zone['zone_high'] + (zone_size * 0.33)   # 33% above zone
                    direction = 'SELL'
                
                # Calculate take profits
                risk_distance = abs(entry_price - stop_loss)
                if direction == 'BUY':
                    tp1 = entry_price + risk_distance
                    tp2 = entry_price + (risk_distance * 2)
                else:
                    tp1 = entry_price - risk_distance  
                    tp2 = entry_price - (risk_distance * 2)
                
                # Create visualization trade
                trade = {
                    'trade_id': f"VIZ_{zone['type']}_{i+1}",
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': tp1,
                    'take_profit_2': tp2,
                    'position_size': 0.1,  # Standard size for visualization
                    'zone_high': zone['zone_high'],
                    'zone_low': zone['zone_low'],
                    'entry_date': actual_end_date,  # Use end date for entry
                    'exit_date': None,  # No exit for visualization
                    'pnl': 0,  # No P&L for visualization
                    'status': 'visualization_only'
                }
                trades.append(trade)

            print(f"   ✅ Created {len(trades)} visualization trades")
            
            return trades
            
        except Exception as e:
            print(f"❌ Simplified backtest failed: {e}")
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
        """Plot zone only if it has verifiable visible foundation"""
        
        zone_high = trade['zone_high']
        zone_low = trade['zone_low']
        direction = trade['direction']
        
        # STRICT VALIDATION: Only plot zones with visible foundations
        zone_start_idx = self.find_zone_formation_index(trade, data)
        
        if zone_start_idx is None:
            print(f"   🚫 REJECTED Trade {trade_num+1} zone - no visible foundation")
            print(f"      Zone: {zone_low:.5f}-{zone_high:.5f}")
            print(f"      Reason: No candles in visualization period overlap with zone")
            return
        
        # Zone validated - proceed with plotting
        print(f"   ✅ VALIDATED Trade {trade_num+1} zone at index {zone_start_idx}")
        
        # Zone styling
        if direction == 'BUY':
            zone_color = self.colors['demand_zone']
            edge_color = '#006666'
        else:
            zone_color = self.colors['supply_zone']
            edge_color = '#CC0000'
        
        # Calculate zone display duration
        entry_date = pd.to_datetime(trade['entry_date'])
        entry_idx = None
        
        for i, row in data.iterrows():
            if pd.to_datetime(row['date']).date() == entry_date.date():
                entry_idx = i
                break
        
        # Zone shows from formation to entry (or end of chart)
        zone_end_idx = entry_idx if entry_idx is not None else len(data)
        zone_width = max(1, zone_end_idx - zone_start_idx)  # Minimum width of 1
        
        # Plot zone rectangle
        rect = patches.Rectangle((zone_start_idx, zone_low), zone_width, zone_high - zone_low,
                                facecolor=zone_color, alpha=0.15,
                                edgecolor=edge_color, linewidth=1.5,
                                label=f'Trade {trade_num+1} Zone')
        ax.add_patch(rect)
        
        # Zone boundary lines
        ax.hlines(zone_high, zone_start_idx, zone_end_idx, colors=edge_color, linewidth=2, alpha=0.7)
        ax.hlines(zone_low, zone_start_idx, zone_end_idx, colors=edge_color, linewidth=2, alpha=0.7)
        
        # Zone information label
        label_x = zone_start_idx + max(1, zone_width * 0.1)  # 10% into zone or 1 candle
        ax.text(label_x, (zone_high + zone_low) / 2, 
                f'T{trade_num+1}\n{zone_low:.5f}-{zone_high:.5f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=zone_color, alpha=0.8),
                fontsize=8, fontweight='bold', ha='left')
        
        print(f"      Zone plotted from index {zone_start_idx} to {zone_end_idx}")
    
    def find_zone_formation_index(self, trade, data):
        """Find zone formation with PRICE-BASED matching (not index-based)"""
        try:
            zone_high = trade['zone_high']
            zone_low = trade['zone_low']
            zone_center = (zone_high + zone_low) / 2
            
            print(f"   🔍 Searching for zone {zone_low:.5f}-{zone_high:.5f} in visible data")
            
            # Method 1: Find ANY candle that falls within zone price range
            found_indices = []
            for i, row in data.iterrows():
                candle_high = row['high']
                candle_low = row['low']
                candle_center = (candle_high + candle_low) / 2
                
                # Check if candle is within or overlaps zone boundaries
                candle_overlaps = (candle_low <= zone_high and candle_high >= zone_low)
                candle_within = (zone_low <= candle_center <= zone_high)
                zone_within_candle = (candle_low <= zone_center <= candle_high)
                
                if candle_overlaps or candle_within or zone_within_candle:
                    found_indices.append((i, abs(candle_center - zone_center)))
                    print(f"      Match found at index {i}: candle {candle_low:.5f}-{candle_high:.5f}")
            
            if found_indices:
                # Return closest match by price
                found_indices.sort(key=lambda x: x[1])  # Sort by distance
                best_match = found_indices[0][0]
                print(f"   ✅ Zone mapped to index {best_match}")
                return best_match
            
            # Method 2: Check if zone is completely outside visible price range
            data_min = data['low'].min()
            data_max = data['high'].max()
            
            if zone_high < data_min:
                print(f"   🚫 Zone {zone_low:.5f}-{zone_high:.5f} BELOW visible range {data_min:.5f}-{data_max:.5f}")
                return None
            elif zone_low > data_max:
                print(f"   🚫 Zone {zone_low:.5f}-{zone_high:.5f} ABOVE visible range {data_min:.5f}-{data_max:.5f}")
                return None
            
            # Method 3: Find closest candle by price distance
            closest_idx = None
            min_distance = float('inf')
            
            for i, row in data.iterrows():
                candle_center = (row['high'] + row['low']) / 2
                distance = abs(candle_center - zone_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i
            
            # Only accept if reasonably close (within 500 pips)
            max_distance = 0.05  # 500 pips tolerance
            if min_distance <= max_distance:
                print(f"   📍 Zone mapped to closest candle at index {closest_idx} (distance: {min_distance:.5f})")
                return closest_idx
            
            # Zone is invalid - too far from any visible data
            print(f"   🚫 Zone {zone_low:.5f}-{zone_high:.5f} has NO visible foundation")
            print(f"      Visible range: {data_min:.5f}-{data_max:.5f}")
            print(f"      Closest distance: {min_distance:.5f} (max: {max_distance:.5f})")
            return None
            
        except Exception as e:
            print(f"   ❌ Zone mapping error: {e}")
            return None
        
    def plot_trade_markers(self, ax, trade, data, trade_num):
        """Plot trade levels synchronized with zone lifetime"""
        
        zone_high = trade['zone_high']
        zone_low = trade['zone_low']
        entry_price = trade['entry_price']
        stop_price = trade['stop_loss']
        tp1_price = trade['take_profit_1']
        tp2_price = trade['take_profit_2']
        direction = trade['direction']
        
        # Find zone formation index (same as zone plotting)
        zone_start_idx = self.find_zone_formation_index(trade, data)
        
        if zone_start_idx is None:
            print(f"   ⚠️  Cannot plot trade levels for T{trade_num+1} - no zone foundation")
            return
        
        # Calculate zone end index (when zone gets invalidated)
        zone_end_idx = self.find_zone_invalidation_index(trade, data, zone_start_idx)
        
        # Ensure minimum width of 1 candle
        if zone_end_idx <= zone_start_idx:
            zone_end_idx = min(zone_start_idx + 5, len(data) - 1)  # Show for at least 5 candles
        
        print(f"   📍 Plotting T{trade_num+1} trade levels from index {zone_start_idx} to {zone_end_idx}:")
        print(f"      Entry: {entry_price:.5f} (5% from zone)")
        print(f"      Stop: {stop_price:.5f} (33% buffer)")
        print(f"      TP1: {tp1_price:.5f} (1:1 RR)")
        print(f"      TP2: {tp2_price:.5f} (1:2 RR)")
        
        # Entry marker styling
        if direction == 'BUY':
            marker = '^'
            entry_color = self.colors['buy_entry']
        else:
            marker = 'v'
            entry_color = self.colors['sell_entry']
        
        # Plot trade levels ONLY during zone lifetime (same as zone)
        # ENTRY LEVEL - Solid line
        ax.hlines(entry_price, zone_start_idx, zone_end_idx, 
                colors=entry_color, linewidth=3, alpha=0.9, linestyle='-')
        
        # STOP LOSS LEVEL - Dashed red line
        ax.hlines(stop_price, zone_start_idx, zone_end_idx, 
                colors=self.colors['stop_loss'], linewidth=2, 
                alpha=0.8, linestyle='--')
        
        # TAKE PROFIT 1 - Dotted green line
        ax.hlines(tp1_price, zone_start_idx, zone_end_idx, 
                colors=self.colors['take_profit'], linewidth=1.5, 
                alpha=0.7, linestyle=':')
        
        # TAKE PROFIT 2 - Solid green line
        ax.hlines(tp2_price, zone_start_idx, zone_end_idx, 
                colors=self.colors['take_profit'], linewidth=2, 
                alpha=0.8, linestyle='-')
        
        # Entry marker at zone formation point
        ax.scatter(zone_start_idx, entry_price, marker=marker, s=200, 
                color=entry_color, edgecolor='white', linewidth=2, 
                zorder=15, alpha=0.9)
        
        # Trade info box positioned at zone start
        zone_size = zone_high - zone_low
        zone_pips = zone_size / 0.0001
        entry_buffer = abs(entry_price - (zone_high if direction == 'BUY' else zone_low)) / 0.0001
        stop_buffer = abs(stop_price - (zone_low if direction == 'BUY' else zone_high)) / 0.0001
        
        info_text = f'T{trade_num+1}\n'
        info_text += f'{direction}\n'
        info_text += f'E: {entry_price:.5f}\n'
        info_text += f'S: {stop_price:.5f}'
        
        # Position info box just to the right of zone start
        info_x = zone_start_idx + max(1, (zone_end_idx - zone_start_idx) * 0.1)
        info_y = entry_price
        
        ax.text(info_x, info_y, info_text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=entry_color, alpha=0.8),
            fontsize=8, fontweight='bold', color='white',
            verticalalignment='center', ha='left')
        
        # Add level labels at the END of each line (when zone expires)
        ax.text(zone_end_idx + 1, entry_price, f'Entry', 
            fontsize=7, color=entry_color, fontweight='bold', 
            verticalalignment='center', ha='left')
        ax.text(zone_end_idx + 1, stop_price, f'Stop', 
            fontsize=7, color=self.colors['stop_loss'], fontweight='bold',
            verticalalignment='center', ha='left')
        ax.text(zone_end_idx + 1, tp1_price, f'TP1', 
            fontsize=7, color=self.colors['take_profit'], fontweight='bold',
            verticalalignment='center', ha='left')
        ax.text(zone_end_idx + 1, tp2_price, f'TP2', 
            fontsize=7, color=self.colors['take_profit'], fontweight='bold',
            verticalalignment='center', ha='left')

    def find_zone_invalidation_index(self, trade, data, zone_start_idx):
        """Find when the zone gets invalidated/tested"""
        zone_high = trade['zone_high']
        zone_low = trade['zone_low']
        direction = trade['direction']
        
        # Calculate 33% test levels (same as risk manager logic)
        zone_size = zone_high - zone_low
        
        if direction == 'BUY':  # R-B-R demand zone
            test_level = zone_high - (zone_size * 0.33)  # 33% down from top
        else:  # D-B-D supply zone
            test_level = zone_low + (zone_size * 0.33)   # 33% up from bottom
        
        # Search for invalidation after zone formation
        for i in range(zone_start_idx + 1, len(data)):
            candle = data.iloc[i]
            
            if direction == 'BUY':
                # Demand zone invalidated when price closes below test level
                if candle['close'] < test_level:
                    print(f"      Zone invalidated at index {i} (close {candle['close']:.5f} < test {test_level:.5f})")
                    return i
            else:
                # Supply zone invalidated when price closes above test level
                if candle['close'] > test_level:
                    print(f"      Zone invalidated at index {i} (close {candle['close']:.5f} > test {test_level:.5f})")
                    return i
        
        # Zone never invalidated in visible period
        print(f"      Zone remains valid through end of chart")
        return len(data) - 1
        
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
    days_back = days_map.get(choice, 180)
    
    # Call the main validation method
    visualizer.validate_recent_trades(days_back)

if __name__ == "__main__":
    validate_trades()