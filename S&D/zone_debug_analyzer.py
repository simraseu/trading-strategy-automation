"""
Zone Debug Analyzer - Standalone Debug Tool
Run this file independently to analyze specific zones with detailed debug output
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from typing import Dict, List, Optional
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from config.settings import ZONE_CONFIG, TREND_CONFIG, RISK_CONFIG

class ZoneDebugAnalyzer:
   """
   Standalone zone analysis tool with detailed debug output
   Analyzes zones and shows complete calculation breakdown
   """
   
   def __init__(self):
       """Initialize the debug analyzer"""
       self.data_loader = DataLoader()
       
   def get_pip_value_for_pair(self, pair: str) -> float:
       """Get correct pip value for currency pair"""
       if 'JPY' in pair.upper():
           return 0.01
       else:
           return 0.0001
   
   def analyze_zones_for_pair(self, pair: str, timeframe: str, days_back: int = 730) -> List[Dict]:
       """
       Analyze all zones for a specific pair with full debug output
       
       Args:
           pair: Currency pair (e.g., 'EURUSD')
           timeframe: Timeframe (e.g., '3D')
           days_back: Days of historical data
           
       Returns:
           List of analyzed zones with debug info
       """
       print(f"\n🔍 ZONE DEBUG ANALYSIS: {pair} {timeframe}")
       print("=" * 60)
       
       # Load and process data
       data = self.data_loader.load_pair_data(pair, timeframe)
       if data is None or len(data) < 100:
           print(f"❌ Insufficient data for {pair} {timeframe}")
           return []
       
       # Limit data if requested
       if days_back < 9999 and len(data) > days_back:
           data = data.tail(days_back)
       
       print(f"📊 Loaded {len(data)} candles")
       print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
       
       # Initialize components
       candle_classifier = CandleClassifier(data)
       classified_data = candle_classifier.classify_all_candles()
       
       zone_detector = ZoneDetector(candle_classifier)
       patterns = zone_detector.detect_all_patterns(classified_data)
       
       trend_classifier = TrendClassifier(data)
       trend_data = trend_classifier.classify_trend_with_filter()
       
       # Get all patterns
       all_patterns = (patterns['dbd_patterns'] + patterns['rbr_patterns'] + 
                      patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', []))
       
       print(f"🎯 Found {len(all_patterns)} total zones")
       print()
       
       # Store data and trend data for trade simulation
       self.data = data
       self.trend_data = trend_data
       
       # Analyze each zone with detailed debug output
       analyzed_zones = []
       for i, zone in enumerate(all_patterns, 1):
           print(f"🔍 ZONE #{i}: {zone['type']} zone")
           
           # Add pair info and actual zone end index to zone
           zone['pair'] = pair
           zone_end_idx = zone.get('end_idx', i * 20 + 200)  # Get real end_idx or estimate
           
           # Analyze this zone
           analysis = self.debug_single_zone(zone, data, trend_data, zone_end_idx)
           if analysis:
               analyzed_zones.append(analysis)
           
           print()  # Spacing between zones
       
       print(f"✅ Analysis complete: {len(analyzed_zones)} zones analyzed")
       return analyzed_zones
   
   def debug_single_zone(self, zone: Dict, data: pd.DataFrame, trend_data: pd.DataFrame, zone_end_idx: int) -> Optional[Dict]:
       """
       Debug a single zone with full calculation breakdown
       
       Args:
           zone: Zone dictionary
           data: OHLC data
           trend_data: Trend classification data
           zone_end_idx: Actual zone end index
           
       Returns:
           Zone analysis with debug info
       """
       try:
           zone_high = zone['zone_high']
           zone_low = zone['zone_low']
           zone_range = zone_high - zone_low
           pair = zone.get('pair', 'EURUSD')
           
           print(f"   Zone High: {zone_high:.6f}")
           print(f"   Zone Low: {zone_low:.6f}")
           print(f"   Zone Range: {zone_range:.6f}")
           
           # Calculate entry and stop prices
           if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
               entry_price = zone_high + (zone_range * 0.05)  # 5% above zone
               direction = 'BUY'
               initial_stop = zone_low - (zone_range * 0.33)  # 33% buffer below
           elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell)
               entry_price = zone_low - (zone_range * 0.05)  # 5% below zone
               direction = 'SELL'
               initial_stop = zone_high + (zone_range * 0.33)  # 33% buffer above
           else:
               print(f"   ❌ Unknown zone type: {zone['type']}")
               return None
           
           # Calculate pip values and distances
           pip_value = self.get_pip_value_for_pair(pair)
           stop_distance_pips = abs(entry_price - initial_stop) / pip_value
           
           print(f"   Entry: {entry_price:.6f}, Stop: {initial_stop:.6f}")
           print(f"   Pip Value: {pip_value}, Stop Distance: {stop_distance_pips:.1f} pips")
           
           # Risk management calculations
           max_risk_amount = 500  # $500 max risk per trade (5% of $10,000)
           
           # Pip value per lot calculation
           if 'JPY' in pair.upper():
               pip_value_per_lot = 1.0  # $1 per pip for JPY pairs
           else:
               pip_value_per_lot = 10.0  # $10 per pip for major pairs
           
           print(f"   Risk Amount: ${max_risk_amount}, Stop Distance: {stop_distance_pips:.1f} pips")
           print(f"   Adjusted Pip Value Per Lot: ${pip_value_per_lot}")
           
           # Position sizing
           if stop_distance_pips > 0:
               proper_position_size = max_risk_amount / (stop_distance_pips * pip_value_per_lot)
               proper_position_size = max(0.01, min(proper_position_size, 1.0))  # Apply limits
               
               print(f"   Calculated Position Size: {proper_position_size:.4f} lots")
               print(f"   Final Position Size: {proper_position_size:.4f} lots")
           else:
               print(f"   ❌ Invalid stop distance: {stop_distance_pips} pips")
               return None
           
           # Target calculation
           risk_distance = abs(entry_price - initial_stop)
           target_rr = 2.5  # 1:2.5 risk reward
           
           if direction == 'BUY':
               target_price = entry_price + (risk_distance * target_rr)
           else:
               target_price = entry_price - (risk_distance * target_rr)
           
           print(f"   Target Price: {target_price:.6f} (2.5R)")
           
           # Check trend alignment
           if zone_end_idx < len(trend_data):
               current_trend = trend_data['trend'].iloc[min(zone_end_idx, len(trend_data) - 1)]
               
               is_aligned = False
               if current_trend == 'bullish':
                   is_aligned = zone['type'] in ['R-B-R', 'D-B-R']
               elif current_trend == 'bearish':
                   is_aligned = zone['type'] in ['D-B-D', 'R-B-D']
               
               trend_status = "✅ ALIGNED" if is_aligned else "❌ NOT ALIGNED"
               print(f"   Trend: {current_trend.upper()} - {trend_status}")
           else:
               print(f"   Trend: Unknown (zone index out of range)")
           
           # Zone formation date
           try:
               if zone_end_idx < len(data):
                   formation_date = data.index[zone_end_idx]
                   print(f"   Formation Date: {formation_date.strftime('%Y-%m-%d')}")
               else:
                   print(f"   Formation Date: Recent (index {zone_end_idx})")
           except:
               print(f"   Formation Date: Unknown")
           
           return {
               'zone_type': zone['type'],
               'pair': pair,
               'direction': direction,
               'zone_high': zone_high,
               'zone_low': zone_low,
               'zone_range': zone_range,
               'entry_price': entry_price,
               'stop_price': initial_stop,
               'target_price': target_price,
               'stop_distance_pips': stop_distance_pips,
               'position_size': proper_position_size,
               'risk_amount': max_risk_amount,
               'pip_value': pip_value,
               'pip_value_per_lot': pip_value_per_lot,
               'zone_end_idx': zone_end_idx,  # Store actual zone end index
               'trend_aligned': is_aligned if zone_end_idx < len(trend_data) else False
           }
           
       except Exception as e:
           print(f"   ❌ Error analyzing zone: {str(e)}")
           return None

   def show_trade_examples(self, analyzed_zones: List[Dict], data: pd.DataFrame, pair: str):
       """
       Show examples of different trade outcomes using REALISTIC simulation
       """
       print(f"\n🎯 TRADE OUTCOME EXAMPLES")
       print("=" * 50)
       
       # Only trade trend-aligned zones (like the core engine)
       aligned_zones = [zone for zone in analyzed_zones if zone.get('trend_aligned', False)]
       
       if not aligned_zones:
           print("   ⚠️  No trend-aligned zones found for trading")
           return
       
       # Build realistic zone activation schedule
       zone_activation_schedule = []
       for zone in aligned_zones:
           zone_end_idx = zone['zone_end_idx']
           activation_idx = zone_end_idx + 1  # Activate 1 candle after formation
           
           zone_activation_schedule.append({
               'zone': zone,
               'zone_end_idx': zone_end_idx,
               'activation_idx': activation_idx
           })
       
       # Sort by activation time
       zone_activation_schedule.sort(key=lambda x: x['activation_idx'])
       
       # Simulate trades with realistic walk-forward logic (like core engine)
       trade_results = []
       used_zones = set()
       
       # Start scanning from a reasonable point in the data
       start_idx = max(200, min(z['activation_idx'] for z in zone_activation_schedule) if zone_activation_schedule else 200)
       
       for current_idx in range(start_idx, len(data) - 20):  # Leave room for trade execution
           for zone_info in zone_activation_schedule:
               zone = zone_info['zone']
               zone_id = f"{zone['zone_type']}_{zone_info['zone_end_idx']}"
               
               # Skip if zone already used or not yet activated
               if (zone_id in used_zones or 
                   current_idx < zone_info['activation_idx']):
                   continue
               
               # Check trend alignment at current time (like core engine)
               if current_idx < len(self.trend_data):
                   current_trend = self.trend_data['trend'].iloc[current_idx]
                   
                   is_aligned = False
                   if current_trend == 'bullish':
                       is_aligned = zone['zone_type'] in ['R-B-R', 'D-B-R']
                   elif current_trend == 'bearish':
                       is_aligned = zone['zone_type'] in ['D-B-D', 'R-B-D']
                   
                   if not is_aligned:
                       continue
               
               # Check if this zone can trigger an entry
               trade_result = self.simulate_single_zone_trade(zone, data, current_idx, pair)
               if trade_result:
                   trade_result['zone_info'] = zone
                   trade_results.append(trade_result)
                   used_zones.add(zone_id)
                   
                   # Stop after finding enough examples
                   if len(trade_results) >= 30:
                       break
           
           if len(trade_results) >= 30:
               break
       
       # Categorize results
       wins = [t for t in trade_results if t['result'] == 'WIN']
       losses = [t for t in trade_results if t['result'] == 'LOSS']
       breakevens = [t for t in trade_results if t['result'] == 'BREAKEVEN']
       neutrals = [t for t in trade_results if t['result'] == 'NEUTRAL']
       
       # Calculate statistics
       total_trades = len(trade_results)
       win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
       be_rate = (len(breakevens) / total_trades * 100) if total_trades > 0 else 0
       loss_rate = (len(losses) / total_trades * 100) if total_trades > 0 else 0
       
       total_pnl = sum(t['pnl'] for t in trade_results)
       gross_profit = sum(t['pnl'] for t in wins)
       gross_loss = abs(sum(t['pnl'] for t in losses))
       profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
       
       print(f"📊 Trade Results Summary:")
       print(f"   💚 Wins: {len(wins)} ({win_rate:.1f}%)")
       print(f"   🔴 Losses: {len(losses)} ({loss_rate:.1f}%)")
       print(f"   ⚖️  Breakevens: {len(breakevens)} ({be_rate:.1f}%)")
       print(f"   ⚪ Neutrals: {len(neutrals)}")
       print(f"   📊 Total Simulated: {total_trades}")
       print(f"   💰 Total P&L: ${total_pnl:.0f}")
       print(f"   📈 Profit Factor: {profit_factor:.2f}")
       
       # Show examples
       if wins:
           print(f"\n💚 WINNING TRADE EXAMPLES:")
           for i, trade in enumerate(wins[:5], 1):
               zone = trade['zone_info']
               print(f"   Win #{i}: {zone['zone_type']} zone")
               print(f"      Entry: {zone['entry_price']:.5f} → Exit: {trade['exit_price']:.5f}")
               print(f"      Result: +{trade['pips']:.1f} pips = ${trade['pnl']:.0f}")
               print(f"      Duration: {trade['duration']} candles")
               print()
       
       if losses:
           print(f"🔴 LOSING TRADE EXAMPLES:")
           for i, trade in enumerate(losses[:5], 1):
               zone = trade['zone_info']
               print(f"   Loss #{i}: {zone['zone_type']} zone")
               print(f"      Entry: {zone['entry_price']:.5f} → Exit: {trade['exit_price']:.5f}")
               print(f"      Result: {trade['pips']:.1f} pips = ${trade['pnl']:.0f}")
               print(f"      Duration: {trade['duration']} candles")
               print()
       
       if breakevens:
           print(f"⚖️  BREAKEVEN TRADE EXAMPLES:")
           for i, trade in enumerate(breakevens[:3], 1):
               zone = trade['zone_info']
               print(f"   BE #{i}: {zone['zone_type']} zone")
               print(f"      Entry: {zone['entry_price']:.5f} → Exit: {trade['exit_price']:.5f}")
               print(f"      Result: 1R hit → moved to breakeven → stopped out")
               print(f"      P&L: ${trade['pnl']:.0f} (commission only)")
               print(f"      Duration: {trade['duration']} candles")
               print()
       
       if len(trade_results) == 0:
           print("   ⚠️  No tradeable outcomes found in the data period")

   def simulate_single_zone_trade(self, zone: Dict, data: pd.DataFrame, current_idx: int, pair: str) -> Optional[Dict]:
       """
       Simulate a single zone trade using realistic entry logic (matching core engine)
       """
       try:
           entry_price = zone['entry_price']
           stop_price = zone['stop_price']
           target_price = zone['target_price']
           direction = zone['direction']
           position_size = zone['position_size']
           
           # Check if current candle can trigger entry (same logic as core engine)
           current_candle = data.iloc[current_idx]
           
           can_enter = False
           if direction == 'BUY':
               # For demand zones: price approaches from ABOVE, triggers buy entry above zone
               if (current_candle['high'] >= entry_price and 
                   current_candle['low'] <= zone['zone_high']):  # Confirms approach from above
                   can_enter = True
           elif direction == 'SELL':
               # For supply zones: price approaches from BELOW, triggers sell entry below zone
               if (current_candle['low'] <= entry_price and 
                   current_candle['high'] >= zone['zone_low']):  # Confirms approach from below
                   can_enter = True
           
           if not can_enter:
               return None
           
           # Execute the trade simulation
           return self.simulate_trade_execution(
               entry_price, stop_price, target_price, direction,
               position_size, data, current_idx, pair
           )
           
       except Exception as e:
           return None

   def simulate_trade_execution(self, entry_price: float, stop_price: float, target_price: float,
                              direction: str, position_size: float, data: pd.DataFrame,
                              entry_idx: int, pair: str) -> Dict:
       """
       Simulate realistic trade execution with break-even management (matching core engine)
       """
       # Get pip value for P&L calculation
       pip_value = self.get_pip_value_for_pair(pair)
       
       # Pip value per lot for P&L
       if 'JPY' in pair.upper():
           pip_value_per_lot = 1.0
       else:
           pip_value_per_lot = 10.0
       
       # Add spread cost (matching core engine)
       spread_pips = 2.0
       if direction == 'BUY':
           entry_price += (spread_pips * pip_value)
       else:
           entry_price -= (spread_pips * pip_value)
       
       risk_distance = abs(entry_price - stop_price)
       current_stop = stop_price
       breakeven_moved = False
       
       # Look ahead for exit (matching core engine limit of 50 candles)
       for exit_idx in range(entry_idx + 1, min(entry_idx + 50, len(data))):
           exit_candle = data.iloc[exit_idx]
           
           # Calculate 1R target for break-even
           one_r_target = entry_price + risk_distance if direction == 'BUY' else entry_price - risk_distance
           
           # Check for 1R hit (break-even trigger) - EXACT logic from core engine
           if not breakeven_moved:
               if direction == 'BUY' and exit_candle['high'] >= one_r_target:
                   current_stop = entry_price  # Move stop to EXACT entry price
                   breakeven_moved = True
               elif direction == 'SELL' and exit_candle['low'] <= one_r_target:
                   current_stop = entry_price  # Move stop to EXACT entry price
                   breakeven_moved = True
           
           # Check exits with WICK-BASED logic (matching core engine)
           if direction == 'BUY':
               # Stop hit
               if exit_candle['low'] <= current_stop:
                   price_diff = current_stop - entry_price
                   pips_moved = price_diff / pip_value
                   gross_pnl = pips_moved * position_size * pip_value_per_lot
                   total_commission = 7.0 * position_size * 2  # Commission on entry and exit
                   net_pnl = gross_pnl - total_commission
                   
                   if breakeven_moved and current_stop == entry_price:
                       result = 'BREAKEVEN'
                   else:
                       result = 'LOSS' if net_pnl < 0 else 'WIN'
                   
                   return {
                       'result': result,
                       'pnl': round(net_pnl, 2),
                       'pips': round(pips_moved, 1),
                       'exit_price': current_stop,
                       'breakeven_moved': breakeven_moved,
                       'duration': exit_idx - entry_idx
                   }
               
               # Target hit
               elif exit_candle['high'] >= target_price:
                   price_diff = target_price - entry_price
                   pips_moved = price_diff / pip_value
                   gross_pnl = pips_moved * position_size * pip_value_per_lot
                   total_commission = 7.0 * position_size * 2
                   net_pnl = gross_pnl - total_commission
                   
                   return {
                       'result': 'WIN',
                       'pnl': round(net_pnl, 2),
                       'pips': round(pips_moved, 1),
                       'exit_price': target_price,
                       'breakeven_moved': breakeven_moved,
                       'duration': exit_idx - entry_idx
                   }
           
           else:  # SELL
               # Stop hit
               if exit_candle['high'] >= current_stop:
                   price_diff = entry_price - current_stop
                   pips_moved = price_diff / pip_value
                   gross_pnl = pips_moved * position_size * pip_value_per_lot
                   total_commission = 7.0 * position_size * 2
                   net_pnl = gross_pnl - total_commission
                   
                   if breakeven_moved and current_stop == entry_price:
                       result = 'BREAKEVEN'
                   else:
                       result = 'LOSS' if net_pnl < 0 else 'WIN'
                   
                   return {
                       'result': result,
                       'pnl': round(net_pnl, 2),
                       'pips': round(pips_moved, 1),
                       'exit_price': current_stop,
                       'breakeven_moved': breakeven_moved,
                       'duration': exit_idx - entry_idx
                   }
               
               # Target hit
               elif exit_candle['low'] <= target_price:
                   price_diff = entry_price - target_price
                   pips_moved = price_diff / pip_value
                   gross_pnl = pips_moved * position_size * pip_value_per_lot
                   total_commission = 7.0 * position_size * 2
                   net_pnl = gross_pnl - total_commission
                   
                   return {
                       'result': 'WIN',
                       'pnl': round(net_pnl, 2),
                       'pips': round(pips_moved, 1),
                       'exit_price': target_price,
                       'breakeven_moved': breakeven_moved,
                       'duration': exit_idx - entry_idx
                   }
       
       # Trade still open at end - assume neutral exit with costs
       return {
           'result': 'NEUTRAL',
           'pnl': -14.0,  # Just commission costs
           'pips': 0.0,
           'exit_price': entry_price,
           'breakeven_moved': breakeven_moved,
           'duration': 50
       }

def main():
   """Main function for zone debug analysis"""
   print("🔍 ZONE DEBUG ANALYZER")
   print("=" * 40)
   
   analyzer = ZoneDebugAnalyzer()
   
   print("\n🎯 SELECT ANALYSIS MODE:")
   print("1. Analyze specific pair/timeframe")
   print("2. Quick analysis (EURUSD 3D)")
   print("3. JPY pair analysis (CHFJPY 1M)")
   
   choice = input("\nEnter choice (1-3): ").strip()
   
   if choice == '1':
       # Custom analysis
       pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
       timeframe = input("Enter timeframe (e.g., 3D): ").strip()
       days_back = input("Enter days back (e.g., 730): ").strip()
       
       try:
           days_back = int(days_back)
           analyzed_zones = analyzer.analyze_zones_for_pair(pair, timeframe, days_back)
           
           # Get the data for trade simulation
           data = analyzer.data_loader.load_pair_data(pair, timeframe)
           if data is not None and analyzed_zones:
               analyzer.show_trade_examples(analyzed_zones, data, pair)
               
       except ValueError:
           print("❌ Invalid days back value")
   
   elif choice == '2':
       # Quick EURUSD analysis
       analyzed_zones = analyzer.analyze_zones_for_pair('EURUSD', '3D', 730)
       
       # Get the data for trade simulation
       data = analyzer.data_loader.load_pair_data('EURUSD', '3D')
       if data is not None and analyzed_zones:
           analyzer.show_trade_examples(analyzed_zones, data, 'EURUSD')
   
   elif choice == '3':
       # JPY pair analysis
       analyzed_zones = analyzer.analyze_zones_for_pair('CHFJPY', '1M', 730)
       
       # Get the data for trade simulation
       data = analyzer.data_loader.load_pair_data('CHFJPY', '1M')
       if data is not None and analyzed_zones:
           analyzer.show_trade_examples(analyzed_zones, data, 'CHFJPY')
   
   else:
       print("❌ Invalid choice")

if __name__ == "__main__":
   main()