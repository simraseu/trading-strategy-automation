"""
Zone Debug Analyzer - Standalone Debug Tool (ALIGNED WITH CORE ENGINE)
Run this file independently to analyze specific zones with detailed debug output
FIXED: Now uses identical logic to core_backtest_engine.py for matching results
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from config.settings import ZONE_CONFIG, TREND_CONFIG, RISK_CONFIG

class ZoneDebugAnalyzer:
   """
   Standalone zone analysis tool with detailed debug output
   FIXED: Now matches core_backtest_engine.py logic exactly
   """
   
   def __init__(self):
       """Initialize the debug analyzer"""
       self.data_loader = DataLoader()
       
   def load_data_with_validation(self, pair: str, timeframe: str, days_back: int = 730) -> Optional[pd.DataFrame]:
       """Load data using SAME method as core engine with INTELLIGENT timeframe-aware filtering"""
       try:            
           # Use updated data loader to get full dataset
           data = self.data_loader.load_pair_data(pair, timeframe)
           
           if data is None or len(data) < 100:
               return None
           
           # TIMEFRAME-AWARE FILTERING: Adjust requirements based on timeframe
           timeframe_multipliers = {
               '1D': 1, '2D': 2, '3D': 3, '4D': 4, '5D': 5,
               '1W': 7, '2W': 14, '3W': 21, '1M': 30,
               'H12': 0.5, 'H8': 0.33, 'H4': 0.17
           }
           
           # Get timeframe multiplier (default to 1 if unknown)
           tf_multiplier = timeframe_multipliers.get(timeframe, 1)
           
           # Calculate minimum candles needed based on timeframe
           min_candles_for_ema200 = max(200, int(200 / tf_multiplier))  # Scale EMA200 requirement
           min_trading_candles = max(50, int(100 / tf_multiplier))      # Scale trading data requirement
           absolute_minimum = min_candles_for_ema200 + min_trading_candles
           
           print(f"   📊 Timeframe {timeframe}: multiplier={tf_multiplier}, min_candles={absolute_minimum}")
           
           # INTELLIGENT FILTERING: Only filter if we have sufficient data
           if days_back < 9999 and len(data) > absolute_minimum * 2:  # Only filter if we have 2x minimum
               # Ensure we have a proper datetime index
               if not isinstance(data.index, pd.DatetimeIndex):
                   print(f"⚠️  Data does not have DatetimeIndex: {type(data.index)}")
                   return None
               
               # Calculate required candles for the requested period
               requested_candles = int(days_back / tf_multiplier)
               buffer_candles = min_candles_for_ema200  # EMA200 buffer
               total_candles_needed = requested_candles + buffer_candles
               
               # Only filter if we have more data than needed
               if len(data) > total_candles_needed:
                   # Take the last N candles instead of date-based filtering
                   data = data.tail(total_candles_needed)
                   print(f"   📊 Filtered to {len(data)} candles (candle-based filtering)")
               else:
                   print(f"   📊 Using all {len(data)} candles (insufficient for filtering)")
               
               print(f"   📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
           else:
               print(f"   📊 Using all {len(data)} candles (no filtering applied)")
           
           # Final validation
           if len(data) < absolute_minimum:
               print(f"⚠️  Insufficient data: {len(data)} < {absolute_minimum} required for {timeframe}")
               return None
           
           return data
           
       except Exception as e:
           print(f"❌ Error loading {pair} {timeframe}: {str(e)}")
           import traceback
           traceback.print_exc()
           return None
   
   def get_pip_value_for_pair(self, pair: str) -> float:
       """Get correct pip value for currency pair"""
       if 'JPY' in pair.upper():
           return 0.01
       else:
           return 0.0001
   
   def analyze_zones_for_pair(self, pair: str, timeframe: str, days_back: int = 730) -> List[Dict]:
       """
       Analyze all zones for a specific pair using EXACT CORE ENGINE LOGIC
       
       Args:
           pair: Currency pair (e.g., 'EURUSD')
           timeframe: Timeframe (e.g., '3D')
           days_back: Days of historical data
           
       Returns:
           List of analyzed zones with debug info
       """
       print(f"\n🔍 ZONE DEBUG ANALYSIS: {pair} {timeframe}")
       print("=" * 60)
       
       # Load data using SAME method as core engine
       data = self.load_data_with_validation(pair, timeframe, days_back)
       if data is None or len(data) < 100:
           print(f"❌ Insufficient data for {pair} {timeframe}")
           return []
       
       print(f"📊 Loaded {len(data)} candles")
       print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
       
       # Initialize components using SAME MODULES as core engine
       candle_classifier = CandleClassifier(data)
       classified_data = candle_classifier.classify_all_candles()
       
       zone_detector = ZoneDetector(candle_classifier)
       patterns = zone_detector.detect_all_patterns(classified_data)
       
       trend_classifier = TrendClassifier(data)
       trend_data = trend_classifier.classify_trend_with_filter()
       
       # Get all patterns (matching core engine)
       all_patterns = (patterns['dbd_patterns'] + patterns['rbr_patterns'] + 
                      patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', []))
       
       print(f"🎯 Found {len(all_patterns)} total zones")
       
       # APPLY SAME FILTERING AS CORE ENGINE
       valid_patterns = [
           pattern for pattern in all_patterns
           if pattern.get('end_idx') is not None  # All properly formed zones
       ]

       print(f"🎯 Using {len(valid_patterns)} patterns (realistic - no hindsight)")   

       if not valid_patterns:
           total_zones = len(all_patterns)
           valid_count = len(valid_patterns)
           print(f"❌ No valid zones: {valid_count}/{total_zones} formed properly")
           return []

       print(f"🎯 {len(valid_patterns)} zones available for trading from {len(all_patterns)} total")
       print()
       
       # Store data and trend data for trade simulation
       self.data = data
       self.trend_data = trend_data
       
       # Analyze each zone with detailed debug output
       analyzed_zones = []
       for i, zone in enumerate(valid_patterns, 1):
           print(f"🔍 ZONE #{i}: {zone['type']} zone")
           
           # Add pair info and get ACTUAL zone end index
           zone['pair'] = pair
           zone_end_idx = zone.get('end_idx')
           if zone_end_idx is None:
               print(f"   ❌ Missing end_idx for zone #{i}")
               continue
           
           # Analyze this zone
           analysis = self.debug_single_zone(zone, data, trend_data, zone_end_idx)
           if analysis:
               analyzed_zones.append(analysis)
           
           print()  # Spacing between zones
       
       print(f"✅ Analysis complete: {len(analyzed_zones)} zones analyzed")
       return analyzed_zones
   
   def debug_single_zone(self, zone: Dict, data: pd.DataFrame, trend_data: pd.DataFrame, zone_end_idx: int) -> Optional[Dict]:
       """
       Debug a single zone with full calculation breakdown using CORE ENGINE LOGIC
       
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
           
           # Calculate entry and stop prices using CORE ENGINE LOGIC
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
           
           # Risk management calculations using CORE ENGINE LOGIC
           max_risk_amount = 500  # $500 max risk per trade (5% of $10,000)
           
           # Pip value per lot calculation
           if 'JPY' in pair.upper():
               pip_value_per_lot = 1.0  # $1 per pip for JPY pairs
           else:
               pip_value_per_lot = 10.0  # $10 per pip for major pairs
           
           print(f"   Risk Amount: ${max_risk_amount}, Stop Distance: {stop_distance_pips:.1f} pips")
           print(f"   Adjusted Pip Value Per Lot: ${pip_value_per_lot}")
           
           # Position sizing using CORE ENGINE LOGIC
           if stop_distance_pips > 0:
               proper_position_size = max_risk_amount / (stop_distance_pips * pip_value_per_lot)
               proper_position_size = max(0.01, min(proper_position_size, 1.0))  # Apply limits
               
               print(f"   Calculated Position Size: {proper_position_size:.4f} lots")
               print(f"   Final Position Size: {proper_position_size:.4f} lots")
           else:
               print(f"   ❌ Invalid stop distance: {stop_distance_pips} pips")
               return None
           
           # Target calculation using CORE ENGINE LOGIC (2.5R)
           risk_distance = abs(entry_price - initial_stop)
           target_rr = 2.5  # 1:2.5 risk reward
           
           if direction == 'BUY':
               target_price = entry_price + (risk_distance * target_rr)
           else:
               target_price = entry_price - (risk_distance * target_rr)
           
           print(f"   Target Price: {target_price:.6f} (2.5R)")
           
           # Check trend alignment using CORE ENGINE LOGIC
           is_aligned = False
           current_trend = 'unknown'
           
           if zone_end_idx < len(trend_data):
               current_trend = trend_data['trend'].iloc[min(zone_end_idx, len(trend_data) - 1)]
               
               # Apply CORE ENGINE trend alignment logic
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
               'zone_end_idx': zone_end_idx,
               'trend_aligned': is_aligned,
               'current_trend': current_trend
           }
           
       except Exception as e:
           print(f"   ❌ Error analyzing zone: {str(e)}")
           return None

   def execute_realistic_trades(self, analyzed_zones: List[Dict], data: pd.DataFrame, pair: str) -> List[Dict]:
       """
       Execute trades using EXACT CORE ENGINE LOGIC for realistic walk-forward simulation
       """
       trades = []
       used_zones = set()
       
       # Apply CORE ENGINE trend filtering first
       current_trend = self.trend_data['trend'].iloc[-1]
       
       aligned_zones = []
       for zone in analyzed_zones:
           zone_type = zone['zone_type']
           
           # Apply CORE ENGINE trend alignment logic
           is_aligned = False
           if current_trend == 'bullish':
               is_aligned = zone_type in ['R-B-R', 'D-B-R']
           elif current_trend == 'bearish':
               is_aligned = zone_type in ['D-B-D', 'R-B-D']
           
           if is_aligned:
               aligned_zones.append(zone)
       
       print(f"   Current trend: {current_trend.upper()}")
       print(f"   Trend-aligned zones: {len(aligned_zones)} of {len(analyzed_zones)}")
       
       if not aligned_zones:
           print("   ⚠️  No trend-aligned zones found for trading")
           return []
       
       # Build realistic zone activation schedule (CORE ENGINE LOGIC)
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
       
       # Simulate trades with realistic walk-forward logic (CORE ENGINE LOGIC)
       start_idx = max(200, min(z['activation_idx'] for z in zone_activation_schedule) if zone_activation_schedule else 200)
       
       for current_idx in range(start_idx, len(data) - 20):  # Leave room for trade execution
           for zone_info in zone_activation_schedule:
               zone = zone_info['zone']
               zone_id = f"{zone['zone_type']}_{zone_info['zone_end_idx']}"
               
               # Skip if zone already used or not yet activated
               if (zone_id in used_zones or 
                   current_idx < zone_info['activation_idx']):
                   continue
               
               # Check trend alignment at current time (CORE ENGINE LOGIC)
               if current_idx < len(self.trend_data):
                   current_moment_trend = self.trend_data['trend'].iloc[current_idx]
                   
                   is_aligned = False
                   if current_moment_trend == 'bullish':
                       is_aligned = zone['zone_type'] in ['R-B-R', 'D-B-R']
                   elif current_moment_trend == 'bearish':
                       is_aligned = zone['zone_type'] in ['D-B-D', 'R-B-D']
                   
                   if not is_aligned:
                       continue
               
               # Execute trade with CORE ENGINE LOGIC
               trade_result = self.execute_single_realistic_trade(zone, data, current_idx, pair)
               if trade_result:
                   trade_result['zone_info'] = zone
                   trades.append(trade_result)
                   used_zones.add(zone_id)
                   
                   # Stop after finding enough examples
                   if len(trades) >= 100:
                       break
           
           if len(trades) >= 100:
               break
       
       return trades

   def execute_single_realistic_trade(self, zone: Dict, data: pd.DataFrame, current_idx: int, pair: str) -> Optional[Dict]:
       """
       Execute single trade using EXACT CORE ENGINE LOGIC
       """
       entry_price = zone['entry_price']
       stop_price = zone['stop_price']
       target_price = zone['target_price']
       direction = zone['direction']
       position_size = zone['position_size']
       
       # Check if current candle can trigger entry (CORE ENGINE LOGIC)
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
       
       # Execute the trade simulation using CORE ENGINE LOGIC
       return self.simulate_trade_execution(
           entry_price, stop_price, target_price, direction,
           position_size, data, current_idx, pair
       )

   def simulate_trade_execution(self, entry_price: float, stop_price: float, target_price: float,
                              direction: str, position_size: float, data: pd.DataFrame,
                              entry_idx: int, pair: str) -> Dict:
       """
       Simulate realistic trade execution with break-even management (EXACT CORE ENGINE LOGIC)
       """
       # Get pip value for P&L calculation
       pip_value = self.get_pip_value_for_pair(pair)
       
       # Pip value per lot for P&L
       if 'JPY' in pair.upper():
           pip_value_per_lot = 1.0
       else:
           pip_value_per_lot = 10.0
       
       # Add spread cost (CORE ENGINE LOGIC)
       spread_pips = 2.0
       if direction == 'BUY':
           entry_price += (spread_pips * pip_value)
       else:
           entry_price -= (spread_pips * pip_value)
       
       risk_distance = abs(entry_price - stop_price)
       current_stop = stop_price
       breakeven_moved = False
       
       # Look ahead for exit (CORE ENGINE limit of 50 candles)
       for exit_idx in range(entry_idx + 1, min(entry_idx + 50, len(data))):
           exit_candle = data.iloc[exit_idx]
           
           # Calculate 1R target for break-even (CORE ENGINE LOGIC)
           one_r_target = entry_price + risk_distance if direction == 'BUY' else entry_price - risk_distance
           
           # Check for 1R hit (break-even trigger) - EXACT CORE ENGINE LOGIC
           if not breakeven_moved:
               if direction == 'BUY' and exit_candle['high'] >= one_r_target:
                   current_stop = entry_price  # Move stop to EXACT entry price
                   breakeven_moved = True
               elif direction == 'SELL' and exit_candle['low'] <= one_r_target:
                   current_stop = entry_price  # Move stop to EXACT entry price
                   breakeven_moved = True
           
           # Check exits with WICK-BASED logic (CORE ENGINE LOGIC)
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
           
           else:  # SELL - EXACT CORE ENGINE LOGIC
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

   def show_trade_examples(self, analyzed_zones: List[Dict], data: pd.DataFrame, pair: str):
       """
       Show examples of different trade outcomes using EXACT CORE ENGINE LOGIC
       """
       print(f"\n🎯 TRADE OUTCOME EXAMPLES (CORE ENGINE SIMULATION)")
       print("=" * 50)
       
       # Execute trades using EXACT CORE ENGINE LOGIC
       trade_results = self.execute_realistic_trades(analyzed_zones, data, pair)
       
       if len(trade_results) == 0:
           print("   ⚠️  No tradeable outcomes found in the data period")
           return
       
       # Categorize results (CORE ENGINE LOGIC)
       wins = [t for t in trade_results if t['result'] == 'WIN']
       losses = [t for t in trade_results if t['result'] == 'LOSS']
       breakevens = [t for t in trade_results if t['result'] == 'BREAKEVEN']
       neutrals = [t for t in trade_results if t['result'] == 'NEUTRAL']
       
       # Calculate statistics (CORE ENGINE LOGIC)
       total_trades = len(trade_results)
       win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
       be_rate = (len(breakevens) / total_trades * 100) if total_trades > 0 else 0
       loss_rate = (len(losses) / total_trades * 100) if total_trades > 0 else 0
       
       total_pnl = sum(t['pnl'] for t in trade_results)
       gross_profit = sum(t['pnl'] for t in wins)
       gross_loss = abs(sum(t['pnl'] for t in losses))
       profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
       
       print(f"📊 Trade Results Summary (MATCHING CORE ENGINE):")
       print(f"   💚 Wins: {len(wins)} ({win_rate:.1f}%)")
       print(f"   🔴 Losses: {len(losses)} ({loss_rate:.1f}%)")
       print(f"   ⚖️  Breakevens: {len(breakevens)} ({be_rate:.1f}%)")
       print(f"   ⚪ Neutrals: {len(neutrals)}")
       print(f"   📊 Total Simulated: {total_trades}")
       print(f"   💰 Total P&L: ${total_pnl:.0f}")
       print(f"   📈 Profit Factor: {profit_factor:.2f}")
       print(f"   📈 Total Return: {(total_pnl/10000)*100:.2f}%")
       
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

def main():
   """Main function for zone debug analysis"""
   print("🔍 ZONE DEBUG ANALYZER (ALIGNED WITH CORE ENGINE)")
   print("=" * 50)
   
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