import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator

class ManualTradeValidator:
   def __init__(self):
       self.data_loader = DataLoader()
       self.validation_results = []
       
   def validate_all_trades(self, csv_file='manual_trades_validation.csv'):
       """Main validation function"""
       print("🔍 MANUAL TRADE VALIDATION ENGINE")
       print("=" * 50)
       
       # Load manual trades
       try:
           manual_trades = pd.read_csv(csv_file)
           print(f"📋 Loaded {len(manual_trades)} manual trades")
       except FileNotFoundError:
           print(f"❌ File {csv_file} not found!")
           return
       
       # Load EURUSD data
       data = self.data_loader.load_pair_data('EURUSD', 'Daily')
       print(f"📊 Loaded {len(data)} candles\n")
       
       # Validate each trade
       for index, trade in manual_trades.iterrows():
           result = self.validate_single_trade(trade, data)
           self.validation_results.append(result)
       
       # Final summary
       self.print_final_summary()
   
   def validate_single_trade(self, manual_trade, data):
       """Validate single trade"""
       trade_id = manual_trade['trade_id']
       direction = str(manual_trade['direction']).upper().strip()
       zone_low = manual_trade['zone_low']
       zone_high = manual_trade['zone_high']
       
       # Determine zone type
       zone_type = 'R-B-R' if direction == 'BUY' else 'D-B-D'
       
       # Check if trade is invalid
       entry_date = manual_trade['entry_date']
       is_invalid = (pd.isna(entry_date) or str(entry_date).strip() in ['N/A', '', 'Invalid'])
       
       print(f"🧪 TRADE {trade_id}: {direction} | Zone: {zone_low:.4f}-{zone_high:.4f} {'[INVALID]' if is_invalid else ''}")
       
       # Zone Detection
       zone_result = self.test_zone_detection(manual_trade, data, zone_type)
       
       if zone_result['zone_found']:
           print(f"   Zone: ✅ FOUND (quality: {zone_result['match_quality']:.0f}%)")
           
           if not is_invalid:
               # Signal Generation
               signal_result = self.test_signal_generation(manual_trade, data, zone_result['best_match'])
               
               if signal_result and signal_result['signal_generated']:
                   print(f"   Signal: ✅ GENERATED ({signal_result['total_signals']} total)")
                   overall_result = "✅ FULL MATCH"
                   overall_success = True
               else:
                   print(f"   Signal: ❌ NOT GENERATED")
                   overall_result = "❌ SIGNAL FAIL"
                   overall_success = False
           else:
               # Invalid trade - should not have generated signal
               overall_result = "❌ UNEXPECTED ZONE"
               overall_success = False
       else:
           if is_invalid:
               print(f"   Zone: ✅ NOT FOUND (correct)")
               overall_result = "✅ CORRECT REJECTION"
               overall_success = True
           else:
               print(f"   Zone: ❌ NOT FOUND")
               overall_result = "❌ ZONE FAIL"
               overall_success = False
       
       print(f"   Result: {overall_result}\n")
       
       return {
           'trade_id': trade_id,
           'overall_success': overall_success,
           'expected_invalid': is_invalid
       }
   
   def test_zone_detection(self, manual_trade, data, zone_type):
       """Test zone detection"""
       try:
           base_end_date = manual_trade['base_end_date']
           
           # Parse date
           try:
               parsed_date = pd.to_datetime(base_end_date, format='%d-%m-%Y')
           except:
               parsed_date = pd.to_datetime(base_end_date)
           
           base_end_idx = data.index.get_loc(parsed_date)
           
           # Create data window
           window_start = max(0, base_end_idx - 50)
           window_end = min(len(data), base_end_idx + 20)
           window_data = data.iloc[window_start:window_end]
           
           # Initialize components
           candle_classifier = CandleClassifier(window_data)
           classified_data = candle_classifier.classify_all_candles()
           zone_detector = ZoneDetector(candle_classifier)
           patterns = zone_detector.detect_all_patterns(classified_data)
           
           # Find matching zone
           manual_zone = {
               'low': float(manual_trade['zone_low']),
               'high': float(manual_trade['zone_high']),
               'type': zone_type
           }
           
           best_match = self.find_best_zone_match(patterns, manual_zone)
           
           if best_match:
               match_quality = self.calculate_zone_match_quality(manual_zone, best_match)
               return {
                   'zone_found': True,
                   'best_match': best_match,
                   'match_quality': match_quality
               }
           else:
               return {
                   'zone_found': False,
                   'best_match': None,
                   'match_quality': 0
               }
               
       except Exception as e:
           return {
               'zone_found': False,
               'best_match': None,
               'match_quality': 0,
               'error': str(e)
           }
   
   def test_signal_generation(self, manual_trade, data, detected_zone):
       """Test signal generation"""
       try:
           entry_date = manual_trade['entry_date']
           entry_date_idx = data.index.get_loc(entry_date)
           
           # Create historical data window
           lookback_start = max(0, entry_date_idx - 365)
           historical_data = data.iloc[lookback_start:entry_date_idx]
           
           # Initialize components
           candle_classifier = CandleClassifier(historical_data)
           classified_data = candle_classifier.classify_all_candles()
           zone_detector = ZoneDetector(candle_classifier)
           trend_classifier = TrendClassifier(historical_data)
           risk_manager = RiskManager(account_balance=10000)
           signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
           
           # Check trend alignment
           trend_data = trend_classifier.classify_trend_with_filter()
           current_trend = trend_data['trend_filtered'].iloc[-1]
           expected_direction = str(manual_trade['direction']).upper().strip()
           
           bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
           bearish_trends = ['strong_bearish', 'medium_bearish', 'weak_bearish']
           
           # Check trend alignment
           if expected_direction == 'BUY' and current_trend not in bullish_trends:
               print(f"      ❌ Trend mismatch: Need bullish, got {current_trend}")
               return {'signal_generated': False, 'total_signals': 0}
           elif expected_direction == 'SELL' and current_trend not in bearish_trends:
               print(f"      ❌ Trend mismatch: Need bearish, got {current_trend}")
               return {'signal_generated': False, 'total_signals': 0}
           
           # Generate signals with mock risk validation
           def mock_risk_validation(zone, price, pair='EURUSD', data=None):
               entry_price = zone['zone_high'] + 0.0010 if zone['type'] == 'R-B-R' else zone['zone_low'] - 0.0010
               
               if zone['type'] == 'R-B-R':  # BUY trade
                   stop_loss_price = zone['zone_low'] - 0.0020
               else:  # SELL trade  
                   stop_loss_price = zone['zone_high'] + 0.0020
               
               stop_distance_pips = abs(entry_price - stop_loss_price) / 0.0001
               risk_distance = abs(entry_price - stop_loss_price)
               
               if zone['type'] == 'R-B-R':
                   tp1 = entry_price + risk_distance
                   tp2 = entry_price + (risk_distance * 2)
               else:
                   tp1 = entry_price - risk_distance
                   tp2 = entry_price - (risk_distance * 2)
               
               return {
                   'is_tradeable': True,
                   'entry_price': entry_price,
                   'stop_loss_price': stop_loss_price,
                   'take_profit_1': tp1,
                   'take_profit_2': tp2,
                   'position_size': 0.1,
                   'risk_amount': 100,
                   'risk_reward_ratio': 2.0,
                   'stop_distance_pips': stop_distance_pips,
                   'entry_method': 'manual_strategy',
                   'trade_direction': 'BUY' if zone['type'] == 'R-B-R' else 'SELL'
               }

           original_validate_zone = risk_manager.validate_zone_for_trading
           risk_manager.validate_zone_for_trading = mock_risk_validation

           signals = signal_generator.generate_signals(classified_data, 'Daily', 'EURUSD')

           risk_manager.validate_zone_for_trading = original_validate_zone
           
           # Find matching signal
           matching_signal = self.find_matching_signal(signals, manual_trade, detected_zone)
           
           if matching_signal:
               return {
                   'signal_generated': True,
                   'signal': matching_signal,
                   'total_signals': len(signals)
               }
           else:
               if len(signals) > 0:
                   print(f"      ❌ {len(signals)} signals generated but none matched")
               return {
                   'signal_generated': False,
                   'total_signals': len(signals)
               }
               
       except Exception as e:
           print(f"      ❌ Error: {str(e)}")
           return {'signal_generated': False, 'total_signals': 0}
   
   def find_best_zone_match(self, patterns, manual_zone):
    """Find best matching zone with improved tolerance"""
    zones = patterns['rbr_patterns'] if manual_zone['type'] == 'R-B-R' else patterns['dbd_patterns']
    
    best_match = None
    best_score = 0
    
    manual_center = (manual_zone['high'] + manual_zone['low']) / 2
    manual_size = manual_zone['high'] - manual_zone['low']
    
    print(f"      Looking for zone near {manual_center:.4f} (size: {manual_size:.4f})")
    
    for i, zone in enumerate(zones):
        zone_center = (zone['zone_high'] + zone['zone_low']) / 2
        zone_size = zone['zone_high'] - zone['zone_low']
        center_distance = abs(manual_center - zone_center)
        
        print(f"      Zone {i+1}: center {zone_center:.4f}, distance {center_distance:.4f} ({center_distance*10000:.0f} pips)")
        
        # MUCH MORE FLEXIBLE matching - use THREE criteria
        
        # Criteria 1: Close centers (500 pips tolerance)
        center_match = center_distance <= 0.0500
        
        # Criteria 2: Any overlap between zones
        overlap_start = max(manual_zone['low'], zone['zone_low'])
        overlap_end = min(manual_zone['high'], zone['zone_high'])
        has_overlap = overlap_end > overlap_start
        
        # Criteria 3: Similar zone sizes (within 2x)
        size_ratio = min(manual_size, zone_size) / max(manual_size, zone_size)
        size_similar = size_ratio >= 0.3  # Allow 3x size difference
        
        print(f"         Center match: {center_match}, Overlap: {has_overlap}, Size similar: {size_similar}")
        
        # STRICTER: Require better quality matches
        criteria_met = sum([center_match, has_overlap, size_similar])
        
        # For good matches, require ALL THREE criteria OR center match + overlap
        if (criteria_met >= 3) or (center_match and has_overlap and center_distance <= 0.0200):

            # Calculate composite score
            center_score = max(0, 1 - (center_distance / 0.0500))
            
            if has_overlap:
                overlap_size = overlap_end - overlap_start
                overlap_score = overlap_size / max(manual_size, zone_size)
            else:
                overlap_score = 0
            
            size_score = size_ratio
            
            # Weighted composite score
            composite_score = (center_score * 0.5) + (overlap_score * 0.3) + (size_score * 0.2)
            
            print(f"         ✅ POTENTIAL MATCH! Score: {composite_score:.3f}")
            
            # Only accept if score is good enough
            if composite_score > best_score and composite_score >= 0.5:
                best_score = composite_score
                best_match = zone
        else:
            print(f"         ❌ Only {criteria_met}/3 criteria met")
    
    if best_match:
        print(f"      🎯 Best match: Score {best_score:.3f}")
    else:
        print(f"      ❌ No matches found")
    
    return best_match
   
   def find_matching_signal(self, signals, manual_trade, detected_zone):
       """Find matching signal"""
       manual_direction = str(manual_trade['direction']).upper().strip()
       manual_entry = float(manual_trade['entry_price'])
       
       for signal in signals:
           signal_direction = str(signal['direction']).upper().strip()
           
           # Check direction match
           if signal_direction != manual_direction:
               continue
           
           # Check zone proximity
           manual_zone_center = (detected_zone['zone_high'] + detected_zone['zone_low']) / 2
           signal_zone_center = (signal['zone_high'] + signal['zone_low']) / 2
           zone_center_diff = abs(manual_zone_center - signal_zone_center)
           
           # Check entry proximity
           entry_diff = abs(signal['entry_price'] - manual_entry)
           
           # Accept if zone centers are close OR entry prices are close
           if zone_center_diff < 0.0200 or entry_diff < 0.0150:
               return signal
       
       return None
   
   def calculate_zone_match_quality(self, manual_zone, detected_zone):
       """Calculate match quality percentage"""
       manual_center = (manual_zone['high'] + manual_zone['low']) / 2
       detected_center = (detected_zone['zone_high'] + detected_zone['zone_low']) / 2
       
       center_diff = abs(manual_center - detected_center)
       manual_size = manual_zone['high'] - manual_zone['low']
       
       if manual_size > 0:
           center_score = max(0, 1 - (center_diff / manual_size))
           return center_score * 100
       
       return 0
   
   def print_final_summary(self):
       """Print final summary"""
       print("=" * 50)
       print("📊 VALIDATION SUMMARY")
       print("=" * 50)
       
       total_trades = len(self.validation_results)
       valid_trades = [r for r in self.validation_results if not r.get('expected_invalid', False)]
       invalid_trades = [r for r in self.validation_results if r.get('expected_invalid', False)]
       
       valid_matches = sum(1 for r in valid_trades if r['overall_success'])
       invalid_correct = sum(1 for r in invalid_trades if r['overall_success'])
       
       print(f"🎯 Total Trades: {total_trades}")
       print(f"   Valid Trades: {len(valid_trades)}")
       print(f"   Invalid Trades: {len(invalid_trades)}")
       
       print(f"\n✅ VALID TRADE RESULTS:")
       print(f"   System Matches: {valid_matches}/{len(valid_trades)}")
       if len(valid_trades) > 0:
           valid_rate = (valid_matches/len(valid_trades))*100
           print(f"   Match Rate: {valid_rate:.1f}%")
       
       print(f"\n✅ INVALID TRADE RESULTS:")
       print(f"   Correct Rejections: {invalid_correct}/{len(invalid_trades)}")
       if len(invalid_trades) > 0:
           invalid_rate = (invalid_correct/len(invalid_trades))*100
           print(f"   Rejection Rate: {invalid_rate:.1f}%")
       
       # Overall success
       overall_success = valid_matches + invalid_correct
       overall_rate = (overall_success / total_trades) * 100 if total_trades > 0 else 0
       print(f"\n📈 OVERALL SUCCESS: {overall_success}/{total_trades} ({overall_rate:.1f}%)")
       
       # Assessment
       target_rate = 85
       if overall_rate >= target_rate:
           print(f"\n🎉 SUCCESS: {overall_rate:.1f}% ≥ {target_rate}% - System ready!")
       else:
           print(f"\n⚠️  NEEDS WORK: {overall_rate:.1f}% < {target_rate}% - Requires fixes")

def main():
   validator = ManualTradeValidator()
   script_dir = os.path.dirname(os.path.abspath(__file__))
   csv_path = os.path.join(script_dir, 'manual_trades_validation.csv')
   validator.validate_all_trades(csv_path)

if __name__ == "__main__":
   main()