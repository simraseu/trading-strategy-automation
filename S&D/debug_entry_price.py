"""
ENTRY PRICE LOGIC DEBUG & FIX
Resolves the entry price calculation mismatch identified in validation
"""

import pandas as pd
import numpy as np
from modules.risk_manager import RiskManager
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

class EntryPriceDebugger:
    """Debug and fix entry price calculation logic"""
    
    def __init__(self):
        self.test_zone = {
            'type': 'R-B-R',
            'zone_high': 1.1050,
            'zone_low': 1.1020,
            'zone_range': 0.0030,
            'leg_out': {'ratio_to_base': 2.5}
        }
    
    def debug_entry_calculation(self):
        """Debug the entry price calculation step by step"""
        
        print("🔍 DEBUGGING ENTRY PRICE CALCULATION")
        print("=" * 50)
        
        # Manual calculation (expected)
        zone_high = self.test_zone['zone_high']
        zone_low = self.test_zone['zone_low'] 
        zone_range = zone_high - zone_low
        
        print(f"📊 Zone Details:")
        print(f"   Zone High: {zone_high:.5f}")
        print(f"   Zone Low: {zone_low:.5f}")
        print(f"   Zone Range: {zone_range:.5f}")
        print(f"   Zone Type: {self.test_zone['type']}")
        
        # Expected calculation (R-B-R = BUY = entry above zone_low)
        if self.test_zone['type'] == 'R-B-R':  # Bullish zone
            # Entry should be 5% above zone_low (into the zone)
            expected_entry_manual = zone_low + (zone_range * 0.05)
            print(f"\n🎯 Manual Calculation (R-B-R):")
            print(f"   Formula: zone_low + (zone_range × 0.05)")
            print(f"   Calculation: {zone_low:.5f} + ({zone_range:.5f} × 0.05)")
            print(f"   Expected Entry: {expected_entry_manual:.5f}")
        
        # Risk Manager calculation (current)
        risk_manager = RiskManager(account_balance=10000)
        calculated_entry = risk_manager.calculate_entry_price_manual(self.test_zone)
        
        print(f"\n🔧 Risk Manager Calculation:")
        print(f"   Calculated Entry: {calculated_entry:.5f}")
        
        # Check the difference
        difference = abs(calculated_entry - expected_entry_manual)
        print(f"\n🔍 Comparison:")
        print(f"   Difference: {difference:.5f}")
        print(f"   Match: {'✅ YES' if difference < 0.00001 else '❌ NO'}")
        
        if difference >= 0.00001:
            print(f"\n🔧 ISSUE IDENTIFIED:")
            self.analyze_risk_manager_logic()
        
        return expected_entry_manual, calculated_entry, difference < 0.00001
    
    def analyze_risk_manager_logic(self):
        """Analyze the risk manager's entry price logic"""
        
        print(f"🔍 Analyzing Risk Manager Entry Logic...")
        
        # Let's check what the risk manager is actually doing
        zone = self.test_zone
        zone_size = zone['zone_high'] - zone['zone_low']
        front_run_distance = zone_size * 0.05
        
        print(f"📊 Expected Logic Breakdown:")
        print(f"   Zone Size: {zone_size:.5f}")
        print(f"   Front Run Distance (5%): {front_run_distance:.5f}")
        
        if zone['type'] == 'R-B-R':  # Bullish demand zone
            expected_entry = zone['zone_high'] + front_run_distance  # This might be wrong!
            print(f"   Current Logic: zone_high + front_run = {zone['zone_high']:.5f} + {front_run_distance:.5f} = {expected_entry:.5f}")
            
            correct_entry = zone['zone_low'] + front_run_distance  # This should be correct!
            print(f"   Correct Logic: zone_low + front_run = {zone['zone_low']:.5f} + {front_run_distance:.5f} = {correct_entry:.5f}")
            
            print(f"\n💡 ISSUE FOUND:")
            print(f"   Risk Manager using: zone_high + front_run")
            print(f"   Should be using: zone_low + front_run")
            print(f"   For R-B-R (bullish), entry should be 5% INTO the zone from the LOW")
    
    def create_fixed_entry_method(self):
        """Create corrected entry price calculation method"""
        
        def calculate_entry_price_fixed(zone: dict) -> float:
            """
            FIXED: Your 5% front-running entry method using CORRECT zone logic
            """
            zone_size = zone['zone_high'] - zone['zone_low']
            front_run_distance = zone_size * 0.05  # 5% of zone size
            
            if zone['type'] == 'R-B-R':  # Bullish demand zone
                # CORRECTED: Entry 5% above the zone LOW (into the zone)
                entry_price = zone['zone_low'] + front_run_distance
                
            elif zone['type'] == 'D-B-D':  # Bearish supply zone
                # CORRECTED: Entry 5% below the zone HIGH (into the zone)
                entry_price = zone['zone_high'] - front_run_distance
            else:
                # For reversal patterns, same logic
                if zone['type'] == 'D-B-R':  # Bullish reversal
                    entry_price = zone['zone_low'] + front_run_distance
                else:  # R-B-D - Bearish reversal
                    entry_price = zone['zone_high'] - front_run_distance
            
            return entry_price
        
        return calculate_entry_price_fixed
    
    def test_fixed_method(self):
        """Test the fixed entry calculation method"""
        
        print(f"\n🔧 TESTING FIXED ENTRY METHOD")
        print("=" * 40)
        
        fixed_method = self.create_fixed_entry_method()
        
        # Test with our sample zone
        fixed_entry = fixed_method(self.test_zone)
        expected_entry = self.test_zone['zone_low'] + ((self.test_zone['zone_high'] - self.test_zone['zone_low']) * 0.05)
        
        print(f"🎯 Fixed Method Test:")
        print(f"   Fixed Entry: {fixed_entry:.5f}")
        print(f"   Expected Entry: {expected_entry:.5f}")
        print(f"   Match: {'✅ PERFECT' if abs(fixed_entry - expected_entry) < 0.00001 else '❌ STILL WRONG'}")
        
        # Test different zone types
        test_zones = [
            {'type': 'R-B-R', 'zone_high': 1.2050, 'zone_low': 1.2020},  # Bullish
            {'type': 'D-B-D', 'zone_high': 1.2050, 'zone_low': 1.2020},  # Bearish
            {'type': 'D-B-R', 'zone_high': 1.2050, 'zone_low': 1.2020},  # Bullish reversal
            {'type': 'R-B-D', 'zone_high': 1.2050, 'zone_low': 1.2020},  # Bearish reversal
        ]
        
        print(f"\n🧪 Testing All Zone Types:")
        for zone in test_zones:
            entry = fixed_method(zone)
            zone_range = zone['zone_high'] - zone['zone_low']
            
            if zone['type'] in ['R-B-R', 'D-B-R']:  # Bullish zones
                expected = zone['zone_low'] + (zone_range * 0.05)
                direction = "BUY (5% above LOW)"
            else:  # Bearish zones
                expected = zone['zone_high'] - (zone_range * 0.05)
                direction = "SELL (5% below HIGH)"
            
            match = abs(entry - expected) < 0.00001
            print(f"   {zone['type']}: {entry:.5f} {direction} {'✅' if match else '❌'}")
        
        return fixed_method


def fix_risk_manager_entry_logic():
    """
    Generate the corrected risk_manager.py entry logic
    """
    
    print(f"\n🛠️  GENERATING CORRECTED RISK MANAGER CODE")
    print("=" * 50)
    
    corrected_method = '''
def calculate_entry_price_manual(self, zone: Dict) -> float:
    """
    CORRECTED: Your 5% front-running entry method using CORRECT zone logic
    
    Args:
        zone: Zone dictionary
        
    Returns:
        Entry price with 5% front-running from proper zone boundary
    """
    zone_size = zone['zone_high'] - zone['zone_low']
    front_run_distance = zone_size * 0.05  # 5% of zone size
    
    if zone['type'] in ['R-B-R', 'D-B-R']:  # Bullish zones (BUY)
        # CORRECTED: Entry 5% above the zone LOW (into the zone from bottom)
        entry_price = zone['zone_low'] + front_run_distance
        
    elif zone['type'] in ['D-B-D', 'R-B-D']:  # Bearish zones (SELL)
        # CORRECTED: Entry 5% below the zone HIGH (into the zone from top)
        entry_price = zone['zone_high'] - front_run_distance
    else:
        # Fallback for any other zone types
        entry_price = zone['zone_low'] + front_run_distance
    
    return entry_price
    '''
    
    print("📝 CORRECTED METHOD:")
    print(corrected_method)
    
    print(f"\n📋 REQUIRED ACTION:")
    print(f"1. Open: modules/risk_manager.py")
    print(f"2. Find: calculate_entry_price_manual method")
    print(f"3. Replace with the corrected code above")
    print(f"4. Re-run validation to confirm fix")
    
    # Save to file
    with open('risk_manager_entry_fix.py', 'w') as f:
        f.write("# CORRECTED ENTRY PRICE METHOD FOR RISK MANAGER\n")
        f.write("# Replace the calculate_entry_price_manual method in risk_manager.py with this:\n\n")
        f.write(corrected_method)
    
    print(f"\n💾 Corrected method saved to: risk_manager_entry_fix.py")


def main_debug_entry():
    """Main debugging function"""
    
    print("🔧 ENTRY PRICE LOGIC DEBUGGER")
    print("🎯 Fixing validation mismatch")
    print("=" * 50)
    
    debugger = EntryPriceDebugger()
    
    # Debug the issue
    expected, calculated, matches = debugger.debug_entry_calculation()
    
    if not matches:
        # Test the fixed method
        debugger.test_fixed_method()
        
        # Generate correction
        fix_risk_manager_entry_logic()
        
        print(f"\n✅ DEBUG COMPLETE!")
        print(f"📋 NEXT STEPS:")
        print(f"   1. Apply the corrected method to risk_manager.py")
        print(f"   2. Re-run: python debug_validation_framework.py")
        print(f"   3. Choose option 1 (Quick Validation)")
        print(f"   4. Confirm all phases show ✅")
    else:
        print(f"✅ Entry logic is already correct!")


if __name__ == "__main__":
    main_debug_entry()