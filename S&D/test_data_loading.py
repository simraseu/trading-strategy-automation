"""
Quick test to verify data loading works with your files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader

def test_data_loading():
    print("🧪 Testing Data Loading")
    print("=" * 40)
    
    loader = DataLoader()
    
    # List available files
    files = loader.list_available_files()
    print(f"📁 Available files: {files}")
    
    # Try to load EURUSD daily data
    try:
        data = loader.load_pair_data('EURUSD', 'Daily')
        print(f"\n📊 EURUSD Daily Data:")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        print(f"\n   First 5 rows:")
        print(data.head())
        
        # Validate data quality
        is_valid = loader.validate_data_quality(data)
        
        if is_valid:
            print("✅ Data loading successful!")
            return True
        else:
            print("❌ Data validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()