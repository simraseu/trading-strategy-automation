import pandas as pd

# Load your data first
df = pd.read_csv('EURUSD.raw_M15_202106170000_202507021515.csv', 
                 sep='\t',
                 skiprows=1,
                 header=None,
                 names=['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD'])

# Convert price columns to numeric
price_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
df[price_cols] = df[price_cols].astype(float)

def get_frankfurt_open_time(date):
    """Returns Frankfurt open time (8:00 GMT+1 CET / GMT+2 CEST) in your data's timezone"""
    dt = pd.to_datetime(date)
    month = dt.month
    
    # Frankfurt DST adjustment (CET/CEST to your GMT+3 data)
    if 3 <= month <= 10:  # Summer (CEST = GMT+2)
        return '09:00:00'  # Frankfurt 8:00 CEST = 9:00 in your GMT+3 data
    else:  # Winter (CET = GMT+1)
        return '10:00:00'  # Frankfurt 8:00 CET = 10:00 in your GMT+3 data

def analyze_daily_extremes_frankfurt_onwards(df):
    """
    Analyzes daily highs/lows from Frankfurt open (8:00 Frankfurt time) onwards only
    """
    
    # Filter data from Frankfurt open onwards for each day
    filtered_data = []
    
    for date in df['DATE'].unique():
        day_data = df[df['DATE'] == date].copy()
        frankfurt_open_time = get_frankfurt_open_time(date)
        
        # Filter from Frankfurt open onwards
        day_data_filtered = day_data[day_data['TIME'] >= frankfurt_open_time].copy()
        
        if len(day_data_filtered) > 0:
            filtered_data.append(day_data_filtered)
    
    # Combine all filtered data
    df_filtered = pd.concat(filtered_data, ignore_index=True)
    
    # Create daily data from filtered timeframe
    daily_data = df_filtered.groupby('DATE').agg({
        'OPEN': 'first',  # First price from Frankfurt open onwards
        'HIGH': 'max', 
        'LOW': 'min',
        'CLOSE': 'last'   # Last price of day
    }).reset_index()
    
    # Classify days based on Frankfurt open vs close
    daily_data['day_type'] = daily_data.apply(
        lambda row: 'bullish' if row['CLOSE'] > row['OPEN'] else 'bearish', 
        axis=1
    )
    
    # Find exact timing of extremes (from Frankfurt session onwards)
    results = []
    
    for _, day_info in daily_data.iterrows():
        date = day_info['DATE']
        frankfurt_open_time = get_frankfurt_open_time(date)
        day_data = df_filtered[df_filtered['DATE'] == date].copy()
        
        if day_info['day_type'] == 'bullish':
            # Find when LOW occurred (from Frankfurt onwards)
            low_time = day_data[day_data['LOW'] == day_info['LOW']]['TIME'].iloc[0]
            extreme_price = day_info['LOW']
            extreme_type = 'LOW'
            extreme_time = low_time
            
        else:  # bearish day
            # Find when HIGH occurred (from Frankfurt onwards)  
            high_time = day_data[day_data['HIGH'] == day_info['HIGH']]['TIME'].iloc[0]
            extreme_price = day_info['HIGH']
            extreme_type = 'HIGH'
            extreme_time = high_time
            
        results.append({
            'DATE': date,
            'DAY_TYPE': day_info['day_type'],
            'EXTREME_TYPE': extreme_type,
            'EXTREME_TIME': extreme_time,
            'FRANKFURT_OPEN_TIME': frankfurt_open_time,
            'EXTREME_PRICE': extreme_price,
            'FRANKFURT_OPEN': day_info['OPEN'],
            'DAY_CLOSE': day_info['CLOSE']
        })
    
    return pd.DataFrame(results)

# Run the analysis
extreme_timing = analyze_daily_extremes_frankfurt_onwards(df)

print("DAILY EXTREME TIMING (FROM FRANKFURT OPEN 08:00 CET/CEST ONWARDS)")
print("=" * 70)
print(extreme_timing.head(10))

# Time distribution analysis
print("\nTIME DISTRIBUTION OF EXTREMES (FRANKFURT SESSION ONWARDS):")
print("=" * 60)

time_distribution = extreme_timing.groupby(['DAY_TYPE', 'EXTREME_TIME']).size().reset_index(name='COUNT')
print(time_distribution.head(20))

# Hour-based summary
extreme_timing['HOUR'] = pd.to_datetime(extreme_timing['EXTREME_TIME']).dt.hour

print("\nHOURLY DISTRIBUTION (FROM FRANKFURT OPEN ONWARDS):")
print("=" * 50)
hourly_summary = extreme_timing.groupby(['DAY_TYPE', 'HOUR']).size().reset_index(name='COUNT')
hourly_pivot = hourly_summary.pivot(index='HOUR', columns='DAY_TYPE', values='COUNT').fillna(0)
print(hourly_pivot)

# Session-based analysis (adjusted for Frankfurt onwards)
def classify_session_frankfurt_onwards(hour):
    if 9 <= hour <= 15:  # Frankfurt/London overlap
        return 'European' 
    elif 16 <= hour <= 23:
        return 'NY'
    else:
        return 'Late_NY/Early_European'

extreme_timing['SESSION'] = extreme_timing['HOUR'].apply(classify_session_frankfurt_onwards)

print("\nSESSION DISTRIBUTION (FRANKFURT ONWARDS):")
print("=" * 45)
session_summary = extreme_timing.groupby(['DAY_TYPE', 'SESSION']).size().reset_index(name='COUNT')
session_pivot = session_summary.pivot(index='SESSION', columns='DAY_TYPE', values='COUNT').fillna(0)
print(session_pivot)



#Visuals: 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for better plots
plt.style.use('default')
sns.set_palette("Set2")

# ===== VISUALIZATION 1: HOURLY BAR CHART =====
fig, ax = plt.subplots(figsize=(14, 8))

# Create side-by-side bars
x_pos = range(len(hourly_pivot.index))
width = 0.35

bars1 = ax.bar([x - width/2 for x in x_pos], hourly_pivot['bearish'], 
               width, label='Bearish Days (Daily HIGH)', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar([x + width/2 for x in x_pos], hourly_pivot['bullish'], 
               width, label='Bullish Days (Daily LOW)', color='#4ECDC4', alpha=0.8)

# Customize the chart
ax.set_xlabel('Hour (GMT+3)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Days', fontsize=12, fontweight='bold')
ax.set_title('When Do Daily Extremes Occur?\n(From Frankfurt Open Onwards)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(hourly_pivot.index)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

plt.tight_layout()
plt.savefig('frankfurt_hourly_extremes.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== VISUALIZATION 2: SESSION DISTRIBUTION PIE CHARTS =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Bullish days pie chart
bullish_session = extreme_timing[extreme_timing['DAY_TYPE'] == 'bullish']['SESSION'].value_counts()
colors1 = ['#4ECDC4', '#45B7D1', '#96CEB4']
wedges1, texts1, autotexts1 = ax1.pie(bullish_session.values, labels=bullish_session.index, 
                                      autopct='%1.1f%%', colors=colors1, startangle=90)
ax1.set_title('Bullish Days: When Daily LOWS Occur', fontsize=14, fontweight='bold')

# Bearish days pie chart  
bearish_session = extreme_timing[extreme_timing['DAY_TYPE'] == 'bearish']['SESSION'].value_counts()
colors2 = ['#FF6B6B', '#FF8E53', '#FF9F43']
wedges2, texts2, autotexts2 = ax2.pie(bearish_session.values, labels=bearish_session.index,
                                      autopct='%1.1f%%', colors=colors2, startangle=90)
ax2.set_title('Bearish Days: When Daily HIGHS Occur', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('frankfurt_session_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== VISUALIZATION 3: INTERACTIVE PLOTLY HEATMAP =====
# Prepare data for heatmap
time_pivot = time_distribution.pivot(index='EXTREME_TIME', columns='DAY_TYPE', values='COUNT').fillna(0)

fig = go.Figure(data=go.Heatmap(
    z=time_pivot.values,
    x=time_pivot.columns,
    y=time_pivot.index,
    colorscale='RdYlBu_r',
    text=time_pivot.values,
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps=False
))

fig.update_layout(
    title='Frankfurt Analysis: Detailed Time Distribution<br><sub>When daily extremes occur (15-minute intervals)</sub>',
    xaxis_title='Day Type',
    yaxis_title='Time (15-min intervals)',
    font=dict(size=12),
    height=800,
    width=600
)

fig.write_html('frankfurt_time_heatmap.html')
fig.show()

# ===== VISUALIZATION 4: SUMMARY STATISTICS DISPLAY =====
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top extreme times for bullish days
bullish_times = extreme_timing[extreme_timing['DAY_TYPE'] == 'bullish']['HOUR'].value_counts().head(5)
ax1.bar(bullish_times.index, bullish_times.values, color='#4ECDC4', alpha=0.8)
ax1.set_title('Top 5 Hours: Daily LOWS (Bullish Days)', fontweight='bold')
ax1.set_xlabel('Hour (GMT+3)')
ax1.set_ylabel('Count')

# Top extreme times for bearish days
bearish_times = extreme_timing[extreme_timing['DAY_TYPE'] == 'bearish']['HOUR'].value_counts().head(5)
ax2.bar(bearish_times.index, bearish_times.values, color='#FF6B6B', alpha=0.8)
ax2.set_title('Top 5 Hours: Daily HIGHS (Bearish Days)', fontweight='bold')
ax2.set_xlabel('Hour (GMT+3)')
ax2.set_ylabel('Count')

# Overall day type distribution
day_type_counts = extreme_timing['DAY_TYPE'].value_counts()
ax3.pie(day_type_counts.values, labels=day_type_counts.index, autopct='%1.1f%%',
        colors=['#4ECDC4', '#FF6B6B'], startangle=90)
ax3.set_title('Overall Day Type Distribution', fontweight='bold')

# Monthly pattern (if enough data)
extreme_timing['MONTH'] = pd.to_datetime(extreme_timing['DATE']).dt.month
monthly_pattern = extreme_timing.groupby(['MONTH', 'DAY_TYPE']).size().unstack(fill_value=0)
if len(monthly_pattern) > 1:
    monthly_pattern.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
    ax4.set_title('Monthly Pattern of Day Types', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
else:
    ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Monthly Analysis', fontweight='bold')

plt.tight_layout()
plt.savefig('frankfurt_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("🎯 ALL VISUALIZATIONS CREATED!")
print("📁 Files saved:")
print("   - frankfurt_hourly_extremes.png")
print("   - frankfurt_session_distribution.png") 
print("   - frankfurt_time_heatmap.html (interactive)")
print("   - frankfurt_complete_analysis.png")