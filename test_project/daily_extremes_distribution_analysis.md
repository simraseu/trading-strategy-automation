# Daily Extremes Distribution Analysis

## PROJECT BACKGROUND
This project was my first ever algorithmic trading system, and was used to gain experience and acts as preparation for my big project. Was done within 1 day, and very satisfied with the result and knowledge gained. I use this experience to move on to the big project of quantifying my current profitable discretionary trading strategy.

## When Do Daily Highs/Lows Occur? (Frankfurt Open Onwards)

### SUMMARY
Total Sample: 1,254 trading days
- Bearish Days (Daily Highs): 642 days
- Bullish Days (Daily Lows): 612 days

### HOURLY BREAKDOWN (Percentages)
Time (GMT+3)    Bearish Days    Bullish Days    Combined
----------------------------------------------------------------
09:00           20.1%           17.5%           37.6%
10:00           19.3%           19.9%           39.2%
11:00            9.3%            9.3%           18.6%
12:00            5.0%            4.4%            9.4%
13:00            3.6%            2.8%            6.4%
14:00            4.8%            3.9%            8.7%
15:00            8.1%            9.2%           17.3%
16:00            5.5%            6.9%           12.4%
17:00            4.2%            5.4%            9.6%
18:00            1.2%            1.5%            2.7%
19:00            0.9%            0.7%            1.6%
20:00            0.8%            0.5%            1.3%
21:00            1.1%            1.0%            2.1%
22:00            0.3%            0.3%            0.6%

### KEY FINDINGS
HIGH PROBABILITY WINDOWS:
- 09:00-10:00: 76.8% of daily extremes occur here
- 09:00-11:00: 95.4% cumulative probability
- 15:00-17:00: Secondary cluster (39.3%)

LOW ACTIVITY PERIODS:
- 18:00-22:00: Only 8.3% of extremes
- Minimal overnight risk after 18:00

TRADING IMPLICATIONS:
1. Position sizing: Reduce size before 09:00 GMT+3
2. Stop management: Tighten stops during 09:00-11:00
3. Entry timing: Avoid new positions 18:00+ 
4. Risk-off periods: 12:00-14:00 (lowest volatility)

### TECHNICAL NOTES
- Data source: Frankfurt open onwards
- Timezone: GMT+3
- Sample period: 2017.01.10 to 2025.07.02
- Methodology: Daily high/low identification

## STRATEGY IMPLEMENTATION RESULTS

📊 FRANKFURT OPENING RANGE STRATEGY RESULTS
============================================================
Strategy: Target=0.5x range, Stop=range opposite
Data Period: 2017.01.10 to 2025.07.02
------------------------------------------------------------
Total Trades: 1652
Winning Trades: 1186 (71.8%)
Losing Trades: 466 (28.2%)
Total Pips: 3545.1
Average Win: 13.7 pips
Average Loss: -27.2 pips
Risk-Reward Ratio: 0.50
------------------------------------------------------------
EXIT REASONS:
  TARGET: 1186 trades (71.8%)
  STOP_LOSS: 466 trades (28.2%)
DIRECTION BREAKDOWN:
  LONG: 801 trades, 70.7% win rate
  SHORT: 851 trades, 72.9% win rate

### STRATEGY VALIDATION
The research directly informed this profitable automated system:
- Leveraged 09:00-11:00 high-volatility window
- Confirmed Frankfurt opening as optimal entry timing
- Applied range-break methodology with disciplined exits
- Achieved consistent profitability over 8+ years of data

### OPTIMIZATION OPPORTUNITIES
This distribution analysis provides the foundation for systematic optimization across 9 key variables:
- Target/Stop multipliers based on range size
- Entry timing refinements using hourly probabilities
- Risk management adjustments for high/low volatility periods
- Parameter optimization framework detailed in development plan

For complete optimization methodology, see: `frankfurt_strategy_development_plan.md`

### NEXT STEPS
Moving forward to quantify and systematize my profitable discretionary trading approach based on learnings from this foundational project.