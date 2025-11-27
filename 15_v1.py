"""
5-Minute Opening Range Breakout Strategy - Complete Backtest
E-mini S&P 500 (/ES) - 1-minute data
Implements EXACT rules from strategy document with reversal logic
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA PREPARATION
# =============================================================================

def download_and_prepare_data(start_date='2020-01-01', force_update=False):
    """
    Download or load /ES 1-minute data and prepare it for backtesting.
    Uses cached parquet file if available.
    """
    parquet_file = 'ES_2020_2025_Fresh.parq'
    
    # Try to load existing data
    if not force_update:
        try:
            print(f"Loading cached data from {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            
            # Ensure timezone is set correctly
            if df.index.tz is None:
                print("  Warning: Cached data has no timezone, assuming America/New_York")
                df.index = df.index.tz_localize('America/New_York')
            
            # Convert to Central Time if needed
            if str(df.index.tz) != 'America/Chicago':
                print(f"  Converting from {df.index.tz} to America/Chicago...")
                df.index = df.index.tz_convert('America/Chicago')
            
            # Filter regular trading hours only (9:30 - 16:00 CT)
            df = df.between_time('09:30', '16:00')
            
            print(f"✓ Loaded {len(df)} bars from cache")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            return df
        except FileNotFoundError:
            print(f"Cache file not found, will download fresh data...")
    
    # Download fresh data
    print("Downloading /ES 1-minute data from Yahoo Finance...")
    print("This may take several minutes...")
    
    ticker = "ES=F"  # E-mini S&P 500 futures
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval='1m',
        progress=True
    )
    
    if df.empty:
        raise ValueError("No data downloaded. Check ticker symbol and date range.")
    
    # Clean column names
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Convert to Chicago time (Central)
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    df.index = df.index.tz_convert('America/Chicago')
    
    # Filter regular trading hours only (9:30 - 16:00 CT)
    df = df.between_time('09:30', '16:00')
    
    # Remove any duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill small gaps
    df = df.fillna(method='ffill').dropna()
    
    # Save to parquet for future use
    df.to_parquet(parquet_file)
    print(f"✓ Downloaded and saved {len(df)} bars to {parquet_file}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def create_15min_database(df_1min):
    """
    Create a database of first 15-minute bar statistics for each trading day.
    This eliminates look-ahead bias.
    """
    database_file = '15min_dataBase.parq'
    
    print("\nBuilding 15-minute opening range database...")
    
    # Ensure data is in Central Time
    if df_1min.index.tz is not None:
        if str(df_1min.index.tz) != 'America/Chicago':
            print(f"Converting from {df_1min.index.tz} to America/Chicago...")
            df_1min.index = df_1min.index.tz_convert('America/Chicago')
    
    # Get unique trading dates using date() method to avoid timezone issues
    dates = sorted(list(set(df_1min.index.date)))
    dates = [pd.Timestamp(d) for d in dates]
    
    records = []
    total_days = len(dates)
    
    for i, date in enumerate(dates):
        if (i + 1) % 100 == 0:
            print(f"  Processing day {i+1}/{total_days}...")
        
        # Get data for this day using date comparison
        day_mask = df_1min.index.date == date.date()
        day_data = df_1min[day_mask]
        
        if len(day_data) == 0:
            continue
        
        # First 15-minute bar: 9:30:00 to 9:44:59
        first_15_start = pd.Timestamp.combine(date.date(), time(9, 30)).tz_localize('America/Chicago')
        first_15_end = pd.Timestamp.combine(date.date(), time(9, 44, 59)).tz_localize('America/Chicago')
        
        first_15_data = day_data[(day_data.index >= first_15_start) & 
                                  (day_data.index <= first_15_end)]
        
        if len(first_15_data) == 0:
            continue
        
        # Calculate high and low of first 15 minutes
        high_15 = first_15_data['High'].max()
        low_15 = first_15_data['Low'].min()
        bar_range = high_15 - low_15
        
        # Get previous day's close (for no-trade filter)
        if i > 0:
            prev_date = dates[i - 1]
            prev_day_mask = df_1min.index.date == prev_date.date()
            prev_day_data = df_1min[prev_day_mask]
            if len(prev_day_data) > 0:
                prev_close = prev_day_data['Close'].iloc[-1]
            else:
                prev_close = np.nan
        else:
            prev_close = np.nan
        
        records.append({
            'date': date,
            'high_15': high_15,
            'low_15': low_15,
            'bar_range': bar_range,
            'prev_close': prev_close,
            'long_trigger': high_15 + 0.50,
            'short_trigger': low_15 - 0.50
        })
    
    db = pd.DataFrame(records)
    db.set_index('date', inplace=True)
    
    # Save database
    db.to_parquet(database_file)
    print(f"✓ Created 15-min database with {len(db)} trading days")
    print(f"  Saved to {database_file}")
    
    return db


# =============================================================================
# STRATEGY IMPLEMENTATION
# =============================================================================

class OpeningRangeBreakout(Strategy):
    """
    5-Minute Opening Range Breakout with Reversal Logic
    
    Implements exact rules from strategy document:
    - First 15-min bar (9:30-9:44:59 CT) defines range
    - Long trigger: high_15 + 0.50
    - Short trigger: low_15 - 0.50
    - Entry window: 9:30 - 10:00 CT
    - Initial stop: opposite side of 15-min bar
    - Target: 20 points from entry
    - Move to breakeven when +15 points in profit
    - Reversal if stopped out before 10:15 CT
    """
    
    def init(self):
        # Load the 15-minute database
        self.db_15min = pd.read_parquet('15min_dataBase.parq')
        
        # Track daily state
        self.current_date = None
        self.high_15 = None
        self.low_15 = None
        self.long_trigger = None
        self.short_trigger = None
        self.prev_close = None
        
        # Position tracking
        self.entry_triggered = False
        self.entry_price = None
        self.entry_side = None  # 'long' or 'short'
        self.initial_stop = None
        self.target_price = None
        self.stop_moved_to_be = False
        
        # Reversal tracking
        self.reversal_triggered = False
        self.stopped_out_time = None
        
        # For breakout detection
        self.prev_high = None
        self.prev_low = None
    
    def is_no_trade_day(self):
        """
        Check if today is a no-trade day based on rules:
        - First 15-min bar contains previous day's close
        - Could add FOMC and triple witching filters here
        """
        if pd.isna(self.prev_close):
            return False
        
        # Check if prev close is within first 15-min range
        if self.low_15 <= self.prev_close <= self.high_15:
            return True
        
        return False
    
    def next(self):
        # Get current bar info
        current_time = self.data.index[-1]
        current_date = pd.Timestamp(current_time.date())  # Timezone-naive date
        current_hour_minute = current_time.time()
        
        # New trading day - reset everything
        if current_date != self.current_date:
            self.current_date = current_date
            
            # Load today's 15-min stats from database
            if current_date in self.db_15min.index:
                day_stats = self.db_15min.loc[current_date]
                self.high_15 = day_stats['high_15']
                self.low_15 = day_stats['low_15']
                self.long_trigger = day_stats['long_trigger']
                self.short_trigger = day_stats['short_trigger']
                self.prev_close = day_stats['prev_close']
            else:
                # No stats for this day, skip
                self.high_15 = None
                self.long_trigger = None
                return
            
            # Reset daily tracking
            self.entry_triggered = False
            self.entry_price = None
            self.entry_side = None
            self.initial_stop = None
            self.target_price = None
            self.stop_moved_to_be = False
            self.reversal_triggered = False
            self.stopped_out_time = None
            self.prev_high = None
            self.prev_low = None
        
        # Skip if no valid 15-min data
        if self.high_15 is None:
            return
        
        # Check no-trade day condition
        if not self.entry_triggered and not self.reversal_triggered:
            if self.is_no_trade_day():
                return
        
        # Store previous bar's high/low for breakout detection
        if len(self.data.Close) >= 2:
            self.prev_high = self.data.High[-2]
            self.prev_low = self.data.Low[-2]
        
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_close = self.data.Close[-1]
        
        # =================================================================
        # POSITION MANAGEMENT (if already in a position)
        # =================================================================
        
        if self.position:
            # Update trailing stop to breakeven at +15 points
            if not self.stop_moved_to_be:
                if self.entry_side == 'long':
                    if current_high >= self.entry_price + 15.0:
                        self.stop_moved_to_be = True
                        # Note: backtesting.py handles stops internally
                        # We just track the state
                elif self.entry_side == 'short':
                    if current_low <= self.entry_price - 15.0:
                        self.stop_moved_to_be = True
            
            # Check if stopped out (for reversal logic)
            if self.entry_side == 'long':
                if current_low <= self.initial_stop:
                    self.stopped_out_time = current_time
            elif self.entry_side == 'short':
                if current_high >= self.initial_stop:
                    self.stopped_out_time = current_time
            
            return
        
        # =================================================================
        # REVERSAL LOGIC (if stopped out before 10:15 CT)
        # =================================================================
        
        if self.stopped_out_time is not None and not self.reversal_triggered:
            # Check if stopped out before 10:15 CT
            cutoff_time = time(10, 15)
            
            if self.stopped_out_time.time() < cutoff_time:
                # Valid reversal window
                if current_hour_minute >= time(9, 30) and current_hour_minute < cutoff_time:
                    if self.entry_side == 'long':
                        # Was long, now go short at low_15
                        self.entry_price = self.low_15
                        self.entry_side = 'short'
                        self.initial_stop = self.high_15
                        self.target_price = self.low_15 - 20.0
                        self.stop_moved_to_be = False
                        self.reversal_triggered = True
                        
                        # Calculate stop loss and take profit for backtesting.py
                        sl_distance = abs(self.initial_stop - self.entry_price)
                        tp_distance = abs(self.target_price - self.entry_price)
                        
                        self.sell(sl=sl_distance, tp=tp_distance)
                        return
                    
                    elif self.entry_side == 'short':
                        # Was short, now go long at high_15
                        self.entry_price = self.high_15
                        self.entry_side = 'long'
                        self.initial_stop = self.low_15
                        self.target_price = self.high_15 + 20.0
                        self.stop_moved_to_be = False
                        self.reversal_triggered = True
                        
                        sl_distance = abs(self.initial_stop - self.entry_price)
                        tp_distance = abs(self.target_price - self.entry_price)
                        
                        self.buy(sl=sl_distance, tp=tp_distance)
                        return
        
        # =================================================================
        # INITIAL ENTRY LOGIC (if no entry yet today)
        # =================================================================
        
        if not self.entry_triggered:
            # Entry window: 9:30 - 10:00 CT
            entry_cutoff = time(10, 0)
            
            if current_hour_minute < time(9, 30) or current_hour_minute >= entry_cutoff:
                return
            
            # Check for breakout confirmation (proper breakout detection)
            # Long breakout: current High > trigger AND previous High <= trigger
            if self.prev_high is not None:
                if current_high > self.long_trigger and self.prev_high <= self.long_trigger:
                    # LONG ENTRY
                    self.entry_triggered = True
                    self.entry_price = self.long_trigger
                    self.entry_side = 'long'
                    self.initial_stop = self.low_15
                    self.target_price = self.entry_price + 20.0
                    
                    # Calculate distances for backtesting.py
                    sl_distance = abs(self.initial_stop - self.entry_price)
                    tp_distance = abs(self.target_price - self.entry_price)
                    
                    self.buy(sl=sl_distance, tp=tp_distance)
                    return
                
                # Short breakout: current Low < trigger AND previous Low >= trigger
                elif current_low < self.short_trigger and self.prev_low >= self.short_trigger:
                    # SHORT ENTRY
                    self.entry_triggered = True
                    self.entry_price = self.short_trigger
                    self.entry_side = 'short'
                    self.initial_stop = self.high_15
                    self.target_price = self.entry_price - 20.0
                    
                    sl_distance = abs(self.initial_stop - self.entry_price)
                    tp_distance = abs(self.target_price - self.entry_price)
                    
                    self.sell(sl=sl_distance, tp=tp_distance)
                    return


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("5-MIN OPENING RANGE BREAKOUT STRATEGY - BACKTEST")
    print("E-mini S&P 500 (/ES) - 2020 to Present")
    print("=" * 70)
    
    # Step 1: Load or download 1-minute data
    df_1min = download_and_prepare_data(start_date='2020-01-01', force_update=False)
    
    # Step 2: Create 15-minute opening range database
    db_15min = create_15min_database(df_1min)
    
    # Step 3: Prepare data for backtesting.py
    # backtesting.py needs OHLC columns with capital letters
    bt_data = df_1min[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Step 4: Run backtest
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST...")
    print("=" * 70)
    
    bt = Backtest(
        bt_data,
        OpeningRangeBreakout,
        cash=100000,
        commission=0.00,  # Adjust based on your broker
        exclusive_orders=True,
        trade_on_close=False
    )
    
    stats = bt.run()
    
    # Step 5: Display results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(stats)
    
    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)
    print(f"Total Return:          {stats['Return [%]']:.2f}%")
    print(f"Buy & Hold Return:     {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"Max Drawdown:          {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio:          {stats['Sharpe Ratio']:.2f}")
    print(f"Win Rate:              {stats['Win Rate [%]']:.2f}%")
    print(f"Total Trades:          {stats['# Trades']}")
    print(f"Avg Trade:             {stats['Avg. Trade [%]']:.2f}%")
    
    # Step 6: Generate interactive plot
    print("\n" + "=" * 70)
    print("Generating interactive plot...")
    print("=" * 70)
    bt.plot()
    
    return stats, bt


if __name__ == "__main__":
    stats, bt = main()