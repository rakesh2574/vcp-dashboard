import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import warnings
import time
import argparse
import json
import os

warnings.filterwarnings('ignore')

# --- Core VCP Logic (Copied from your original script) ---

# TradingView data fetcher
try:
    from tvDatafeed import TvDatafeed, Interval

    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False


@dataclass
class ResistanceZone:
    """Represents a resistance zone"""
    resistance_price: float
    zone_top: float
    zone_bottom: float
    resistance_timestamp: pd.Timestamp
    resistance_bar_index: int


@dataclass
class TouchPoint:
    """Represents a touch of resistance zone"""
    touch_timestamp: pd.Timestamp
    touch_price: float
    touch_bar_index: int


@dataclass
class SwingLow:
    """Represents a swing low between touches"""
    swinglow_timestamp: pd.Timestamp
    swinglow_price: float
    swinglow_bar_index: int


@dataclass
class VCPPattern:
    """Complete VCP Pattern with all components"""
    symbol: str
    exchange: str
    resistance: ResistanceZone
    swinglow1: Optional[SwingLow]
    touch1: Optional[TouchPoint]
    swinglow2: Optional[SwingLow]
    touch2: Optional[TouchPoint]
    swinglow3: Optional[SwingLow]
    touch3: Optional[TouchPoint]
    breakout_timestamp: Optional[pd.Timestamp]
    breakout_price: Optional[float]
    breakout_bar_index: Optional[int]
    pattern_complete: bool
    pattern_valid: bool
    validation_message: str
    current_price: float
    trade_status: str
    target_price: Optional[float]
    stoploss_price: Optional[float]
    target_hit_timestamp: Optional[pd.Timestamp]
    stoploss_hit_timestamp: Optional[pd.Timestamp]
    max_price_after_breakout: Optional[float]
    min_price_after_breakout: Optional[float]
    trade_outcome_pct: Optional[float]


class TradingViewDataFetcher:
    """Fetches OHLC data from TradingView"""

    def __init__(self, headless_mode=False):
        if not TV_AVAILABLE:
            raise ImportError("TradingView DataFeed not available")
        # In headless mode, you might want to suppress browser pop-ups if tvdatafeed uses selenium with a browser
        self.tv = TvDatafeed()
        self._setup_intervals()

    def _setup_intervals(self):
        self.interval_map = {
            '1m': Interval.in_1_minute, '3m': Interval.in_3_minute, '5m': Interval.in_5_minute,
            '15m': Interval.in_15_minute, '30m': Interval.in_30_minute, '1H': Interval.in_1_hour,
            '2H': getattr(Interval, 'in_2_hour', Interval.in_1_hour), '4H': Interval.in_4_hour,
            '1D': Interval.in_daily, '1W': Interval.in_weekly, '1M': Interval.in_monthly
        }

    def get_data(self, symbol: str, exchange: str = 'BINANCE', timeframe: str = '1H',
                 n_bars: int = 500) -> pd.DataFrame:
        interval = self.interval_map.get(timeframe, self.interval_map['1H'])
        try:
            data = self.tv.get_hist(symbol, exchange, interval, n_bars=n_bars)
            if data is None or data.empty:
                raise ValueError(f"No data for {symbol} on {exchange}")
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna()
            if len(data) < 50:
                raise ValueError(f"Insufficient data: only {len(data)} bars")
            return data
        except Exception as e:
            raise Exception(f"Error fetching {symbol}: {e}")


class VCPDetector:
    """Detects VCP patterns in price data with trade tracking"""

    def __init__(self, swing_length: int = 10, zone_percent: float = 1.0, max_resistance_levels: int = 50,
                 target_percent: float = 10.0, stoploss_percent: float = 5.0):
        self.swing_length = swing_length
        self.zone_percent = zone_percent
        self.max_resistance_levels = max_resistance_levels
        self.target_percent = target_percent
        self.stoploss_percent = stoploss_percent

    def find_swing_highs(self, df: pd.DataFrame) -> List[Tuple[int, float, pd.Timestamp]]:
        swing_highs = []
        length = min(self.swing_length, len(df) // 4)
        for i in range(length, len(df) - length):
            high_price = df['high'].iloc[i]
            is_swing_high = all(df['high'].iloc[j] <= high_price for j in range(i - length, i + length + 1) if j != i)
            if is_swing_high:
                swing_highs.append((i, high_price, df.index[i]))
        return swing_highs

    def create_resistance_zones(self, swing_highs: List[Tuple[int, float, pd.Timestamp]]) -> List[ResistanceZone]:
        zones = []
        for bar_index, high_price, timestamp in swing_highs:
            zone = ResistanceZone(
                resistance_price=high_price, zone_top=high_price,
                zone_bottom=high_price * (1 - self.zone_percent / 100),
                resistance_timestamp=timestamp, resistance_bar_index=bar_index
            )
            zones.append(zone)
        return zones[-self.max_resistance_levels:]

    def find_touches_and_breakout(self, df: pd.DataFrame, zone: ResistanceZone, start_index: int) -> Tuple[
        List[TouchPoint], Optional[int]]:
        touches, left_zone = [], False
        for i in range(start_index + 1, len(df)):
            high, close = df['high'].iloc[i], df['close'].iloc[i]
            if high < zone.zone_bottom: left_zone = True
            if left_zone and zone.zone_bottom <= high <= zone.zone_top and close <= zone.zone_top:
                touches.append(TouchPoint(df.index[i], high, i))
                left_zone = False
            if close > zone.zone_top: return touches, i
            if len(touches) >= 3:
                for j in range(i + 1, min(i + 50, len(df))):
                    if df['close'].iloc[j] > zone.zone_top: return touches, j
                return touches, None
        return touches, None

    def find_minimum_between(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[SwingLow]:
        if start_idx >= end_idx or start_idx < 0 or end_idx > len(df): return None
        segment = df.iloc[start_idx + 1:end_idx]
        if segment.empty: return None
        min_idx, min_price = segment['low'].idxmin(), segment['low'].min()
        bar_index = df.index.get_loc(min_idx)
        return SwingLow(min_idx, min_price, bar_index)

    def track_trade_outcome(self, df: pd.DataFrame, pattern: VCPPattern) -> VCPPattern:
        if not pattern.breakout_bar_index: return pattern
        pattern.target_price = pattern.resistance.resistance_price * (1 + self.target_percent / 100)
        pattern.stoploss_price = pattern.resistance.resistance_price * (1 - self.stoploss_percent / 100)
        post_breakout_data = df.iloc[pattern.breakout_bar_index + 1:]
        if post_breakout_data.empty:
            pattern.trade_status = "Ongoing"
            return pattern
        pattern.max_price_after_breakout, pattern.min_price_after_breakout = post_breakout_data['high'].max(), \
        post_breakout_data['low'].min()
        target_hit, sl_hit = False, False
        for idx, row in post_breakout_data.iterrows():
            if row['low'] <= pattern.stoploss_price and not target_hit:
                pattern.stoploss_hit_timestamp, pattern.trade_outcome_pct, pattern.trade_status, sl_hit = idx, -self.stoploss_percent, "SL Hit", True
                break
            if row['high'] >= pattern.target_price and not sl_hit:
                pattern.target_hit_timestamp, pattern.trade_outcome_pct, pattern.trade_status, target_hit = idx, self.target_percent, "Target Hit", True
                break
        if not target_hit and not sl_hit:
            current_price = post_breakout_data['close'].iloc[-1]
            pattern.trade_outcome_pct = ((
                                                     current_price - pattern.resistance.resistance_price) / pattern.resistance.resistance_price * 100)
            pattern.trade_status = "Ongoing"
        return pattern

    def validate_vcp_pattern(self, pattern: VCPPattern) -> Tuple[bool, str]:
        if not pattern.touch3: return False, "Incomplete: <3 touches"
        if not all([pattern.swinglow1, pattern.swinglow2, pattern.swinglow3]): return False, "Incomplete: Missing lows"
        low1, low2, low3 = pattern.swinglow1.swinglow_price, pattern.swinglow2.swinglow_price, pattern.swinglow3.swinglow_price
        if not (low1 < low2 < low3): return False, f"No contraction: {low1:.2f}→{low2:.2f}→{low3:.2f}"
        res_price = pattern.resistance.resistance_price
        cont1, cont2, cont3 = (res_price - low1) / res_price * 100, (res_price - low2) / res_price * 100, (
                    res_price - low3) / res_price * 100
        msg = f"{cont1:.1f}%→{cont2:.1f}%→{cont3:.1f}%"
        if not pattern.breakout_timestamp: return True, f"Setup Ready: {msg}"
        return True, f"Valid VCP: {msg}"

    def detect_vcp_patterns(self, df: pd.DataFrame, symbol: str, exchange: str, current_price: float) -> List[
        VCPPattern]:
        patterns = []
        swing_highs = self.find_swing_highs(df)
        resistance_zones = self.create_resistance_zones(swing_highs)
        for zone in resistance_zones:
            touches, breakout_idx = self.find_touches_and_breakout(df, zone, zone.resistance_bar_index)
            pattern = VCPPattern(
                symbol=symbol, exchange=exchange, resistance=zone, swinglow1=None, touch1=None,
                swinglow2=None, touch2=None, swinglow3=None, touch3=None, breakout_timestamp=None,
                breakout_price=None, breakout_bar_index=None, pattern_complete=False, pattern_valid=False,
                validation_message="", current_price=current_price, trade_status="Not Valid", target_price=None,
                stoploss_price=None, target_hit_timestamp=None, stoploss_hit_timestamp=None,
                max_price_after_breakout=None, min_price_after_breakout=None, trade_outcome_pct=None
            )
            if touches: pattern.touch1, pattern.swinglow1 = touches[0], self.find_minimum_between(df,
                                                                                                  zone.resistance_bar_index,
                                                                                                  touches[
                                                                                                      0].touch_bar_index)
            if len(touches) >= 2: pattern.touch2, pattern.swinglow2 = touches[1], self.find_minimum_between(df, touches[
                0].touch_bar_index, touches[1].touch_bar_index)
            if len(touches) >= 3: pattern.touch3, pattern.swinglow3, pattern.pattern_complete = touches[
                2], self.find_minimum_between(df, touches[1].touch_bar_index, touches[2].touch_bar_index), True
            if breakout_idx is not None:
                pattern.breakout_timestamp, pattern.breakout_price, pattern.breakout_bar_index = df.index[breakout_idx], \
                df['close'].iloc[breakout_idx], breakout_idx
            pattern.pattern_valid, pattern.validation_message = self.validate_vcp_pattern(pattern)
            if pattern.pattern_valid:
                if pattern.touch3 and not pattern.breakout_timestamp:
                    pattern.trade_status = "Setup Ready"
                elif pattern.breakout_timestamp:
                    pattern = self.track_trade_outcome(df, pattern)
            if pattern.touch1: patterns.append(pattern)
        return patterns


class VCPAnalyzer:
    """Main analyzer class for VCP patterns"""

    def __init__(self, swing_length: int = 10, zone_percent: float = 1.0, target_percent: float = 10.0,
                 stoploss_percent: float = 5.0, headless_mode=False):
        self.data_fetcher = TradingViewDataFetcher(headless_mode) if TV_AVAILABLE else None
        self.vcp_detector = VCPDetector(swing_length, zone_percent, 50, target_percent, stoploss_percent)
        self.headless_mode = headless_mode

    def analyze_multiple_symbols(self, symbols: List[str], exchange: str = 'BINANCE', timeframe: str = '1H',
                                 n_bars: int = 500) -> Dict:
        if not self.data_fetcher: raise Exception("TradingView data fetcher not available")
        all_patterns, successful_symbols, failed_symbols = [], [], []

        progress_bar = None
        if not self.headless_mode:
            progress_bar = st.progress(0, text="Initializing analysis...")

        total_symbols = len(symbols)

        for i, symbol in enumerate(symbols):
            try:
                if self.headless_mode:
                    print(f"Analyzing {i + 1}/{total_symbols}: {symbol}...")
                else:
                    progress_text = f"Analyzing {i + 1}/{total_symbols}: {symbol}"
                    progress_bar.progress((i + 1) / total_symbols, text=progress_text)

                df = self.data_fetcher.get_data(symbol, exchange, timeframe, n_bars)
                current_price = df['close'].iloc[-1]
                patterns = self.vcp_detector.detect_vcp_patterns(df, symbol, exchange, current_price)
                all_patterns.extend(patterns)
                successful_symbols.append(symbol)
            except Exception as e:
                failed_symbols.append({'symbol': symbol, 'error': str(e)})
                if self.headless_mode:
                    print(f"  ERROR for {symbol}: {e}")

        if progress_bar:
            progress_bar.empty()

        valid_patterns = [p for p in all_patterns if p.pattern_valid]
        return {
            'all_patterns': all_patterns, 'valid_patterns': valid_patterns,
            'setup_ready_patterns': [p for p in valid_patterns if p.trade_status == "Setup Ready"],
            'ongoing_patterns': [p for p in valid_patterns if p.trade_status == "Ongoing"],
            'target_hit_patterns': [p for p in valid_patterns if p.trade_status == "Target Hit"],
            'sl_hit_patterns': [p for p in valid_patterns if p.trade_status == "SL Hit"],
            'successful_symbols': successful_symbols, 'failed_symbols': failed_symbols,
            'total_patterns': len(all_patterns), 'valid_pattern_count': len(valid_patterns),
            'exchange': exchange, 'timeframe': timeframe, 'analysis_timestamp': datetime.now()
        }


def create_consolidated_table(patterns: List[VCPPattern]) -> pd.DataFrame:
    """Create a consolidated table of all VCP patterns"""
    data = []
    for p in patterns:
        row = {
            'Symbol': p.symbol, 'Exchange': p.exchange, 'Current Price': f"{p.current_price:.4f}",
            'Pattern Status': '✅ Valid' if p.pattern_valid else '❌ Invalid',
            'Trade Status': p.trade_status, 'Validation': p.validation_message,
            'Resistance': f"{p.resistance.resistance_price:.4f}",
            'Zone Top': f"{p.resistance.zone_top:.4f}", 'Zone Bottom': f"{p.resistance.zone_bottom:.4f}",
            'Resistance Time': p.resistance.resistance_timestamp.strftime('%Y-%m-%d %H:%M'),
            'SwingLow1 Price': f"{p.swinglow1.swinglow_price:.4f}" if p.swinglow1 else '-',
            'SwingLow1 Time': p.swinglow1.swinglow_timestamp.strftime('%Y-%m-%d %H:%M') if p.swinglow1 else '-',
            'Touch1 Price': f"{p.touch1.touch_price:.4f}" if p.touch1 else '-',
            'Touch1 Time': p.touch1.touch_timestamp.strftime('%Y-%m-%d %H:%M') if p.touch1 else '-',
            'SwingLow2 Price': f"{p.swinglow2.swinglow_price:.4f}" if p.swinglow2 else '-',
            'SwingLow2 Time': p.swinglow2.swinglow_timestamp.strftime('%Y-%m-%d %H:%M') if p.swinglow2 else '-',
            'Touch2 Price': f"{p.touch2.touch_price:.4f}" if p.touch2 else '-',
            'Touch2 Time': p.touch2.touch_timestamp.strftime('%Y-%m-%d %H:%M') if p.touch2 else '-',
            'SwingLow3 Price': f"{p.swinglow3.swinglow_price:.4f}" if p.swinglow3 else '-',
            'SwingLow3 Time': p.swinglow3.swinglow_timestamp.strftime('%Y-%m-%d %H:%M') if p.swinglow3 else '-',
            'Touch3 Price': f"{p.touch3.touch_price:.4f}" if p.touch3 else '-',
            'Touch3 Time': p.touch3.touch_timestamp.strftime('%Y-%m-%d %H:%M') if p.touch3 else '-',
            'Breakout Price': f"{p.breakout_price:.4f}" if p.breakout_timestamp else '-',
            'Breakout Time': p.breakout_timestamp.strftime('%Y-%m-%d %H:%M') if p.breakout_timestamp else '-',
            'Target': f"{p.target_price:.4f}" if p.target_price else '-',
            'Stop Loss': f"{p.stoploss_price:.4f}" if p.stoploss_price else '-',
            'P&L %': f"{p.trade_outcome_pct:.2f}%" if p.trade_outcome_pct is not None else '-',
            'Target Hit Time': p.target_hit_timestamp.strftime('%Y-%m-%d %H:%M') if p.target_hit_timestamp else '-',
            'SL Hit Time': p.stoploss_hit_timestamp.strftime('%Y-%m-%d %H:%M') if p.stoploss_hit_timestamp else '-',
        }
        data.append(row)
    return pd.DataFrame(data)


# --- Configuration Management ---
CONFIG_FILE = "vcp_config.json"
DEFAULT_CONFIG = {
    "symbols_input": "TATACHEM",
    "exchange": "NSE", "timeframe": "1H", "n_bars": 500,
    "swing_length": 10, "zone_percent": 1.0, "target_percent": 10.0, "stoploss_percent": 5.0
}


def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


# --- Headless Execution Logic ---
def run_headless():
    """Runs the analysis and saves the CSV without launching the web app."""
    print("--- VCP Auto-Analyzer (Headless Mode) ---")

    if not TV_AVAILABLE:
        print("❌ ERROR: TradingView DataFeed Required. Please install with: pip install tvdatafeed")
        return

    config = load_config()
    print(f"Loaded configuration from {CONFIG_FILE}")

    symbols = [s.strip().upper() for s in config['symbols_input'].split(',') if s.strip()]
    if not symbols:
        print("❌ No symbols configured. Please set them in vcp_config.json or via the web UI.")
        return

    try:
        analyzer = VCPAnalyzer(
            swing_length=config['swing_length'],
            zone_percent=config['zone_percent'],
            target_percent=config['target_percent'],
            stoploss_percent=config['stoploss_percent'],
            headless_mode=True
        )
        results = analyzer.analyze_multiple_symbols(symbols, config['exchange'], config['timeframe'], config['n_bars'])

        print(f"\nAnalysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Found {results['valid_pattern_count']} valid patterns out of {results['total_patterns']} detected.")

        output_filename = "vcp_analysis_results.csv"
        if results['all_patterns']:
            df_to_save = create_consolidated_table(results['all_patterns'])
            df_to_save.to_csv(output_filename, index=False)
            print(f"✅ Results successfully saved to `{output_filename}`.")
        else:
            pd.DataFrame().to_csv(output_filename, index=False)
            print(f"No patterns found. `{output_filename}` has been cleared.")

        if results['failed_symbols']:
            print("\n⚠️ Failed Symbols:")
            for item in results['failed_symbols']:
                print(f"  - {item['symbol']}: {item['error']}")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


# --- Streamlit UI & Application Logic ---
def run_streamlit_app():
    st.set_page_config(page_title="VCP Auto-Analyzer", page_icon="📈", layout="wide")

    st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .stDataFrame { font-size: 12px; }
        .dataframe td, .dataframe th { padding: 4px !important; font-size: 12px !important; border-color: #333 !important; }
        div[data-testid="metric-container"] { background-color: #1a1c24; border: 1px solid #464646; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #FAFAFA; }
    </style>
    """, unsafe_allow_html=True)

    def check_password():
        if "password_correct" not in st.session_state:
            st.session_state.password_correct = False
        if st.session_state.password_correct:
            return True
        st.header("🔑 Admin Access")
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        if st.button("Login"):
            if username == "sherlock" and password == "irene":
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("😕 Incorrect username or password")
        return False

    def render_config_page():
        st.title("⚙️ VCP Analyzer Configuration")
        cfg = load_config()
        with st.form("config_form"):
            st.header("📋 Instruments")
            symbols_input = st.text_area("Symbols (comma-separated)", value=cfg['symbols_input'], height=150)
            exchange = st.selectbox("Exchange", ["NSE", "BSE", "BINANCE", "NASDAQ", "NYSE", "FX_IDC"],
                                    index=["NSE", "BSE", "BINANCE", "NASDAQ", "NYSE", "FX_IDC"].index(cfg['exchange']))
            st.header("⏱️ Timeframe & Pattern Settings")
            timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1H", "4H", "1D"],
                                     index=["5m", "15m", "30m", "1H", "4H", "1D"].index(cfg['timeframe']))
            n_bars = st.slider("Historical Bars", 200, 2000, cfg['n_bars'], 50)
            swing_length = st.slider("Swing High Length", 5, 30, cfg['swing_length'], 1)
            zone_percent = st.slider("Zone Size (%)", 0.5, 3.0, cfg['zone_percent'], 0.1)
            st.header("💰 Trade Management")
            target_percent = st.slider("Target (%)", 5.0, 30.0, cfg['target_percent'], 1.0)
            stoploss_percent = st.slider("Stop Loss (%)", 2.0, 15.0, cfg['stoploss_percent'], 1.0)
            if st.form_submit_button("💾 Save Configuration"):
                new_config = {
                    "symbols_input": symbols_input, "exchange": exchange, "timeframe": timeframe,
                    "n_bars": n_bars, "swing_length": swing_length, "zone_percent": zone_percent,
                    "target_percent": target_percent, "stoploss_percent": stoploss_percent
                }
                save_config(new_config)
                st.success(
                    f"Configuration saved to {CONFIG_FILE}. The main page and headless mode will use these settings.")

    def render_main_page():
        st.title("📈 VCP Auto-Analyzer Dashboard")
        if not TV_AVAILABLE:
            st.error("❌ **TradingView DataFeed Required**. Please install with: `pip install tvDatafeed`")
            st.stop()
        config = load_config()
        st.info(
            f"**Running with parameters from `{CONFIG_FILE}`:** Exchange: `{config['exchange']}`, Timeframe: `{config['timeframe']}`. To change, go to the [config page](/?page=config).")
        symbols = [s.strip().upper() for s in config['symbols_input'].split(',') if s.strip()]
        if not symbols:
            st.error("❌ No symbols configured. Please set them on the config page.")
            return
        try:
            analyzer = VCPAnalyzer(config['swing_length'], config['zone_percent'], config['target_percent'],
                                   config['stoploss_percent'])
            results = analyzer.analyze_multiple_symbols(symbols, config['exchange'], config['timeframe'],
                                                        config['n_bars'])
            st.success(
                f"Analysis complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Found {results['valid_pattern_count']} valid patterns.")
            output_filename = "vcp_analysis_results.csv"
            if results['all_patterns']:
                df_to_save = create_consolidated_table(results['all_patterns'])
                df_to_save.to_csv(output_filename, index=False)
                st.markdown(f"💾 Results automatically saved to `{output_filename}`.")
            else:
                pd.DataFrame().to_csv(output_filename, index=False)
                st.markdown(f"No patterns found. `{output_filename}` has been cleared.")
            st.header("📊 Analysis Results")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Analyzed", len(results['successful_symbols']))
            with col2:
                st.metric("Valid Patterns", results['valid_pattern_count'])
            with col3:
                st.metric("🎯 Setup Ready", len(results['setup_ready_patterns']))
            with col4:
                st.metric("⏳ Ongoing", len(results['ongoing_patterns']))
            with col5:
                st.metric("✅ Target Hit", len(results['target_hit_patterns']))
            with col6:
                st.metric("🛑 SL Hit", len(results['sl_hit_patterns']))
            if results['failed_symbols']:
                with st.expander("⚠️ Failed Symbols"):
                    st.json({f['symbol']: f['error'] for f in results['failed_symbols']})
            tab1, tab2, tab3, tab4 = st.tabs(["📋 All Patterns", "🎯 Setup Ready", "⏳ Ongoing", "✅ Completed"])
            with tab1:
                if results['all_patterns']: st.dataframe(create_consolidated_table(results['all_patterns']),
                                                         use_container_width=True, height=600)
            with tab2:
                if results['setup_ready_patterns']: st.dataframe(
                    create_consolidated_table(results['setup_ready_patterns']), use_container_width=True, height=400)
            with tab3:
                if results['ongoing_patterns']: st.dataframe(create_consolidated_table(results['ongoing_patterns']),
                                                             use_container_width=True, height=400)
            with tab4:
                completed = results['target_hit_patterns'] + results['sl_hit_patterns']
                if completed: st.dataframe(create_consolidated_table(completed), use_container_width=True, height=400)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    query_params = st.query_params
    if query_params.get("page") == "config":
        if check_password():
            render_config_page()
    else:
        render_main_page()


# --- Main App Router ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCP Pattern Analyzer. Runs as a Streamlit app by default.")
    parser.add_argument("--headless", action="store_true",
                        help="Run the analysis in the terminal without launching the web UI.")
    args = parser.parse_args()

    if args.headless:
        run_headless()
    else:
        run_streamlit_app()

