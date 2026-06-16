#!/usr/bin/env python3
"""Virtual day trading portfolio with ORB and mean reversion signals."""

import json
import re
import sys
import os
from datetime import datetime
from typing import Optional
import pandas as pd

PORTFOLIO_FILE = "portfolio_data.json"
STARTING_CASH  = 100_000.0
SNAPSHOT_FILE  = "snapshots.json"

# ─── Auto-trade configuration ─────────────────────────────────────────────────
AUTO_POSITION_PCT = 0.05   # target position size as % of total portfolio value
AUTO_CASH_RESERVE = 0.15   # minimum cash floor as % of STARTING_CASH
AUTO_WEAK_BUY     = False  # if False, skip WEAK-strength buy signals
AUTO_ADD_TO_POS   = False  # if False, skip BUY signals for already-held symbols

# ANSI colors
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
B  = "\033[1m"    # bold
D  = "\033[2m"    # dim
Z  = "\033[0m"    # reset

_ANSI = re.compile(r"\033\[[0-9;]*m")

def _vlen(s: str) -> int:
    """Visible length of a string (strips ANSI escape codes)."""
    return len(_ANSI.sub("", s))

def _ljust(s: str, width: int) -> str:
    """Left-justify accounting for invisible ANSI chars."""
    return s + " " * max(0, width - _vlen(s))


# ─── Portfolio persistence ────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": {}, "trades": []}


def save_portfolio(data: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_snapshots() -> list:
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE) as f:
            return json.load(f)
    return []


def save_snapshots(snaps: list):
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(snaps, f, indent=2)


def _portfolio_value(data: dict) -> tuple:
    """Return (total_value, positions_value) using live prices."""
    mv = 0.0
    for sym, pos in data["positions"].items():
        price = get_price(sym)
        mv += (price or pos["avg_cost"]) * pos["shares"]
    return round(data["cash"] + mv, 2), round(mv, 2)


def _max_drawdown(values: list) -> float:
    """Max peak-to-trough drawdown as a percentage."""
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


# ─── Price / data helpers ─────────────────────────────────────────────────────

def get_price(symbol: str) -> Optional[float]:
    try:
        import yfinance as yf
        p = yf.Ticker(symbol.upper()).fast_info.last_price
        return round(float(p), 2) if p and p > 0 else None
    except Exception:
        return None


def _et_now() -> "pd.Timestamp":
    """Current time in US/Eastern."""
    return pd.Timestamp.now(tz="America/New_York")


def _next_preopen_dt() -> "pd.Timestamp":
    """Next 09:20 ET (10 min before open) on a weekday."""
    now = _et_now()
    pre = pd.Timestamp(f"{now.date()} 09:20", tz="America/New_York")
    if now < pre and now.dayofweek < 5:
        return pre
    candidate = now + pd.Timedelta(days=1)
    while candidate.dayofweek >= 5:
        candidate += pd.Timedelta(days=1)
    return pd.Timestamp(f"{candidate.date()} 09:20", tz="America/New_York")


def _download(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def _is_potentially_delisted(symbol: str) -> bool:
    """Return True when no price AND no recent daily OHLC data can be fetched."""
    if get_price(symbol) is not None:
        return False
    df = _download(symbol, period="5d", interval="1d")
    return df is None or df.empty


_NASDAQ_EXCHANGES = {"NMS", "NGM", "NCM", "NasdaqGS", "NasdaqGM", "NasdaqCM"}

def fetch_nasdaq_most_active(n: int = 50) -> list:
    """Return up to n most-active Nasdaq tickers by volume via yfinance screen."""
    try:
        import yfinance as yf
        result = yf.screen("most_actives", count=150)
        quotes = result.get("quotes", [])

        def _valid(sym: str) -> bool:
            return bool(sym) and not any(c in sym for c in (".", "^", "/"))

        nasdaq = [q["symbol"] for q in quotes
                  if _valid(q.get("symbol", "")) and q.get("exchange", "") in _NASDAQ_EXCHANGES]
        if len(nasdaq) >= 5:
            return nasdaq[:n]
        # Fallback: screener returned different exchange codes — accept everything
        return [q["symbol"] for q in quotes if _valid(q.get("symbol", ""))][:n]
    except Exception:
        return []


def get_watchlist(data: Optional[dict] = None) -> list:
    """Return the persisted dynamic watchlist from portfolio data, or empty list."""
    if data and "watchlist" in data:
        return data["watchlist"]
    return []


# ─── ORB strategy ─────────────────────────────────────────────────────────────
#
# Opening Range Breakout: mark the high/low of the first 30 minutes (9:30–10:00 ET).
# A close above OR high = bullish breakout; below OR low = bearish breakout.
# Entry is confirmed only AFTER the range has fully formed (≥ 25 bars).

def calc_orb(symbol: str) -> Optional[dict]:
    df = _download(symbol, period="1d", interval="1m")
    if df is None or len(df) < 5:
        return None

    current = float(df["Close"].iloc[-1])

    try:
        et_idx    = df.index.tz_convert("America/New_York")
        today     = et_idx[0].date()
        t_open    = pd.Timestamp(f"{today} 09:30", tz="America/New_York")
        t_orclose = pd.Timestamp(f"{today} 10:00", tz="America/New_York")
        orb_df    = df[(et_idx >= t_open) & (et_idx < t_orclose)]
    except Exception:
        orb_df = df.head(30)

    if orb_df.empty:
        orb_df = df.head(30)

    bars     = len(orb_df)
    orb_high = float(orb_df["High"].max())
    orb_low  = float(orb_df["Low"].min())

    if bars < 25:
        signal = "PENDING"
    elif current > orb_high:
        signal = "BREAK_UP"
    elif current < orb_low:
        signal = "BREAK_DOWN"
    else:
        pct_pos = (current - orb_low) / (orb_high - orb_low) * 100 if orb_high > orb_low else 50
        signal  = "UPPER" if pct_pos > 60 else ("LOWER" if pct_pos < 40 else "NEUTRAL")

    return {
        "orb_high":  round(orb_high, 2),
        "orb_low":   round(orb_low, 2),
        "current":   round(current, 2),
        "signal":    signal,
        "bars":      bars,
        "pct_above": round((current - orb_high) / orb_high * 100, 2) if current > orb_high else None,
        "pct_below": round((orb_low - current) / orb_low * 100, 2)  if current < orb_low  else None,
    }


# ─── Mean Reversion strategy ──────────────────────────────────────────────────
#
# Uses RSI-14 and Bollinger Bands (20, 2σ) on daily closes.
# Oversold (RSI < 30 or price near/below lower BB) → buy signal.
# Overbought (RSI > 70 or price near/above upper BB) → sell signal.
# Acts as a FILTER on ORB: confirms momentum when MR agrees, flags conflict when it disagrees.

def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def _bollinger(closes: pd.Series, period: int = 20, n_std: float = 2.0):
    ma  = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return ma + n_std * std, ma, ma - n_std * std


def calc_mean_reversion(symbol: str) -> Optional[dict]:
    df = _download(symbol, period="60d", interval="1d")
    if df is None or len(df) < 20:
        return None

    closes  = df["Close"]
    rsi_val = float(_rsi(closes).iloc[-1])
    bb_u, bb_m, bb_l = _bollinger(closes)
    current = float(closes.iloc[-1])
    u, m, l = float(bb_u.iloc[-1]), float(bb_m.iloc[-1]), float(bb_l.iloc[-1])
    bb_pct  = (current - l) / (u - l) * 100 if u > l else 50.0

    if   rsi_val < 30 or bb_pct <  5: signal = "OVERSOLD"
    elif rsi_val > 70 or bb_pct > 95: signal = "OVERBOUGHT"
    elif rsi_val < 40 or bb_pct < 20: signal = "WEAK"
    elif rsi_val > 60 or bb_pct > 80: signal = "STRONG"
    else:                              signal = "NEUTRAL"

    return {
        "rsi":    round(rsi_val, 1),
        "bb_up":  round(u, 2),
        "bb_mid": round(m, 2),
        "bb_low": round(l, 2),
        "bb_pct": round(bb_pct, 1),
        "signal": signal,
    }


# ─── VWAP strategy ───────────────────────────────────────────────────────────
#
# Intraday Volume Weighted Average Price: cumulative (price × volume) / cumulative volume.
# Price above VWAP signals bullish intraday bias; below signals bearish.
# Used as a confirmation layer — agrees with ORB breakout or flags counter-trend risk.

def calc_vwap(symbol: str) -> Optional[dict]:
    df = _download(symbol, period="1d", interval="1m")
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    tp   = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    current  = float(df["Close"].iloc[-1])
    vwap_val = float(vwap.iloc[-1])
    pct = (current - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0.0
    if   pct >  0.3: signal = "ABOVE"
    elif pct < -0.3: signal = "BELOW"
    else:            signal = "AT"
    return {"vwap": round(vwap_val, 2), "current": round(current, 2),
            "pct":  round(pct, 2),       "signal":  signal}


# ─── Gap strategy ─────────────────────────────────────────────────────────────
#
# Gap up/down vs prior close at today's open, confirmed by above-average volume.
# GAP_UP (>2% + 1.5× avg vol) aligns with or opposes the ORB direction.
# Agrees with BUY/SELL → upgrades conviction; opposes → CONFLICT.

def calc_gap(symbol: str) -> Optional[dict]:
    df = _download(symbol, period="5d", interval="1d")
    if df is None or len(df) < 2:
        return None
    prev_close = float(df["Close"].iloc[-2])
    today_open = float(df["Open"].iloc[-1])
    today_vol  = float(df["Volume"].iloc[-1])
    avg_vol    = float(df["Volume"].iloc[:-1].mean())
    gap_pct    = (today_open - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
    vol_ratio  = today_vol / avg_vol if avg_vol > 0 else 1.0
    if   gap_pct >  2.0 and vol_ratio > 1.5: signal = "GAP_UP"
    elif gap_pct < -2.0 and vol_ratio > 1.5: signal = "GAP_DOWN"
    elif gap_pct >  0.5:                     signal = "SMALL_GAP_UP"
    elif gap_pct < -0.5:                     signal = "SMALL_GAP_DOWN"
    else:                                    signal = "NO_GAP"
    return {"gap_pct":    round(gap_pct, 2),   "vol_ratio":  round(vol_ratio, 2),
            "prev_close": round(prev_close, 2), "today_open": round(today_open, 2),
            "signal":     signal}


# ─── Signal combination ───────────────────────────────────────────────────────

_STRENGTH_LADDER = ["NEUTRAL", "WEAK", "MODERATE", "STRONG"]

def _upgrade_strength(strength: str) -> str:
    idx = _STRENGTH_LADDER.index(strength) if strength in _STRENGTH_LADDER else 0
    return _STRENGTH_LADDER[min(idx + 1, len(_STRENGTH_LADDER) - 1)]


def combine_signals(orb_signal: str, mr_signal: str,
                    vwap_signal: Optional[str] = None,
                    gap_signal:  Optional[str] = None) -> tuple:
    """Return (action, strength): action in BUY/SELL/CONFLICT/HOLD."""
    up      = orb_signal == "BREAK_UP"
    down    = orb_signal == "BREAK_DOWN"
    bull_mr = mr_signal in ("OVERSOLD", "WEAK")
    bear_mr = mr_signal in ("OVERBOUGHT", "STRONG")

    if up and bear_mr:     action, strength = "CONFLICT", "WEAK"
    elif down and bull_mr: action, strength = "CONFLICT", "WEAK"
    elif up:               action, strength = "BUY",  "STRONG" if bull_mr else "MODERATE"
    elif down:             action, strength = "SELL", "STRONG" if bear_mr else "MODERATE"
    elif mr_signal == "OVERSOLD":   action, strength = "BUY",  "WEAK"
    elif mr_signal == "OVERBOUGHT": action, strength = "SELL", "WEAK"
    else:                           action, strength = "HOLD", "NEUTRAL"

    # ── VWAP + Gap modifiers ───────────────────────────────────────────────────
    gap_up_strong   = gap_signal == "GAP_UP"
    gap_down_strong = gap_signal == "GAP_DOWN"
    gap_up          = gap_signal in ("GAP_UP", "SMALL_GAP_UP")
    gap_down        = gap_signal in ("GAP_DOWN", "SMALL_GAP_DOWN")
    vwap_above      = vwap_signal == "ABOVE"
    vwap_below      = vwap_signal == "BELOW"

    if action == "BUY":
        if gap_down:
            action = "CONFLICT"
        elif gap_up_strong and vwap_above:
            strength = "STRONG"
        elif gap_up or vwap_above:
            strength = _upgrade_strength(strength)
    elif action == "SELL":
        if gap_up:
            action = "CONFLICT"
        elif gap_down_strong and vwap_below:
            strength = "STRONG"
        elif gap_down or vwap_below:
            strength = _upgrade_strength(strength)
    elif action == "HOLD":
        if gap_up_strong and vwap_above:
            action, strength = "BUY",  "WEAK"
        elif gap_down_strong and vwap_below:
            action, strength = "SELL", "WEAK"

    return action, strength


# ─── Formatting ───────────────────────────────────────────────────────────────

def fmt_currency(v: float) -> str:
    c    = G if v >= 0 else R
    sign = "+" if v > 0 else ""
    return f"{c}{sign}${v:,.2f}{Z}"

def fmt_price(v: float) -> str:
    return f"${v:,.2f}"

def fmt_orb(signal: str) -> str:
    return {
        "BREAK_UP":   f"{G}BREAK ^{Z}",
        "BREAK_DOWN": f"{R}BREAK v{Z}",
        "PENDING":    f"{D}PENDING{Z}",
        "UPPER":      f"{Y}UPPER  {Z}",
        "LOWER":      f"{Y}LOWER  {Z}",
        "NEUTRAL":    f"{D}NEUTRAL{Z}",
    }.get(signal, f"{D}{signal}{Z}")

def fmt_mr(signal: str, rsi: float) -> str:
    base = f"RSI {rsi:4.1f}"
    return {
        "OVERSOLD":   f"{G}{base}  OVERSOLD  {Z}",
        "OVERBOUGHT": f"{R}{base}  OVERBOUGHT{Z}",
        "WEAK":       f"{C}{base}  WEAK      {Z}",
        "STRONG":     f"{Y}{base}  STRONG    {Z}",
        "NEUTRAL":    f"{D}{base}  NEUTRAL   {Z}",
    }.get(signal, f"{D}{base}  {signal}{Z}")

def fmt_action(action: str, strength: str) -> str:
    stars = "**" if strength == "STRONG" else ("*" if strength == "MODERATE" else " ")
    if action == "BUY":      return f"{G}{B}{stars} BUY {Z}"
    if action == "SELL":     return f"{R}{B}{stars} SELL{Z}"
    if action == "CONFLICT": return f"{Y}CONFLICT{Z}"
    return f"{D}   HOLD {Z}"

def fmt_vwap(signal: str, pct: float) -> str:
    pct_s = f"{pct:+.1f}%"
    return {
        "ABOVE": f"{G}ABOVE {pct_s}{Z}",
        "BELOW": f"{R}BELOW {pct_s}{Z}",
        "AT":    f"{D}AT    {pct_s}{Z}",
    }.get(signal, f"{D}{signal}{Z}")

def fmt_gap(signal: str, pct: float) -> str:
    pct_s = f"{pct:+.1f}%"
    return {
        "GAP_UP":         f"{G}GAP ^ {pct_s}{Z}",
        "GAP_DOWN":       f"{R}GAP v {pct_s}{Z}",
        "SMALL_GAP_UP":   f"{Y}gap ^ {pct_s}{Z}",
        "SMALL_GAP_DOWN": f"{Y}gap v {pct_s}{Z}",
        "NO_GAP":         f"{D}no gap     {Z}",
    }.get(signal, f"{D}{signal}{Z}")


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_buy(data: dict, symbol: str, shares: float, price: Optional[float] = None):
    symbol = symbol.upper()
    if price is None:
        print(f"  Fetching price for {symbol}...")
        price = get_price(symbol)
        if price is None:
            print(f"  Could not fetch price. Use: buy {symbol} {shares} <price>")
            return
    cost = price * shares
    if cost > data["cash"]:
        print(f"  Insufficient cash. Need {fmt_price(cost)}, have {fmt_price(data['cash'])}")
        return
    data["cash"] -= cost
    pos = data["positions"].get(symbol)
    if pos:
        total = pos["shares"] + shares
        pos["avg_cost"] = (pos["avg_cost"] * pos["shares"] + cost) / total
        pos["shares"]   = total
    else:
        data["positions"][symbol] = {"shares": shares, "avg_cost": price}
    data["trades"].append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": "BUY", "symbol": symbol,
        "shares": shares, "price": price, "total": cost,
    })
    save_portfolio(data)
    print(f"  BUY  {shares} {symbol} @ {fmt_price(price)} = {fmt_price(cost)}")
    print(f"  Cash remaining: {fmt_price(data['cash'])}")


def cmd_sell(data: dict, symbol: str, shares: float, price: Optional[float] = None):
    symbol = symbol.upper()
    pos = data["positions"].get(symbol)
    if not pos or pos["shares"] < shares:
        held = pos["shares"] if pos else 0
        print(f"  Not enough shares. Holding {held} {symbol}")
        return
    if price is None:
        print(f"  Fetching price for {symbol}...")
        price = get_price(symbol)
        if price is None:
            print(f"  Could not fetch price. Use: sell {symbol} {shares} <price>")
            return
    proceeds = price * shares
    gain     = (price - pos["avg_cost"]) * shares
    pos["shares"] -= shares
    if pos["shares"] == 0:
        del data["positions"][symbol]
    data["cash"] += proceeds
    data["trades"].append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": "SELL", "symbol": symbol,
        "shares": shares, "price": price, "total": proceeds,
        "realized_pnl": round(gain, 2),
    })
    save_portfolio(data)
    print(f"  SELL {shares} {symbol} @ {fmt_price(price)} = {fmt_price(proceeds)}  P&L: {fmt_currency(gain)}")
    print(f"  Cash remaining: {fmt_price(data['cash'])}")


def cmd_portfolio(data: dict):
    print("\n" + "=" * 62)
    print("  VIRTUAL PORTFOLIO")
    print("=" * 62)
    total_mv = total_cb = 0.0
    rows = []
    for sym, pos in data["positions"].items():
        price = get_price(sym)
        mv    = (price or pos["avg_cost"]) * pos["shares"]
        unr   = (price - pos["avg_cost"]) * pos["shares"] if price else 0.0
        total_mv += mv
        total_cb += pos["avg_cost"] * pos["shares"]
        rows.append((sym, pos["shares"], pos["avg_cost"], price, mv, unr))
    if rows:
        print(f"\n  {'Symbol':<8} {'Shares':>8} {'AvgCost':>10} {'Price':>10} {'MktValue':>12} {'Unrealized':>18}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*18}")
        for sym, sh, ac, pr, mv, unr in rows:
            pr_s = fmt_price(pr) if pr else "  N/A  "
            print(f"  {sym:<8} {sh:>8.2f} {fmt_price(ac):>10} {pr_s:>10} {fmt_price(mv):>12} {_ljust(fmt_currency(unr), 27)}")
    else:
        print("\n  No open positions.")
    tv  = data["cash"] + total_mv
    pnl = tv - STARTING_CASH
    print(f"\n  Cash:             {fmt_price(data['cash']):>12}")
    print(f"  Positions Value:  {fmt_price(total_mv):>12}")
    print(f"  Total Value:      {fmt_price(tv):>12}")
    print(f"  Unrealized P&L:   {_ljust(fmt_currency(total_mv - total_cb), 21)}")
    print(f"  Total P&L:        {_ljust(fmt_currency(pnl), 21)}")
    print("=" * 62 + "\n")


def cmd_history(data: dict, limit: int = 20):
    trades = data["trades"][-limit:]
    if not trades:
        print("  No trades yet.\n")
        return
    print(f"\n  {'Time':<20} {'Act':<5} {'Symbol':<7} {'Shares':>7} {'Price':>10} {'Total':>12} {'P&L':>12}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*10} {'-'*12} {'-'*12}")
    for t in trades:
        pnl = t.get("realized_pnl")
        ps  = fmt_currency(pnl) if pnl is not None else ""
        print(f"  {t['time']:<20} {t['action']:<5} {t['symbol']:<7} {t['shares']:>7.2f} "
              f"{fmt_price(t['price']):>10} {fmt_price(t['total']):>12} {_ljust(ps, 21)}")
    print()


def cmd_scan(symbols: Optional[list] = None):
    data = load_portfolio()
    wl   = get_watchlist(data)
    if not wl and not symbols:
        import sys as _sys
        _sys.stderr.write("  fetching Nasdaq most active...\r"); _sys.stderr.flush()
        wl = fetch_nasdaq_most_active(50)
        _sys.stderr.write(" " * 50 + "\r"); _sys.stderr.flush()
    syms = [s.upper() for s in (symbols or wl)]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n  SIGNAL SCAN  {now}")
    print("  " + "=" * 96)
    hdr = (f"  {'Symbol':<7}  {'Price':>8}  "
           f"{'ORB (30-min)':^13}  {'Mean Reversion':^22}  "
           f"{'VWAP':^11}  {'Gap':^11}  {'Action':^10}")
    print(hdr)
    print(f"  {'-'*7}  {'-'*8}  {'-'*13}  {'-'*22}  {'-'*11}  {'-'*11}  {'-'*10}")

    for sym in syms:
        import sys as _sys; _sys.stderr.write(f"  scanning {sym}...\r"); _sys.stderr.flush()
        orb  = calc_orb(sym)
        mr   = calc_mean_reversion(sym)
        vwap = calc_vwap(sym)
        gap  = calc_gap(sym)

        price_s  = fmt_price(orb["current"] if orb else (mr["bb_mid"] if mr else 0))
        orb_s    = fmt_orb(orb["signal"]) if orb else f"{D}N/A    {Z}"
        mr_s     = fmt_mr(mr["signal"], mr["rsi"]) if mr else f"{D}N/A              {Z}"
        vwap_s   = fmt_vwap(vwap["signal"], vwap["pct"]) if vwap else f"{D}N/A       {Z}"
        gap_s    = fmt_gap(gap["signal"], gap["gap_pct"]) if gap else f"{D}N/A       {Z}"
        vwap_sig = vwap["signal"] if vwap else None
        gap_sig  = gap["signal"]  if gap  else None
        action_s = (fmt_action(*combine_signals(orb["signal"], mr["signal"], vwap_sig, gap_sig))
                    if orb and mr else f"{D}N/A{Z}")

        print(f"  {sym:<7}  {price_s:>8}  {_ljust(orb_s,22)}  {_ljust(mr_s,31)}  "
              f"{_ljust(vwap_s,20)}  {_ljust(gap_s,20)}  {action_s}")

    print("  " + "=" * 96 + "\n")


def cmd_signals(symbol: str):
    symbol = symbol.upper()
    print(f"\n  Fetching data for {symbol}...\n")
    orb  = calc_orb(symbol)
    mr   = calc_mean_reversion(symbol)
    vwap = calc_vwap(symbol)
    gap  = calc_gap(symbol)

    print(f"  {B}{symbol} - Signal Detail{Z}")
    print("  " + "-" * 46)

    # ── ORB ──
    if orb:
        print(f"\n  {C}Opening Range Breakout  (first 30 min){Z}")
        print(f"    OR High:    {fmt_price(orb['orb_high'])}")
        print(f"    OR Low:     {fmt_price(orb['orb_low'])}")
        print(f"    Current:    {fmt_price(orb['current'])}")
        print(f"    Bars built: {orb['bars']}")
        if orb["pct_above"] is not None:
            print(f"    Distance:   {G}+{orb['pct_above']}% above OR high{Z}")
        elif orb["pct_below"] is not None:
            print(f"    Distance:   {R}-{orb['pct_below']}% below OR low{Z}")
        print(f"    Signal:     {fmt_orb(orb['signal'])}")
    else:
        print(f"\n  {D}ORB: no intraday data available{Z}")

    # ── Mean Reversion ──
    if mr:
        print(f"\n  {C}Mean Reversion  (RSI-14  +  Bollinger 20,2std){Z}")
        rsi_note = (f"  {G}← OVERSOLD{Z}" if mr["rsi"] < 30
                    else (f"  {R}← OVERBOUGHT{Z}" if mr["rsi"] > 70 else ""))
        print(f"    RSI-14:     {mr['rsi']:.1f}{rsi_note}")
        print(f"    BB Upper:   {fmt_price(mr['bb_up'])}")
        print(f"    BB Mid:     {fmt_price(mr['bb_mid'])}")
        print(f"    BB Lower:   {fmt_price(mr['bb_low'])}")
        print(f"    BB %B:      {mr['bb_pct']:.1f}%  (0% = lower band, 100% = upper band)")
        print(f"    Signal:     {fmt_mr(mr['signal'], mr['rsi'])}")
    else:
        print(f"\n  {D}MR: no daily data available{Z}")

    # ── VWAP ──
    if vwap:
        print(f"\n  {C}VWAP  (intraday, 1-min bars){Z}")
        print(f"    VWAP:       {fmt_price(vwap['vwap'])}")
        print(f"    Current:    {fmt_price(vwap['current'])}")
        dist_c = G if vwap["pct"] >= 0 else R
        print(f"    Distance:   {dist_c}{vwap['pct']:+.2f}%{Z} from VWAP")
        print(f"    Signal:     {fmt_vwap(vwap['signal'], vwap['pct'])}")
    else:
        print(f"\n  {D}VWAP: no intraday data available{Z}")

    # ── Gap ──
    if gap:
        print(f"\n  {C}Gap  (vs prior close){Z}")
        print(f"    Prev Close: {fmt_price(gap['prev_close'])}")
        print(f"    Today Open: {fmt_price(gap['today_open'])}")
        gap_c = G if gap["gap_pct"] >= 0 else R
        print(f"    Gap:        {gap_c}{gap['gap_pct']:+.2f}%{Z}")
        print(f"    Vol Ratio:  {gap['vol_ratio']:.1f}x average")
        print(f"    Signal:     {fmt_gap(gap['signal'], gap['gap_pct'])}")
    else:
        print(f"\n  {D}Gap: no daily data available{Z}")

    # ── Combined ──
    if orb and mr:
        vwap_sig = vwap["signal"] if vwap else None
        gap_sig  = gap["signal"]  if gap  else None
        action, strength = combine_signals(orb["signal"], mr["signal"], vwap_sig, gap_sig)
        print(f"\n  {B}Combined Signal:  {fmt_action(action, strength)}{Z}")
        if action == "CONFLICT":
            print(f"  {Y}  Signals disagree — sit out or reduce size{Z}")
        elif strength == "STRONG":
            print(f"  {G}  All signals aligned — highest-conviction setup{Z}")
        elif strength == "MODERATE":
            print(f"  {C}  ORB confirmed, partial agreement — standard momentum trade{Z}")
        elif strength == "WEAK" and action in ("BUY", "SELL"):
            print(f"  {D}  Weak signal, limited confirmation — smaller size{Z}")
    print()


def cmd_stats(data: dict):
    sells = [t for t in data["trades"] if t["action"] == "SELL" and "realized_pnl" in t]
    if not sells:
        print("  No completed trades yet.\n")
        return

    pnls   = [t["realized_pnl"] for t in sells]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate      = len(wins) / len(pnls) * 100
    avg_win       = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss      = sum(losses) / len(losses) if losses else 0.0
    profit_factor = sum(wins)   / abs(sum(losses)) if losses else float("inf")
    expectancy    = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    best_trade    = max(pnls)
    worst_trade   = min(pnls)
    total_realized = sum(pnls)

    total_val, _ = _portfolio_value(data)
    total_return  = total_val - STARTING_CASH
    total_ret_pct = total_return / STARTING_CASH * 100

    snaps  = load_snapshots()
    values = [s["total_value"] for s in snaps]
    max_dd = _max_drawdown(values) if len(values) >= 2 else None

    print(f"\n  {B}PERFORMANCE STATS{Z}")
    print("  " + "=" * 46)

    print(f"\n  {C}Returns{Z}")
    print(f"    Starting Cash:    {fmt_price(STARTING_CASH):>12}")
    print(f"    Current Value:    {fmt_price(total_val):>12}")
    print(f"    Total Return:     {_ljust(fmt_currency(total_return), 20)} ({total_ret_pct:+.2f}%)")
    print(f"    Realized P&L:     {_ljust(fmt_currency(total_realized), 20)}")
    if max_dd is not None:
        dd_str = f"{R}-{max_dd:.2f}%{Z}" if max_dd > 0 else f"{G}0.00%{Z}"
        print(f"    Max Drawdown:     {dd_str}  (from {len(snaps)} snapshots)")

    print(f"\n  {C}Trade Record{Z}")
    print(f"    Total Trades:     {len(pnls)}")
    print(f"    Winners:          {len(wins)}   ({win_rate:.1f}%)")
    print(f"    Losers:           {len(losses)}   ({100-win_rate:.1f}%)")
    print(f"    Avg Win:          {_ljust(fmt_currency(avg_win), 20)}")
    print(f"    Avg Loss:         {_ljust(fmt_currency(avg_loss), 20)}")
    print(f"    Win/Loss Ratio:   {abs(avg_win/avg_loss):.2f}x" if avg_loss else "    Win/Loss Ratio:   N/A (no losses)")

    pf_str = f"{profit_factor:.2f}x" if profit_factor != float("inf") else "inf (no losses)"
    print(f"    Profit Factor:    {pf_str}")
    print(f"    Expectancy/trade: {_ljust(fmt_currency(expectancy), 20)}")

    print(f"\n  {C}Best / Worst{Z}")
    best_t  = max(sells, key=lambda t: t["realized_pnl"])
    worst_t = min(sells, key=lambda t: t["realized_pnl"])
    print(f"    Best Trade:  {_ljust(fmt_currency(best_trade), 20)}  {best_t['symbol']} on {best_t['time'][:10]}")
    print(f"    Worst Trade: {_ljust(fmt_currency(worst_trade), 20)}  {worst_t['symbol']} on {worst_t['time'][:10]}")

    print("  " + "=" * 46 + "\n")


def cmd_snapshot(data: dict):
    print("  Fetching live prices...")
    total_val, mv = _portfolio_value(data)
    today = datetime.now().strftime("%Y-%m-%d")
    snap  = {
        "date":             today,
        "time":             datetime.now().isoformat(timespec="seconds"),
        "total_value":      total_val,
        "cash":             round(data["cash"], 2),
        "positions_value":  mv,
    }
    snaps = [s for s in load_snapshots() if s["date"] != today]
    snaps.append(snap)
    snaps.sort(key=lambda s: s["date"])
    save_snapshots(snaps)
    pnl = total_val - STARTING_CASH
    print(f"  Snapshot saved  {today}  |  Total: {fmt_price(total_val)}  |  P&L: {fmt_currency(pnl)}")


def cmd_equity():
    snaps = load_snapshots()
    if not snaps:
        print("  No snapshots yet. Run 'snapshot' to record today's value.\n")
        return

    values    = [s["total_value"] for s in snaps]
    min_v     = min(values)
    max_v     = max(values)
    bar_width = 30

    print(f"\n  {B}EQUITY CURVE{Z}  ({len(snaps)} snapshots)")
    print("  " + "=" * 68)
    print(f"  {'Date':<12} {'Value':>12} {'Daily Chg':>12}   {'Chart (|= baseline)'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}   {'-'*bar_width}")

    prev = STARTING_CASH
    base_pos = int((STARTING_CASH - min_v) / (max_v - min_v) * bar_width) if max_v > min_v else bar_width // 2

    for s in snaps:
        v      = s["total_value"]
        change = v - prev
        prev   = v

        if max_v > min_v:
            bar_len = int((v - min_v) / (max_v - min_v) * bar_width)
        else:
            bar_len = bar_width // 2

        color  = G if v >= STARTING_CASH else R
        bar    = color + "=" * bar_len + Z
        marker = f"{color}|{Z}" if bar_len == base_pos else " "

        change_s = fmt_currency(change) if change != 0 else f"{D}  --{Z}"
        print(f"  {s['date']:<12} {fmt_price(v):>12} {_ljust(change_s, 21)}   {bar}{marker}")

    total_return = values[-1] - STARTING_CASH
    total_pct    = total_return / STARTING_CASH * 100
    max_dd       = _max_drawdown(values)
    dd_str       = f"{R}-{max_dd:.2f}%{Z}" if max_dd > 0 else f"{G}0.00%{Z}"

    print(f"\n  Total Return:   {_ljust(fmt_currency(total_return), 20)} ({total_pct:+.2f}%)")
    print(f"  Max Drawdown:   {dd_str}")
    print("  " + "=" * 68 + "\n")


def cmd_reset():
    if os.path.exists(PORTFOLIO_FILE):
        c = input("  Reset portfolio? All data will be lost. (yes/no): ").strip().lower()
        if c == "yes":
            os.remove(PORTFOLIO_FILE)
            print(f"  Portfolio reset with {fmt_price(STARTING_CASH)} starting cash.")
        else:
            print("  Cancelled.")
    else:
        print("  No portfolio data found.")


def _auto_shares(price: float, budget: float, strength: str) -> int:
    """Shares to buy given a per-trade dollar budget and signal strength."""
    mult = {"STRONG": 1.0, "MODERATE": 0.6, "WEAK": 0.3}.get(strength, 0.0)
    return max(0, int(budget * mult / price))


def _is_actionable_buy(sym: str, strength: str, positions: dict) -> bool:
    """True if this BUY signal passes the filter rules."""
    if strength == "WEAK" and not AUTO_WEAK_BUY:
        return False
    if not AUTO_ADD_TO_POS and sym in positions:
        return False
    return True


def cmd_autotrade(data: dict, symbols: Optional[list] = None, dry_run: bool = False):
    import sys as _sys
    syms = [s.upper() for s in (symbols or get_watchlist(data))]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = f"{Y}DRY RUN{Z}" if dry_run else f"{G}LIVE{Z}"
    print(f"\n  AUTO-TRADE  {now}  [{mode}]")
    print("  " + "=" * 76)

    cash_floor = STARTING_CASH * AUTO_CASH_RESERVE
    buys = sells = skips = 0

    # ── Pre-scan: detect and liquidate potentially delisted positions ─────────
    delisted_syms = [sym for sym in list(data["positions"])
                     if _is_potentially_delisted(sym)]
    if delisted_syms:
        print(f"  {R}Potentially delisted: {', '.join(delisted_syms)}{Z}")
        for sym in delisted_syms:
            pos = data["positions"].get(sym)
            if pos:
                sell_price = pos["avg_cost"]
                proceeds   = round(sell_price * pos["shares"], 2)
                print(f"  {sym:<7}  {R}FORCE LIQUIDATE{Z}  {pos['shares']} sh "
                      f"@ {fmt_price(sell_price)} (last known cost)  = {fmt_price(proceeds)}")
                if not dry_run:
                    data["cash"] += proceeds
                    data["trades"].append({
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "action": "SELL", "symbol": sym,
                        "shares": pos["shares"], "price": sell_price,
                        "total": proceeds, "realized_pnl": 0.0,
                        "note": "force-liquidated: potentially delisted",
                    })
                    del data["positions"][sym]
                sells += 1
        if not dry_run:
            save_portfolio(data)
        print()

    # ── Refresh watchlist from Nasdaq most active (skip when explicit syms given) ──
    if not symbols:
        _sys.stderr.write("  fetching Nasdaq most active...\r"); _sys.stderr.flush()
        active = fetch_nasdaq_most_active(50)
        _sys.stderr.write(" " * 50 + "\r"); _sys.stderr.flush()
        if active:
            new_wl = [s for s in active if s not in delisted_syms]
            syms   = new_wl
            if not dry_run:
                data["watchlist"] = new_wl
                save_portfolio(data)
            print(f"  {C}Watchlist refreshed:{Z} {len(new_wl)} Nasdaq most-active symbols")
        else:
            print(f"  {Y}Could not fetch Nasdaq most active — using current watchlist{Z}")
        print()

    # ── Pass 1: fetch all signals up front ───────────────────────────────────
    scan_results = []
    for sym in syms:
        _sys.stderr.write(f"  scanning {sym}...\r"); _sys.stderr.flush()
        scan_results.append((sym, calc_orb(sym), calc_mean_reversion(sym),
                             calc_vwap(sym), calc_gap(sym)))
    _sys.stderr.write(" " * 50 + "\r"); _sys.stderr.flush()

    # ── Divide available cash evenly among qualifying BUY signals ────────────
    n_buys = 0
    for sym, orb, mr, vwap, gap in scan_results:
        if orb and mr:
            _act, _str = combine_signals(orb["signal"], mr["signal"],
                                         vwap["signal"] if vwap else None,
                                         gap["signal"]  if gap  else None)
            if _act == "BUY" and _is_actionable_buy(sym, _str, data["positions"]):
                n_buys += 1
    available_cash = max(0.0, data["cash"] - cash_floor)
    per_trade      = available_cash / n_buys if n_buys > 0 else 0.0

    print(f"  Scanned: {len(scan_results)}  |  Buy signals: {n_buys}  |  "
          f"Deployable cash: {fmt_price(available_cash)}  |  Per trade: {fmt_price(per_trade)}")
    print("  " + "-" * 76)

    # ── Pass 2: execute ───────────────────────────────────────────────────────
    for sym, orb, mr, vwap, gap in scan_results:
        if not orb or not mr:
            print(f"  {sym:<7}  {D}no data — skipped{Z}")
            skips += 1
            continue

        price    = orb["current"]
        vwap_sig = vwap["signal"] if vwap else None
        gap_sig  = gap["signal"]  if gap  else None
        action, strength = combine_signals(orb["signal"], mr["signal"], vwap_sig, gap_sig)

        if action == "SELL":
            pos = data["positions"].get(sym)
            if pos and pos["shares"] > 0:
                gain = (price - pos["avg_cost"]) * pos["shares"]
                print(f"  {sym:<7}  {R}AUTO SELL{Z}  {strength:<8}  "
                      f"{pos['shares']} sh @ {fmt_price(price)}  P&L: {fmt_currency(gain)}")
                if not dry_run:
                    cmd_sell(data, sym, pos["shares"], price)
                sells += 1
            else:
                print(f"  {sym:<7}  {D}SELL signal — not holding{Z}")
                skips += 1

        elif action == "BUY":
            if not _is_actionable_buy(sym, strength, data["positions"]):
                reason = "WEAK signal" if (strength == "WEAK" and not AUTO_WEAK_BUY) else "already holding"
                print(f"  {sym:<7}  {D}BUY ({strength}) — {reason}, skipped{Z}")
                skips += 1
                continue
            shares = _auto_shares(price, per_trade, strength)
            cost   = shares * price
            if shares < 1:
                print(f"  {sym:<7}  {Y}BUY signal — price too high for 1 share within budget{Z}")
                skips += 1
                continue
            if data["cash"] - cost < cash_floor:
                print(f"  {sym:<7}  {Y}BUY signal — would breach cash reserve floor{Z}")
                skips += 1
                continue
            print(f"  {sym:<7}  {G}AUTO BUY {Z}  {strength:<8}  "
                  f"{shares} sh @ {fmt_price(price)}  = {fmt_price(cost)}")
            if not dry_run:
                cmd_buy(data, sym, shares, price)
            buys += 1

        else:
            print(f"  {sym:<7}  {D}{action:<8} — no action{Z}")
            skips += 1

    total_val, pos_val = _portfolio_value(data)
    pnl = total_val - STARTING_CASH
    print("  " + "=" * 76)
    print(f"  Buys: {buys}  |  Sells: {sells}  |  Skipped: {skips}  "
          f"|  Cash: {fmt_price(data['cash'])}")

    if data["positions"]:
        print(f"\n  {'Symbol':<8} {'Shares':>8} {'AvgCost':>10} {'Price':>10} {'MktValue':>12} {'Unrealized':>18}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*18}")
        for sym, pos in data["positions"].items():
            price = get_price(sym)
            mv    = (price or pos["avg_cost"]) * pos["shares"]
            unr   = (price - pos["avg_cost"]) * pos["shares"] if price else 0.0
            pr_s  = fmt_price(price) if price else f"{D}N/A{Z}"
            print(f"  {sym:<8} {pos['shares']:>8.2f} {fmt_price(pos['avg_cost']):>10} "
                  f"{pr_s:>10} {fmt_price(mv):>12} {_ljust(fmt_currency(unr), 27)}")
        print()

    print(f"  Portfolio Value: {fmt_price(total_val)}  "
          f"(Positions: {fmt_price(pos_val)}  |  P&L: {fmt_currency(pnl)})")
    if dry_run:
        print(f"  {Y}Dry run — no orders were placed.{Z}")
    print()


def cmd_watch(data: dict, interval_min: int = 5, symbols: Optional[list] = None,
              dry_run: bool = False):
    import time
    print(f"\n  AUTO-WATCH  interval={interval_min} min  Press Ctrl+C to stop.\n")
    try:
        while True:
            now   = _et_now()
            pre   = pd.Timestamp(f"{now.date()} 09:20", tz="America/New_York")
            close = pd.Timestamp(f"{now.date()} 16:00", tz="America/New_York")

            if now.dayofweek >= 5 or now < pre:
                next_pre = _next_preopen_dt()
                wait_sec = max(0, (next_pre - now).total_seconds())
                print(f"  {Y}Market not open.{Z} Waiting until "
                      f"{next_pre.strftime('%a %Y-%m-%d %H:%M')} ET  "
                      f"({int(wait_sec / 60)} min)  Ctrl+C to stop.")
                time.sleep(wait_sec)
                continue

            if now >= close:
                next_pre = _next_preopen_dt()
                wait_sec = max(0, (next_pre - now).total_seconds())
                print(f"  {R}Market closed.{Z} Resuming at "
                      f"{next_pre.strftime('%a %Y-%m-%d %H:%M')} ET  "
                      f"({int(wait_sec / 60)} min)  Ctrl+C to stop.")
                time.sleep(wait_sec)
                continue

            data = load_portfolio()
            cmd_autotrade(data, symbols=symbols, dry_run=dry_run)

            now       = _et_now()
            close     = pd.Timestamp(f"{now.date()} 16:00", tz="America/New_York")
            remaining = max(0, (close - now).total_seconds())
            sleep_sec = min(interval_min * 60, remaining)

            if sleep_sec < 30:
                print(f"  {Y}Market closing — pausing auto-watch.{Z}")
                continue

            print(f"  Next scan in {interval_min} min...  (market closes 16:00 ET)")
            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("\n  Auto-watch stopped.\n")


def print_help():
    print(f"""
  {B}Portfolio{Z}
    buy <SYM> <SHARES> [PRICE]    Buy shares (live price if omitted)
    sell <SYM> <SHARES> [PRICE]   Sell shares (live price if omitted)
    portfolio                      Holdings, P&L
    history [N]                    Last N trades (default 20)
    reset                          Wipe and restart with ${STARTING_CASH:,.0f}

  {B}Signals{Z}
    scan [SYM SYM ...]             ORB + mean reversion for watchlist (or given symbols)
    signals <SYM>                  Detailed breakdown for one symbol

  {B}Automation{Z}
    autotrade [--dry] [SYM ...]    Scan signals and auto-execute BUY/SELL orders
                                   Force-liquidates delisted holdings; refreshes
                                   watchlist from Nasdaq most-active by volume
    watch [MIN] [--dry] [SYM ...]  Repeat autotrade every MIN minutes (default 5)
                                   Starts automatically at 09:20 ET (10 min pre-open)
                                   Pauses at market close (16:00 ET); resumes next day
    Config: AUTO_POSITION_PCT={AUTO_POSITION_PCT*100:.0f}%  AUTO_CASH_RESERVE={AUTO_CASH_RESERVE*100:.0f}%  AUTO_WEAK_BUY={AUTO_WEAK_BUY}  AUTO_ADD_TO_POS={AUTO_ADD_TO_POS}

  {B}Performance{Z}
    stats                          Win rate, avg win/loss, profit factor, max drawdown
    snapshot                       Save today's portfolio value (builds equity curve)
    equity                         Show equity curve across all snapshots

  {B}Watchlist:{Z}  Auto-populated from Nasdaq most-active by volume each autotrade cycle

  {B}Strategy logic{Z}
    ORB:    breakout above/below the first 30-min range → momentum entry
    MR:     RSI-14 + Bollinger 20,2σ → confirms or filters ORB signal
    VWAP:   intraday fair value; above = bullish bias, below = bearish bias
    Gap:    gap >2% + 1.5× avg volume → confirms or opposes ORB direction
    ** BUY/SELL = signals agree  |  CONFLICT = sit out  |  VWAP/Gap upgrade strength
""")


def interactive_mode(data: dict):
    print(f"\n  Virtual Day Trading Portfolio  (starting cash: {fmt_price(STARTING_CASH)})")
    print('  Type "help" for commands.\n')
    while True:
        try:
            line = input("portfolio> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        parts = line.split()
        cmd   = parts[0].lower()
        if cmd in ("exit", "quit"):
            break
        elif cmd == "help":
            print_help()
        elif cmd == "portfolio":
            cmd_portfolio(data)
        elif cmd == "history":
            cmd_history(data, int(parts[1]) if len(parts) > 1 else 20)
        elif cmd == "reset":
            cmd_reset(); data = load_portfolio()
        elif cmd == "buy":
            if len(parts) < 3: print("  Usage: buy <SYM> <SHARES> [PRICE]"); continue
            cmd_buy(data, parts[1], float(parts[2]), float(parts[3]) if len(parts) > 3 else None)
        elif cmd == "sell":
            if len(parts) < 3: print("  Usage: sell <SYM> <SHARES> [PRICE]"); continue
            cmd_sell(data, parts[1], float(parts[2]), float(parts[3]) if len(parts) > 3 else None)
        elif cmd == "scan":
            cmd_scan(parts[1:] if len(parts) > 1 else None)
        elif cmd == "signals":
            if len(parts) < 2: print("  Usage: signals <SYM>"); continue
            cmd_signals(parts[1])
        elif cmd == "autotrade":
            rest    = parts[1:]
            dry     = "--dry" in rest
            symbols = [s for s in rest if s != "--dry"] or None
            cmd_autotrade(data, symbols=symbols, dry_run=dry)
        elif cmd == "watch":
            rest    = parts[1:]
            dry     = "--dry" in rest
            rest    = [s for s in rest if s != "--dry"]
            try:    interval = int(rest[0]); syms = rest[1:] or None
            except (IndexError, ValueError): interval = 5; syms = rest or None
            cmd_watch(data, interval_min=interval, symbols=syms, dry_run=dry)
        elif cmd == "stats":
            cmd_stats(data)
        elif cmd == "snapshot":
            cmd_snapshot(data)
        elif cmd == "equity":
            cmd_equity()
        else:
            print(f'  Unknown command: "{cmd}". Type "help" for commands.')


def main():
    data = load_portfolio()
    if len(sys.argv) > 1:
        cmd  = sys.argv[1].lower()
        args = sys.argv[2:]
        if cmd == "buy" and len(args) >= 2:
            cmd_buy(data, args[0], float(args[1]), float(args[2]) if len(args) > 2 else None)
        elif cmd == "sell" and len(args) >= 2:
            cmd_sell(data, args[0], float(args[1]), float(args[2]) if len(args) > 2 else None)
        elif cmd == "portfolio":
            cmd_portfolio(data)
        elif cmd == "history":
            cmd_history(data, int(args[0]) if args else 20)
        elif cmd == "scan":
            cmd_scan(args if args else None)
        elif cmd == "signals" and args:
            cmd_signals(args[0])
        elif cmd == "autotrade":
            dry     = "--dry" in args
            symbols = [s for s in args if s != "--dry"] or None
            cmd_autotrade(data, symbols=symbols, dry_run=dry)
        elif cmd == "watch":
            rest = [s for s in args if s != "--dry"]
            dry  = "--dry" in args
            try:    interval = int(rest[0]); syms = rest[1:] or None
            except (IndexError, ValueError): interval = 5; syms = rest or None
            cmd_watch(data, interval_min=interval, symbols=syms, dry_run=dry)
        elif cmd == "stats":
            cmd_stats(data)
        elif cmd == "snapshot":
            cmd_snapshot(data)
        elif cmd == "equity":
            cmd_equity()
        elif cmd == "reset":
            cmd_reset()
        else:
            print_help()
    else:
        interactive_mode(data)


if __name__ == "__main__":
    main()