#!/usr/bin/env python3
"""
scanner.py
GitHub Actions friendly scanner:
- scans CoinGecko ranks (default 40..100)
- uses 1-hour timeframe (hourly OHLCV), RSI(14), simple divergence, ATR
- computes confidence, suggests Entry/SL/TP, sorts by confidence
- sends HTML email (top picks CSV attached)
- token-bucket rate limiter + retries for Actions
Configure secrets in GitHub: EMAIL_FROM, EMAIL_PASS, EMAIL_TO (optional)
"""

import os, time, sys, math, json
from datetime import datetime, timedelta
from pathlib import Path
import requests
import pandas as pd
import numpy as np

# optional exchange orderflow (best-effort)
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    HAS_CCXT = False

# ta indicators
try:
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
except Exception:
    print("Missing 'ta' package. Make sure workflow installs 'ta'.")
    sys.exit(1)

# ---------------- CONFIG (env overrides available) ----------------
RANK_START = int(os.getenv("RANK_START", "40"))
RANK_END   = int(os.getenv("RANK_END", "100"))   # inclusive
DAYS_LOOKBACK = int(os.getenv("DAYS_LOOKBACK", "3"))  # fetch enough history for hourly bars
REQUESTS_PER_MIN = int(os.getenv("REQUESTS_PER_MIN", "40"))
SLEEP_BASE = float(os.getenv("SLEEP_BASE", "1.5"))
TOP_N_EMAIL = int(os.getenv("TOP_N_EMAIL", "5"))
OUTPUT_DIR = Path("scan_output"); OUTPUT_DIR.mkdir(exist_ok=True)
RAW_DIR = Path("raw"); RAW_DIR.mkdir(exist_ok=True)

VOL_RATIO_THRESH = float(os.getenv("VOL_RATIO_THRESH", "1.5"))
RSI_WINDOW = int(os.getenv("RSI_WINDOW", "14"))

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO") or EMAIL_FROM

# Confidence weights
W_ORDERFLOW = float(os.getenv("W_ORDERFLOW", "0.5"))
W_VOL       = float(os.getenv("W_VOL", "0.3"))
W_RSI       = float(os.getenv("W_RSI", "0.15"))
W_DIV_BONUS  = float(os.getenv("W_DIV_BONUS", "0.05"))
# ------------------------------------------------------------------

HEADERS = {"Accept":"application/json", "User-Agent":"gh-scanner-v2/1.0"}

# Token-bucket limiter
class TokenBucket:
    def __init__(self, rate_per_min):
        self.capacity = rate_per_min
        self.tokens = rate_per_min
        self.refill_interval = 60.0 / rate_per_min
        self.last = time.time()
    def take(self):
        now = time.time()
        elapsed = now - self.last
        if elapsed > 0:
            add = elapsed / self.refill_interval
            self.tokens = min(self.capacity, self.tokens + add)
            self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return
        wait = (1 - self.tokens) * self.refill_interval
        time.sleep(wait + 0.05)
        self.tokens = max(0, self.tokens - 1)
        self.last = time.time()

bucket = TokenBucket(REQUESTS_PER_MIN)
session = requests.Session()
session.headers.update(HEADERS)
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))

def safe_get(url, params=None, timeout=20):
    bucket.take()
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        wait = int(ra) if ra and ra.isdigit() else 10
        print(f"[429] sleeping {wait}s then retry")
        time.sleep(wait)
        bucket.take()
        r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# --- CoinGecko helpers (hourly) ---
def fetch_market_coins(top_n_needed):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    per_page = 100
    coins = []
    page = 1
    while len(coins) < top_n_needed:
        params = {"vs_currency":"usd","order":"market_cap_desc","per_page":per_page,"page":page,"sparkline":"false"}
        data = safe_get(url, params=params)
        if not data: break
        coins.extend(data)
        if len(data) < per_page: break
        page += 1
        time.sleep(0.1)
    return coins

def fetch_market_chart(coin_id, days=DAYS_LOOKBACK):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":days}
    return safe_get(url, params=params)

def build_hourly_df(chart_json):
    prices = chart_json.get("prices", [])
    vols   = chart_json.get("total_volumes", [])
    if not prices or not vols: return None
    dfp = pd.DataFrame(prices, columns=["ts","price"])
    dfv = pd.DataFrame(vols, columns=["ts","volume"])
    dfp['ts'] = pd.to_datetime(dfp['ts'], unit='ms')
    dfv['ts'] = pd.to_datetime(dfv['ts'], unit='ms')
    merged = pd.merge_asof(dfp.sort_values('ts'), dfv.sort_values('ts'), on='ts', direction='nearest', tolerance=pd.Timedelta("5m"))
    merged.set_index('ts', inplace=True)
    hh = merged['price'].resample('1H').ohlc().join(merged['volume'].resample('1H').sum())
    hh.columns = ['open','high','low','close','volume']
    hh = hh.dropna()
    return hh

# --- Indicators & signals ---
def compute_rsi(series, window=RSI_WINDOW):
    return RSIIndicator(series, window=window).rsi()

def detect_divergence(df, rsi):
    tmp = df.copy(); tmp['rsi'] = rsi
    lows = tmp['low'].rolling(3, center=True).apply(lambda arr: 1 if arr[1] == arr.min() else 0).dropna()
    highs= tmp['high'].rolling(3, center=True).apply(lambda arr: 1 if arr[1] == arr.max() else 0).dropna()
    low_idxs = [i for i,v in lows.items() if v==1]
    high_idxs = [i for i,v in highs.items() if v==1]
    if len(low_idxs) >= 2:
        p1,p2 = low_idxs[-2], low_idxs[-1]
        if tmp.loc[p2,'low'] < tmp.loc[p1,'low'] and tmp.loc[p2,'rsi'] > tmp.loc[p1,'rsi']:
            return "bullish_div"
    if len(high_idxs) >= 2:
        p1,p2 = high_idxs[-2], high_idxs[-1]
        if tmp.loc[p2,'high'] > tmp.loc[p1,'high'] and tmp.loc[p2,'rsi'] < tmp.loc[p1,'rsi']:
            return "bearish_div"
    return None

def decide_signal(df_hourly):
    if df_hourly is None or len(df_hourly) < 30:
        return None
    rsi = compute_rsi(df_hourly['close'])
    last_rsi = float(rsi.iloc[-1])
    vol_ratio = float(df_hourly['volume'].iloc[-1]) / float(df_hourly['volume'].iloc[-25:-1].mean()) if len(df_hourly) >= 25 else 0.0
    try:
        price_change_24h = (df_hourly['close'].iloc[-1] - df_hourly['close'].iloc[-24]) / df_hourly['close'].iloc[-24]
    except Exception:
        price_change_24h = 0.0
    divergence = detect_divergence(df_hourly, rsi)
    signal = "neutral"; reasons=[]
    if divergence == "bullish_div":
        signal="bullish"; reasons.append("rsi_div")
    if divergence == "bearish_div":
        signal="bearish"; reasons.append("rsi_div")
    if last_rsi < 35 and vol_ratio >= VOL_RATIO_THRESH and price_change_24h >= 0.02:
        signal="bullish"; reasons.append("rsi_oversold+vol")
    if last_rsi > 65 and vol_ratio >= VOL_RATIO_THRESH and price_change_24h <= -0.02:
        signal="bearish"; reasons.append("rsi_overbought+vol")
    if signal == "neutral":
        return None
    return {"rsi": last_rsi, "vol_ratio": vol_ratio, "price_change_24h": price_change_24h, "divergence": divergence, "signal": signal, "reasons": reasons}

# --- Orderflow (best-effort) ---
def fetch_orderflow_delta(symbol):
    if not HAS_CCXT:
        return None
    ex = ccxt.binance({'enableRateLimit': True, 'timeout': 20000})
    pair = f"{symbol}/USDT"
    try:
        since = int((datetime.utcnow() - timedelta(minutes=30)).timestamp() * 1000)
        trades = ex.fetch_trades(pair, since=since, limit=500)
    except Exception:
        return None
    buy = sum([t.get('amount',0) for t in trades if t.get('side') == 'buy'])
    sell= sum([t.get('amount',0) for t in trades if t.get('side') == 'sell'])
    return buy - sell

# --- Confidence & trade suggestion ---
def compute_confidence(df):
    df = df.copy()
    if 'orderflow_delta' in df.columns and not df['orderflow_delta'].isna().all():
        max_of = df['orderflow_delta'].abs().max()
        df['of_norm'] = df['orderflow_delta'].abs() / (max_of if max_of and max_of>0 else 1.0)
    else:
        df['of_norm'] = 0.0
    df['vol_norm'] = df['vol_ratio'].fillna(0).apply(lambda x: min(x/3.0, 1.0))
    def rsi_score(row):
        r = row.get('rsi', None)
        if r is None or math.isnan(r): return 0.0
        if row['signal'] == 'bullish':
            return max(0.0, min((40.0 - r)/20.0, 1.0))
        else:
            return max(0.0, min((r - 60.0)/20.0, 1.0))
    df['rsi_score'] = df.apply(rsi_score, axis=1)
    df['div_bonus'] = df['divergence'].notna().astype(float) * W_DIV_BONUS
    df['raw_conf'] = (W_ORDERFLOW * df['of_norm']) + (W_VOL * df['vol_norm']) + (W_RSI * df['rsi_score']) + df['div_bonus']
    max_raw = df['raw_conf'].max() if len(df) > 0 else 1.0
    df['confidence'] = (df['raw_conf'] / (max_raw if max_raw>0 else 1.0) * 100).round(2)
    return df

def add_trade_suggestions(df):
    df = df.copy()
    atrs=[]; entries=[]; sls=[]; tp1s=[]; tp2s=[]
    for _, row in df.iterrows():
        coin_id = row['id']
        try:
            chart = fetch_market_chart(coin_id, days=DAYS_LOOKBACK)
            df_h = build_hourly_df(chart)
            atr_series = AverageTrueRange(df_h['high'], df_h['low'], df_h['close'], window=14).average_true_range()
            atr = float(atr_series.iloc[-1]) if len(atr_series)>0 else None
        except Exception:
            atr = None
        atrs.append(atr)
        entry = float(row.get('close')) if 'close' in row and row.get('close') is not None else None
        if atr is None or entry is None:
            sl=tp1=tp2=None
        else:
            if row['signal'] == 'bullish':
                sl = entry - atr; tp1 = entry + 2*atr; tp2 = entry + 3*atr
            else:
                sl = entry + atr; tp1 = entry - 2*atr; tp2 = entry - 3*atr
        entries.append(entry); sls.append(sl); tp1s.append(tp1); tp2s.append(tp2)
    df['ATR'] = atrs; df['Entry'] = entries; df['SL'] = sls; df['TP1'] = tp1s; df['TP2'] = tp2s
    def rr_calc(r):
        try:
            risk = abs(r['Entry'] - r['SL'])
            reward = abs(r['TP2'] - r['Entry'])
            return round(reward / risk, 2) if risk and risk>0 else None
        except Exception:
            return None
    df['RR'] = df.apply(rr_calc, axis=1)
    return df

# --- Email (HTML + CSV) ---
def send_html_email(subject, df_top, csv_bytes):
    if not EMAIL_FROM or not EMAIL_PASS or not EMAIL_TO:
        print("Email not configured; skipping email.")
        return False
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication
    html_table = df_top.to_html(index=False, float_format="%.6f")
    html_body = f"<html><body><h3>Crypto Scanner — Top {len(df_top)} picks</h3><p>UTC {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>{html_table}</body></html>"
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM; msg['To'] = EMAIL_TO; msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    part = MIMEApplication(csv_bytes, Name="top_picks.csv")
    part['Content-Disposition'] = 'attachment; filename="top_picks.csv"'
    msg.attach(part)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=60) as s:
            s.login(EMAIL_FROM, EMAIL_PASS)
            s.send_message(msg)
        print("Email sent to", EMAIL_TO); return True
    except Exception as e:
        print("Email failed:", e); return False

# --- Main runner ---
def main():
    print("Scanner start:", datetime.utcnow().isoformat())
    desired_top = RANK_END
    coins = fetch_market_coins(desired_top)
    start_idx = max(0, RANK_START - 1)
    end_idx = min(len(coins), RANK_END)
    universe = coins[start_idx:end_idx]
    rows=[]
    for c in universe:
        coin_id = c.get('id'); symbol = c.get('symbol','').upper()
        try:
            chart = fetch_market_chart(coin_id, days=DAYS_LOOKBACK)
            df_h = build_hourly_df(chart)
            if df_h is None or len(df_h)<30:
                time.sleep(SLEEP_BASE); continue
            sig = decide_signal(df_h)
            if not sig:
                time.sleep(SLEEP_BASE); continue
            close_price = float(df_h['close'].iloc[-1])
            of_delta = None
            try:
                of_delta = fetch_orderflow_delta(symbol)
            except Exception:
                of_delta = None
            row = {"id": coin_id, "symbol": symbol, "close": close_price,
                   "signal": sig['signal'], "rsi": round(sig['rsi'],2), "vol_ratio": round(sig['vol_ratio'],3),
                   "price_change_24h": round(sig['price_change_24h'],4), "divergence": sig['divergence'],
                   "reasons": ";".join(sig['reasons'])}
            row['orderflow_delta'] = of_delta if of_delta is not None else np.nan
            rows.append(row)
        except Exception as e:
            print("Error", coin_id, e)
        time.sleep(SLEEP_BASE)

    if not rows:
        print("No signals found."); return

    df = pd.DataFrame(rows)
    df = compute_confidence(df)
    df['close'] = df['close'].astype(float)
    df = add_trade_suggestions(df)
    df = df.sort_values('confidence', ascending=False).reset_index(drop=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"scan_{RANK_START}_{RANK_END}_{ts}.csv"
    df.to_csv(out_file, index=False)
    print("Saved:", out_file)

    top_df = df.head(TOP_N_EMAIL)[['symbol','signal','Entry','SL','TP1','TP2','RR','confidence','reasons']]
    csv_bytes = top_df.to_csv(index=False).encode('utf-8')
    if len(top_df)>0:
        subj = f"[Scanner] Top {len(top_df)} picks — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        send_html_email(subj, top_df, csv_bytes)

    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
