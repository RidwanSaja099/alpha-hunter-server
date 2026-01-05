import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. ALAT BANTU HITUNG (ULTIMATE V5 - SNIPER MODE)
# ==========================================

def hitung_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def hitung_bollinger(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + (std * 2), sma - (std * 2)

# [BARU] Hitung Bandwidth (Deteksi Squeeze)
def hitung_bollinger_bandwidth(upper, lower, middle):
    return ((upper - lower) / middle) * 100

def hitung_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def hitung_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

# [BARU] RVOL (Relative Volume) - Validasi Volume Ledakan
def hitung_rvol(volume, window=20):
    avg_vol = volume.rolling(window=window).mean()
    rvol = volume / avg_vol
    return rvol

# [BARU] Smart Money Flow (Deteksi Akumulasi Bandar)
def hitung_smart_money_flow(df, period=20):
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']
    # Intraday Intensity Index
    range_hl = (high - low).replace(0, 0.001)
    iii = ((2 * close - high - low) / range_hl) * vol
    smf = iii.rolling(window=period).sum()
    return smf

def hitung_stochastic(high, low, close, k_window=14, d_window=3):
    low_min = low.rolling(window=k_window).min()
    high_max = high.rolling(window=k_window).max()
    denom = (high_max - low_min).replace(0, 0.001)
    k_percent = 100 * ((close - low_min) / denom)
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def hitung_vwap(df):
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).cumsum() / v.cumsum()

def hitung_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx

def hitung_cmf(high, low, close, volume, period=20):
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, 0.001)
    mfv = mfv * volume
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf

def hitung_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def hitung_fibonacci_levels(df, lookback=120):
    recent_high = df['High'].tail(lookback).max()
    recent_low = df['Low'].tail(lookback).min()
    diff = recent_high - recent_low
    return {
        '0.0': recent_low,
        '0.382': recent_low + 0.382 * diff,
        '0.5': recent_low + 0.5 * diff,
        '0.618': recent_low + 0.618 * diff,
        '0.786': recent_low + 0.786 * diff,
        '1.0': recent_high
    }

# DETEKSI POLA CANDLESTICK
def deteksi_candle_pattern(row, prev_row):
    open_p, close_p = row['Open'], row['Close']
    high_p, low_p = row['High'], row['Low']
    
    body = abs(close_p - open_p)
    range_len = high_p - low_p
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    
    pola = []
    if body <= (range_len * 0.1) and range_len > 0: pola.append("Doji")
    if lower_shadow >= (body * 2) and upper_shadow <= (body * 0.5) and range_len > 0: pola.append("Hammer")
    if body > (range_len * 0.85) and range_len > 0:
        if close_p > open_p: pola.append("Bullish Marubozu")
        else: pola.append("Bearish Marubozu")
    if (prev_row['Close'] < prev_row['Open']) and (close_p > open_p): 
        if close_p > prev_row['Open'] and open_p < prev_row['Close']:
            pola.append("Bullish Engulfing")
    return pola

# ==========================================
# 2. FUNGSI AMBIL BERITA
# ==========================================
def ambil_berita_saham(ticker):
    try:
        kode_bersih = ticker.replace(".JK", "")
        if not ticker.endswith(".JK"): ticker += ".JK"
        stock = yf.Ticker(kode_bersih)
        raw_news = stock.news
        berita_bersih = []
        if raw_news:
            for n in raw_news[:5]: 
                link = n.get('link', '')
                if not link: link = f"https://finance.yahoo.com/quote/{kode_bersih}"
                ts = n.get('providerPublishTime', 0)
                try: tgl = datetime.fromtimestamp(ts).strftime('%d %b %H:%M')
                except: tgl = "Terkini"
                berita_bersih.append({
                    "title": n.get('title', 'Update Pasar'),
                    "publisher": n.get('publisher', 'Yahoo Finance'),
                    "link": link,
                    "date": tgl
                })
        if not berita_bersih:
            berita_bersih.append({"title": f"Info {kode_bersih}", "publisher": "System", "link": f"https://finance.yahoo.com/quote/{kode_bersih}", "date": "Now"})
        return berita_bersih
    except: return []

# ==========================================
# 3. OTAK UTAMA: ANALISA MULTI-STRATEGY (MTF + VOLUME + SMART MONEY)
# ==========================================
def analisa_multistrategy(ticker):
    try:
        if not ticker.endswith(".JK"): ticker += ".JK"
        stock = yf.Ticker(ticker)
        
        # --- [MTF] AMBIL DATA DAILY & WEEKLY ---
        df = stock.history(period="1y", interval="1d")
        df_weekly = stock.history(period="2y", interval="1wk")
        info = stock.info 
        
        if df.empty or len(df) < 60: 
            return {"verdict": "SKIP", "reason": "Data Kurang", "score": 0, "type": "UNKNOWN", "last_price": 0, "change_pct": 0, "support": 0}

        # --- DATA MENTAH ---
        last = df.iloc[-1]
        prev = df.iloc[-2]
        last_price = float(last['Close'])
        prev_close = float(prev['Close'])
        change_pct = (last_price - prev_close) / prev_close
        
        open_price = last['Open']
        high_price = last['High']
        low_price = last['Low']
        volume_now = last['Volume']

        # [BARU] GAP DETECTION
        gap_percent = (open_price - prev_close) / prev_close
        is_gap_up = gap_percent > 0.01 # Gap naik > 1%

        # --- INDIKATOR DAILY ---
        sma_5 = df['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) > 200 else 0
        
        # [BARU] GOLDEN CROSS
        prev_sma50 = df['Close'].rolling(window=50).mean().iloc[-2]
        prev_sma200 = df['Close'].rolling(window=200).mean().iloc[-2] if len(df) > 200 else 0
        is_golden_cross = (prev_sma50 < prev_sma200) and (sma_50 > sma_200)

        vol_avg = df['Volume'].rolling(window=20).mean().iloc[-1]
        rsi = hitung_rsi(df['Close']).iloc[-1]
        
        bb_upper, bb_lower = hitung_bollinger(df['Close'])
        last_bb_lower = bb_lower.iloc[-1]
        last_bb_upper = bb_upper.iloc[-1]
        
        # [BARU] SQUEEZE DETECTOR
        bandwidth = hitung_bollinger_bandwidth(bb_upper, bb_lower, sma_20).iloc[-1]
        is_squeeze = bandwidth < 5.0 # Bandwidth sempit < 5%

        macd, macd_signal = hitung_macd(df['Close'])
        last_macd = macd.iloc[-1]
        last_signal = macd_signal.iloc[-1]
        
        # [BARU] RVOL (RELATIVE VOLUME)
        rvol_series = hitung_rvol(df['Volume'])
        last_rvol = rvol_series.iloc[-1] # > 1.5 artinya volume tinggi

        # [BARU] SMART MONEY FLOW
        smf_series = hitung_smart_money_flow(df)
        last_smf = smf_series.iloc[-1]
        is_smart_money_in = last_smf > 0 and last_smf > smf_series.iloc[-2]

        stoch_k, stoch_d = hitung_stochastic(df['High'], df['Low'], df['Close'])
        last_k = stoch_k.iloc[-1]
        last_d = stoch_d.iloc[-1]
        
        vwap_series = hitung_vwap(df)
        last_vwap = vwap_series.iloc[-1]

        adx = hitung_adx(df['High'], df['Low'], df['Close']).iloc[-1]
        cmf = hitung_cmf(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
        atr = hitung_atr(df['High'], df['Low'], df['Close']).iloc[-1]
        fibs = hitung_fibonacci_levels(df)
        pola_candle = deteksi_candle_pattern(last, prev)

        # --- ANALISA WEEKLY ---
        weekly_trend = "NEUTRAL"
        if not df_weekly.empty and len(df_weekly) > 50:
            w_sma_50 = df_weekly['Close'].rolling(window=50).mean().iloc[-1]
            w_close = df_weekly['Close'].iloc[-1]
            if w_close > w_sma_50: weekly_trend = "BULLISH"
            else: weekly_trend = "BEARISH"

        pe = info.get('trailingPE', 100) if info.get('trailingPE') else 100
        pbv = info.get('priceToBook', 10) if info.get('priceToBook') else 10

        # --- FILTER MARKET REGIME ---
        is_trending = adx > 25
        money_inflow = cmf > 0.05
        is_uptrend_ma = last_price > sma_50

        # ==========================================
        # SCORING ENGINE V5 (LEBIH PRESISI)
        # ==========================================
        scores = {"BSJP": 0, "BPJS": 0, "SCALPING": 0, "SWING": 0, "ARA": 0, "INVEST": 0}
        reasons = []

        # 1. SCORE DASAR (WEEKLY TREND)
        if weekly_trend == "BULLISH": 
            for k in scores: scores[k] += 10
            reasons.append("Weekly Uptrend")
        elif weekly_trend == "BEARISH":
            for k in scores: scores[k] -= 10
        
        # 2. SCORE VOLUME & SMART MONEY (VALIDASI UTAMA)
        if last_rvol > 1.5: 
            for k in ["SCALPING", "ARA", "BSJP"]: scores[k] += 15
            reasons.append("Volume Spike")
        if is_smart_money_in:
            for k in ["SWING", "BSJP", "SCALPING"]: scores[k] += 15
            reasons.append("Smart Money In")
        
        # 3. ARA HUNTER (Agresif)
        if change_pct > 0.10 and last_rvol > 2.0: scores["ARA"] += 40
        if is_gap_up and last_price > open_price: scores["ARA"] += 20
        if last_price >= (high_price * 0.98): scores["ARA"] += 25
        if money_inflow: scores["ARA"] += 15

        # 4. SCALPING (Cepat)
        if atr > (last_price * 0.015): scores["SCALPING"] += 20 
        if last_price > last_vwap: scores["SCALPING"] += 30 
        if last_rvol > 1.2: scores["SCALPING"] += 20
        if "Bullish Marubozu" in pola_candle: scores["SCALPING"] += 15

        # 5. SWING (Santai & Trend)
        if is_trending and is_uptrend_ma: scores["SWING"] += 35
        if is_golden_cross: 
            scores["SWING"] += 30
            reasons.append("Golden Cross")
        if is_squeeze:
            scores["SWING"] += 20
            reasons.append("Bollinger Squeeze")
        if last_macd > last_signal: scores["SWING"] += 15
        
        # Pullback Strategy (Buy on Weakness)
        dist_to_ma20 = (last_price - sma_20) / sma_20
        if is_uptrend_ma and -0.03 < dist_to_ma20 < 0.02: 
            scores["SWING"] += 25
            reasons.append("Rebound MA20")

        # 6. BSJP (Beli Sore Jual Pagi)
        if last_price > open_price and last_price >= (high_price * 0.97): scores["BSJP"] += 35
        if last_rvol > 1.0 and is_smart_money_in: scores["BSJP"] += 30
        if last_price > sma_5: scores["BSJP"] += 20

        # 7. BPJS (Beli Pagi Jual Sore)
        if is_gap_up and last_rvol > 2.0: 
            scores["BPJS"] += 40
            reasons.append("Gap Up + Vol Besar")
        if rsi < 30 and "Hammer" in pola_candle: 
            scores["BPJS"] += 30
            reasons.append("Oversold Reversal")
        
        # 8. INVEST
        if pe < 15 and pe > 0: scores["INVEST"] += 25
        if pbv < 1.5: scores["INVEST"] += 25
        if last_price > sma_200: scores["INVEST"] += 30 
        if weekly_trend == "BULLISH": scores["INVEST"] += 20

        # ==========================================
        # KEPUTUSAN FINAL
        # ==========================================
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Filter Bearish: Jika trend turun, kurangi skor swing drastis
        if weekly_trend == "BEARISH" and best_type == "SWING":
            best_score -= 30
            reasons.append("⚠️ Weekly Bearish")

        # SUPPORT & RESISTANCE DINAMIS
        candidates = [sma_20, sma_50, last_bb_lower, last_vwap, fibs['0.5'], fibs['0.618']]
        valid_supports = [x for x in candidates if x < (last_price * 0.995)]
        harga_support = max(valid_supports) if valid_supports else (last_price - (atr * 2))

        # Stop Loss & Target
        stop_loss = harga_support - (atr * 0.5)
        risk = last_price - stop_loss
        target_price = last_price + (risk * 2.5) # RR 1:2.5

        # VERDICT (Threshold diperketat untuk akurasi)
        verdict = "WAIT"
        if best_score >= 90: verdict = "STRONG BUY 🔥"
        elif best_score >= 70: verdict = "BUY ✅"
        elif best_score >= 50: verdict = "NEUTRAL ⚠️"
        else: verdict = "AVOID / SELL"
        
        if is_smart_money_in: reasons.append("Bandar Masuk")

        return {
            "score": int(best_score),
            "verdict": verdict,
            "type": best_type,
            "reason": " | ".join(reasons[:3]), 
            "last_price": int(last_price),
            "change_pct": round(change_pct * 100, 2),
            "support": int(harga_support),
            "stop_loss": int(stop_loss),
            "target_price": int(target_price)
        }

    except Exception as e:
        return {
            "score": 0, "verdict": "ERROR", "type": "ERROR", 
            "reason": str(e), "last_price": 0, "change_pct": 0, "support": 0, "stop_loss":0, "target_price":0
        }
